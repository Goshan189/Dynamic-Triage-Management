"""
engine.py
---------
SimPy discrete-event simulation engine.

Simulates 480 minutes of hospital operation.
- Poisson arrivals (exponential inter-arrival, mean=3 min)
- Patient type distribution: high=30%, medium=35%, low=35%
- Each patient: vitals → ML → route → move corridor-by-corridor
- Re-routing check at every arrive()
- Collects: avg wait per priority, total reroutes, peak corridor density,
            bed occupancy timeline

Imports: graph/ and ml/ only (no visualisation/).
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import simpy

from graph.graph import HospitalGraph
from graph.layout import (
    AVOIDANCE_THRESHOLD,
    PRIORITY_DESTINATIONS,
    SERVICE_TIMES_MIN,
)
from graph.tracker import PatientTracker
from ml.ml_bridge import MLBridge
from simulation.router import Router


# ──────────────────────────────────────────────────────────────────────────────
# Simulation configuration
# ──────────────────────────────────────────────────────────────────────────────
SIM_DURATION_MIN    = 480.0           # 8-hour shift
ARRIVAL_MEAN_MIN    = 3.0             # Poisson mean inter-arrival (minutes)
PRIORITY_WEIGHTS    = {"high": 0.30, "medium": 0.35, "low": 0.35}
PRIORITIES          = ["high", "medium", "low"]
PRIORITY_WTS_LIST   = [PRIORITY_WEIGHTS[p] for p in PRIORITIES]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SimMetrics:
    wait_times:          Dict[str, List[float]]  = field(default_factory=lambda: defaultdict(list))
    total_reroutes:      int                     = 0
    peak_edge_density:   Dict[Tuple[str,str], float] = field(default_factory=dict)
    bed_timeline:        Dict[str, List[Tuple[float, int]]] = field(default_factory=lambda: defaultdict(list))
    routing_log:         List[Dict]              = field(default_factory=list)
    patients_processed:  int                     = 0


# ──────────────────────────────────────────────────────────────────────────────
# SimPy process
# ──────────────────────────────────────────────────────────────────────────────
class SimulationEngine:

    def __init__(
        self,
        graph:   HospitalGraph,
        tracker: PatientTracker,
        bridge:  MLBridge,
        on_routing_event: Optional[Callable[[Dict], None]] = None,
        seed:   int = 42,
    ) -> None:
        self._graph   = graph
        self._tracker = tracker
        self._bridge  = bridge
        self._metrics = SimMetrics()
        self._rng     = random.Random(seed)
        self._patient_counter = 0
        self._on_routing_event = on_routing_event   # hook for live_view
        self._env: Optional[simpy.Environment] = None

    # ── public ────────────────────────────────────────────────────────────────
    def run(self, duration_min: float = SIM_DURATION_MIN, rt_factor: Optional[float] = None) -> SimMetrics:
        if rt_factor is not None and rt_factor > 0:
            import simpy.rt
            self._env = simpy.rt.RealtimeEnvironment(factor=rt_factor, strict=False)
        else:
            self._env = simpy.Environment()
            
        self._env.process(self._arrival_generator(self._env, duration_min))
        self._env.run(until=duration_min)
        return self._metrics

    @property
    def metrics(self) -> SimMetrics:
        return self._metrics

    @property
    def env(self) -> Optional[simpy.Environment]:
        return self._env

    # ── SimPy generators ──────────────────────────────────────────────────────
    def _arrival_generator(self, env: simpy.Environment, until: float):
        """Poisson arrival stream."""
        while True:
            inter = self._rng.expovariate(1.0 / ARRIVAL_MEAN_MIN)
            yield env.timeout(inter)
            if env.now >= until:
                break
            # Spawn a patient process
            priority = self._rng.choices(PRIORITIES, weights=PRIORITY_WTS_LIST)[0]
            env.process(self._patient_process(env, priority))

    def _patient_process(self, env: simpy.Environment, priority: str):
        """Full patient lifecycle."""
        # 1. Generate vitals and classify
        vitals   = self._bridge.generate_synthetic_vitals()
        result   = self._bridge.classify(vitals)
        priority = result["priority"]   # override with ML result

        # 2. Plan route from entrance
        route = Router.compute_route("entrance", priority, self._graph)
        dest  = route.path[-1]

        # 3. Register patient
        pid = f"P{self._patient_counter:05d}"
        self._patient_counter += 1
        arrival_time = env.now
        self._tracker.register(pid, priority, route.path, arrival_time)

        # Log routing decision
        self._log_route(env.now, pid, priority, route)

        # Snapshot bed timeline for entrance
        self._snapshot_bed_timeline(env.now, route.path[0])

        # 4. Walk corridor by corridor
        path = list(route.path)
        current_idx = 0

        while current_idx < len(path) - 1:
            src = path[current_idx]
            dst = path[current_idx + 1]

            # Depart from current node
            _, _, travel_s = self._tracker.depart(pid)
            travel_min = travel_s / 60.0

            # Update peak edge density
            self._update_peak_density(src, dst)

            # Simulate travel time
            yield env.timeout(travel_min)

            # Arrive at next node
            arrived_at = self._tracker.arrive(pid)
            self._snapshot_bed_timeline(env.now, arrived_at)

            # Check if reroute needed
            state = self._tracker.get_state(pid)
            if state is None:
                return

            remaining = path[current_idx + 1:]
            if Router.needs_reroute(arrived_at, remaining, priority, self._graph):
                new_route = Router.compute_route(
                    arrived_at, priority, self._graph,
                    preferred_dest=path[-1]
                )
                self._tracker.reroute(pid, new_route.path)
                path = self._tracker.get_state(pid)["path"]
                current_idx = self._tracker.get_state(pid)["path_index"]
                self._metrics.total_reroutes += 1
                self._log_route(env.now, pid, priority, new_route, is_reroute=True)
            else:
                current_idx += 1
                # Refresh path in case it was updated
                state = self._tracker.get_state(pid)
                if state:
                    path = state["path"]

        # 5. Service time at final destination
        final_node = path[-1]
        service_min = SERVICE_TIMES_MIN.get(final_node, 30.0)
        service_min = self._rng.expovariate(1.0 / service_min)
        wait_start = env.now
        yield env.timeout(service_min)

        # Record wait time (service time in this simplified model = total wait)
        self._metrics.wait_times[priority].append(service_min)

        # 6. Discharge
        try:
            self._tracker.discharge(pid)
        except Exception:
            pass
        self._snapshot_bed_timeline(env.now, final_node)
        self._metrics.patients_processed += 1

    # ── helpers ───────────────────────────────────────────────────────────────
    def _log_route(
        self,
        sim_time:   float,
        pid:        str,
        priority:   str,
        route:      Any,
        is_reroute: bool = False,
    ) -> None:
        entry = {
            "sim_time":   sim_time,
            "patient_id": pid,
            "priority":   priority,
            "path":       route.path,
            "eta_s":      route.eta_seconds,
            "reasoning":  route.reasoning,
            "is_reroute": is_reroute,
        }
        self._metrics.routing_log.append(entry)
        if self._on_routing_event:
            self._on_routing_event(entry)

    def _update_peak_density(self, src: str, dst: str) -> None:
        d = self._graph.get_edge_density(src, dst)
        key = (src, dst)
        if self._metrics.peak_edge_density.get(key, 0.0) < d:
            self._metrics.peak_edge_density[key] = d

    def _snapshot_bed_timeline(self, sim_time: float, node: str) -> None:
        info = self._graph.zoom_level_3(node)
        self._metrics.bed_timeline[node].append(
            (sim_time, info["beds_occupied"])
        )


# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(metrics: SimMetrics) -> None:
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Patients processed : {metrics.patients_processed}")
    print(f"Total reroutes     : {metrics.total_reroutes}")
    print()
    print("Average service time by priority:")
    for p in PRIORITIES:
        times = metrics.wait_times[p]
        avg = sum(times) / len(times) if times else 0.0
        print(f"  {p:8s}: {avg:.1f} min  (n={len(times)})")
    print()
    print("Peak corridor density:")
    for (src, dst), d in sorted(
        metrics.peak_edge_density.items(), key=lambda x: -x[1]
    )[:10]:
        color = HospitalGraph.density_color(d)
        print(f"  {src:15s} → {dst:15s}: {d:.2f}  [{color}]")
    print("=" * 60)
