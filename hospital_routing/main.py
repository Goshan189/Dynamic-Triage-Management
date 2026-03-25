"""
main.py
-------
Full pipeline entry point for the Hospital Routing System.

Usage:
    python main.py               # headless simulation (no GUI)
    python main.py --live        # simulation + live animated graph
    python main.py --headless    # explicit headless mode
    python main.py --duration 60 # custom simulation duration (minutes)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── path setup so 'graph', 'ml', 'simulation', 'visualisation' are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hospital Routing Simulation")
    p.add_argument("--live",      action="store_true", help="Show live animated graph")
    p.add_argument("--headless",  action="store_true", help="Run headless (no GUI)")
    p.add_argument("--duration",  type=float, default=480.0,
                   help="Simulation duration in minutes (default: 480)")
    p.add_argument("--speed",     type=float, default=0.2,
                   help="Real seconds per simulation minute for live view (0 for instant). Default: 0.2")
    p.add_argument("--data-path", type=str, default=None,
                   help="Path to data.csv (auto-detected if omitted)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    live = args.live and not args.headless

    print("=" * 60)
    print("  Hospital Routing System — Dynamic Patient Prioritization")
    print("=" * 60)

    # ── 1. Build graph -----------------------------------------------------
    print("\n[1/5] Building hospital graph...")
    from graph.graph import HospitalGraph
    graph = HospitalGraph()
    print(
        f"      Loaded {graph.G.number_of_nodes()} nodes, "
        f"{graph.G.number_of_edges()} edges."
    )

    # ── 2. Build tracker ---------------------------------------------------
    print("[2/5] Initialising patient tracker...")
    from graph.tracker import PatientTracker
    tracker = PatientTracker(graph)

    # ── 3. Train ML model -------------------------------------------------
    print("[3/5] Training ML triage model (this may take a moment)...")
    t0 = time.perf_counter()
    from ml.ml_bridge import MLBridge
    bridge = MLBridge(data_path=args.data_path)
    dt = time.perf_counter() - t0
    print(
        f"      Model: {bridge.model_name} | "
        f"Accuracy: {bridge.accuracy:.1%} | "
        f"Trained in {dt:.1f}s"
    )

    # ── 4. Build simulation engine -----------------------------------------
    print("[4/5] Setting up simulation engine...")
    from simulation.engine import SimulationEngine, print_summary

    routing_events = []   # shared log (engine writes, view reads)

    def on_routing_event(evt):
        routing_events.append(evt)

    engine = SimulationEngine(
        graph=graph,
        tracker=tracker,
        bridge=bridge,
        on_routing_event=on_routing_event,
    )

    # ── 5. Run ─────────────────────────────────────────────────────────────
    if live:
        print("[5/5] Starting live visualisation (close window to stop)...")
        from visualisation.live_view import LiveView

        view = LiveView(graph, tracker, interval_ms=40)

        # Hook the routing event callback so the side panel stays updated
        engine_on_event = engine._on_routing_event

        def combined_event(evt):
            if engine_on_event:
                engine_on_event(evt)
            view.on_routing_event(evt)
            # Update sim clock reference in the view
            view.sim_time_min = evt.get("sim_time", 0.0)

        engine._on_routing_event = combined_event

        def run_sim():
            rt = args.speed if args.speed > 0 else None
            metrics = engine.run(duration_min=args.duration, rt_factor=rt)
            view.sim_running = False
            print_summary(metrics)

        view.start(sim_runner=run_sim)

    else:
        print(
            f"[5/5] Running headless simulation "
            f"({args.duration:.0f} min)..."
        )
        t0 = time.perf_counter()
        metrics = engine.run(duration_min=args.duration)
        dt = time.perf_counter() - t0
        print(f"      Finished in {dt:.1f}s real time.")
        print_summary(metrics)


if __name__ == "__main__":
    main()
