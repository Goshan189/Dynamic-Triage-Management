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
import threading

# ── path setup so 'graph', 'ml', 'simulation', 'visualisation' are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hospital Routing Simulation")
    p.add_argument("--live",      action="store_true", help="Show live animated graph")
    p.add_argument("--headless",  action="store_true", help="Run headless (no GUI)")
    p.add_argument(
        "--interactive", action="store_true", help="Enable interactive terminal for manual patient injections."
    )
    p.add_argument("--duration",  type=float, default=480.0,
                   help="Simulation duration in minutes (default: 480)")
    p.add_argument("--speed",     type=float, default=0.6,
                   help="Real-time seconds per simulation minute (default: 0.6)")
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

    # ── interactive terminal daemon ───────────────────────────────────────────
    if args.interactive:
        def interactive_loop():
            print("\n" + "="*50)
            print(" INTERACTIVE MODE ENABLED")
            print(" Commands:")
            print("  spawn         -> Random synthetic patient")
            print("  spawn high    -> Force High-Risk patient")
            print("  spawn row <N> -> Inject patient from data.csv row N")
            print("  surge <N>     -> Queue N random patients instantly")
            print("  ambient <0-1> -> Fills hospital with low-priority non-emergency patients")
            print("="*50 + "\n")
            while True:
                try:
                    cmd = input(">>> ").strip().lower()
                    if not cmd:
                        continue
                    if cmd == "spawn":
                        engine.inject_patient()
                    elif cmd == "spawn high":
                        engine.inject_patient(priority="high")
                    elif cmd.startswith("spawn row "):
                        try:
                            row_idx = int(cmd.split()[-1])
                            bridge = engine._bridge
                            v = bridge.get_patient_row(row_idx)
                            engine.inject_patient(vitals=v)
                        except ValueError:
                            print("Invalid row number.")
                    elif cmd.startswith("surge "):
                        try:
                            n = int(cmd.split()[-1])
                            for _ in range(n):
                                engine.inject_patient()
                        except ValueError:
                            print("Invalid surge amount.")
                    elif cmd.startswith("ambient "):
                        try:
                            target_density = float(cmd.split()[-1])
                            target_density = max(0.0, min(1.0, target_density))
                            
                            from graph.layout import NODE_DEF_MAP
                            total_capacity = sum(nd.beds_total for nd in NODE_DEF_MAP.values())
                            active = engine._tracker.active_patient_count()
                            target_patients = int(total_capacity * target_density)
                            diff = target_patients - active
                            
                            if diff > 0:
                                print(f"[System] Spawning {diff} general checkup patients to reach {target_density:.0%} capacity.")
                                for _ in range(diff):
                                    engine.inject_patient(priority="low")
                            else:
                                print(f"[System] Density is already at or above target.")
                        except ValueError:
                            print("Invalid ambient value.")
                    else:
                        print("Unknown command. Try 'spawn', 'ambient <0-1>', etc.")
                except Exception as e:
                    print(f"Interactive Error: {e}")

        t = threading.Thread(target=interactive_loop, daemon=True)
        t.start()

    # ── go ────────────────────────────────────────────────────────────────────
    print(f"[5/5] Running {'headless' if args.headless else 'live'} simulation", end="")
    if args.interactive:
        print(" (interactive mode)", end="")
    print(f" ({args.duration:.0f} min)...")

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
