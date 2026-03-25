================================================================================
  HOSPITAL ROUTING SYSTEM — TECHNICAL SPECIFICATION & DEEP DIVE
================================================================================

This document provides a highly granular, code-level deep dive into the architecture, variable structures, and mathematical algorithms that drive the Hospital Routing Simulation.

================================================================================
1. MACHINE LEARNING & CLINICAL TRIAGE LAYER (`ml_bridge.py`, `ml_triage_model.py`)
================================================================================
The ML layer acts as the initial "Admissions Brain," evaluating incoming patients to calculate priority triage classes and valid physical destinations within the hospital.

1.1. Core Vitals Array (Patient Input Data)
The system processes synthetic patient records containing physiological and behavioral markers. The key variables driving the system are psychometric scores, chosen because standard hospital simulations rarely factor in behavioral emergency metrics:
- `cesd` (Center for Epidemiologic Studies Depression Scale): Measures depression severity (Scale 0-60). Scores > 45 indicate critical intervention (ICU target).
- `stai_t` (State-Trait Anxiety Inventory): Measures systemic anxiety (Scale 20-80). Scores > 50 map to Surgical Wards for isolated physical monitoring.
- `mbi_ex` (Maslach Burnout Inventory - Exhaustion): Measures physical/mental burnout (Scale 0-54). Scores > 40 map to Emergency for acute burnout intervention.
- `health` (Self-Scored Health): Scale 1-5. Scores < 3 trigger routing to the Diagnostics wing (`radiology`, `lab`).

1.2. The XGBoost Engine (`TriageModel.classify`)
- Instead of using raw threshold logic, an XGBoost classifier analyzes the vitals dynamically and outputs an explicit Priority string (`high`, `medium`, `low`) alongside an exact `priority_score` percentage indicating prediction certainty.
- Variable Choice: XGBoost handles non-linear psychometric correlations perfectly (e.g., high anxiety + low health mapping correctly to High Risk trauma, rather than standard OPD).

1.3. Matrix Output Arrays (`get_clinical_allocation`)
- To facilitate Game Theory, the ML engine CANNOT output a rigid single target string (e.g., `preferred = "opd_a"`).
- Instead, the logic calculates an Array Matrix of acceptable clinical sinks (`List[str]`). 
- Why? If a patient only has mild anxiety (`stai_t > 35`), both Outpatient Department A (`opd_a`) and B (`opd_b`) are equally valid. Outputting `["opd_a", "opd_b"]` hands control over to the Dijkstra Game Theory routing engine to mathematically load-balance between the redundant sinks.


================================================================================
2. GAME THEORY TOPOLOGY & FACILITY LOCATION (`graph.py`, `layout.py`)
================================================================================
The hospital is physically structured as a Network Directed Graph. 

2.1. Node Data Structure (`NodeDef`)
Each room in the hospital is constructed using a strictly defined tuple of integers and floats:
- `beds_total` (int): Represents the hard physical limit of a room. If `beds_occupied >= beds_total`, the routing engine physically rejects the path to enforce queue limits.
- `staff` (int): Internal variable scaling service processing limits.
- `pos` (Tuple[float, float]): X/Y Cartesian coordinates rendering the physical geometry onto the Matplotlib GUI.
- `walk_speed_factor` (float): Modifies patient traversal velocity directly within the Edge (e.g., ICU corridors have a 0.7 drag to represent slow-moving Gurneys).

2.2. Facility Location Optimization (`_BASE_EDGES`)
Game Theory is not just about routing algorithms; it is actively built into the structural geometry of the hospital floorplan via Facility Location weighting.
- The `distance_m` variable on every edge dictates baseline algorithmic traversing cost. 
- Why these specific values? The architecture explicitly guarantees High-Risk routing priority by physically slashing the edge-lengths between Acute trauma targets:
  - `entrance -> triage`: 10m
  - `triage -> emergency`: 8m
  - `emergency -> icu`: 12m
  - `emergency -> ward_surgical`: 15m (An explicit "express corridor" carved natively through the graph linking Trauma to Surgery, bypassing Outpatient distances outright).

2.3. Service Timers (`SERVICE_TIMES_MIN`)
- Each node demands an exclusive `service_min` lock when a patient enters. 
- During development, `emergency` was locked at `180.0` minutes, blocking a bed for 3 Sim-Hours and artificially crashing graph capacity. By slashing the emergency lock to `25.0` minutes, High-Risk patients cycle rapidly, keeping the node fluid and departure lists clear.


================================================================================
3. DIJKSTRA & NASH EQUILIBRIUM PATHFINDING (`router.py`)
================================================================================
The `Router` engine calculates mathematical paths from the patient's current (X,Y) to their destination Matrix array.

3.1. Valid Target Filtering (`available_dests`)
- First, the algorithm intersects the ML target `List[str]` against the real-time physical capacities of the network `graph.node_states[d]`. If `beds_occupied < beds_total` is False, the node is stripped from the matrix, automatically diverting capacity-deadlocked traffic.

3.2. Wardrop's User Nash Equilibrium
When evaluating redundant matrices (e.g., choosing dynamically between `opd_a` and `opd_b`), the router doesn't blindly calculate shortest physical distance. It executes a dual-factor Game Theory equation:
- `result.eta_seconds`: The baseline Dijkstra traversal penalty traversing edge weights (`distance_m`).
- `queue_wait_penalty`: Calculated precisely as `queue_len * (base_service_time_seconds / 5.0)`.
- `total_equilibrium_cost`: The sum of the shortest physical distance PLUS the inherent queue service backlog of the destination.
- Why? This ensures true dynamic Nash Equilibrium. If `opd_a` is physically closer (lower `eta_seconds`) but has a massive queue backlog (huge `queue_wait_penalty`), the total cost mathematically forces the routing iteration to select `opd_b`, successfully bleeding density off the congested node and physically load-balancing the hospital perfectly.

3.3. Triage Cascading Fallback Loop
- If the entire valid Matrix evaluates to `beds_total` overflow (maximum capacity lock across the facility), the variable `available_dests` resets to `["triage"]`. The patient physically walks to `triage` and enters a recursive locked generator inside `engine.py`.


================================================================================
4. SIMPY ASYNCHRONOUS PHYSICS CORE (`engine.py`)
================================================================================
The `SimPy` asynchronous generator handles the "Time" physics in the universe.

4.1. Simulation Clock and Execution Speed
- The environment runs on `env.now`, operating entirely chronologically in simulation-minutes.
- The terminal CLI injects `VISUAL_PACE_MULTIPLIER`. 850 simulated seconds might map proportionally to 1 real-life second at 60 FPS plotting, mathematically stretching walking distances into fluid Matplotlib micro-animations.

4.2. Movement Generators (`_patient_process`)
- Yields: In Python, `yield env.timeout(travel_min)` suspends the specific patient's processing thread context while effectively "moving" forward in simulation time. This acts as the physical walking animation timer.
- Dynamic Re-routing listener: At the end of every `yield`, the script executes `Router.needs_reroute(arrived_at, remaining, priority, self._graph)`. If a corridor suddenly flooded while the patient was walking, they calculate a fresh Dijkstra matrix exactly where they are currently standing, overriding their former flight plan.

4.3. Pure Sandbox Terminal Memory Management
- The system actively removed mathematical Poisson distribution (artificial background generation) to establish a Pure Sandbox execution loop. `spawn` or `surge 100` specifically calls the `Tracker` to serialize unique `PID` identifiers into memory. 
- Garbage Paging: As patients hit `Phase 5: Discharge`, `self._tracker.discharge(pid)` physically deletes their underlying class properties from Python's system RAM. This prevents heap exhaustion. However, memory tracking hooks (`_discharged_count += 1`) aggressively pull out and archive their life-cycle math to keep rendering persistent Live Dashboard scores correctly indefinitely.
