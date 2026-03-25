"""
layout.py
---------
Nash-equilibrium-derived hospital layout.
All 11 nodes, 17 directed edges, 5 zones.
This module is pure data — no runtime logic, no imports from other
project modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Density thresholds (Google Maps style)
# ──────────────────────────────────────────────────────────────────────────────
DENSITY_GREEN  = 0.30   # ≤ 0.30 → green
DENSITY_AMBER  = 0.55   # ≤ 0.55 → amber
DENSITY_ORANGE = 0.75   # ≤ 0.75 → orange
DENSITY_RED    = 0.90   # ≤ 1.00 → red

# Edge density avoidance thresholds per priority
AVOIDANCE_THRESHOLD = {
    "high":   0.90,   # catastrophic only
    "medium": 0.75,
    "low":    0.55,
}

# Weight penalty multipliers per priority
WEIGHT_PENALTY = {
    "high":   1.2,
    "medium": 3.0,
    "low":    6.0,
}

SPILLOVER_FACTOR = 0.35

# Walking speed (m/s) — used for base_time_s = distance / speed
WALK_SPEED = 1.2  # m/s

# ──────────────────────────────────────────────────────────────────────────────
# Visual Pacing
# Because the simulation scales time linearly, a 20-second walk completes in 
# <0.05 real seconds, making patients visually teleport. We significantly
# inflate travel times so patients have a visible tracking trajectory.
# ──────────────────────────────────────────────────────────────────────────────
VISUAL_PACE_MULTIPLIER = 12.0

# ──────────────────────────────────────────────────────────────────────────────
# Zone definitions
# ──────────────────────────────────────────────────────────────────────────────
ZONES: Dict[str, List[str]] = {
    "entry":       ["entrance", "triage"],
    "emergency":   ["emergency", "icu"],
    "diagnostics": ["radiology", "lab", "pharmacy"],
    "outpatient":  ["opd_a", "opd_b"],
    "ward":        ["ward_surgical", "ward_general"],
}

# Reverse lookup: node → zone
NODE_ZONE: Dict[str, str] = {
    node: zone
    for zone, nodes in ZONES.items()
    for node in nodes
}


# ──────────────────────────────────────────────────────────────────────────────
# Node static definitions
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class NodeDef:
    name:              str
    zone:              str
    node_type:         str
    beds_total:        int
    staff:             int
    walk_speed_factor: float
    pos:               Tuple[float, float]


NODE_DEFS: List[NodeDef] = [
    NodeDef("entrance",      "entry",       "entry",      0,  3, 1.0, (0.0,  3.0)),
    NodeDef("triage",        "entry",       "triage",     0,  6, 1.0, (2.0,  3.0)),
    NodeDef("emergency",     "emergency",   "emergency", 15, 12, 0.9, (4.0,  4.5)),
    NodeDef("icu",           "emergency",   "critical",   8, 10, 0.7, (6.0,  5.5)),
    NodeDef("radiology",     "diagnostics", "diagnostic", 0,  6, 1.0, (4.0,  1.5)),
    NodeDef("lab",           "diagnostics", "diagnostic", 0,  8, 1.0, (6.0,  2.5)),
    NodeDef("pharmacy",      "diagnostics", "support",    0,  5, 1.0, (7.5,  1.5)),
    NodeDef("opd_a",         "outpatient",  "outpatient",20,  8, 1.0, (8.0,  4.5)),
    NodeDef("opd_b",         "outpatient",  "outpatient",20,  8, 1.0, (8.0,  2.0)),
    NodeDef("ward_surgical", "ward",        "ward",      30, 10, 0.9, (10.0, 5.0)),
    NodeDef("ward_general",  "ward",        "ward",      30, 10, 0.9, (10.0, 1.5)),
]

# Quick dict lookup
NODE_DEF_MAP: Dict[str, NodeDef] = {n.name: n for n in NODE_DEFS}


# ──────────────────────────────────────────────────────────────────────────────
# Edge static definitions
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EdgeDef:
    src:              str
    dst:              str
    distance_m:       float
    base_time_s:      float   # = distance_m / WALK_SPEED  (pre-computed)
    corridor_capacity: int


def _edge(src: str, dst: str, dist_m: float, cap: int) -> EdgeDef:
    raw_s = dist_m / WALK_SPEED
    paced_s = raw_s * VISUAL_PACE_MULTIPLIER
    return EdgeDef(src, dst, dist_m, round(paced_s, 1), cap)


_BASE_EDGES: List[EdgeDef] = [
    _edge("entrance",      "triage",         20, 12),
    _edge("triage",        "emergency",       35,  8),
    _edge("triage",        "radiology",       40,  8),
    _edge("triage",        "opd_a",           60, 10),
    _edge("triage",        "opd_b",           55, 10),
    _edge("emergency",     "icu",             30,  6),
    _edge("emergency",     "radiology",       50,  8),
    _edge("emergency",     "lab",             45,  8),
    _edge("radiology",     "lab",             28,  8),
    _edge("lab",           "icu",             40,  6),
    _edge("lab",           "pharmacy",        32,  8),
    _edge("lab",           "opd_b",           38,  8),
    _edge("pharmacy",      "opd_b",           22,  8),
    _edge("pharmacy",      "opd_a",           35,  8),
    _edge("opd_a",         "ward_surgical",   45,  8),
    _edge("opd_b",         "ward_general",    45,  8),
    _edge("ward_surgical", "ward_general",    55,  6),
]

EDGE_DEFS: List[EdgeDef] = []
for ed in _BASE_EDGES:
    EDGE_DEFS.append(ed)
    EDGE_DEFS.append(EdgeDef(ed.dst, ed.src, ed.distance_m, ed.base_time_s, ed.corridor_capacity))

# Quick lookup: (src, dst) → EdgeDef
EDGE_DEF_MAP: Dict[Tuple[str, str], EdgeDef] = {
    (e.src, e.dst): e for e in EDGE_DEFS
}


# ──────────────────────────────────────────────────────────────────────────────
# Priority → valid destination nodes
# ──────────────────────────────────────────────────────────────────────────────
PRIORITY_DESTINATIONS: Dict[str, List[str]] = {
    "high":   ["emergency", "icu"],
    "medium": ["emergency", "radiology", "lab", "opd_a", "opd_b"],
    "low":    ["opd_a", "opd_b", "pharmacy", "ward_surgical", "ward_general"],
}

# Service times in minutes per destination type
SERVICE_TIMES_MIN: Dict[str, float] = {
    "emergency":    180.0,
    "icu":          120.0,
    "ward_surgical": 90.0,
    "ward_general":  90.0,
    "opd_a":         30.0,
    "opd_b":         30.0,
    "radiology":     20.0,
    "lab":           15.0,
    "pharmacy":      10.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Zone bounding boxes for visualisation [(xmin, ymin, xmax, ymax)]
# ──────────────────────────────────────────────────────────────────────────────
ZONE_BBOX: Dict[str, Tuple[float, float, float, float]] = {
    "entry":       (-0.7, 2.2, 3.0, 3.8),
    "emergency":   (3.2, 3.8, 7.2, 6.3),
    "diagnostics": (3.2, 0.7, 8.2, 3.3),
    "outpatient":  (7.2, 1.2, 9.0, 5.3),
    "ward":        (9.2, 0.7, 11.0, 5.8),
}
