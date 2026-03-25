"""
graph.py
--------
Builds the NetworkX directed graph from layout.py definitions.
Manages the weight engine, spillover calculation, and hierarchical
zoom-level reads.

No runtime imports from simulation/, ml/, or visualisation/.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import networkx as nx

from graph.layout import (
    DENSITY_AMBER,
    DENSITY_GREEN,
    DENSITY_ORANGE,
    DENSITY_RED,
    EDGE_DEF_MAP,
    EDGE_DEFS,
    NODE_DEF_MAP,
    NODE_DEFS,
    SPILLOVER_FACTOR,
    WEIGHT_PENALTY,
    ZONES,
    EdgeDef,
    NodeDef,
)


# ──────────────────────────────────────────────────────────────────────────────
# Runtime edge state  (mutable, threadsafe via lock)
# ──────────────────────────────────────────────────────────────────────────────
class EdgeState:
    __slots__ = ("active_count", "density", "effective_weight", "_base")

    def __init__(self, base_time_s: float) -> None:
        self._base           = base_time_s
        self.active_count    = 0
        self.density         = 0.0
        self.effective_weight = base_time_s   # starts equal to base


# ──────────────────────────────────────────────────────────────────────────────
# Runtime node state  (mutable)
# ──────────────────────────────────────────────────────────────────────────────
class NodeState:
    __slots__ = ("beds_occupied", "queue_length", "density_score", "status")

    def __init__(self) -> None:
        self.beds_occupied = 0
        self.queue_length  = 0
        self.density_score = 0.0
        self.status        = "available"


# ──────────────────────────────────────────────────────────────────────────────
# HospitalGraph
# ──────────────────────────────────────────────────────────────────────────────
class HospitalGraph:
    """
    Central graph object.  Wraps a NetworkX DiGraph and exposes:
      - update_edge_density(src, dst)  — recompute after count changes
      - compute_spillover(src, dst)    — adjacency-weighted penalty
      - effective_weight(src, dst, priority) — weight for Dijkstra
      - zoom_level_1()   — zone aggregate density dict
      - zoom_level_2()   — full edge list with live weights
      - zoom_level_3(node) — per-node detail
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self.G            = nx.DiGraph()
        self.node_states: Dict[str, NodeState] = {}
        self.edge_states: Dict[Tuple[str, str], EdgeState] = {}

        self._build()

    # ── construction ──────────────────────────────────────────────────────────
    def _build(self) -> None:
        for nd in NODE_DEFS:
            self.G.add_node(nd.name)
            self.node_states[nd.name] = NodeState()

        for ed in EDGE_DEFS:
            self.G.add_edge(
                ed.src, ed.dst,
                distance_m=ed.distance_m,
                base_time_s=ed.base_time_s,
                corridor_capacity=ed.corridor_capacity,
            )
            self.edge_states[(ed.src, ed.dst)] = EdgeState(ed.base_time_s)

    # ── density update ────────────────────────────────────────────────────────
    def update_edge_density(self, src: str, dst: str) -> None:
        """Recompute density and effective_weight from live active_count."""
        with self._lock:
            es  = self.edge_states[(src, dst)]
            ed  = EDGE_DEF_MAP[(src, dst)]
            es.density = min(1.0, es.active_count / ed.corridor_capacity)

    def _compute_effective_weight(
        self, es: EdgeState, base: float, priority: str
    ) -> float:
        penalty = WEIGHT_PENALTY[priority]
        spillover = self._spillover_unlocked(es)
        return base * (1 + (es.density + spillover) * penalty)

    # ── spillover ─────────────────────────────────────────────────────────────
    def _spillover_unlocked(self, es: EdgeState) -> float:
        """
        Called while lock IS held or not needed (read-only edge states).
        Average density of OTHER edges adjacent to each endpoint,
        multiplied by SPILLOVER_FACTOR.
        """
        return 0.0   # placeholder; real impl below via compute_spillover()

    def compute_spillover(self, src: str, dst: str) -> float:
        """
        Spillover penalty for edge (src→dst):
        average density of all other edges incident to src or dst,
        times SPILLOVER_FACTOR.
        """
        with self._lock:
            adjacent_densities: List[float] = []
            for (u, v), other_es in self.edge_states.items():
                if (u, v) == (src, dst):
                    continue
                if u == src or v == src or u == dst or v == dst:
                    adjacent_densities.append(other_es.density)
            if not adjacent_densities:
                return 0.0
            return (sum(adjacent_densities) / len(adjacent_densities)) * SPILLOVER_FACTOR

    # ── effective weight for Dijkstra ─────────────────────────────────────────
    def effective_weight(self, src: str, dst: str, priority: str) -> float:
        """
        Returns the weight Dijkstra should use for edge (src→dst) given priority.
        HIGH priority: use base_time_s UNLESS density ≥ 0.90 (catastrophic).
        MEDIUM/LOW:    use base × (1 + (density + spillover) × penalty)
        """
        ed = EDGE_DEF_MAP[(src, dst)]
        base = ed.base_time_s
        with self._lock:
            es = self.edge_states[(src, dst)]
            density = es.density

        spillover = self.compute_spillover(src, dst)

        if priority == "high":
            if density >= 0.90:
                # catastrophic — penalise like medium but we still route through
                penalty = WEIGHT_PENALTY["medium"]
                return base * (1 + (density + spillover) * penalty)
            return base   # raw time, ignore density

        penalty = WEIGHT_PENALTY[priority]
        return base * (1 + (density + spillover) * penalty)

    # ── public read/write helpers ─────────────────────────────────────────────
    def increment_edge(self, src: str, dst: str) -> None:
        with self._lock:
            self.edge_states[(src, dst)].active_count += 1
        self.update_edge_density(src, dst)

    def decrement_edge(self, src: str, dst: str) -> None:
        with self._lock:
            es = self.edge_states[(src, dst)]
            es.active_count = max(0, es.active_count - 1)
        self.update_edge_density(src, dst)

    def increment_node_beds(self, node: str) -> None:
        with self._lock:
            ns = self.node_states[node]
            ns.beds_occupied += 1
            self._refresh_node_status(node, ns)

    def decrement_node_beds(self, node: str) -> None:
        with self._lock:
            ns = self.node_states[node]
            ns.beds_occupied = max(0, ns.beds_occupied - 1)
            self._refresh_node_status(node, ns)

    def increment_node_queue(self, node: str) -> None:
        with self._lock:
            self.node_states[node].queue_length += 1

    def decrement_node_queue(self, node: str) -> None:
        with self._lock:
            ns = self.node_states[node]
            ns.queue_length = max(0, ns.queue_length - 1)

    def _refresh_node_status(self, node: str, ns: NodeState) -> None:
        """Update status based on occupancy percentage (beds_total > 0 only)."""
        nd = NODE_DEF_MAP[node]
        if nd.beds_total == 0:
            ns.status = "available"
            ns.density_score = 0.0
            return
        ratio = ns.beds_occupied / nd.beds_total
        ns.density_score = ratio
        if ratio < 0.40:
            ns.status = "available"
        elif ratio < 0.70:
            ns.status = "busy"
        elif ratio < 0.90:
            ns.status = "near_full"
        else:
            ns.status = "full"

    # ── zoom level reads ──────────────────────────────────────────────────────
    def zoom_level_1(self) -> Dict[str, float]:
        """
        Level 1 (hospital view): zone-aggregate density.
        Returns {zone_name: avg_density} based on edge densities within zone.
        """
        zone_densities: Dict[str, List[float]] = {z: [] for z in ZONES}
        with self._lock:
            for (src, dst), es in self.edge_states.items():
                src_zone = self._node_zone(src)
                dst_zone = self._node_zone(dst)
                zone_densities[src_zone].append(es.density)
                if dst_zone != src_zone:
                    zone_densities[dst_zone].append(es.density)
        return {
            z: (sum(vals) / len(vals) if vals else 0.0)
            for z, vals in zone_densities.items()
        }

    def zoom_level_2(self) -> List[Dict]:
        """
        Level 2 (routing view): full edge list with live density and weights.
        """
        result = []
        with self._lock:
            for (src, dst), es in self.edge_states.items():
                ed = EDGE_DEF_MAP[(src, dst)]
                result.append({
                    "src":               src,
                    "dst":               dst,
                    "base_time_s":       ed.base_time_s,
                    "corridor_capacity": ed.corridor_capacity,
                    "active_count":      es.active_count,
                    "density":           es.density,
                })
        return result

    def zoom_level_3(self, node: str) -> Dict:
        """
        Level 3 (department view): per-node detail.
        """
        nd = NODE_DEF_MAP[node]
        with self._lock:
            ns = self.node_states[node]
            return {
                "name":          node,
                "zone":          nd.zone,
                "type":          nd.node_type,
                "beds_total":    nd.beds_total,
                "beds_occupied": ns.beds_occupied,
                "queue_length":  ns.queue_length,
                "staff":         nd.staff,
                "status":        ns.status,
                "density_score": ns.density_score,
                "pos":           nd.pos,
            }

    def get_edge_density(self, src: str, dst: str) -> float:
        with self._lock:
            return self.edge_states[(src, dst)].density

    def get_edge_active_count(self, src: str, dst: str) -> int:
        with self._lock:
            return self.edge_states[(src, dst)].active_count

    @staticmethod
    def density_color(density: float) -> str:
        """Map density float to colour label."""
        if density <= DENSITY_GREEN:
            return "green"
        elif density <= DENSITY_AMBER:
            return "amber"
        elif density <= DENSITY_ORANGE:
            return "orange"
        else:
            return "red"

    @staticmethod
    def _node_zone(node: str) -> str:
        for zone, members in ZONES.items():
            if node in members:
                return zone
        return "unknown"
