"""
router.py
---------
Dijkstra-based patient router with priority-aware weight selection and
reasoning generation.

Public API:
    Router.compute_route(
        src:       str,          # current node
        priority:  str,          # "high" | "medium" | "low"
        graph:     HospitalGraph,
    ) -> RouteResult(path, eta_seconds, reasoning)

    Router.needs_reroute(
        current_node: str,
        remaining_path: List[str],
        priority: str,
        graph: HospitalGraph,
    ) -> bool

Imports: graph/ only.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graph.graph import HospitalGraph
from graph.layout import (
    AVOIDANCE_THRESHOLD,
    DENSITY_RED,
    EDGE_DEF_MAP,
    NODE_DEF_MAP,
    PRIORITY_DESTINATIONS,
    WEIGHT_PENALTY,
)


@dataclass
class RouteResult:
    path:        List[str]
    eta_seconds: float
    reasoning:   str


class Router:
    """
    Stateless router.  All methods are classmethods for easy use.
    """

    # ── main entry point ──────────────────────────────────────────────────────
    @classmethod
    def compute_route(
        cls,
        src:      str,
        priority: str,
        graph:    HospitalGraph,
        preferred_dest: Optional[str] = None,
    ) -> RouteResult:
        """
        Compute the optimal route from src to any valid destination for this
        priority level.

        Returns the best RouteResult among all possible destinations.
        If preferred_dest is given, try it first; fall back to all valid dests.
        """
        valid_dests = PRIORITY_DESTINATIONS[priority]

        # Filter out full destinations
        available_dests = []
        for d in valid_dests:
            nd = NODE_DEF_MAP[d]
            ns = graph.node_states[d]
            # beds_total == 0 means it's a queue without a hard limit
            if nd.beds_total == 0 or ns.beds_occupied < nd.beds_total:
                available_dests.append(d)

        if not available_dests:
            # Fallback to triage waiting room
            available_dests = ["triage"]

        # Build list of destinations to try
        if preferred_dest and preferred_dest in available_dests:
            dests_to_try = [preferred_dest] + [
                d for d in available_dests if d != preferred_dest
            ]
        else:
            dests_to_try = list(available_dests)

        best: Optional[RouteResult] = None

        for dest in dests_to_try:
            if dest == src:
                continue
            result = cls._dijkstra(src, dest, priority, graph)
            if result is None:
                continue
            if best is None or result.eta_seconds < best.eta_seconds:
                best = result
            if preferred_dest and dest == preferred_dest and result is not None:
                # Accept preferred destination unless it's clearly worse
                if best.path[-1] == preferred_dest:
                    break

        if best is None:
            # Fallback: just stay put (should never happen in a connected graph)
            return RouteResult(
                path=[src],
                eta_seconds=0.0,
                reasoning=f"No route found from {src} for priority {priority}.",
            )

        return best

    # ── reroute check ─────────────────────────────────────────────────────────
    @classmethod
    def needs_reroute(
        cls,
        current_node:   str,
        remaining_path: List[str],
        priority:       str,
        graph:          HospitalGraph,
    ) -> bool:
        """
        After arrive(), check if any edge on the remaining path exceeds
        the avoidance threshold for this priority.
        """
        threshold = AVOIDANCE_THRESHOLD[priority]
        for i in range(len(remaining_path) - 1):
            src = remaining_path[i]
            dst = remaining_path[i + 1]
            density = graph.get_edge_density(src, dst)
            if density > threshold:
                return True
        return False

    # ── exit route ────────────────────────────────────────────────────────────
    @classmethod
    def compute_exit_route(
        cls, src: str, priority: str, graph: HospitalGraph, dest: str = "entrance"
    ) -> RouteResult:
        """
        Compute an exit route back to the entrance. Uses low priority routing
        to avoid cutting off critical inbound patients.
        """
        result = cls._dijkstra(src, dest, "low", graph)
        if result is None:
            return RouteResult([src, dest], 0.0, "Forced generic exit route")
        result.reasoning = f"Exit Routing | {result.reasoning}"
        return result

    # ── Dijkstra implementation ───────────────────────────────────────────────
    @classmethod
    def _dijkstra(
        cls,
        src:      str,
        dst:      str,
        priority: str,
        graph:    HospitalGraph,
    ) -> Optional[RouteResult]:
        """
        Single-source Dijkstra from src to dst using priority-aware weights.
        Returns RouteResult or None if no path exists.
        """
        # dist, node, predecessor
        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Optional[str]] = {src: None}
        heap: List[Tuple[float, str]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == dst:
                break
            for v in graph.G.successors(u):
                w = graph.effective_weight(u, v, priority)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dst not in dist:
            return None

        # Reconstruct path
        path: List[str] = []
        cur: Optional[str] = dst
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()

        eta = dist[dst]

        # Build reasoning string
        reasoning = cls._build_reasoning(path, priority, graph, eta)

        return RouteResult(path=path, eta_seconds=eta, reasoning=reasoning)

    # ── reasoning builder ─────────────────────────────────────────────────────
    @classmethod
    def _build_reasoning(
        cls,
        path:     List[str],
        priority: str,
        graph:    HospitalGraph,
        eta:      float,
    ) -> str:
        parts: List[str] = []

        parts.append(f"Priority={priority.upper()}")
        parts.append(f"Route: {' → '.join(path)}")
        parts.append(f"ETA: {eta:.1f}s")

        # Describe weight mode
        penalty = WEIGHT_PENALTY[priority]
        if priority == "high":
            # Check if any edge is catastrophic
            catastrophic = any(
                graph.get_edge_density(path[i], path[i + 1]) >= 0.90
                for i in range(len(path) - 1)
                if (path[i], path[i + 1]) in EDGE_DEF_MAP
            )
            if catastrophic:
                parts.append("Mode: effective weights (catastrophic density detected)")
            else:
                parts.append("Mode: raw base-time weights (high priority)")
        else:
            parts.append(f"Mode: effective weights (penalty×{penalty})")

        # Edge density summary
        edge_summaries: List[str] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) not in EDGE_DEF_MAP:
                continue
            density = graph.get_edge_density(u, v)
            color   = graph.density_color(density)
            edge_summaries.append(f"{u}→{v}:{color}({density:.2f})")
        if edge_summaries:
            parts.append("Edges: " + ", ".join(edge_summaries))

        return " | ".join(parts)
