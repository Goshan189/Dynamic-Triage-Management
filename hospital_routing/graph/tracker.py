"""
tracker.py
----------
Patient instance tracking.

Every patient exists in exactly one of two states:
  AT_NODE    — inside a department (affects beds_occupied / queue_length)
  IN_TRANSIT — on a corridor segment (affects edge active_count / density)

Public API:
  register(patient_id, priority, path)  → registers at entrance
  depart(patient_id)                    → moves patient from current node to next edge
  arrive(patient_id)                    → moves patient from edge into next node
  discharge(patient_id)                 → removes patient from system
  reroute(patient_id, new_path)         → replaces remaining path (keeps current position)
  get_state(patient_id) → dict
  all_in_transit() → list of transit info dicts (for visualisation)
  active_patient_count() → int

No imports from simulation/, ml/, or visualisation/.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from graph.graph import HospitalGraph
from graph.layout import EDGE_DEF_MAP, NODE_DEF_MAP


class PatientStatus(Enum):
    AT_NODE    = auto()
    IN_TRANSIT = auto()
    DISCHARGED = auto()


@dataclass
class Patient:
    patient_id:     str
    priority:       str
    path:           List[str]           # full planned path
    path_index:     int = 0             # index of CURRENT node in path
    status:         PatientStatus = PatientStatus.AT_NODE
    current_node:   Optional[str] = None
    transit_src:    Optional[str] = None
    transit_dst:    Optional[str] = None
    transit_start:  float = 0.0         # real time (seconds) transit began
    transit_duration: float = 0.0       # base_time_s for this segment
    arrival_time:   float = 0.0         # sim time (minutes) of first registration
    registration_time: float = 0.0      # real wall time of registration


class PatientTracker:
    """
    Thread-safe tracker for all patient instances.
    Drives HospitalGraph state changes via depart()/arrive().
    """

    def __init__(self, graph: HospitalGraph) -> None:
        self._graph   = graph
        self._lock    = threading.Lock()
        self._patients: Dict[str, Patient] = {}

    # ── registration ──────────────────────────────────────────────────────────
    def register(
        self,
        patient_id:    str,
        priority:      str,
        path:          List[str],
        arrival_time:  float = 0.0,
    ) -> None:
        """
        Register a new patient.  Patient starts AT_NODE at path[0] (entrance).
        Increments queue_length at entrance (no bed — entry node).
        """
        if not path:
            raise ValueError("Path must have at least one node.")
        start_node = path[0]
        p = Patient(
            patient_id=patient_id,
            priority=priority,
            path=path,
            path_index=0,
            status=PatientStatus.AT_NODE,
            current_node=start_node,
            arrival_time=arrival_time,
            registration_time=time.monotonic(),
        )
        with self._lock:
            self._patients[patient_id] = p

        nd = NODE_DEF_MAP[start_node]
        if nd.beds_total > 0:
            self._graph.increment_node_beds(start_node)
        else:
            self._graph.increment_node_queue(start_node)

    # ── movement ──────────────────────────────────────────────────────────────
    def depart(self, patient_id: str) -> Tuple[str, str, float]:
        """
        Move patient from current node onto the next corridor segment.
        Returns (src, dst, base_time_s).
        Removes patient from node, adds to edge.
        """
        with self._lock:
            p = self._patients[patient_id]
            if p.status != PatientStatus.AT_NODE:
                raise RuntimeError(
                    f"Patient {patient_id} is not AT_NODE (status={p.status})"
                )
            if p.path_index >= len(p.path) - 1:
                raise RuntimeError(
                    f"Patient {patient_id} is already at final destination"
                )

            src = p.path[p.path_index]
            dst = p.path[p.path_index + 1]

            # Compute travel time for this segment
            ed = EDGE_DEF_MAP[(src, dst)]
            transit_duration = ed.base_time_s

            # Update patient state
            p.status           = PatientStatus.IN_TRANSIT
            p.current_node     = None
            p.transit_src      = src
            p.transit_dst      = dst
            p.transit_start    = time.monotonic()
            p.transit_duration = transit_duration

        # Release from source node
        nd = NODE_DEF_MAP[src]
        if nd.beds_total > 0:
            self._graph.decrement_node_beds(src)
        else:
            self._graph.decrement_node_queue(src)

        # Add to edge
        self._graph.increment_edge(src, dst)

        return src, dst, transit_duration

    def arrive(self, patient_id: str) -> str:
        """
        Move patient from corridor into destination node.
        Returns destination node name.
        Removes from edge, adds to node, advances path_index.
        """
        with self._lock:
            p = self._patients[patient_id]
            if p.status != PatientStatus.IN_TRANSIT:
                raise RuntimeError(
                    f"Patient {patient_id} is not IN_TRANSIT (status={p.status})"
                )
            src = p.transit_src
            dst = p.transit_dst

            # Advance path index
            p.path_index   += 1
            p.status        = PatientStatus.AT_NODE
            p.current_node  = dst
            p.transit_src   = None
            p.transit_dst   = None

        # Remove from edge immediately (corridor clears)
        self._graph.decrement_edge(src, dst)

        # Add to destination node
        nd = NODE_DEF_MAP[dst]
        if nd.beds_total > 0:
            self._graph.increment_node_beds(dst)
        else:
            self._graph.increment_node_queue(dst)

        return dst

    # ── reroute ───────────────────────────────────────────────────────────────
    def reroute(self, patient_id: str, new_path_from_current: List[str]) -> None:
        """
        Replace remaining path starting from the patient's current position.
        Patient must be AT_NODE.
        new_path_from_current must start with the patient's current node.
        """
        with self._lock:
            p = self._patients[patient_id]
            if p.status != PatientStatus.AT_NODE:
                raise RuntimeError(
                    f"Cannot reroute patient {patient_id} while IN_TRANSIT"
                )
            if not new_path_from_current:
                raise ValueError("New path must not be empty.")
            if new_path_from_current[0] != p.current_node:
                raise ValueError(
                    f"New path must start at current node {p.current_node}, "
                    f"got {new_path_from_current[0]}"
                )
            # Preserve the already-traversed prefix
            prefix = p.path[: p.path_index]
            p.path       = prefix + new_path_from_current
            p.path_index = len(prefix)

    # ── discharge ─────────────────────────────────────────────────────────────
    def discharge(self, patient_id: str) -> None:
        """
        Remove patient from system.  Patient must be AT_NODE at their final
        destination (last element of path).
        """
        with self._lock:
            p = self._patients.get(patient_id)
            if p is None:
                return
            if p.status == PatientStatus.AT_NODE and p.current_node:
                node = p.current_node
                p.status       = PatientStatus.DISCHARGED
                p.current_node = None
            elif p.status == PatientStatus.IN_TRANSIT:
                # Emergency discharge mid-transit — clear edge
                src, dst = p.transit_src, p.transit_dst
                p.status = PatientStatus.DISCHARGED
            else:
                del self._patients[patient_id]
                return

        if p.status == PatientStatus.DISCHARGED:
            if 'node' in dir():           # AT_NODE branch
                nd = NODE_DEF_MAP[node]
                if nd.beds_total > 0:
                    self._graph.decrement_node_beds(node)
                else:
                    self._graph.decrement_node_queue(node)
            del self._patients[patient_id]

    # ── queries ───────────────────────────────────────────────────────────────
    def get_state(self, patient_id: str) -> Optional[Dict]:
        with self._lock:
            p = self._patients.get(patient_id)
            if p is None:
                return None
            return {
                "patient_id":       p.patient_id,
                "priority":         p.priority,
                "status":           p.status.name,
                "current_node":     p.current_node,
                "transit_src":      p.transit_src,
                "transit_dst":      p.transit_dst,
                "path":             list(p.path),
                "path_index":       p.path_index,
                "transit_start":    p.transit_start,
                "transit_duration": p.transit_duration,
            }

    def all_in_transit(self) -> List[Dict]:
        """Return list of patients currently IN_TRANSIT (for visualisation)."""
        now = time.monotonic()
        with self._lock:
            result = []
            for p in self._patients.values():
                if p.status == PatientStatus.IN_TRANSIT:
                    elapsed  = now - p.transit_start
                    progress = min(1.0, elapsed / p.transit_duration) if p.transit_duration > 0 else 0.0
                    result.append({
                        "patient_id": p.patient_id,
                        "priority":   p.priority,
                        "src":        p.transit_src,
                        "dst":        p.transit_dst,
                        "progress":   progress,
                    })
        return result

    def active_patient_count(self) -> int:
        with self._lock:
            return sum(
                1 for p in self._patients.values()
                if p.status != PatientStatus.DISCHARGED
            )

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._patients.keys())
