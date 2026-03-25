"""
live_view.py
------------
Animated hospital graph visualisation using matplotlib.FuncAnimation.

Passive observer — reads from HospitalGraph and PatientTracker only.
Never writes to graph or tracker state.

Features:
  - Hospital graph: nodes at (x,y) from layout.py
  - Node fill colour = bed occupancy level (green/amber/orange/red)
  - Node label = dept name + beds_occupied/beds_total + queue length
  - Edge colour + stroke width = corridor density (Google Maps style)
  - Moving dots on edges = patients IN_TRANSIT (animated by progress)
  - Dashed zone boundary rectangles
  - Side panel: clock, active patient count, zone density, last 5 routing decisions
  - When a corridor turns red the NEXT patient routed around it takes a blue path

Imports: graph/ and simulation/tracker as read-only.
"""

from __future__ import annotations

import collections
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from graph.graph import HospitalGraph
from graph.layout import (
    DENSITY_AMBER,
    DENSITY_GREEN,
    DENSITY_ORANGE,
    DENSITY_RED,
    EDGE_DEF_MAP,
    EDGE_DEFS,
    NODE_DEF_MAP,
    NODE_DEFS,
    ZONE_BBOX,
    ZONES,
)
from graph.tracker import PatientTracker


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────
def _edge_color(density: float) -> str:
    if density <= DENSITY_GREEN:
        return "#2ecc71"    # green
    elif density <= DENSITY_AMBER:
        return "#f1c40f"    # amber/yellow
    elif density <= DENSITY_ORANGE:
        return "#e67e22"    # orange
    else:
        return "#e74c3c"    # red


def _edge_lw(density: float) -> float:
    """Line width 1.5 → 5.0 proportional to density."""
    return 1.5 + density * 3.5


def _node_face_color(occupancy_ratio: float) -> str:
    if occupancy_ratio < 0.40:
        return "#27ae60"    # green
    elif occupancy_ratio < 0.70:
        return "#f39c12"    # amber
    elif occupancy_ratio < 0.90:
        return "#e67e22"    # orange
    else:
        return "#c0392b"    # red


def _priority_dot_color(priority: str) -> str:
    return {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}.get(priority, "#95a5a6")


# ──────────────────────────────────────────────────────────────────────────────
# LiveView
# ──────────────────────────────────────────────────────────────────────────────
class LiveView:
    """
    Reads live state from HospitalGraph and PatientTracker on every animation
    frame.  Maintains a routing event log for the side panel.
    """

    def __init__(
        self,
        graph:   HospitalGraph,
        tracker: PatientTracker,
        interval_ms: int = 40,
    ) -> None:
        self._graph   = graph
        self._tracker = tracker
        self._interval = interval_ms

        # Routing event queue (written by engine callback)
        self._route_log: collections.deque = collections.deque(maxlen=5)
        self._log_lock  = threading.Lock()

        # Sim clock (set externally)
        self.sim_time_min: float = 0.0
        self.sim_running:  bool  = True

        # Build position map
        self._pos: Dict[str, Tuple[float, float]] = {
            nd.name: nd.pos for nd in NODE_DEFS
        }

        # Set up figure
        self._fig = plt.figure(figsize=(18, 10), facecolor="#0f111a")
        # Main graph axes + side panel
        self._ax_graph = self._fig.add_axes([0.01, 0.01, 0.68, 0.98])
        self._ax_panel = self._fig.add_axes([0.70, 0.01, 0.29, 0.98])

        for ax in (self._ax_graph, self._ax_panel):
            ax.set_facecolor("#0f111a")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        self._ax_graph.set_xlim(-1.2, 12.0)
        self._ax_graph.set_ylim(-0.5, 7.0)
        self._ax_graph.set_aspect("equal")

        # Persistent artists (redrawn each frame)
        self._edge_lines:       Dict[Tuple[str,str], Any] = {}
        self._edge_glow_lines:  Dict[Tuple[str,str], Any] = {}
        self._node_patches:     Dict[str, Any] = {}
        self._node_texts:       Dict[str, Any] = {}
        self._dot_scatter       = None
        self._glow_scatter      = None
        self._anim: Optional[Any] = None

        self._init_static_elements()

    # ── routing event hook ────────────────────────────────────────────────────
    def on_routing_event(self, event: Dict) -> None:
        """Called by the engine when a routing decision is made."""
        with self._log_lock:
            self._route_log.append(event)

    # ── static elements (drawn once setup, updated each frame) ───────────────
    def _init_static_elements(self) -> None:
        ax = self._ax_graph

        # Zone boundary boxes
        zone_colors = {
            "entry":       "#16213e",
            "emergency":   "#2d0a0a",
            "diagnostics": "#0a1a2d",
            "outpatient":  "#0d2e0d",
            "ward":        "#2d2a0a",
        }
        zone_label_colors = {
            "entry":       "#7f8c8d",
            "emergency":   "#e74c3c",
            "diagnostics": "#3498db",
            "outpatient":  "#2ecc71",
            "ward":        "#f39c12",
        }
        for zone, (x0, y0, x1, y1) in ZONE_BBOX.items():
            rect = mpatches.FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0,
                boxstyle="round,pad=0.05",
                linewidth=1.5,
                edgecolor=zone_label_colors.get(zone, "#7f8c8d"),
                facecolor=zone_colors.get(zone, "#16213e"),
                linestyle="--",
                alpha=0.5,
                zorder=0,
            )
            ax.add_patch(rect)
            ax.text(
                (x0 + x1) / 2, y1 + 0.05,
                zone.upper(),
                ha="center", va="bottom",
                fontsize=7, color=zone_label_colors.get(zone, "#7f8c8d"),
                fontweight="bold",
                zorder=1,
            )

        # Edges (initial — updated each frame)
        for ed in EDGE_DEFS:
            sx, sy = self._pos[ed.src]
            dx, dy = self._pos[ed.dst]
            
            # Static background track
            ax.plot(
                [sx, dx], [sy, dy],
                color="#2c3e50", linewidth=5.0,
                alpha=0.25, zorder=1,
                solid_capstyle="round",
            )
            
            # Dynamic glow line
            glow_line, = ax.plot(
                [sx, dx], [sy, dy],
                color="#2ecc71", linewidth=5.5,
                alpha=0.15, zorder=2,
                solid_capstyle="round",
            )
            self._edge_glow_lines[(ed.src, ed.dst)] = glow_line

            # Core line
            line, = ax.plot(
                [sx, dx], [sy, dy],
                color="#2ecc71", linewidth=1.5,
                alpha=0.9, zorder=3,
                solid_capstyle="round",
            )
            self._edge_lines[(ed.src, ed.dst)] = line

            # Arrow annotation at midpoint
            mid_x, mid_y = (sx + dx) / 2, (sy + dy) / 2
            ax.annotate(
                "", xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#4a4a6a",
                    lw=0.8,
                    mutation_scale=8,
                ),
                zorder=4,
            )

        # Nodes
        for nd in NODE_DEFS:
            x, y = nd.pos
            circle = plt.Circle(
                (x, y), 0.35,
                color="#27ae60", zorder=5,
                linewidth=2, edgecolor="#ecf0f1",
            )
            ax.add_patch(circle)
            self._node_patches[nd.name] = circle

            label = ax.text(
                x, y - 0.52,
                self._node_label(nd.name),
                ha="center", va="top",
                fontsize=6.5, color="#ecf0f1",
                zorder=6,
            )
            self._node_texts[nd.name] = label

        # Dot scatter for in-transit patients (empty initially)
        self._glow_scatter = ax.scatter(
            [], [], c=[], s=450, zorder=9,
            edgecolors="none", alpha=0.35,
        )
        self._dot_scatter = ax.scatter(
            [], [], c=[], s=160, zorder=10,
            edgecolors="#ffffff", linewidths=1.2,
        )

        ax.set_title(
            "Hospital Routing — Live View",
            color="#ecf0f1", fontsize=14, pad=8, fontweight="bold",
        )

    # ── animation ─────────────────────────────────────────────────────────────
    def _update_frame(self, frame: int) -> List:
        changed: List[Any] = []

        # -- Update edges --
        for (src, dst), line in self._edge_lines.items():
            glow = self._edge_glow_lines[(src, dst)]
            density = self._graph.get_edge_density(src, dst)
            color   = _edge_color(density)
            lw      = _edge_lw(density)
            
            line.set_color(color)
            line.set_linewidth(lw)
            
            glow.set_color(color)
            glow.set_linewidth(lw + 4.0)
            
            changed.extend([line, glow])

        # -- Update nodes --
        for nd in NODE_DEFS:
            info  = self._graph.zoom_level_3(nd.name)
            ratio = (info["beds_occupied"] / nd.beds_total) if nd.beds_total > 0 else 0.0
            fc    = _node_face_color(ratio)
            patch = self._node_patches[nd.name]
            
            patch.set_facecolor(fc)
            
            if info["status"] == "full":
                patch.set_edgecolor("#ff0000")
                patch.set_linewidth(3 + 3 * np.abs(np.sin(frame * 0.15)))
            else:
                patch.set_edgecolor("#ecf0f1")
                patch.set_linewidth(2)
                
            changed.append(patch)

            # Label
            txt = self._node_texts[nd.name]
            txt.set_text(self._node_label(nd.name))
            changed.append(txt)

        # -- Moving dots --
        transit = self._tracker.all_in_transit()
        xs, ys, colors = [], [], []
        for t in transit:
            sx, sy = self._pos.get(t["src"], (0, 0))
            dx, dy = self._pos.get(t["dst"], (0, 0))
            p = t["progress"]
            # Perpendicular offset so dots don't overlap the edge line
            nx_off = -(dy - sy) * 0.07
            ny_off =  (dx - sx) * 0.07
            xs.append(sx + (dx - sx) * p + nx_off)
            ys.append(sy + (dy - sy) * p + ny_off)
            colors.append(_priority_dot_color(t["priority"]))

        if xs:
            pts = np.c_[xs, ys]
            self._dot_scatter.set_offsets(pts)
            self._dot_scatter.set_facecolor(colors)
            self._glow_scatter.set_offsets(pts)
            self._glow_scatter.set_facecolor(colors)
        else:
            empty = np.empty((0, 2))
            self._dot_scatter.set_offsets(empty)
            self._glow_scatter.set_offsets(empty)
        changed.extend([self._dot_scatter, self._glow_scatter])

        # -- Side panel --
        self._draw_panel()

        return changed

    def _draw_panel(self) -> None:
        ax = self._ax_panel
        ax.cla()
        ax.set_facecolor("#0f111a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        y = 0.97
        lh = 0.035   # line height

        def txt(text, x=0.03, color="#ecf0f1", size=9, bold=False):
            nonlocal y
            ax.text(
                x, y, text,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=size, color=color,
                fontweight="bold" if bold else "normal",
            )
            y -= lh

        txt("HOSPITAL DASHBOARD", size=11, color="#9b59b6", bold=True)
        y -= 0.01

        # Sim clock
        mins = int(self.sim_time_min)
        txt(f"⏱  Sim time: {mins // 60:02d}h {mins % 60:02d}m", color="#3498db")
        txt(f"👥 Active patients: {self._tracker.active_patient_count()}", color="#ecf0f1")
        y -= 0.01

        # Zone density
        txt("ZONE DENSITY", size=8, color="#9b59b6", bold=True)
        zone_densities = self._graph.zoom_level_1()
        zone_colors = {
            "entry": "#7f8c8d", "emergency": "#e74c3c",
            "diagnostics": "#3498db", "outpatient": "#2ecc71",
            "ward": "#f39c12",
        }
        for zone, density in zone_densities.items():
            bar_len = int(density * 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            color_label = (
                "🟢" if density <= DENSITY_GREEN else
                "🟡" if density <= DENSITY_AMBER else
                "🟠" if density <= DENSITY_ORANGE else "🔴"
            )
            txt(
                f"{color_label} {zone[:10]:10s} {bar} {density:.2f}",
                color=zone_colors.get(zone, "#7f8c8d"), size=7.5
            )
        y -= 0.01

        # Department status
        txt("DEPARTMENT STATUS", size=8, color="#9b59b6", bold=True)
        for nd in NODE_DEFS:
            info = self._graph.zoom_level_3(nd.name)
            if nd.beds_total > 0:
                occ_str = f"{info['beds_occupied']:2d}/{nd.beds_total:2d}"
            else:
                occ_str = f"Q:{info['queue_length']:2d}"
            status = info["status"]
            scol = {
                "available": "#2ecc71", "busy": "#f39c12",
                "near_full": "#e67e22", "full": "#e74c3c",
            }.get(status, "#7f8c8d")
            txt(
                f"{nd.name[:14]:14s} {occ_str}  [{status[:8]}]",
                color=scol, size=7,
            )
        y -= 0.01

        # Last 5 routing decisions as Event Feed
        txt("LIVE EVENT FEED", size=8, color="#9b59b6", bold=True)
        with self._log_lock:
            log_entries = list(self._route_log)

        for entry in reversed(log_entries[-6:]):
            p = entry['priority'].upper()
            p_color = _priority_dot_color(entry["priority"])
            t_min = int(entry["sim_time"])
            dst = entry['path'][-1].upper() if entry['path'] else 'UNKNOWN'
            reason = entry.get("reasoning", "")
            
            if entry.get("is_reroute"):
                if "Exit Routing" in reason:
                    msg = f"[{t_min}m] {p} patient successfully discharged."
                else:
                    msg = f"[{t_min}m] ⚠️ REROUTE: {p} patient sent to {dst}."
            else:
                msg = f"[{t_min}m] New {p} risk patient routed to {dst}."

            txt(msg, color=p_color, size=7.5)
            
            if entry.get("is_reroute") and "Exit" not in reason:
                if "|" in reason:
                    shorthand = reason.split("|")[-1].strip()
                    txt(f"      ↳ {shorthand[:42]}", color="#7f8c8d", size=6.5)
            y -= 0.005
            
        y -= 0.02
        txt("PRIORITY LEGEND", size=8, color="#9b59b6", bold=True)
        txt("🔴 High-Risk Patient (Fastest route)", color="#e74c3c", size=7)
        txt("🟠 Medium-Risk Patient", color="#f39c12", size=7)
        txt("🔵 Low-Risk Patient (Ambient crowd)", color="#3498db", size=7)

    def _node_label(self, name: str) -> str:
        nd   = NODE_DEF_MAP[name]
        info = self._graph.zoom_level_3(name)
        if nd.beds_total > 0:
            occ = f"{info['beds_occupied']}/{nd.beds_total}"
        else:
            occ = f"Q:{info['queue_length']}"
        return f"{name}\n{occ}"

    # ── public start ──────────────────────────────────────────────────────────
    def start(self, sim_runner: Optional[Callable] = None) -> None:
        """
        Start the animation.  If sim_runner is provided, it is called in a
        daemon thread so the simulation runs concurrently with the display.
        """
        if sim_runner:
            t = threading.Thread(target=sim_runner, daemon=True)
            t.start()

        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            interval=self._interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
