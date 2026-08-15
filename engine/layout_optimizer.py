"""
Voltforge AI - Topological Auto-Placement & Schematic Layout Optimizer
Uses force-directed graph optimization and Manhattan crossing minimization for EDA circuits.
"""

import math
from typing import Any, Dict, List, Tuple


class LayoutOptimizer:
    """Calculates optimal (x, y) coordinates for schematic nodes to minimize wire overlaps."""

    GRID_SNAP = 20

    @classmethod
    def optimize_layout(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        canvas_width: int = 1200,
        canvas_height: int = 800,
        iterations: int = 50
    ) -> List[Dict[str, Any]]:
        """Apply force-directed auto-layout to component positions."""
        if not components:
            return []

        # 1. Initialize positions
        positions: Dict[str, Tuple[float, float]] = {}
        for idx, comp in enumerate(components):
            cid = str(comp.get("id") or f"c_{idx}")
            x = float(comp.get("x") or (100 + (idx % 4) * 220))
            y = float(comp.get("y") or (100 + (idx // 4) * 180))
            positions[cid] = (x, y)

        # 2. Build adjacency graph from wires
        adjacency: Dict[str, List[str]] = {cid: [] for cid in positions}
        for w in wires:
            src = str(w.get("fromNodeId") or w.get("from") or "")
            dst = str(w.get("toNodeId") or w.get("to") or "")
            if src in adjacency and dst in adjacency and src != dst:
                adjacency[src].append(dst)
                adjacency[dst].append(src)

        k_repulse = 4000.0
        k_spring = 0.05
        ideal_dist = 180.0

        # 3. Iterative Force Simulation
        for _ in range(iterations):
            forces: Dict[str, Tuple[float, float]] = {cid: (0.0, 0.0) for cid in positions}

            # Repulsive forces (Coulomb)
            cids = list(positions.keys())
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    c1, c2 = cids[i], cids[j]
                    x1, y1 = positions[c1]
                    x2, y2 = positions[c2]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = math.hypot(dx, dy) or 1.0

                    if dist < 400.0:
                        f = k_repulse / (dist * dist)
                        fx = (dx / dist) * f
                        fy = (dy / dist) * f
                        forces[c1] = (forces[c1][0] + fx, forces[c1][1] + fy)
                        forces[c2] = (forces[c2][0] - fx, forces[c2][1] - fy)

            # Attractive forces along wires (Hooke)
            for src, neighbors in adjacency.items():
                x1, y1 = positions[src]
                for dst in neighbors:
                    x2, y2 = positions[dst]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = math.hypot(dx, dy) or 1.0
                    delta = dist - ideal_dist
                    f = k_spring * delta
                    fx = (dx / dist) * f
                    fy = (dy / dist) * f
                    forces[src] = (forces[src][0] + fx, forces[src][1] + fy)

            # Apply displacement with damping & bounds
            for cid in positions:
                fx, fy = forces[cid]
                # Cap displacement per step
                fx = max(-30.0, min(30.0, fx))
                fy = max(-30.0, min(30.0, fy))
                new_x = max(50.0, min(canvas_width - 150.0, positions[cid][0] + fx))
                new_y = max(50.0, min(canvas_height - 150.0, positions[cid][1] + fy))
                positions[cid] = (new_x, new_y)

        # 4. Snap to schematic grid and return updated components
        updated_components: List[Dict[str, Any]] = []
        for idx, comp in enumerate(components):
            cid = str(comp.get("id") or f"c_{idx}")
            x, y = positions.get(cid, (comp.get("x", 0), comp.get("y", 0)))
            snapped_x = round(x / cls.GRID_SNAP) * cls.GRID_SNAP
            snapped_y = round(y / cls.GRID_SNAP) * cls.GRID_SNAP

            new_comp = dict(comp)
            new_comp["x"] = snapped_x
            new_comp["y"] = snapped_y
            updated_components.append(new_comp)

        return updated_components
