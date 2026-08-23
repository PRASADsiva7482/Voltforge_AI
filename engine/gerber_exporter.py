"""
RS-274X Gerber and Excellon drill package export.

This exporter is intentionally small, but it produces a coherent fabrication
bundle from the PCB layout state: copper traces, pads, vias, masks, silkscreen,
board outline, drill file, and a manifest.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any, Dict, Iterable, List, Tuple

Point = Tuple[float, float]


class GerberExporter:
    @classmethod
    def generate_gerber_zip(
        cls,
        board_width_mm: float,
        board_height_mm: float,
        footprints: List[Dict[str, Any]],
        traces: List[Dict[str, Any]],
        vias: List[Dict[str, Any]],
        project_name: str = "VoltForge_PCB",
    ) -> bytes:
        safe_name = cls._safe_name(project_name)
        manifest = {
            "projectName": safe_name,
            "format": "RS-274X",
            "units": "millimeters",
            "boardWidth_mm": board_width_mm,
            "boardHeight_mm": board_height_mm,
            "footprintCount": len(footprints),
            "traceCount": len(traces),
            "viaCount": len(vias),
            "layers": [
                "F.Cu",
                "B.Cu",
                "F.Mask",
                "B.Mask",
                "F.SilkS",
                "B.SilkS",
                "Edge.Cuts",
                "PTH.drl",
            ],
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{safe_name}-F_Cu.gtl", cls._generate_copper_layer(traces, footprints, vias, "F.Cu"))
            archive.writestr(f"{safe_name}-B_Cu.gbl", cls._generate_copper_layer(traces, footprints, vias, "B.Cu"))
            archive.writestr(f"{safe_name}-F_Mask.gts", cls._generate_mask_layer(footprints, vias, "F.Mask"))
            archive.writestr(f"{safe_name}-B_Mask.gbs", cls._generate_mask_layer(footprints, vias, "B.Mask"))
            archive.writestr(f"{safe_name}-F_SilkS.gto", cls._generate_silkscreen_layer(footprints, "F.SilkS"))
            archive.writestr(f"{safe_name}-B_SilkS.gbo", cls._generate_silkscreen_layer(footprints, "B.SilkS"))
            archive.writestr(f"{safe_name}-Edge_Cuts.gml", cls._generate_edge_cuts(board_width_mm, board_height_mm))
            archive.writestr(f"{safe_name}-PTH.drl", cls._generate_drill_file(footprints, vias))
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    @classmethod
    def _generate_copper_layer(
        cls,
        traces: List[Dict[str, Any]],
        footprints: List[Dict[str, Any]],
        vias: List[Dict[str, Any]],
        layer_name: str,
    ) -> str:
        lines = cls._gerber_header(f"Copper {layer_name}")
        lines.extend([
            "%ADD10C,0.1524*%",
            "%ADD11C,0.2540*%",
            "%ADD12C,0.6000*%",
            "%ADD20R,1.4000X1.4000*%",
            "%ADD21C,1.4000*%",
            "D11*",
        ])

        aperture_by_width = {0.1524: "D10", 0.254: "D11", 0.6: "D12"}
        for trace in traces:
            if trace.get("layer", "F.Cu") != layer_name:
                continue
            width = cls._float(trace.get("width_mm"), 0.254)
            aperture = cls._nearest_aperture(width, aperture_by_width)
            points = cls._points(trace.get("points", []))
            if len(points) < 2:
                continue
            lines.append(f"{aperture}*")
            lines.append(cls._coord(points[0], "D02"))
            for point in points[1:]:
                lines.append(cls._coord(point, "D01"))

        lines.append("D20*")
        for footprint in footprints:
            fx = cls._float(footprint.get("x"), 0)
            fy = cls._float(footprint.get("y"), 0)
            for pad in footprint.get("pads", []):
                shape = str(pad.get("shape", "rect")).lower()
                lines.append("D21*" if shape == "circle" else "D20*")
                lines.append(cls._coord((fx + cls._float(pad.get("x"), 0), fy + cls._float(pad.get("y"), 0)), "D03"))

        lines.append("D12*")
        for via in vias:
            lines.append(cls._coord((cls._float(via.get("x"), 0), cls._float(via.get("y"), 0)), "D03"))

        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_mask_layer(
        cls,
        footprints: List[Dict[str, Any]],
        vias: List[Dict[str, Any]],
        layer_name: str,
    ) -> str:
        lines = cls._gerber_header(layer_name)
        lines.extend([
            "%ADD20R,1.7000X1.7000*%",
            "%ADD21C,1.7000*%",
            "%ADD22C,0.9000*%",
        ])

        for footprint in footprints:
            fx = cls._float(footprint.get("x"), 0)
            fy = cls._float(footprint.get("y"), 0)
            for pad in footprint.get("pads", []):
                shape = str(pad.get("shape", "rect")).lower()
                lines.append("D21*" if shape == "circle" else "D20*")
                lines.append(cls._coord((fx + cls._float(pad.get("x"), 0), fy + cls._float(pad.get("y"), 0)), "D03"))

        lines.append("D22*")
        for via in vias:
            lines.append(cls._coord((cls._float(via.get("x"), 0), cls._float(via.get("y"), 0)), "D03"))

        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_silkscreen_layer(cls, footprints: List[Dict[str, Any]], layer_name: str) -> str:
        lines = cls._gerber_header(layer_name)
        lines.extend(["%ADD10C,0.1500*%", "D10*"])
        for footprint in footprints:
            fx = cls._float(footprint.get("x"), 0)
            fy = cls._float(footprint.get("y"), 0)
            fw = cls._float(footprint.get("width"), 10)
            fh = cls._float(footprint.get("height"), 10)
            corners = [
                (fx - fw / 2, fy - fh / 2),
                (fx + fw / 2, fy - fh / 2),
                (fx + fw / 2, fy + fh / 2),
                (fx - fw / 2, fy + fh / 2),
                (fx - fw / 2, fy - fh / 2),
            ]
            lines.append(cls._coord(corners[0], "D02"))
            for corner in corners[1:]:
                lines.append(cls._coord(corner, "D01"))
        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_edge_cuts(cls, width_mm: float, height_mm: float) -> str:
        lines = cls._gerber_header("Edge.Cuts")
        lines.extend(["%ADD10C,0.1000*%", "D10*"])
        outline = [(0, 0), (width_mm, 0), (width_mm, height_mm), (0, height_mm), (0, 0)]
        lines.append(cls._coord(outline[0], "D02"))
        for point in outline[1:]:
            lines.append(cls._coord(point, "D01"))
        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_drill_file(cls, footprints: List[Dict[str, Any]], vias: List[Dict[str, Any]]) -> str:
        drill_points: Dict[float, List[Point]] = {}

        for via in vias:
            diameter = cls._float(via.get("drill_mm"), 0.3)
            drill_points.setdefault(diameter, []).append((cls._float(via.get("x"), 0), cls._float(via.get("y"), 0)))

        for footprint in footprints:
            fx = cls._float(footprint.get("x"), 0)
            fy = cls._float(footprint.get("y"), 0)
            for pad in footprint.get("pads", []):
                diameter = pad.get("drillDiameter")
                if diameter is None:
                    continue
                drill_points.setdefault(cls._float(diameter, 0.8), []).append((
                    fx + cls._float(pad.get("x"), 0),
                    fy + cls._float(pad.get("y"), 0),
                ))

        lines = [
            "M48",
            "; VoltForge Excellon drill file",
            "METRIC,TZ",
        ]
        tools = sorted(drill_points)
        for index, diameter in enumerate(tools, start=1):
            lines.append(f"T{index}C{diameter:.3f}")
        lines.append("%")

        for index, diameter in enumerate(tools, start=1):
            lines.append(f"T{index}")
            for point in drill_points[diameter]:
                lines.append(f"X{point[0]:.4f}Y{point[1]:.4f}")

        lines.append("M30")
        return "\n".join(lines)

    @staticmethod
    def _gerber_header(layer_name: str) -> List[str]:
        return [
            f"G04 VoltForge EDA - {layer_name} *",
            "%FSLAX25Y25*%",
            "%MOMM*%",
        ]

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name or "VoltForge_PCB")
        return safe or "VoltForge_PCB"

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _points(cls, raw_points: Iterable[Dict[str, Any]]) -> List[Point]:
        return [
            (cls._float(point.get("x"), 0), cls._float(point.get("y"), 0))
            for point in raw_points
        ]

    @staticmethod
    def _coord(point: Point, operation: str) -> str:
        return f"X{int(round(point[0] * 100000)):07d}Y{int(round(point[1] * 100000)):07d}{operation}*"

    @staticmethod
    def _nearest_aperture(width_mm: float, apertures: Dict[float, str]) -> str:
        selected_width = min(apertures, key=lambda candidate: abs(candidate - width_mm))
        return apertures[selected_width]
