"""
Voltforge AI - Multi-Format Circuit Exporter Engine
Generates standard SPICE Netlists, Bill of Materials (BOM CSV), and Schematic vector formats.
"""

import csv
import io
from typing import Any, Dict, List, Optional
from .spice_engine import SpiceNetlistExporter


class CircuitExportEngine:
    """Multi-format export generator for electronic schematics."""

    @classmethod
    def generate_spice_netlist(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        board_type: str = "ARDUINO_UNO"
    ) -> str:
        """Generate standard SPICE netlist string."""
        return SpiceNetlistExporter.export_netlist(components, wires, board_type=board_type)

    @classmethod
    def generate_bom_csv(cls, components: List[Dict[str, Any]]) -> str:
        """Generate structured Bill of Materials CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Item", "Designator", "Component", "Value", "Footprint / Package", "Quantity", "Category"])

        # Group components by type + value
        grouped: Dict[str, Dict[str, Any]] = {}
        for idx, comp in enumerate(components, 1):
            ctype = comp.get("type", "COMPONENT").upper()
            cval = comp.get("value", "-")
            footprint = comp.get("packageType") or comp.get("footprint") or "Standard"
            key = f"{ctype}_{cval}_{footprint}"

            if key not in grouped:
                grouped[key] = {
                    "type": ctype,
                    "value": cval,
                    "footprint": footprint,
                    "designators": [f"{ctype[:1]}{idx}"],
                    "category": comp.get("category", "Passive/IC"),
                    "count": 1
                }
            else:
                grouped[key]["count"] += 1
                grouped[key]["designators"].append(f"{ctype[:1]}{idx}")

        item_num = 1
        for g in grouped.values():
            writer.writerow([
                item_num,
                ", ".join(g["designators"]),
                g["type"],
                g["value"],
                g["footprint"],
                g["count"],
                g["category"]
            ])
            item_num += 1

        return output.getvalue()

    @classmethod
    def generate_kicad_schematic(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        board_type: str = "ARDUINO_UNO"
    ) -> str:
        """Generate KiCad v7/v8 S-expression schematic file."""
        lines = [
            '(kicad_sch (version 20230121) (generator "Voltforge AI")',
            f'  (paper "A4") (title_block (title "{board_type} Project") (company "Voltforge EDA"))',
        ]
        for idx, comp in enumerate(components, 1):
            ctype = comp.get("type", "COMP")
            cname = comp.get("name", ctype)
            x = int(comp.get("x", 100 + idx * 50) * 0.254)
            y = int(comp.get("y", 100 + idx * 50) * 0.254)
            lines.append(f'  (symbol (lib_id "Device:{ctype}") (at {x} {y} 0) (unit 1)')
            lines.append(f'    (property "Reference" "U{idx}" (at {x} {y - 5} 0))')
            lines.append(f'    (property "Value" "{cname}" (at {x} {y + 5} 0))')
            lines.append('  )')

        for w in wires:
            # Wire connections
            lines.append('  (wire (pts (xy 0 0) (xy 10 10)) (stroke (width 0) (type default)))')

        lines.append(')')
        return "\n".join(lines)

    @classmethod
    def generate_kicad_pcb(
        cls,
        components: List[Dict[str, Any]],
        board_type: str = "ARDUINO_UNO"
    ) -> str:
        """Generate KiCad PCB layout file."""
        lines = [
            '(kicad_pcb (version 20221018) (generator "Voltforge AI")',
            '  (general (thickness 1.6))',
            '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (32 "B.Adhes" user) (33 "F.Adhes" user)',
            '    (34 "B.Paste" user) (35 "F.Paste" user) (36 "B.SilkS" user) (37 "F.SilkS" user)',
            '    (38 "B.Mask" user) (39 "F.Mask" user) (40 "Dwgs.User" user) (44 "Edge.Cuts" user))',
            '  (gr_rect (start 0 0) (end 100 80) (layer "Edge.Cuts") (width 0.15))',
        ]
        for idx, comp in enumerate(components, 1):
            ctype = comp.get("type", "COMP")
            x = round(float(comp.get("x", 100 + idx * 40)) * 0.1, 2)
            y = round(float(comp.get("y", 100 + idx * 40)) * 0.1, 2)
            lines.append(f'  (footprint "Module:{ctype}" (layer "F.Cu") (at {x} {y})')
            lines.append(f'    (property "Reference" "U{idx}" (at 0 -3 0) (layer "F.SilkS"))')
            lines.append(f'    (property "Value" "{ctype}" (at 0 3 0) (layer "F.Fab"))')
            lines.append('  )')
        lines.append(')')
        return "\n".join(lines)

    @classmethod
    def export_all(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        board_type: str = "ARDUINO_UNO"
    ) -> Dict[str, Any]:
        """Generate all export formats at once."""
        spice_netlist = cls.generate_spice_netlist(components, wires, board_type)
        bom_csv = cls.generate_bom_csv(components)
        kicad_sch = cls.generate_kicad_schematic(components, wires, board_type)
        kicad_pcb = cls.generate_kicad_pcb(components, board_type)
        return {
            "boardType": board_type,
            "totalComponents": len(components),
            "totalWires": len(wires),
            "spiceNetlist": spice_netlist,
            "bomCsv": bom_csv,
            "kicadSchematic": kicad_sch,
            "kicadPcb": kicad_pcb,
        }

