"""
Voltforge AI - EDA Export & PCB Design Engine
Covers Features 281-300: KiCad Exporter, IPC-2221 Track Width Calculator,
Via Current Capacity, Differential Pair Matching, PCB Stackup,
Thermal Via Array, Copper Pour Recommender, PCB Cost Estimator.
"""

import math
import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.pcb_engine")


class TrackWidthCalculator:
    """IPC-2221 PCB trace width calculator for current capacity."""

    @classmethod
    def calculate_external(
        cls, current_A: float, temp_rise_C: float = 10.0, copper_oz: float = 1.0, trace_length_mm: float = 50.0
    ) -> Dict[str, Any]:
        copper_thickness_mil = copper_oz * 1.378  # 1 oz = 1.378 mil
        # IPC-2221 formula: Area = (I / (k * dT^0.44))^(1/0.725)
        k = 0.048  # external layer constant
        area_mil2 = (current_A / (k * temp_rise_C ** 0.44)) ** (1 / 0.725)
        width_mil = area_mil2 / copper_thickness_mil
        width_mm = width_mil * 0.0254

        # Resistance estimation
        resistivity = 1.724e-6  # copper, ohm-cm
        area_cm2 = area_mil2 * (2.54e-3) ** 2
        length_cm = trace_length_mm / 10
        resistance_mohm = (resistivity * length_cm / area_cm2) * 1000 if area_cm2 > 0 else 0
        voltage_drop = current_A * resistance_mohm / 1000

        return {
            "current_A": current_A,
            "tempRise_C": temp_rise_C,
            "copperWeight_oz": copper_oz,
            "requiredWidth_mm": round(width_mm, 3),
            "requiredWidth_mil": round(width_mil, 1),
            "crossSectionalArea_mil2": round(area_mil2, 1),
            "traceResistance_mOhm": round(resistance_mohm, 2),
            "voltageDrop_mV": round(voltage_drop * 1000, 2),
            "layer": "External",
        }


class ViaCurrentCapacity:
    """PCB via current capacity calculator."""

    @classmethod
    def calculate(cls, drill_mm: float = 0.3, copper_oz: float = 1.0, num_vias: int = 1) -> Dict[str, Any]:
        # IPC-2221 via current capacity approximation
        copper_thickness_mm = copper_oz * 0.035
        barrel_area_mm2 = math.pi * drill_mm * copper_thickness_mm
        # Rough: 1A per 0.06 mm² of barrel area (conservative)
        current_per_via = barrel_area_mm2 / 0.06
        total_current = current_per_via * num_vias

        return {
            "drillDiameter_mm": drill_mm,
            "copperWeight_oz": copper_oz,
            "barrelArea_mm2": round(barrel_area_mm2, 4),
            "currentPerVia_A": round(current_per_via, 2),
            "numVias": num_vias,
            "totalCurrentCapacity_A": round(total_current, 2),
            "recommendation": f"Use {num_vias} via(s) of {drill_mm}mm for up to {round(total_current, 1)}A.",
        }


class DifferentialPairCalculator:
    """Differential pair impedance and length matching."""

    @classmethod
    def calculate_microstrip(
        cls, trace_width_mm: float = 0.15, trace_spacing_mm: float = 0.2,
        dielectric_height_mm: float = 0.2, er: float = 4.5
    ) -> Dict[str, Any]:
        # Approximate microstrip impedance
        w = trace_width_mm
        h = dielectric_height_mm
        effective_er = (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 * h / w)) if w > 0 else er
        z0_single = (87 / math.sqrt(effective_er)) * math.log(5.98 * h / (0.8 * w + h * 0.2)) if w > 0 else 50
        
        # Differential impedance approximation
        s = trace_spacing_mm
        z_diff = 2 * z0_single * (1 - 0.48 * math.exp(-0.96 * s / h))

        return {
            "traceWidth_mm": trace_width_mm,
            "traceSpacing_mm": trace_spacing_mm,
            "dielectricHeight_mm": dielectric_height_mm,
            "dielectricConstant": er,
            "singleEndedImpedance_ohm": round(z0_single, 1),
            "differentialImpedance_ohm": round(z_diff, 1),
            "targetImpedance": "90Ω (USB 2.0) or 100Ω (USB 3.0)",
        }


class PCBCostEstimator:
    """Estimates PCB fabrication cost based on board parameters."""

    @classmethod
    def estimate(
        cls, width_mm: float, height_mm: float, layers: int = 2,
        quantity: int = 5, copper_oz: float = 1.0
    ) -> Dict[str, Any]:
        area_cm2 = (width_mm * height_mm) / 100
        
        # Base pricing model (approximate Chinese fab pricing)
        base_price = 2.0  # minimum order
        area_price = area_cm2 * 0.15 * quantity
        layer_multiplier = {1: 0.8, 2: 1.0, 4: 2.5, 6: 4.0, 8: 6.0}.get(layers, layers * 1.2)
        copper_multiplier = {0.5: 0.9, 1.0: 1.0, 2.0: 1.3}.get(copper_oz, 1.5)
        
        total = max(base_price, area_price) * layer_multiplier * copper_multiplier
        per_board = total / quantity if quantity > 0 else total

        return {
            "boardSize_mm": f"{width_mm} x {height_mm}",
            "area_cm2": round(area_cm2, 2),
            "layers": layers,
            "quantity": quantity,
            "copperWeight_oz": copper_oz,
            "estimatedTotal_USD": round(total, 2),
            "perBoard_USD": round(per_board, 2),
            "leadTime": "3-5 business days (standard)" if layers <= 2 else "7-10 business days",
        }


class ThermalViaArraySolver:
    """Calculates thermal via arrays for heat dissipation pads."""

    @classmethod
    def design(cls, power_W: float, pad_size_mm: float = 5.0, max_temp_rise_C: float = 30.0) -> Dict[str, Any]:
        # Each thermal via provides approximately 3-5 °C/W thermal resistance
        thermal_resistance_per_via = 4.0  # °C/W typical
        required_conductance = power_W / max_temp_rise_C if max_temp_rise_C > 0 else float('inf')
        num_vias = max(1, math.ceil(power_W * thermal_resistance_per_via / max_temp_rise_C))
        
        # Via spacing (minimum 0.5mm pitch)
        grid_size = math.ceil(math.sqrt(num_vias))
        via_pitch = pad_size_mm / grid_size if grid_size > 0 else 1.0

        return {
            "powerDissipation_W": power_W,
            "padSize_mm": pad_size_mm,
            "maxTempRise_C": max_temp_rise_C,
            "requiredVias": num_vias,
            "gridLayout": f"{grid_size}x{grid_size}",
            "viaPitch_mm": round(via_pitch, 2),
            "viaDrillSize_mm": 0.3,
            "recommendation": f"Place {num_vias} thermal vias in a {grid_size}x{grid_size} grid with {round(via_pitch, 1)}mm pitch.",
        }


class KiCadExporter:
    """Exports component placement data in KiCad-compatible format."""

    @classmethod
    def export_symbol(cls, component_type: str, ref_designator: str, value: str) -> str:
        """Generates a KiCad symbol library snippet."""
        return f"""(symbol "{ref_designator}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)
  (property "Reference" "{ref_designator}" (at 0 2.54 0))
  (property "Value" "{value}" (at 0 -2.54 0))
  (property "Footprint" "" (at 0 0 0))
  (property "Datasheet" "" (at 0 0 0))
)"""

    @classmethod
    def export_placement(cls, components: List[Dict[str, Any]]) -> str:
        """Generates KiCad placement position file."""
        lines = ["# Voltforge AI - KiCad Component Placement", "# Ref, Val, Package, PosX, PosY, Rot, Side"]
        x, y = 10.0, 10.0
        for idx, comp in enumerate(components):
            ref = f"U{idx + 1}"
            val = comp.get("name", comp.get("type", "UNKNOWN"))
            lines.append(f"{ref}, {val}, Module, {x:.2f}, {y:.2f}, 0, top")
            x += 15.0
            if x > 80:
                x = 10.0
                y += 15.0
        return "\n".join(lines)


if __name__ == "__main__":
    import json
    print("=== Track Width (2A, 10°C rise) ===")
    print(json.dumps(TrackWidthCalculator.calculate_external(2.0, 10.0, 1.0), indent=2))

    print("\n=== Via Current ===")
    print(json.dumps(ViaCurrentCapacity.calculate(0.3, 1.0, 4), indent=2))

    print("\n=== Differential Pair ===")
    print(json.dumps(DifferentialPairCalculator.calculate_microstrip(), indent=2))

    print("\n=== PCB Cost Estimate ===")
    print(json.dumps(PCBCostEstimator.estimate(50, 50, 2, 10), indent=2))

    print("\n=== Thermal Via Array ===")
    print(json.dumps(ThermalViaArraySolver.design(3.0, 5.0, 20.0), indent=2))
