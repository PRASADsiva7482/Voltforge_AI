import math
from typing import Any, Dict, Optional


class ElectricalPhysicsSolver:
    """Symbolic and numerical electrical physics calculation engine."""

    @classmethod
    def calculate_ohms_law(
        cls,
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        resistance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Solves V = I * R given 2 known variables."""
        if voltage is not None and current is not None:
            resistance = voltage / current if current != 0 else float('inf')
            power = voltage * current
            return {"voltage": voltage, "current": current, "resistance": resistance, "power": power}
        elif voltage is not None and resistance is not None:
            current = voltage / resistance if resistance != 0 else 0
            power = (voltage ** 2) / resistance if resistance != 0 else 0
            return {"voltage": voltage, "current": current, "resistance": resistance, "power": power}
        elif current is not None and resistance is not None:
            voltage = current * resistance
            power = (current ** 2) * resistance
            return {"voltage": voltage, "current": current, "resistance": resistance, "power": power}
        return {"error": "Provide at least 2 variables out of (voltage, current, resistance)."}

    @classmethod
    def calculate_led_resistor(
        cls,
        v_supply: float = 5.0,
        v_forward: float = 2.0,
        i_forward_ma: float = 20.0
    ) -> Dict[str, Any]:
        """Calculates LED current-limiting resistor R = (V_supply - V_forward) / I_forward."""
        i_amps = i_forward_ma / 1000.0
        v_drop = v_supply - v_forward
        if v_drop <= 0:
            return {"error": f"Supply voltage ({v_supply}V) must exceed LED forward voltage ({v_forward}V)."}
        r_exact = v_drop / i_amps
        power_watts = (i_amps ** 2) * r_exact
        
        # Pick standard E24 resistor value
        e24_series = [150, 180, 220, 270, 330, 390, 470, 560, 680, 820, 1000]
        r_recommended = min(e24_series, key=lambda x: abs(x - r_exact))

        return {
            "vSupply": v_supply,
            "vForward": v_forward,
            "iForwardmA": i_forward_ma,
            "exactResistance": round(r_exact, 2),
            "recommendedResistor": f"{r_recommended} ohm",
            "powerDissipation": round(power_watts, 4),
            "isSafeQuarterWatt": power_watts < 0.25,
        }

    @classmethod
    def calculate_voltage_divider(
        cls,
        v_in: float = 5.0,
        r1: float = 10000.0,
        r2: float = 20000.0
    ) -> Dict[str, Any]:
        """Calculates V_out = V_in * (R2 / (R1 + R2)) for level shifting."""
        if (r1 + r2) == 0:
            return {"error": "R1 + R2 cannot be zero."}
        v_out = v_in * (r2 / (r1 + r2))
        i_total_ma = (v_in / (r1 + r2)) * 1000.0
        return {
            "vIn": v_in,
            "r1": r1,
            "r2": r2,
            "vOut": round(v_out, 3),
            "currentDrawmA": round(i_total_ma, 3),
            "is3V3Safe": round(v_out, 1) <= 3.3,
        }

    @classmethod
    def calculate_rc_cutoff(cls, resistance: float, capacitance_farads: float) -> Dict[str, Any]:
        """Calculates RC filter cutoff frequency f_c = 1 / (2 * pi * R * C)."""
        if resistance <= 0 or capacitance_farads <= 0:
            return {"error": "Resistance and capacitance must be positive."}
        f_cutoff = 1.0 / (2.0 * math.pi * resistance * capacitance_farads)
        return {
            "resistance": resistance,
            "capacitanceFarads": capacitance_farads,
            "cutoffFrequencyHz": round(f_cutoff, 2),
        }


if __name__ == "__main__":
    print("LED Resistor for 5V -> 2V LED (20mA):", ElectricalPhysicsSolver.calculate_led_resistor(5.0, 2.0, 20.0))
    print("Voltage Divider 5V -> 3.3V (10k/20k):", ElectricalPhysicsSolver.calculate_voltage_divider(5.0, 10000, 20000))
