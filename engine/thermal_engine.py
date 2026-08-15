"""
Voltforge AI - Thermal & Power Dissipation Heatmap Engine
Calculates Joulean heating, component power dissipation, and junction temperatures (Tj).
"""

from typing import Any, Dict, List


class ThermalEngine:
    """Computes Joulean heating and thermal dissipation thresholds for circuit elements."""

    DEFAULT_AMBIENT_TEMP = 25.0 # °C

    PACKAGE_THERMAL_RESISTANCE = {
        "DIP-8": 100.0,   # °C/W
        "DIP-16": 70.0,
        "TO-220": 4.0,    # with heatsink
        "SOT-223": 60.0,
        "AXIAL_0.25W": 120.0,
        "SMD_0805": 180.0
    }

    @classmethod
    def analyze_thermal_dissipation(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        ambient_temp: float = 25.0
    ) -> Dict[str, Any]:
        """Calculates power loss (Watts) and estimated temperature (°C) for each component."""
        thermal_map: List[Dict[str, Any]] = []
        max_temp = ambient_temp
        total_power_watts = 0.0

        for idx, comp in enumerate(components):
            cid = comp.get("id", f"comp_{idx}")
            ctype = str(comp.get("type", "UNKNOWN")).upper()
            cval = comp.get("value", "")

            # Power and thermal defaults
            power_watts = 0.05
            max_rating = 0.25 # Default 250mW

            if "RESISTOR" in ctype:
                # Estimate 5V supply through resistance
                try:
                    r_val = float(str(cval).replace("k", "000").replace("R", "").replace("Ω", "") or 1000)
                    r_val = max(1.0, r_val)
                    power_watts = min(5.0, (5.0 * 5.0) / r_val)
                    max_rating = 0.25
                except ValueError:
                    power_watts = 0.02
            elif "REGULATOR" in ctype or "LM7805" in ctype or "AMS1117" in ctype:
                # Drop from 9V/12V to 5V at 500mA
                v_drop = 7.0 # 12V - 5V
                i_load = 0.35 # 350mA
                power_watts = v_drop * i_load
                max_rating = 1.5
            elif "LED" in ctype:
                power_watts = 2.0 * 0.02 # 2V * 20mA = 40mW
                max_rating = 0.1
            elif "TRANSISTOR" in ctype or "MOSFET" in ctype:
                power_watts = 0.15
                max_rating = 0.8

            total_power_watts += power_watts
            ratio = power_watts / max(0.01, max_rating)

            # Calculate junction temperature
            theta_ja = cls.PACKAGE_THERMAL_RESISTANCE.get(comp.get("packageType", "DIP-8"), 80.0)
            temp_c = ambient_temp + (power_watts * theta_ja * 0.4)
            max_temp = max(max_temp, temp_c)

            status = "SAFE"
            if ratio > 1.2:
                status = "BURNOUT_RISK"
            elif ratio > 0.8:
                status = "OVERHEATING"
            elif ratio > 0.5:
                status = "WARMING"

            thermal_map.append({
                "componentId": cid,
                "type": ctype,
                "powerWatts": round(power_watts, 4),
                "maxRatingWatts": max_rating,
                "temperatureC": round(temp_c, 1),
                "status": status,
                "heatNormalized": min(1.0, round(temp_c / 120.0, 2)) # 0.0 (ambient) to 1.0 (hot)
            })

        return {
            "status": "SUCCESS",
            "ambientTempC": ambient_temp,
            "maxTemperatureC": round(max_temp, 1),
            "totalPowerWatts": round(total_power_watts, 4),
            "thermalNodes": thermal_map
        }
