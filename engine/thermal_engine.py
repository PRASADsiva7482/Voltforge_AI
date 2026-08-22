"""Thermal analysis driven by solved circuit power.

The browser solver is the source of truth for electrical operating points.
This module only estimates temperature from those powers and component/package
thermal limits; it never invents a global supply or load current.
"""

from typing import Any, Dict, List, Optional


class ThermalEngine:
    DEFAULT_AMBIENT_TEMP = 25.0

    PACKAGE_THERMAL_RESISTANCE = {
        "DIP-8": 100.0,
        "DIP-16": 70.0,
        "TO-220": 4.0,
        "SOT-223": 60.0,
        "AXIAL_0.25W": 120.0,
        "SMD_0805": 180.0,
    }

    @classmethod
    def analyze_thermal_dissipation(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        ambient_temp: float = DEFAULT_AMBIENT_TEMP,
        simulation_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a thermal map for the supplied solved or unsolved circuit.

        ``simulation_state.componentPower`` is expected to be keyed by canvas
        component ID. When it is absent, power is reported as zero and the
        result is explicitly marked ``unsolved`` instead of presenting a fake
        5 V operating point as a measurement.
        """
        del wires  # Reserved for future spatial thermal coupling.
        state = simulation_state or {}
        solved_power = state.get("componentPower") or {}
        thermal_map: List[Dict[str, Any]] = []
        max_temp = float(ambient_temp)
        total_power_watts = 0.0

        for index, component in enumerate(components):
            component_id = str(component.get("id", f"comp_{index}"))
            component_type = str(component.get("type", "UNKNOWN")).upper()
            power_watts = max(0.0, float(solved_power.get(component_id, 0.0) or 0.0))
            max_rating = float(
                component.get("maxPowerWatts")
                or component.get("powerRatingWatts")
                or cls._default_power_rating(component_type)
            )
            package = str(component.get("packageType") or component.get("footprint") or "DIP-8")
            theta_ja = cls.PACKAGE_THERMAL_RESISTANCE.get(package, 80.0)
            temperature_c = float(ambient_temp) + power_watts * theta_ja * 0.4
            max_temp = max(max_temp, temperature_c)
            total_power_watts += power_watts
            ratio = power_watts / max(max_rating, 1e-9)

            if ratio > 1.2:
                status = "BURNOUT_RISK"
            elif ratio > 0.8:
                status = "OVERHEATING"
            elif ratio > 0.5:
                status = "WARMING"
            else:
                status = "SAFE"

            thermal_map.append({
                "componentId": component_id,
                "type": component_type,
                "powerWatts": round(power_watts, 6),
                "maxRatingWatts": max_rating,
                "temperatureC": round(temperature_c, 2),
                "status": status,
                "heatNormalized": min(1.0, round(max(0.0, temperature_c - ambient_temp) / 100.0, 3)),
                "dataSource": "solver" if component_id in solved_power else "unsolved",
            })

        return {
            "status": "SUCCESS",
            "ambientTempC": ambient_temp,
            "maxTemperatureC": round(max_temp, 2),
            "totalPowerWatts": round(total_power_watts, 6),
            "thermalNodes": thermal_map,
            "dataSource": "solver" if solved_power else "unsolved",
        }

    @staticmethod
    def _default_power_rating(component_type: str) -> float:
        if "LED" in component_type:
            return 0.1
        if "REGULATOR" in component_type or "LM7805" in component_type or "AMS1117" in component_type:
            return 1.5
        if "TRANSISTOR" in component_type or "MOSFET" in component_type:
            return 0.8
        return 0.25
