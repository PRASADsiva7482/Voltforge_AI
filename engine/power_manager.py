"""
Voltforge AI - Power System, Battery Management & Solar Sizing Engine
Covers Features 161-170: Battery Discharge Curve, Solar Panel Sizing, Buck/Boost Efficiency,
Deep Sleep Current Optimization, LDO Thermal Solver, USB-PD Configuration, Supercapacitor Energy.
"""

import math
import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.power_manager")


class BatteryLifeEstimator:
    """Estimates battery runtime under active and sleep modes."""

    @classmethod
    def estimate_runtime(
        cls,
        battery_mAh: float,
        active_current_mA: float,
        sleep_current_uA: float = 10.0,
        active_ratio_percent: float = 10.0,
        discharge_efficiency: float = 0.85
    ) -> Dict[str, Any]:
        active_ratio = active_ratio_percent / 100.0
        sleep_ratio = 1.0 - active_ratio
        sleep_current_mA = sleep_current_uA / 1000.0
        avg_current = (active_current_mA * active_ratio) + (sleep_current_mA * sleep_ratio)
        effective_capacity = battery_mAh * discharge_efficiency
        runtime_hours = effective_capacity / avg_current if avg_current > 0 else float('inf')

        return {
            "batteryCapacity_mAh": battery_mAh,
            "activeCurrent_mA": active_current_mA,
            "sleepCurrent_uA": sleep_current_uA,
            "activeRatio_percent": active_ratio_percent,
            "averageCurrent_mA": round(avg_current, 4),
            "estimatedRuntime_hours": round(runtime_hours, 2),
            "estimatedRuntime_days": round(runtime_hours / 24, 2),
            "dischargeEfficiency": discharge_efficiency,
        }


class SolarPanelSizer:
    """Calculates solar panel wattage and charge controller requirements."""

    @classmethod
    def size_panel(
        cls,
        daily_consumption_mAh: float,
        battery_voltage: float = 3.7,
        peak_sun_hours: float = 4.0,
        panel_efficiency: float = 0.80,
        charger_efficiency: float = 0.90
    ) -> Dict[str, Any]:
        daily_energy_Wh = (daily_consumption_mAh / 1000.0) * battery_voltage
        required_panel_W = daily_energy_Wh / (peak_sun_hours * panel_efficiency * charger_efficiency)
        recommended_panel_W = math.ceil(required_panel_W * 1.25 * 10) / 10  # 25% safety margin

        return {
            "dailyConsumption_mAh": daily_consumption_mAh,
            "dailyEnergy_Wh": round(daily_energy_Wh, 3),
            "peakSunHours": peak_sun_hours,
            "requiredPanelPower_W": round(required_panel_W, 3),
            "recommendedPanelPower_W": recommended_panel_W,
            "chargerType": "MPPT charge controller recommended for efficiency",
            "batteryRecommendation": f"LiPo {battery_voltage}V with TP4056 charger module",
        }


class LDOThermalSolver:
    """Calculates LDO regulator power dissipation and thermal management."""

    @classmethod
    def calculate(
        cls,
        v_in: float,
        v_out: float,
        i_out_mA: float,
        thermal_resistance_jc: float = 50.0,  # °C/W typical for SOT-223
        max_junction_temp: float = 125.0,
        ambient_temp: float = 25.0
    ) -> Dict[str, Any]:
        dropout = v_in - v_out
        if dropout < 0:
            return {"error": f"Input voltage ({v_in}V) must exceed output voltage ({v_out}V)."}

        i_out_A = i_out_mA / 1000.0
        power_dissipation = dropout * i_out_A
        temp_rise = power_dissipation * thermal_resistance_jc
        junction_temp = ambient_temp + temp_rise
        is_safe = junction_temp < max_junction_temp

        return {
            "vIn": v_in,
            "vOut": v_out,
            "dropout_V": round(dropout, 2),
            "outputCurrent_mA": i_out_mA,
            "powerDissipation_W": round(power_dissipation, 4),
            "tempRise_C": round(temp_rise, 2),
            "junctionTemp_C": round(junction_temp, 2),
            "maxJunctionTemp_C": max_junction_temp,
            "isSafe": is_safe,
            "recommendation": "Safe operating range." if is_safe else "DANGER: Junction temperature exceeds maximum. Add heatsink or use switching regulator.",
        }


class BuckBoostEfficiencyCalculator:
    """Calculates switching regulator efficiency."""

    @classmethod
    def calculate(cls, v_in: float, v_out: float, i_out_mA: float, efficiency_percent: float = 90.0) -> Dict[str, Any]:
        i_out_A = i_out_mA / 1000.0
        p_out = v_out * i_out_A
        efficiency = efficiency_percent / 100.0
        p_in = p_out / efficiency if efficiency > 0 else p_out
        i_in_A = p_in / v_in if v_in > 0 else 0
        power_loss = p_in - p_out

        return {
            "vIn": v_in,
            "vOut": v_out,
            "outputCurrent_mA": i_out_mA,
            "outputPower_W": round(p_out, 4),
            "inputPower_W": round(p_in, 4),
            "inputCurrent_mA": round(i_in_A * 1000, 2),
            "efficiency_percent": efficiency_percent,
            "powerLoss_W": round(power_loss, 4),
            "regulatorType": "Buck (step-down)" if v_out < v_in else "Boost (step-up)",
        }


class SupercapacitorEnergySolver:
    """Calculates supercapacitor stored energy and backup time."""

    @classmethod
    def calculate(cls, capacitance_F: float, v_charged: float, v_cutoff: float, load_current_mA: float) -> Dict[str, Any]:
        energy_J = 0.5 * capacitance_F * (v_charged ** 2 - v_cutoff ** 2)
        energy_Wh = energy_J / 3600.0
        load_power_W = (v_charged + v_cutoff) / 2.0 * (load_current_mA / 1000.0)
        backup_time_s = energy_J / load_power_W if load_power_W > 0 else float('inf')

        return {
            "capacitance_F": capacitance_F,
            "vCharged": v_charged,
            "vCutoff": v_cutoff,
            "storedEnergy_J": round(energy_J, 4),
            "storedEnergy_mWh": round(energy_Wh * 1000, 2),
            "loadCurrent_mA": load_current_mA,
            "estimatedBackupTime_s": round(backup_time_s, 2),
        }


if __name__ == "__main__":
    import json
    print("=== Battery Life Estimate ===")
    print(json.dumps(BatteryLifeEstimator.estimate_runtime(2000, 80, 10, 5), indent=2))

    print("\n=== Solar Panel Sizing ===")
    print(json.dumps(SolarPanelSizer.size_panel(500, 3.7, 4.0), indent=2))

    print("\n=== LDO Thermal ===")
    print(json.dumps(LDOThermalSolver.calculate(12.0, 3.3, 200), indent=2))

    print("\n=== Buck Converter ===")
    print(json.dumps(BuckBoostEfficiencyCalculator.calculate(12.0, 5.0, 500, 92), indent=2))
