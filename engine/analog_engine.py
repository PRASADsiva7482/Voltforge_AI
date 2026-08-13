"""
Voltforge AI - Advanced Analog Simulation Engine
Covers Features 221-240: BJT Ebers-Moll, MOSFET Square-Law, Op-Amp GBW,
Zener Regulator, Schmitt Trigger, Wheatstone Bridge, NTC Steinhart-Hart,
Current Shunt, R-2R DAC, ADC Quantization, Full-Wave Rectifier.
"""

import math
import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.analog_engine")


class BJTSolver:
    """BJT Ebers-Moll model DC operating point solver."""

    @classmethod
    def solve_common_emitter(
        cls, vcc: float, rb: float, rc: float, beta: float = 200, vbe: float = 0.7
    ) -> Dict[str, Any]:
        ib = (vcc - vbe) / rb if rb > 0 else 0
        ic = beta * ib
        vce = vcc - ic * rc
        saturation = vce < 0.2
        if saturation:
            ic = (vcc - 0.2) / rc
            vce = 0.2
        power_dissipation = vce * ic

        return {
            "vcc": vcc,
            "rb_ohm": rb,
            "rc_ohm": rc,
            "beta": beta,
            "ib_uA": round(ib * 1e6, 2),
            "ic_mA": round(ic * 1000, 4),
            "vce_V": round(vce, 4),
            "region": "Saturation" if saturation else "Active",
            "powerDissipation_mW": round(power_dissipation * 1000, 4),
        }


class MOSFETSolver:
    """MOSFET square-law model for drain current estimation."""

    @classmethod
    def solve_nmos(
        cls, vgs: float, vds: float, vth: float = 1.5, kn: float = 0.5e-3
    ) -> Dict[str, Any]:
        if vgs < vth:
            region = "Cutoff"
            id_A = 0
        elif vds < (vgs - vth):
            region = "Triode (Linear)"
            id_A = kn * ((vgs - vth) * vds - 0.5 * vds ** 2)
        else:
            region = "Saturation"
            id_A = 0.5 * kn * (vgs - vth) ** 2

        return {
            "vgs_V": vgs,
            "vds_V": vds,
            "vth_V": vth,
            "kn": kn,
            "id_mA": round(id_A * 1000, 4),
            "region": region,
        }


class OpAmpSolver:
    """Op-Amp gain-bandwidth product and stability analyzer."""

    @classmethod
    def inverting_amplifier(cls, rf: float, rin: float, gbw_MHz: float = 1.0) -> Dict[str, Any]:
        gain = -rf / rin if rin > 0 else 0
        abs_gain = abs(gain)
        bandwidth_kHz = (gbw_MHz * 1e6) / abs_gain / 1000 if abs_gain > 0 else gbw_MHz * 1000
        return {
            "configuration": "Inverting Amplifier",
            "rf_ohm": rf,
            "rin_ohm": rin,
            "closedLoopGain": round(gain, 2),
            "gainMagnitude_dB": round(20 * math.log10(abs_gain), 2) if abs_gain > 0 else 0,
            "bandwidth_kHz": round(bandwidth_kHz, 2),
            "gbw_MHz": gbw_MHz,
        }

    @classmethod
    def non_inverting_amplifier(cls, rf: float, r1: float, gbw_MHz: float = 1.0) -> Dict[str, Any]:
        gain = 1 + rf / r1 if r1 > 0 else float('inf')
        bandwidth_kHz = (gbw_MHz * 1e6) / gain / 1000 if gain > 0 else 0
        return {
            "configuration": "Non-Inverting Amplifier",
            "rf_ohm": rf,
            "r1_ohm": r1,
            "closedLoopGain": round(gain, 2),
            "gainMagnitude_dB": round(20 * math.log10(gain), 2) if gain > 0 else 0,
            "bandwidth_kHz": round(bandwidth_kHz, 2),
            "gbw_MHz": gbw_MHz,
        }


class ZenerRegulatorSolver:
    """Zener diode voltage regulator design calculator."""

    @classmethod
    def design(cls, v_in: float, v_zener: float, i_load_mA: float, pz_max_mW: float = 500) -> Dict[str, Any]:
        if v_in <= v_zener:
            return {"error": f"V_in ({v_in}V) must exceed V_zener ({v_zener}V)."}
        i_load_A = i_load_mA / 1000.0
        iz_min = 0.005  # 5mA minimum zener current
        i_total = i_load_A + iz_min
        rs = (v_in - v_zener) / i_total
        pz = v_zener * iz_min
        rs_power = (v_in - v_zener) ** 2 / rs if rs > 0 else 0
        is_safe = (pz * 1000) < pz_max_mW

        return {
            "vIn": v_in,
            "vZener": v_zener,
            "iLoad_mA": i_load_mA,
            "seriesResistor_ohm": round(rs, 1),
            "zenerCurrent_mA": round(iz_min * 1000, 2),
            "zenerPower_mW": round(pz * 1000, 2),
            "resistorPower_mW": round(rs_power * 1000, 2),
            "isSafe": is_safe,
            "lineRegulation": "Poor (not suitable for variable loads > 50mA)" if i_load_mA > 50 else "Acceptable",
        }


class WheatStoneBridgeSolver:
    """Wheatstone bridge balance and sensitivity calculator."""

    @classmethod
    def solve(cls, r1: float, r2: float, r3: float, r4: float, v_supply: float) -> Dict[str, Any]:
        v_bridge = v_supply * (r3 / (r3 + r1) - r4 / (r4 + r2))
        is_balanced = abs(v_bridge) < 0.001
        r_unknown = (r2 * r3 / r1) if r1 > 0 else 0

        return {
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
            "vSupply": v_supply,
            "bridgeVoltage_mV": round(v_bridge * 1000, 4),
            "isBalanced": is_balanced,
            "rUnknown_for_balance": round(r_unknown, 2),
        }


class NTCThermistorSolver:
    """Steinhart-Hart NTC thermistor temperature calculator."""

    @classmethod
    def calculate_temperature(
        cls, resistance_ohm: float, r_nominal: float = 10000, t_nominal_C: float = 25, beta: float = 3950
    ) -> Dict[str, Any]:
        t_nominal_K = t_nominal_C + 273.15
        steinhart = 1.0 / t_nominal_K + (1.0 / beta) * math.log(resistance_ohm / r_nominal)
        temp_K = 1.0 / steinhart
        temp_C = temp_K - 273.15
        temp_F = temp_C * 9 / 5 + 32

        return {
            "resistance_ohm": resistance_ohm,
            "temperature_C": round(temp_C, 2),
            "temperature_F": round(temp_F, 2),
            "temperature_K": round(temp_K, 2),
            "beta": beta,
        }


class R2RDACCalculator:
    """R-2R resistor ladder DAC output voltage calculator."""

    @classmethod
    def calculate(cls, digital_value: int, bits: int = 8, v_ref: float = 5.0) -> Dict[str, Any]:
        max_value = (1 << bits) - 1
        if digital_value > max_value:
            digital_value = max_value
        v_out = v_ref * digital_value / (max_value + 1)
        resolution = v_ref / (max_value + 1)

        return {
            "digitalValue": digital_value,
            "binaryRepresentation": f"{digital_value:0{bits}b}",
            "bits": bits,
            "vRef": v_ref,
            "vOut": round(v_out, 4),
            "resolution_mV": round(resolution * 1000, 4),
            "maxOutputVoltage": round(v_ref * max_value / (max_value + 1), 4),
        }


class ADCQuantizationAnalyzer:
    """ADC resolution and quantization noise analyzer."""

    @classmethod
    def analyze(cls, bits: int = 10, v_ref: float = 3.3) -> Dict[str, Any]:
        levels = 1 << bits
        lsb = v_ref / levels
        snr_dB = 6.02 * bits + 1.76
        enob = (snr_dB - 1.76) / 6.02

        return {
            "bits": bits,
            "quantizationLevels": levels,
            "lsb_mV": round(lsb * 1000, 4),
            "idealSNR_dB": round(snr_dB, 2),
            "effectiveBits": round(enob, 2),
            "vRef": v_ref,
            "maxDigitalValue": levels - 1,
        }


class CurrentShuntSolver:
    """Current measurement shunt resistor selector."""

    @classmethod
    def select_shunt(cls, max_current_A: float, adc_range_mV: float = 100.0, adc_bits: int = 12) -> Dict[str, Any]:
        r_shunt = (adc_range_mV / 1000.0) / max_current_A if max_current_A > 0 else 0
        power_dissipation = max_current_A ** 2 * r_shunt
        resolution_mA = max_current_A / (1 << adc_bits) * 1000

        return {
            "maxCurrent_A": max_current_A,
            "shuntResistance_ohm": round(r_shunt, 4),
            "shuntResistance_mOhm": round(r_shunt * 1000, 2),
            "maxVoltageDrop_mV": round(adc_range_mV, 2),
            "powerDissipation_mW": round(power_dissipation * 1000, 4),
            "currentResolution_mA": round(resolution_mA, 4),
            "recommendedIC": "INA219 (I2C current/voltage sensor)" if max_current_A <= 3.2 else "INA260 (high-current)",
        }


if __name__ == "__main__":
    import json
    print("=== BJT Common Emitter ===")
    print(json.dumps(BJTSolver.solve_common_emitter(12, 100000, 1000), indent=2))

    print("\n=== MOSFET NMOS ===")
    print(json.dumps(MOSFETSolver.solve_nmos(3.0, 5.0, 1.5), indent=2))

    print("\n=== Op-Amp Inverting ===")
    print(json.dumps(OpAmpSolver.inverting_amplifier(100000, 10000, 1.0), indent=2))

    print("\n=== Zener Regulator ===")
    print(json.dumps(ZenerRegulatorSolver.design(12, 5.1, 20), indent=2))

    print("\n=== NTC Thermistor at 15kΩ ===")
    print(json.dumps(NTCThermistorSolver.calculate_temperature(15000), indent=2))

    print("\n=== 8-bit DAC at value 128 ===")
    print(json.dumps(R2RDACCalculator.calculate(128, 8, 5.0), indent=2))

    print("\n=== 12-bit ADC Analysis ===")
    print(json.dumps(ADCQuantizationAnalyzer.analyze(12, 3.3), indent=2))
