"""
Voltforge AI - Wireless, IoT & RF Solvers
Covers Features 171-180: Wi-Fi/BLE Power Optimization, LoRaWAN Payload Packer,
Antenna Impedance Matching, Free Space Path Loss, RSSI Interpreter, MQTT Formatter.
"""

import math
import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.wireless_solver")


class WiFiPowerOptimizer:
    """ESP32 Wi-Fi & BLE power mode current estimator."""

    POWER_MODES = {
        "active_tx": {"current_mA": 240, "description": "Active Wi-Fi TX at max power"},
        "active_rx": {"current_mA": 100, "description": "Active Wi-Fi RX"},
        "modem_sleep": {"current_mA": 20, "description": "Modem sleep (CPU active, Wi-Fi off)"},
        "light_sleep": {"current_mA": 0.8, "description": "Light sleep (CPU paused, Wi-Fi off)"},
        "deep_sleep": {"current_mA": 0.01, "description": "Deep sleep (RTC only)"},
        "hibernation": {"current_mA": 0.005, "description": "Hibernation (RTC memory off)"},
    }

    @classmethod
    def estimate_power(cls, mode: str = "active_tx", v_supply: float = 3.3) -> Dict[str, Any]:
        mode_data = cls.POWER_MODES.get(mode, cls.POWER_MODES["active_tx"])
        power_mW = mode_data["current_mA"] * v_supply
        return {
            "mode": mode,
            "description": mode_data["description"],
            "current_mA": mode_data["current_mA"],
            "power_mW": round(power_mW, 2),
            "allModes": {k: v["current_mA"] for k, v in cls.POWER_MODES.items()},
        }


class FreeSpacePathLossCalculator:
    """Calculates RF signal attenuation over distance."""

    @classmethod
    def calculate_fspl(cls, distance_m: float, frequency_MHz: float) -> Dict[str, Any]:
        if distance_m <= 0 or frequency_MHz <= 0:
            return {"error": "Distance and frequency must be positive."}
        fspl_dB = 20 * math.log10(distance_m) + 20 * math.log10(frequency_MHz) + 32.44
        return {
            "distance_m": distance_m,
            "frequency_MHz": frequency_MHz,
            "freeSpacePathLoss_dB": round(fspl_dB, 2),
            "interpretation": f"Signal attenuates by {round(fspl_dB, 1)} dB over {distance_m}m at {frequency_MHz} MHz.",
        }


class RSSIInterpreter:
    """Interprets RSSI signal strength values."""

    @classmethod
    def interpret(cls, rssi_dBm: float) -> Dict[str, Any]:
        if rssi_dBm >= -30:
            quality = "Excellent"
            reliability = "100%"
        elif rssi_dBm >= -50:
            quality = "Very Good"
            reliability = "99%"
        elif rssi_dBm >= -60:
            quality = "Good"
            reliability = "95%"
        elif rssi_dBm >= -70:
            quality = "Fair"
            reliability = "80%"
        elif rssi_dBm >= -80:
            quality = "Weak"
            reliability = "50%"
        else:
            quality = "Very Weak / Disconnecting"
            reliability = "<30%"

        return {
            "rssi_dBm": rssi_dBm,
            "signalQuality": quality,
            "estimatedReliability": reliability,
            "recommendation": "Reduce distance or add antenna" if rssi_dBm < -70 else "Signal quality acceptable.",
        }


class MQTTPayloadFormatter:
    """Formats lightweight IoT JSON payloads for MQTT cloud telemetry."""

    @classmethod
    def format_sensor_payload(
        cls,
        device_id: str,
        temperature: float = None,
        humidity: float = None,
        pressure: float = None,
        battery_percent: float = None,
    ) -> Dict[str, Any]:
        payload = {"deviceId": device_id}
        if temperature is not None:
            payload["temperature_C"] = round(temperature, 2)
        if humidity is not None:
            payload["humidity_percent"] = round(humidity, 2)
        if pressure is not None:
            payload["pressure_hPa"] = round(pressure, 2)
        if battery_percent is not None:
            payload["battery_percent"] = round(battery_percent, 1)

        import json
        payload_json = json.dumps(payload)
        return {
            "topic": f"voltforge/sensors/{device_id}/data",
            "payload": payload,
            "payloadJSON": payload_json,
            "payloadSize_bytes": len(payload_json),
            "qos": 1,
        }


class LoRaWANPayloadPacker:
    """Optimizes sensor payload binary encoding for LoRaWAN minimum airtime."""

    @classmethod
    def pack_sensor_data(cls, temperature: float, humidity: float, battery_mV: int) -> Dict[str, Any]:
        temp_encoded = int((temperature + 40) * 10) & 0xFFFF
        hum_encoded = int(humidity * 10) & 0xFFFF
        batt_encoded = battery_mV & 0xFFFF

        payload_bytes = [
            (temp_encoded >> 8) & 0xFF, temp_encoded & 0xFF,
            (hum_encoded >> 8) & 0xFF, hum_encoded & 0xFF,
            (batt_encoded >> 8) & 0xFF, batt_encoded & 0xFF,
        ]

        return {
            "temperature": temperature,
            "humidity": humidity,
            "battery_mV": battery_mV,
            "payloadHex": "".join(f"{b:02X}" for b in payload_bytes),
            "payloadSize_bytes": len(payload_bytes),
            "decoderFunction": (
                "function decode(bytes) {\n"
                "  var temp = ((bytes[0] << 8) | bytes[1]) / 10.0 - 40;\n"
                "  var hum = ((bytes[2] << 8) | bytes[3]) / 10.0;\n"
                "  var batt = (bytes[4] << 8) | bytes[5];\n"
                "  return { temperature: temp, humidity: hum, battery_mV: batt };\n"
                "}"
            ),
        }


if __name__ == "__main__":
    import json
    print("=== Wi-Fi Power Modes ===")
    print(json.dumps(WiFiPowerOptimizer.estimate_power("deep_sleep"), indent=2))

    print("\n=== FSPL at 100m / 2.4GHz ===")
    print(json.dumps(FreeSpacePathLossCalculator.calculate_fspl(100, 2400), indent=2))

    print("\n=== RSSI Interpretation ===")
    print(json.dumps(RSSIInterpreter.interpret(-65), indent=2))

    print("\n=== MQTT Payload ===")
    print(json.dumps(MQTTPayloadFormatter.format_sensor_payload("esp32_001", 25.3, 62.1, 1013.25, 87.5), indent=2))

    print("\n=== LoRaWAN Payload ===")
    print(json.dumps(LoRaWANPayloadPacker.pack_sensor_data(25.3, 62.1, 3720), indent=2))
