import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger("voltforge-ai.circuit_verifier")

THIRTY_THREE_VOLT_BOARDS = {
    "ESP32", "ESP32_C3", "ESP32_C6", "ESP32_H2", "ESP32_S2", "ESP32_S3", "ESP32_WROOM", "ESP32_WROVER",
    "RASPBERRY_PI_PICO", "RASPBERRY_PI_PICO_2", "RASPBERRY_PI_PICO_W",
    "STM32_BLUE_PILL", "STM32_BLACK_PILL",
    "TEENSY_4_0", "TEENSY_4_1",
    "SEEED_XIAO_ESP32C3", "SEEED_XIAO_ESP32S3", "SEEED_XIAO_RP2040"
}

FIVE_VOLT_BOARDS = {
    "ARDUINO_UNO", "ARDUINO_MEGA", "ARDUINO_LEONARDO", "ARDUINO_NANO", "ATMEL_AVR_ATMEGA328P"
}

I2C_COMPONENTS = {"MPU6050", "BME280", "BMP280", "OLED_DISPLAY", "LCD_1602_I2C", "RTC_DS3231", "PCA9685"}
INDUCTIVE_COMPONENTS = {"RELAY_MODULE", "DC_MOTOR", "SOLENOID", "STEPPER_MOTOR"}


class ElectricalVerifier:
    """Deterministic circuit rule verification engine for Voltforge canvas state."""

    @classmethod
    def verify_circuit(
        self,
        board_type: str = "ARDUINO_UNO",
        components: List[Dict[str, Any]] = None,
        wires: List[Dict[str, Any]] = None,
        code: str = ""
    ) -> Dict[str, Any]:
        components = components or []
        wires = wires or []
        
        issues: List[Dict[str, Any]] = []
        additions: List[Dict[str, Any]] = []
        removals: List[Dict[str, Any]] = []
        value_changes: List[Dict[str, Any]] = []
        code_fixes: List[Dict[str, Any]] = []
        wire_suggestions: List[Dict[str, str]] = []

        component_types = {comp.get("type", "").upper() for comp in components if isinstance(comp, dict)}
        board_upper = (board_type or "ARDUINO_UNO").upper()

        is_3v3_board = any(b in board_upper for b in THIRTY_THREE_VOLT_BOARDS)
        is_5v_board = any(b in board_upper for b in FIVE_VOLT_BOARDS)

        # 1. Voltage Mismatch Checks
        if is_3v3_board:
            for comp in components:
                ctype = comp.get("type", "").upper()
                cname = comp.get("name", comp.get("id", "Component"))
                if ctype in {"5V_SENSOR", "HC_SR04_5V", "5V_RELAY"}:
                    issues.append({
                        "severity": "CRITICAL",
                        "message": f"{cname} operate at 5V logic which can damage 3.3V GPIOs on {board_type}.",
                        "suggestedFix": "Add a logic level converter or 10k/20k voltage divider on signal lines."
                    })
                    additions.append({
                        "type": "LOGIC_LEVEL_CONVERTER",
                        "reason": f"Level convert 5V signals from {cname} to 3.3V GPIOs."
                    })

        # 2. I2C Bus Pull-up Resistor Check
        has_i2c_device = bool(component_types.intersection(I2C_COMPONENTS))
        has_resistor = any("RESISTOR" in ctype for ctype in component_types)
        if has_i2c_device and not has_resistor:
            issues.append({
                "severity": "WARNING",
                "message": "I2C devices require pull-up resistors (4.7k ohm) on SDA and SCL lines for reliable signal integrity.",
                "suggestedFix": "Add two 4.7k ohm pull-up resistors connecting SDA and SCL to VCC."
            })
            additions.append({
                "type": "RESISTOR",
                "value": "4.7k",
                "reason": "Pull-up resistor for I2C SDA line."
            })
            additions.append({
                "type": "RESISTOR",
                "value": "4.7k",
                "reason": "Pull-up resistor for I2C SCL line."
            })

        # 3. Inductive Load Flyback Diode Check
        has_inductive = bool(component_types.intersection(INDUCTIVE_COMPONENTS))
        has_diode = any("DIODE" in ctype for ctype in component_types)
        if has_inductive and not has_diode:
            issues.append({
                "severity": "HIGH",
                "message": "Inductive loads (motors, relays) generate reverse EMF spikes when switched off.",
                "suggestedFix": "Place a flyback diode (1N4007) in anti-parallel across the motor/relay coil."
            })
            additions.append({
                "type": "1N4007_DIODE",
                "reason": "Flyback diode protection against voltage spikes."
            })

        # 4. LED Current Limiting Resistor Check
        if "LED" in component_types and not has_resistor:
            issues.append({
                "severity": "HIGH",
                "message": "LED connected without current-limiting resistor.",
                "suggestedFix": "Connect a 220 ohm or 330 ohm resistor in series with LED anode."
            })
            value_changes.append({
                "target": "LED",
                "suggestedResistor": "220 ohm"
            })

        # Calculate Safety Score
        score = 100
        for issue in issues:
            sev = issue.get("severity", "")
            if sev == "CRITICAL":
                score -= 30
            elif sev == "HIGH":
                score -= 20
            elif sev == "WARNING":
                score -= 10

        safety_score = max(10, min(100, score))
        general_feedback = (
            f"Circuit safety verification complete for {board_type}. Safety score: {safety_score}/100. "
            + ("No critical electrical issues detected." if not issues else f"{len(issues)} electrical issues detected.")
        )

        return {
            "safetyScore": safety_score,
            "generalFeedback": general_feedback,
            "issues": issues,
            "wireSuggestions": wire_suggestions,
            "additions": additions,
            "removals": removals,
            "valueChanges": value_changes,
            "codeFixes": code_fixes,
        }


if __name__ == "__main__":
    sample_res = ElectricalVerifier.verify_circuit(
        board_type="ESP32",
        components=[{"type": "MPU6050"}, {"type": "RELAY_MODULE"}, {"type": "LED"}]
    )
    import json
    print("Circuit Verifier Sample Result:\n", json.dumps(sample_res, indent=2))
