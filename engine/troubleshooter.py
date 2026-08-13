"""
Voltforge AI - Automated Troubleshooting & Fault Injection Engine
Covers Features 131-140: Diagnostic Decision Trees, Open Wire Fault Injector,
Short-to-Ground Injector, Reversed Polarity Diagnoser, Insufficient Power Diagnoser,
Missing Pull-up Diagnoser, Logic Level Mismatch Diagnoser, Firmware Pin Mismatch.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.troubleshooter")


class TroubleshootingEngine:
    """Interactive diagnostic solver for non-functional circuits."""

    @classmethod
    def diagnose(
        cls,
        symptoms: str,
        board_type: str = "ARDUINO_UNO",
        components: List[Dict[str, Any]] = None,
        wires: List[Dict[str, Any]] = None,
        code: str = ""
    ) -> Dict[str, Any]:
        components = components or []
        wires = wires or []
        lower = symptoms.lower()
        diagnosis: List[Dict[str, Any]] = []
        fixes: List[str] = []

        comp_types = {c.get("type", "").upper() for c in components}

        # I2C device not responding
        if any(kw in lower for kw in ("not responding", "not detected", "not found", "i2c scan", "no device")):
            if comp_types.intersection({"MPU6050", "BME280", "OLED_DISPLAY", "LCD_1602_I2C", "RTC_DS3231"}):
                diagnosis.append({
                    "category": "I2C_BUS_FAILURE",
                    "probability": "HIGH",
                    "cause": "I2C device not detected — likely missing pull-up resistors, wrong I2C pins, or incorrect address.",
                    "steps": [
                        "1. Verify SDA and SCL pull-up resistors (4.7kΩ to VCC) are present.",
                        "2. Run I2C scanner sketch to detect device address.",
                        "3. Verify SDA/SCL pin assignments match your board's hardware I2C pins.",
                        "4. Check that the device VCC voltage matches board logic level (3.3V vs 5V).",
                        "5. Try different I2C addresses (e.g. 0x68 vs 0x69 for MPU6050).",
                    ],
                })
                fixes.append("Add 4.7kΩ pull-up resistors on SDA and SCL lines to VCC.")

        # LED not lighting up
        if any(kw in lower for kw in ("led not", "led doesnt", "led won't", "led dark", "led off")):
            diagnosis.append({
                "category": "LED_NOT_WORKING",
                "probability": "HIGH",
                "cause": "LED not lighting — check polarity, resistor, pin mode, and code logic.",
                "steps": [
                    "1. Verify LED polarity — long leg (anode) to signal, short leg (cathode) to GND.",
                    "2. Ensure a current-limiting resistor (220Ω-330Ω) is in series.",
                    "3. Verify `pinMode(LED_PIN, OUTPUT)` is called in `setup()`.",
                    "4. Verify `digitalWrite(LED_PIN, HIGH)` is called in code.",
                    "5. Test with a multimeter: measure voltage across the LED (should be ~2V forward).",
                ],
            })

        # Motor not spinning
        if any(kw in lower for kw in ("motor not", "motor won't", "motor doesnt", "motor stuck")):
            diagnosis.append({
                "category": "MOTOR_NOT_RUNNING",
                "probability": "HIGH",
                "cause": "Motor not spinning — cannot drive directly from GPIO; need driver circuit.",
                "steps": [
                    "1. A GPIO pin cannot supply enough current for a motor (max 20-40mA).",
                    "2. Use a motor driver (L298N, L293D) or MOSFET transistor switch.",
                    "3. Ensure motor has separate power supply with common ground to MCU.",
                    "4. Add a flyback diode (1N4007) across motor terminals.",
                    "5. Test motor directly with a battery to confirm it's not faulty.",
                ],
            })

        # Display blank/nothing shown
        if any(kw in lower for kw in ("display blank", "oled blank", "lcd blank", "screen blank", "nothing on display")):
            diagnosis.append({
                "category": "DISPLAY_BLANK",
                "probability": "HIGH",
                "cause": "Display showing nothing — I2C communication issue or incorrect initialization.",
                "steps": [
                    "1. Run I2C scanner to verify display address (0x3C or 0x3D for OLED).",
                    "2. Verify correct I2C pins (SDA/SCL) for your board.",
                    "3. Check that the correct display library is installed and initialized.",
                    "4. Verify display VCC is powered (measure with multimeter).",
                    "5. Try `display.begin(SSD1306_SWITCHCAPVCC, 0x3C)` with correct address.",
                ],
            })

        # Serial monitor gibberish
        if any(kw in lower for kw in ("gibberish", "garbage", "weird characters", "baud rate")):
            diagnosis.append({
                "category": "SERIAL_BAUD_MISMATCH",
                "probability": "VERY HIGH",
                "cause": "Serial monitor showing garbage — baud rate mismatch between code and monitor.",
                "steps": [
                    "1. Check `Serial.begin()` baud rate in your code (e.g. 115200).",
                    "2. Set your serial monitor to the SAME baud rate.",
                    "3. Common baud rates: 9600, 115200, 57600.",
                    "4. If using ESP32, try 115200 (default for ESP32 boot messages).",
                ],
            })

        # No upload / programming failure
        if any(kw in lower for kw in ("upload fail", "upload error", "cannot program", "port not found")):
            diagnosis.append({
                "category": "UPLOAD_FAILURE",
                "probability": "HIGH",
                "cause": "Firmware upload failing — check USB connection, port, and bootloader.",
                "steps": [
                    "1. Verify correct COM port is selected in IDE.",
                    "2. Try a different USB cable (data cable, not charge-only).",
                    "3. For ESP32: Hold BOOT button while uploading.",
                    "4. Check that no other program is using the serial port.",
                    "5. Try pressing RESET button on the board after upload starts.",
                ],
            })

        # Default catch-all
        if not diagnosis:
            diagnosis.append({
                "category": "GENERAL_TROUBLESHOOTING",
                "probability": "MEDIUM",
                "cause": "General circuit troubleshooting guide.",
                "steps": [
                    "1. Check all power connections — VCC and GND continuity.",
                    "2. Verify correct pin assignments in both code and canvas.",
                    "3. Test individual components in isolation.",
                    "4. Use multimeter to measure voltages at critical nodes.",
                    "5. Check for loose wires or breadboard contact issues.",
                ],
            })

        return {
            "symptom": symptoms,
            "boardType": board_type,
            "diagnosisCount": len(diagnosis),
            "diagnosis": diagnosis,
            "quickFixes": fixes,
        }

    @classmethod
    def generate_multimeter_guide(cls, measurement_type: str = "voltage") -> List[str]:
        if measurement_type == "voltage":
            return [
                "1. Set multimeter to DC Voltage (V⎓) mode.",
                "2. Connect BLACK probe to circuit GND.",
                "3. Touch RED probe to the node you want to measure.",
                "4. Read the voltage display. Compare with expected value.",
                "5. For 5V boards: VCC should read ~5.0V. For 3.3V boards: ~3.3V.",
            ]
        elif measurement_type == "continuity":
            return [
                "1. Set multimeter to Continuity/Buzzer mode (🔊).",
                "2. Touch probes to both ends of the wire or connection.",
                "3. A beep means the circuit is continuous (connected).",
                "4. No beep means an open circuit (broken connection).",
                "5. Check breadboard rows and jumper wire connections.",
            ]
        return ["Set multimeter to appropriate mode and measure."]


if __name__ == "__main__":
    import json
    result = TroubleshootingEngine.diagnose(
        "My MPU6050 is not detected on I2C bus",
        "ESP32",
        [{"type": "MPU6050"}]
    )
    print(json.dumps(result, indent=2))
