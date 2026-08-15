"""
Voltforge AI - Natural Language Text-to-Circuit Synthesizer Engine
Converts human natural language prompts into complete schematic topologies and firmware code.
"""

from typing import Any, Dict, List


class CircuitSynthesizer:
    """Synthesizes complete circuit topologies and firmware from natural language descriptions."""

    @classmethod
    def synthesize_circuit(
        cls,
        prompt: str,
        preferred_board: str = "ARDUINO_UNO"
    ) -> Dict[str, Any]:
        """Convert natural language requirement into circuit components, wires, and firmware."""
        p_lower = prompt.lower()
        board = preferred_board

        if "esp32" in p_lower:
            board = "ESP32_DEVKIT_V1"
        elif "pico" in p_lower or "rp2040" in p_lower:
            board = "RPI_PICO"
        elif "nano" in p_lower:
            board = "ARDUINO_NANO"
        elif "mega" in p_lower:
            board = "ARDUINO_MEGA_2560"

        components: List[Dict[str, Any]] = [
            {"id": "mcu_main", "type": board, "name": "Main MCU", "x": 100, "y": 150}
        ]
        wires: List[Dict[str, Any]] = []

        code_includes: List[str] = ["#include <Arduino.h>"]
        setup_code: List[str] = ["  Serial.begin(115200);"]
        loop_code: List[str] = []

        x_offset = 350
        comp_idx = 1

        # Check for Temperature / Humidity Sensors
        if "temp" in p_lower or "dht" in p_lower or "humidity" in p_lower:
            cid = f"sensor_dht_{comp_idx}"
            components.append({
                "id": cid,
                "type": "SENSOR_DHT22",
                "name": "DHT22 Sensor",
                "x": x_offset,
                "y": 100
            })
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "5V", "toNodeId": cid, "toPinId": "VCC", "color": "#ef4444"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "GND", "toNodeId": cid, "toPinId": "GND", "color": "#000000"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "D2", "toNodeId": cid, "toPinId": "DATA", "color": "#3b82f6"})
            code_includes.append("#include <DHT.h>\n#define DHTPIN 2\n#define DHTTYPE DHT22\nDHT dht(DHTPIN, DHTTYPE);")
            setup_code.append("  dht.begin();")
            loop_code.append("  float temp = dht.readTemperature();\n  Serial.print(\"Temp: \"); Serial.println(temp);")
            x_offset += 160
            comp_idx += 1

        # Check for OLED / LCD Display
        if "oled" in p_lower or "display" in p_lower or "screen" in p_lower:
            cid = f"disp_oled_{comp_idx}"
            components.append({
                "id": cid,
                "type": "DISPLAY_OLED_SSD1306",
                "name": "SSD1306 OLED",
                "x": x_offset,
                "y": 100
            })
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "3V3", "toNodeId": cid, "toPinId": "VCC", "color": "#ef4444"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "GND", "toNodeId": cid, "toPinId": "GND", "color": "#000000"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "SDA", "toNodeId": cid, "toPinId": "SDA", "color": "#10b981"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "SCL", "toNodeId": cid, "toPinId": "SCL", "color": "#8b5cf6"})
            code_includes.append("#include <Wire.h>\n#include <Adafruit_SSD1306.h>\nAdafruit_SSD1306 display(128, 64, &Wire, -1);")
            setup_code.append("  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);\n  display.clearDisplay();")
            loop_code.append("  display.setCursor(0, 0);\n  display.setTextSize(1);\n  display.println(\"Voltforge Online\");\n  display.display();")
            x_offset += 160
            comp_idx += 1

        # Check for Buzzer / Alarm
        if "buzzer" in p_lower or "alarm" in p_lower or "sound" in p_lower:
            cid = f"act_buzzer_{comp_idx}"
            components.append({
                "id": cid,
                "type": "BUZZER",
                "name": "Piezo Buzzer",
                "x": x_offset,
                "y": 280
            })
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "D8", "toNodeId": cid, "toPinId": "POS", "color": "#f59e0b"})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "GND", "toNodeId": cid, "toPinId": "NEG", "color": "#000000"})
            setup_code.append("  pinMode(8, OUTPUT);")
            loop_code.append("  tone(8, 1000, 200);\n  delay(1000);")
            x_offset += 140
            comp_idx += 1

        # Check for LED / Indicator
        if "led" in p_lower or "light" in p_lower:
            cid_r = f"res_{comp_idx}"
            cid_led = f"led_{comp_idx}"
            components.append({"id": cid_r, "type": "RESISTOR", "value": "220", "x": x_offset, "y": 240})
            components.append({"id": cid_led, "type": "LED", "value": "RED", "x": x_offset + 90, "y": 240})
            wires.append({"fromNodeId": "mcu_main", "fromPinId": "D13", "toNodeId": cid_r, "toPinId": "PIN1", "color": "#eab308"})
            wires.append({"fromNodeId": cid_r, "fromPinId": "PIN2", "toNodeId": cid_led, "toPinId": "ANODE", "color": "#eab308"})
            wires.append({"fromNodeId": cid_led, "fromPinId": "CATHODE", "toNodeId": "mcu_main", "toPinId": "GND", "color": "#000000"})
            setup_code.append("  pinMode(13, OUTPUT);")
            loop_code.append("  digitalWrite(13, HIGH);\n  delay(500);\n  digitalWrite(13, LOW);\n  delay(500);")

        generated_code = f"""{chr(10).join(code_includes)}

void setup() {{
{chr(10).join(setup_code)}
}}

void loop() {{
{chr(10).join(loop_code) or "  delay(100);"}
}}
"""

        return {
            "status": "SUCCESS",
            "prompt": prompt,
            "boardType": board,
            "components": components,
            "wires": wires,
            "generatedFirmware": generated_code.strip(),
            "summary": f"Synthesized {len(components)} components and {len(wires)} connections for '{prompt}'."
        }
