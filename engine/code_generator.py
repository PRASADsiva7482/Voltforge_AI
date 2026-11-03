import logging
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.code_generator")


class FirmwareCodeGenerator:
    """Generates compilable firmware code for microcontrollers based on canvas state and user prompts."""

    @classmethod
    def generate(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]],
        board_type: str = "ARDUINO_UNO",
        additional_instructions: str = ""
    ) -> str:
        board_upper = (board_type or "ARDUINO_UNO").upper()
        comp_types = {comp.get("type", "").upper() for comp in components if isinstance(comp, dict)}

        includes = ["#include <Arduino.h>"]
        defines = []
        setup_lines = []
        loop_lines = []

        # Parse component pin definitions
        for idx, comp in enumerate(components):
            ctype = comp.get("type", "").upper()
            cname = comp.get("name", f"comp_{idx}")
            pin = comp.get("pin", f"{idx + 2}")

            if "LED" in ctype:
                defines.append(f"#define LED_PIN {pin}")
                setup_lines.append("  pinMode(LED_PIN, OUTPUT);")
                loop_lines.append("  digitalWrite(LED_PIN, HIGH);\n  delay(1000);\n  digitalWrite(LED_PIN, LOW);\n  delay(1000);")
            elif "BUTTON" in ctype:
                defines.append(f"#define BUTTON_PIN {pin}")
                setup_lines.append("  pinMode(BUTTON_PIN, INPUT_PULLUP);")
                loop_lines.append("  if (digitalRead(BUTTON_PIN) == LOW) {\n    // Button pressed\n  }")
            elif "MPU6050" in ctype:
                includes.append("#include <Wire.h>\n#include <MPU6050.h>")
                setup_lines.append("  Wire.begin();\n  // MPU6050 initialization")
            elif "OLED" in ctype:
                includes.append("#include <Wire.h>\n#include <Adafruit_SSD1306.h>")
                setup_lines.append("  // OLED initialization")

        if not setup_lines:
            setup_lines.append("  // Initialize system pins\n  Serial.begin(115200);")

        if not loop_lines:
            loop_lines.append("  // Main firmware execution loop\n  delay(10);")

        includes_str = "\n".join(sorted(list(set(includes))))
        defines_str = "\n".join(defines)
        setup_str = "\n".join(setup_lines)
        loop_str = "\n".join(loop_lines)

        return f"""/*
 * Voltforge Generated Firmware
 * Target Board: {board_type}
 */

{includes_str}

{defines_str}

void setup() {{
{setup_str}
}}

void loop() {{
{loop_str}
}}
"""


if __name__ == "__main__":
    code = FirmwareCodeGenerator.generate(
        components=[{"type": "LED", "pin": "13"}, {"type": "BUTTON", "pin": "2"}],
        wires=[],
        board_type="ARDUINO_UNO"
    )
    print("Generated Code Sample:\n", code)
