"""
VoltForge AI - Electronics Knowledge Engine
Provides hardware coverage, component specifications, pin routing rules,
and circuit safety verifications.
"""

from typing import Any, Dict, List, Optional

UI_BOARDS = [
    ("ARDUINO_UNO", "Arduino Uno R3", "Arduino"),
    ("ARDUINO_UNO_R4", "Arduino Uno R4", "Arduino"),
    ("ARDUINO_NANO", "Arduino Nano", "Arduino"),
    ("ARDUINO_NANO_EVERY", "Arduino Nano Every", "Arduino"),
    ("ARDUINO_NANO_33_IOT", "Arduino Nano 33 IoT", "Arduino"),
    ("ARDUINO_MEGA", "Arduino Mega 2560", "Arduino"),
    ("ARDUINO_LEONARDO", "Arduino Leonardo", "Arduino"),
    ("ARDUINO_MICRO", "Arduino Micro", "Arduino"),
    ("ARDUINO_DUE", "Arduino Due", "Arduino"),
    ("ARDUINO_GIGA_R1", "Arduino GIGA R1 WiFi", "Arduino"),
    ("ARDUINO_PORTENTA_H7", "Arduino Portenta H7", "Arduino"),
    ("ESP8266", "ESP8266 NodeMCU", "ESP8266"),
    ("ESP8266_WEMOS_D1_MINI", "WeMos D1 Mini", "ESP8266"),
    ("ESP8266_ESP01", "ESP-01", "ESP8266"),
    ("ESP8266_ESP12E", "ESP-12E", "ESP8266"),
    ("ESP32", "ESP32 DevKit V1", "ESP32"),
    ("ESP32_WROOM", "ESP32-WROOM", "ESP32"),
    ("ESP32_WROVER", "ESP32-WROVER", "ESP32"),
    ("ESP32_S2", "ESP32-S2", "ESP32"),
    ("ESP32_S3", "ESP32-S3", "ESP32"),
    ("ESP32_C3", "ESP32-C3", "ESP32"),
    ("ESP32_C6", "ESP32-C6", "ESP32"),
    ("ESP32_H2", "ESP32-H2", "ESP32"),
    ("RASPBERRY_PI_PICO", "Raspberry Pi Pico", "Raspberry Pi Pico"),
    ("RASPBERRY_PI_PICO_W", "Raspberry Pi Pico W", "Raspberry Pi Pico"),
    ("RASPBERRY_PI_PICO_2", "Raspberry Pi Pico 2", "Raspberry Pi Pico"),
    ("STM32_BLUE_PILL", "STM32 Blue Pill", "STM32"),
    ("STM32_BLACK_PILL", "STM32 Black Pill", "STM32"),
    ("TEENSY_4_0", "Teensy 4.0", "Teensy"),
    ("TEENSY_4_1", "Teensy 4.1", "Teensy"),
]

UI_COMPONENTS = [
    {
        "componentType": "LED_STANDARD",
        "name": "Standard 5mm LED",
        "terminals": ["Anode (+)", "Cathode (-)"],
        "forwardVoltageTypicalV": 2.0,
        "forwardCurrentNominalMa": 20.0
    },
    {
        "componentType": "RESISTOR_THROUGH_HOLE",
        "name": "Through-Hole Resistor",
        "terminals": ["Pin 1", "Pin 2"],
        "defaultResistanceOhms": 220
    },
    {
        "componentType": "OLED_DISPLAY",
        "name": "0.96 inch I2C OLED (SSD1306)",
        "terminals": ["VCC", "GND", "SCL", "SDA"],
        "i2cAddress": "0x3C"
    },
    {
        "componentType": "RELAY_SINGLE",
        "name": "5V 1-Channel Relay Module",
        "terminals": ["VCC", "GND", "IN", "NO", "COM", "NC"]
    },
    {
        "componentType": "SERVO_MOTOR",
        "name": "Micro Servo SG90",
        "terminals": ["VCC (5V)", "GND", "SIGNAL (PWM)"]
    },
    {
        "componentType": "DHT22",
        "name": "DHT22 Temperature & Humidity Sensor",
        "terminals": ["VCC", "DATA", "NC", "GND"]
    },
    {
        "componentType": "POTENTIOMETER",
        "name": "Rotary Potentiometer (10k)",
        "terminals": ["VCC", "WIPER (SIG)", "GND"]
    },
    {
        "componentType": "ULTRASONIC_HC_SR04",
        "name": "HC-SR04 Ultrasonic Distance Sensor",
        "terminals": ["VCC", "TRIG", "ECHO", "GND"]
    },
    {
        "componentType": "BUZZER_PIEZO",
        "name": "Active / Passive Piezo Buzzer",
        "terminals": ["Positive (+)", "Negative (-)"]
    }
]

def get_hardware_coverage_report() -> Dict[str, Any]:
    # A catalog name is not evidence of validated electrical support.
    entries = [{"boardType": board_id, "displayName": name, "family": family,
                "status": "unsupported", "reasonCode": "NO_CURATED_EVIDENCE",
                "reason": "This runtime has no curated electrical verification record for this board."}
               for board_id, name, family in UI_BOARDS]
    return {"schemaVersion": 1, "reportId": "local-runtime-coverage", "reportVersion": "2",
            "entryCount": len(entries), "entries": entries,
            "summary": {"verified": 0, "variantRequired": 0, "unsupported": len(entries)},
            # Retain legacy inventory fields, without claiming verification.
            "totalBoards": len(entries), "boards": [dict(entry, boardId=entry["boardType"], name=entry["displayName"]) for entry in entries]}


def get_component_coverage_report() -> Dict[str, Any]:
    entries = [{"componentType": c["componentType"], "displayName": c["name"],
                "category": "component", "status": "unsupported",
                "reasonCode": "NO_CURATED_EVIDENCE", "reason": "Generic catalog metadata is not a verified part specification.",
                "selectionGroup": None, "selectionRequirements": [],
                "physicalIdentitySelected": False, "aiElectricalClaimsAllowed": False}
               for c in UI_COMPONENTS]
    return {"schemaVersion": 1, "reportId": "local-component-coverage", "reportVersion": "2",
            "entryCount": len(entries), "entries": entries, "genericLabelsMaySelectCandidate": False,
            "summary": {"verified": 0, "variantRequired": 0, "simulationOnly": 0,
                        "unsupported": len(entries), "distinctCuratedExactCandidates": 0},
            "totalComponents": len(UI_COMPONENTS), "components": UI_COMPONENTS}


def resolve_pin_suggestions(
    component_type: str, board_type: str = "unspecified",
    component_id: Optional[str] = None, board_id: Optional[str] = None,
    component_pins: Optional[List[Dict[str, Any]]] = None,
    board_pins: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Suggest only explicit I2C signal endpoints supplied by the real canvas.

    No inferred voltage rails, motor drive, missing resistor, or fallback board.
    These are reviewed suggestions, not electrical compatibility certification.
    """
    if not component_id or not board_id or board_type == "unspecified":
        return []
    if not any(kind in component_type.upper() for kind in ("OLED", "I2C")):
        return []

    def pin_id(pins, signal):
        matches = [pin.get("id") for pin in (pins or []) if isinstance(pin, dict)
                   and str(pin.get("name", "")).upper() == signal and pin.get("id")]
        return matches[0] if len(matches) == 1 else None

    suggestions = []
    for signal, color in (("SDA", "#00AAFF"), ("SCL", "#FFAA00")):
        source, target = pin_id(component_pins, signal), pin_id(board_pins, signal)
        if not source or not target:
            return []  # No partial or guessed I2C pin mapping.
        reason = f"Review {signal} signal connection; verify voltage levels, pull-ups and selected board variant before applying. Power wiring is not included."
        suggestions.append({"fromComponentId": component_id, "fromPin": source,
                            "toComponentId": board_id, "toPin": target, "color": color,
                            "from": f"{component_id}:{source}", "to": f"{board_id}:{target}",
                            "description": reason, "reason": reason})
    return suggestions


def workspace_wire_suggestions(board_type, components, wires):
    boards = [c for c in components if (c.get("type") or c.get("componentType")) == board_type and c.get("id")]
    if len(boards) != 1:
        return []
    board = boards[0]
    suggestions = []
    for component in components:
        if component.get("id") == board["id"]:
            continue
        for proposal in resolve_pin_suggestions(
            component.get("type") or component.get("componentType") or "", board_type,
            component.get("id"), board["id"], component.get("pins"), board.get("pins"),
        ):
            source = (proposal["fromComponentId"], proposal["fromPin"])
            target = (proposal["toComponentId"], proposal["toPin"])
            def endpoint(wire, side):
                node = wire.get(side + "NodeId") or wire.get(side + "ComponentId") or wire.get(side + "Component")
                pin = wire.get(side + "PinId") or wire.get(side + "Pin")
                if not node and isinstance(wire.get(side), str) and ":" in wire[side]:
                    return tuple(wire[side].rsplit(":", 1))
                return node, pin
            # Do not replace or duplicate existing endpoint connections.
            if any(source in (endpoint(w, "from"), endpoint(w, "to")) for w in wires):
                continue
            if any(target in (endpoint(w, "from"), endpoint(w, "to")) for w in wires):
                continue
            suggestions.append(proposal)
    return suggestions

def validate_circuit_state(
    board_type: str,
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Bounded heuristics, not a full netlist solver or electrical safety certification."""
    issues = []
    has_mcu = bool(board_type and board_type != "unspecified")
    has_components = len(components) > 0

    def endpoint_pin(wire, side):
        # The Playground sends fromPin/toPin; saved circuits use *PinId.
        # Older API clients sent a combined component:pin endpoint.
        return str(wire.get(side + "Pin") or wire.get(side + "PinId") or
                   str(wire.get(side, "")).rsplit(":", 1)[-1]).upper()
    
    # 1. Short Circuit Detection (Direct VCC to GND wire)
    for wire in wires:
        from_pin = endpoint_pin(wire, "from")
        to_pin = endpoint_pin(wire, "to")
        if ("VCC" in from_pin or "5V" in from_pin or "3V3" in from_pin) and ("GND" in to_pin):
            issues.append({
                "severity": "CRITICAL",
                "message": f"Direct short circuit detected between power rail '{from_pin}' and ground '{to_pin}'!",
                "componentId": wire.get("id", "wire-short")
            })
        elif ("GND" in from_pin) and ("VCC" in to_pin or "5V" in to_pin or "3V3" in to_pin):
            issues.append({
                "severity": "CRITICAL",
                "message": f"Direct short circuit detected between ground '{from_pin}' and power rail '{to_pin}'!",
                "componentId": wire.get("id", "wire-short")
            })
            
    # 2. LED Overcurrent Check (LED connected directly without resistor)
    led_components = [c for c in components if str(c.get("type", "")).upper().startswith(("LED_", "LED-")) or str(c.get("type", "")).upper() == "LED"]
    resistors = [c for c in components if "RESISTOR" in str(c.get("type", "")).upper()]
    
    if led_components and not resistors:
        issues.append({
            "severity": "HIGH",
            "message": "No current-limiting resistor is present. Check whether the LED requires an external resistor; this check does not determine connectivity or module-internal protection.",
            "componentId": led_components[0].get("id", "led-missing-resistor")
        })

    # 3. Floating Microcontroller Ground
    if has_mcu and has_components and len(wires) > 0:
        has_gnd_wire = any("GND" in endpoint_pin(w, "from") or "GND" in endpoint_pin(w, "to") for w in wires)
        if not has_gnd_wire:
            issues.append({
                "severity": "WARNING",
                "message": "No common ground (GND) connection detected between the board and peripherals.",
                "componentId": "board-floating-ground"
            })
            
    score = 100
    for issue in issues:
        sev = issue.get("severity")
        if sev == "CRITICAL":
            score -= 40
        elif sev == "HIGH":
            score -= 20
        elif sev == "WARNING":
            score -= 10
            
    score = max(0, min(100, score))
    is_valid = not any(i.get("severity") == "CRITICAL" for i in issues)
    
    return {
        "isValid": is_valid,
        "safetyScore": score,
        "generalFeedback": (
            f"Basic heuristic checks found {len(issues)} potential issue(s). Verify connectivity, ratings and operating conditions with the simulator and component documentation."
            if issues else "No issues found by the limited checks. This is not a complete electrical validation."
        ),
        "issues": issues,
        "suggestedFixes": [i["message"] for i in issues if i.get("severity") in ("CRITICAL", "HIGH")]
    }
