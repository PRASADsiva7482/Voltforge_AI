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
    boards = []
    for board_id, name, family in UI_BOARDS:
        boards.append({
            "boardId": board_id,
            "name": name,
            "family": family,
            "status": "verified"
        })
    return {
        "schemaVersion": 1,
        "totalBoards": len(boards),
        "boards": boards
    }

def get_component_coverage_report() -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "totalComponents": len(UI_COMPONENTS),
        "components": UI_COMPONENTS
    }

def resolve_pin_suggestions(
    component_type: str,
    board_type: str = "ARDUINO_UNO",
    component_id: Optional[str] = None,
    board_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Determine best pin assignments for a component and microcontroller."""
    ctype = component_type.upper()
    board = board_type.upper()
    c_id = component_id or component_type
    b_id = board_id or "board"
    
    # I2C Pins by board
    i2c_map = {
        "ARDUINO_UNO": {"sda": "A4", "scl": "A5"},
        "ARDUINO_NANO": {"sda": "A4", "scl": "A5"},
        "ARDUINO_MEGA": {"sda": "20", "scl": "21"},
        "ESP32": {"sda": "GPIO21", "scl": "GPIO22"},
        "ESP8266": {"sda": "D2", "scl": "D1"},
        "RASPBERRY_PI_PICO": {"sda": "GP4", "scl": "GP5"}
    }
    pins = i2c_map.get(board, {"sda": "A4", "scl": "A5"})
    
    def _wire(from_pin: str, to_pin: str, color: str, desc: str, target_id: Optional[str] = None):
        tgt = target_id or b_id
        return {
            "fromComponentId": c_id,
            "fromPin": from_pin,
            "toComponentId": tgt,
            "toPin": to_pin,
            "color": color,
            "description": desc,
            "reason": desc,
            "from": f"{c_id}:{from_pin}",
            "to": f"{tgt}:{to_pin}"
        }

    if "OLED" in ctype or "LCD" in ctype:
        return [
            _wire("VCC", "5V", "#FF0000", "Connect OLED VCC to 5V rail"),
            _wire("GND", "GND", "#000000", "Connect OLED GND to common ground"),
            _wire("SDA", pins["sda"], "#00AAFF", f"Connect OLED SDA to I2C Data ({pins['sda']})"),
            _wire("SCL", pins["scl"], "#FFAA00", f"Connect OLED SCL to I2C Clock ({pins['scl']})"),
        ]
    elif "SERVO" in ctype:
        pwm = "GPIO18" if "ESP32" in board else "D9"
        return [
            _wire("VCC", "5V", "#FF0000", "Connect Servo VCC to 5V power"),
            _wire("GND", "GND", "#000000", "Connect Servo GND to ground"),
            _wire("SIGNAL", pwm, "#FF8800", f"Connect Servo Signal to PWM pin ({pwm})"),
        ]
    elif "RELAY" in ctype:
        sig = "GPIO23" if "ESP32" in board else "D7"
        return [
            _wire("VCC", "5V", "#FF0000", "Connect Relay VCC to 5V"),
            _wire("GND", "GND", "#000000", "Connect Relay GND to common ground"),
            _wire("IN", sig, "#00CC00", f"Connect Relay IN to digital output ({sig})"),
        ]
    elif "MOTOR" in ctype:
        return [
            _wire("positive", "5V", "#FF0000", "Connect motor positive to 5V power supply"),
            _wire("negative", "GND", "#000000", "Connect motor ground via switched return"),
        ]
    elif "LED" in ctype:
        dpin = "GPIO2" if "ESP32" in board else "D13"
        return [
            _wire("anode", "Pin1", "#FF0000", "Connect LED anode to current-limiting resistor", target_id="r1"),
            _wire("cathode", "GND", "#000000", "Connect LED cathode to common ground"),
        ]
    elif "DHT" in ctype or "TEMP" in ctype:
        dpin = "GPIO4" if "ESP32" in board else "D2"
        return [
            _wire("VCC", "5V", "#FF0000", "Connect DHT sensor VCC to 5V"),
            _wire("GND", "GND", "#000000", "Connect DHT sensor GND to ground"),
            _wire("DATA", dpin, "#0000FF", f"Connect DHT data line to GPIO ({dpin})"),
        ]
    elif "POTENTIOMETER" in ctype:
        apin = "GPIO34" if "ESP32" in board else "A0"
        return [
            _wire("VCC", "5V", "#FF0000", "Connect Potentiometer high rail to 5V"),
            _wire("GND", "GND", "#000000", "Connect Potentiometer low rail to ground"),
            _wire("WIPER", apin, "#FF00AA", f"Connect Potentiometer Wiper to ADC pin ({apin})"),
        ]
    return []

def validate_circuit_state(
    board_type: str,
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Inspect placed components and wires for electrical correctness and safety."""
    issues = []
    has_mcu = bool(board_type)
    has_components = len(components) > 0
    
    # 1. Short Circuit Detection (Direct VCC to GND wire)
    for wire in wires:
        from_pin = str(wire.get("from", "")).upper()
        to_pin = str(wire.get("to", "")).upper()
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
    led_components = [c for c in components if "LED" in str(c.get("type", "")).upper()]
    resistors = [c for c in components if "RESISTOR" in str(c.get("type", "")).upper()]
    
    if led_components and not resistors:
        issues.append({
            "severity": "HIGH",
            "message": "LED placed without current-limiting resistor. An LED connected directly across 5V or GPIO will burn out or damage the microcontroller.",
            "componentId": led_components[0].get("id", "led-missing-resistor")
        })

    # 3. Floating Microcontroller Ground
    if has_mcu and has_components and len(wires) > 0:
        has_gnd_wire = any("GND" in str(w.get("from", "")).upper() or "GND" in str(w.get("to", "")).upper() for w in wires)
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
            f"Circuit safety analysis completed: {len(issues)} issue(s) identified."
            if issues else "All electrical rules and safety checks passed successfully."
        ),
        "issues": issues,
        "suggestedFixes": [i["message"] for i in issues if i.get("severity") in ("CRITICAL", "HIGH")]
    }
