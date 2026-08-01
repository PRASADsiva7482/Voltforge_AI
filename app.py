import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voltforge-ai")

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.txt")
qa_dataset: List[Dict[str, str]] = []


def load_dataset() -> None:
    global qa_dataset
    qa_dataset = []
    if not os.path.exists(DATASET_PATH):
        logger.warning("dataset.txt not found at %s", DATASET_PATH)
        return

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as handle:
            content = handle.read()
        for chunk in content.split("[Q]"):
            if not chunk.strip() or "[A]" not in chunk:
                continue
            question, answer = chunk.split("[A]", 1)
            qa_dataset.append({"question": question.strip(), "answer": answer.strip()})
        logger.info("Loaded %s local QA templates", len(qa_dataset))
    except Exception as exc:
        logger.error("Failed to load dataset.txt: %s", exc)


load_dataset()

app = FastAPI(title="VoltForge AI Microservice", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/voltForge-ai/api/v1/model")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""
    canvasContext: Optional[str] = ""
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    hasCode: bool = False
    generatedCode: Optional[str] = None
    confidence: float = 0.8
    citations: List[Dict[str, str]] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    context: Optional[str] = ""
    compilerDiagnostics: List[str] = Field(default_factory=list)


class CodeReviewRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    code: str = ""
    componentTypes: List[str] = Field(default_factory=list)
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    circuitDescription: Optional[str] = ""
    compilerDiagnostics: List[str] = Field(default_factory=list)


class SchematicToCodeRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    additionalInstructions: Optional[str] = ""


class GenerateCodeRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    prompt: Optional[str] = ""
    componentTypes: List[str] = Field(default_factory=list)


DOMAIN_TERMS = {
    "arduino",
    "esp32",
    "esp8266",
    "rp2040",
    "pico",
    "stm32",
    "teensy",
    "microcontroller",
    "mcu",
    "gpio",
    "pin",
    "pins",
    "circuit",
    "schematic",
    "wire",
    "wiring",
    "ground",
    "gnd",
    "vcc",
    "5v",
    "3.3v",
    "sensor",
    "led",
    "resistor",
    "motor",
    "relay",
    "servo",
    "lcd",
    "i2c",
    "spi",
    "dht",
    "firmware",
    "sketch",
    "cpp",
    "c++",
    "compile",
    "serial",
    "analogread",
    "digitalwrite",
    "pinmode",
    "voltforge",
}

OUT_OF_DOMAIN_TERMS = {
    "recipe",
    "cookie",
    "cake",
    "movie",
    "song",
    "poem",
    "essay",
    "history of rome",
    "stock",
    "weather",
    "football",
    "cricket score",
    "dating",
    "travel",
}

MCU_PREFIXES = (
    "ARDUINO",
    "ESP32",
    "ESP8266",
    "RASPBERRY_PI_PICO",
    "STM32",
    "TEENSY",
    "SEEED_XIAO",
    "ADAFRUIT_FEATHER",
    "SPARKFUN_THING_PLUS",
    "ATMEL_AVR",
)


def safe_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def wire_from_component(wire: Dict[str, Any]) -> str:
    return str(
        wire.get("fromComponent")
        or wire.get("fromComponentId")
        or wire.get("fromNodeId")
        or ""
    )


def wire_to_component(wire: Dict[str, Any]) -> str:
    return str(
        wire.get("toComponent")
        or wire.get("toComponentId")
        or wire.get("toNodeId")
        or ""
    )


def wire_from_pin(wire: Dict[str, Any]) -> str:
    return str(wire.get("fromPin") or wire.get("fromPinId") or "")


def wire_to_pin(wire: Dict[str, Any]) -> str:
    return str(wire.get("toPin") or wire.get("toPinId") or "")


def normalize_pin(pin: Any, default_kind: str = "digital") -> str:
    text = safe_lower(pin).replace(" ", "").replace("_", "")
    if not text:
        return ""
    if re.fullmatch(r"\d+", text):
        return f"a{text}" if default_kind == "analog" else f"d{text}"
    if re.fullmatch(r"d\d+", text) or re.fullmatch(r"a\d+", text):
        return text
    if text.startswith("gpio") and text[4:].isdigit():
        return f"d{text[4:]}"
    return text


def code_pin(pin: str) -> str:
    normalized = normalize_pin(pin)
    if normalized.startswith("d") and normalized[1:].isdigit():
        return normalized[1:]
    if normalized.startswith("a") and normalized[1:].isdigit():
        return normalized.upper()
    return pin


def display_pin(pin: str) -> str:
    normalized = normalize_pin(pin)
    if normalized.startswith("d") and normalized[1:].isdigit():
        return f"D{normalized[1:]}"
    if normalized.startswith("a") and normalized[1:].isdigit():
        return normalized.upper()
    return str(pin).upper()


def component_label(component: Dict[str, Any]) -> str:
    return f"{component.get('id', '')} {component.get('type', '')} {component.get('name', '')}".lower()


def component_type(component: Dict[str, Any]) -> str:
    return str(component.get("type") or component.get("name") or "").upper()


def is_mcu(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    name = str(component.get("name") or "").upper()
    return kind.startswith(MCU_PREFIXES) or any(prefix in name for prefix in MCU_PREFIXES)


def is_led(component: Dict[str, Any]) -> bool:
    return "LED" in component_type(component) or "led" in component_label(component)


def is_resistor(component: Dict[str, Any]) -> bool:
    return "RESISTOR" in component_type(component) or "resistor" in component_label(component)


def is_motor(component: Dict[str, Any]) -> bool:
    return "MOTOR" in component_type(component)


def is_servo(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    return "SERVO" in kind


def is_relay(component: Dict[str, Any]) -> bool:
    return "RELAY" in component_type(component)


def is_dht(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    return "DHT11" in kind or "DHT22" in kind


def is_i2c_display(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    label = component_label(component)
    return "I2C" in kind or "OLED" in kind or "LCD_I2C" in kind or "lcd i2c" in label


def pins_for(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    pins = component.get("pins") or []
    return pins if isinstance(pins, list) else []


def pin_label(component: Dict[str, Any], pin_id: str) -> str:
    normalized = safe_lower(pin_id)
    for pin in pins_for(component):
        if safe_lower(pin.get("id")) == normalized:
            return f"{pin.get('id', '')} {pin.get('name', '')} {pin.get('type', '')}".lower()
    return normalized


def pin_matches(component: Dict[str, Any], pin_id: str, roles: List[str]) -> bool:
    label = pin_label(component, pin_id)
    return any(role.lower() in label for role in roles)


def find_role_pin(component: Dict[str, Any], roles: List[str], fallback: Optional[str] = None) -> Optional[str]:
    for pin in pins_for(component):
        label = f"{pin.get('id', '')} {pin.get('name', '')} {pin.get('type', '')}".lower()
        if any(role.lower() in label for role in roles):
            return str(pin.get("id"))
    return fallback


def is_power_pin(component: Dict[str, Any], pin_id: str) -> bool:
    label = pin_label(component, pin_id)
    if any(blocked in label for blocked in ("reset", "rst", "aref", "vref", "en", "boot")):
        return False
    return any(token in label for token in ("5v", "3v3", "3.3v", "vin", "vcc", "vdd", "vbus", "vsys", "bat", "power"))


def is_ground_pin(component: Dict[str, Any], pin_id: str) -> bool:
    label = pin_label(component, pin_id)
    return any(token in label for token in ("gnd", "ground", "vss", "neg", "-"))


def is_signal_pin(component: Dict[str, Any], pin_id: str) -> bool:
    return not is_power_pin(component, pin_id) and not is_ground_pin(component, pin_id)


def find_mcu(components: List[Dict[str, Any]], board_type: str = "ARDUINO_UNO") -> Optional[Dict[str, Any]]:
    for component in components:
        if is_mcu(component):
            return component
    for component in components:
        if component_type(component) == board_type.upper():
            return component
    return None


def component_by_id(components: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(component.get("id")): component for component in components if component.get("id")}


def build_connections(wires: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    connections: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for wire in wires:
        a_comp, a_pin = wire_from_component(wire), wire_from_pin(wire)
        b_comp, b_pin = wire_to_component(wire), wire_to_pin(wire)
        if not a_comp or not b_comp or not a_pin or not b_pin:
            continue
        connections.setdefault(a_comp, {}).setdefault(a_pin, []).append(
            {"component": b_comp, "pin": b_pin, "wire": wire}
        )
        connections.setdefault(b_comp, {}).setdefault(b_pin, []).append(
            {"component": a_comp, "pin": a_pin, "wire": wire}
        )
    return connections


def pin_is_connected(connections: Dict[str, Dict[str, List[Dict[str, Any]]]], comp_id: str, pin_id: Optional[str]) -> bool:
    return bool(pin_id and connections.get(comp_id, {}).get(pin_id))


def wire_exists(wires: List[Dict[str, Any]], a_comp: str, a_pin: str, b_comp: str, b_pin: str) -> bool:
    want = {
        (a_comp, normalize_pin(a_pin), b_comp, normalize_pin(b_pin)),
        (b_comp, normalize_pin(b_pin), a_comp, normalize_pin(a_pin)),
    }
    for wire in wires:
        current = (
            wire_from_component(wire),
            normalize_pin(wire_from_pin(wire)),
            wire_to_component(wire),
            normalize_pin(wire_to_pin(wire)),
        )
        if current in want:
            return True
    return False


def make_wire_suggestion(
    from_component_id: str,
    from_pin: str,
    to_component_id: str,
    to_pin: str,
    color: str,
    description: str,
) -> Dict[str, str]:
    return {
        "fromComponentId": from_component_id,
        "fromPin": from_pin,
        "toComponentId": to_component_id,
        "toPin": to_pin,
        "color": color,
        "description": description,
    }


def expected_i2c_pins(board_type: str) -> Tuple[str, str]:
    board = (board_type or "").upper()
    if board.startswith("ESP32"):
        return "d21", "d22"
    if board.startswith("ESP8266"):
        return "d2", "d1"
    if "PICO" in board or "XIAO" in board:
        return "d4", "d5"
    return "a4", "a5"


def default_signal_pin(board_type: str, used: Set[str], preferred: Optional[str] = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    board = (board_type or "").upper()
    if board.startswith("ESP32"):
        candidates.extend(["d21", "d22", "d23", "d19", "d18", "d5", "d4", "d2", "d15"])
    else:
        candidates.extend(["d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13"])
    for candidate in candidates:
        if normalize_pin(candidate) not in used:
            used.add(normalize_pin(candidate))
            return candidate
    return candidates[-1]


def mcu_power_pin(mcu: Optional[Dict[str, Any]], board_type: str) -> str:
    if mcu:
        if (board_type or "").upper().startswith(("ESP", "RASPBERRY_PI_PICO", "STM32", "TEENSY")):
            found = find_role_pin(mcu, ["3v3", "3.3v"], None)
            if found:
                return found
        return find_role_pin(mcu, ["5v", "vbus", "vin", "3v3", "3.3v"], "5v") or "5v"
    return "3v3" if (board_type or "").upper().startswith(("ESP", "RASPBERRY_PI_PICO")) else "5v"


def mcu_ground_pin(mcu: Optional[Dict[str, Any]]) -> str:
    return find_role_pin(mcu, ["gnd", "ground"], "gnd") if mcu else "gnd"


def mcu_pin_connected_to_component(
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
    target: Dict[str, Any],
    target_roles: List[str],
) -> Optional[str]:
    mcu = find_mcu(components)
    if not mcu:
        return None
    mcu_id = str(mcu.get("id"))
    target_id = str(target.get("id"))
    by_id = component_by_id(components)

    for wire in wires:
        pairs = [
            (wire_from_component(wire), wire_from_pin(wire), wire_to_component(wire), wire_to_pin(wire)),
            (wire_to_component(wire), wire_to_pin(wire), wire_from_component(wire), wire_from_pin(wire)),
        ]
        for left_comp, left_pin, right_comp, right_pin in pairs:
            if left_comp == mcu_id and right_comp == target_id and pin_matches(target, right_pin, target_roles):
                return normalize_pin(left_pin)

    connections = build_connections(wires)
    for target_pin, neighbors in connections.get(target_id, {}).items():
        if not pin_matches(target, target_pin, target_roles):
            continue
        for neighbor in neighbors:
            resistor = by_id.get(neighbor["component"])
            if not resistor or not is_resistor(resistor):
                continue
            for resistor_pin, resistor_neighbors in connections.get(str(resistor.get("id")), {}).items():
                if resistor_pin == neighbor["pin"]:
                    continue
                for next_neighbor in resistor_neighbors:
                    if next_neighbor["component"] == mcu_id:
                        return normalize_pin(next_neighbor["pin"])
    return None


def extract_pin_usages(code: str) -> List[Dict[str, Any]]:
    if not code:
        return []

    constants: Dict[str, str] = {}
    for match in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)\s+([ADad]?\d+)\b", code, re.MULTILINE):
        constants[match.group(1)] = match.group(2)
    for match in re.finditer(
        r"\b(?:const\s+)?(?:int|byte|uint8_t|uint16_t|long)\s+([A-Za-z_]\w*)\s*=\s*([ADad]?\d+)\b",
        code,
    ):
        constants[match.group(1)] = match.group(2)

    usages: List[Dict[str, Any]] = []
    call_pattern = re.compile(
        r"\b(pinMode|digitalWrite|digitalRead|analogRead|analogWrite|tone|noTone)\s*\(\s*([A-Za-z_]\w*|[ADad]?\d+)"
    )
    line_starts = [0]
    for match in re.finditer(r"\n", code):
        line_starts.append(match.end())

    def line_number(position: int) -> int:
        line = 1
        for start in line_starts:
            if start <= position:
                line += 1
            else:
                break
        return max(1, line - 1)

    for match in call_pattern.finditer(code):
        fn = match.group(1)
        raw = match.group(2)
        resolved = constants.get(raw, raw)
        kind = "analog" if fn == "analogRead" and re.fullmatch(r"\d+", resolved) else "digital"
        normalized = normalize_pin(resolved, kind)
        if normalized:
            usages.append({"pin": normalized, "raw": raw, "function": fn, "line": line_number(match.start())})
    return usages


def connected_mcu_signal_pins(components: List[Dict[str, Any]], wires: List[Dict[str, Any]]) -> Set[str]:
    mcu = find_mcu(components)
    if not mcu:
        return set()
    mcu_id = str(mcu.get("id"))
    pins: Set[str] = set()
    for wire in wires:
        for comp_id, pin_id in (
            (wire_from_component(wire), wire_from_pin(wire)),
            (wire_to_component(wire), wire_to_pin(wire)),
        ):
            if comp_id == mcu_id and is_signal_pin(mcu, pin_id):
                pins.add(normalize_pin(pin_id))
    return pins


def analyze_code_against_context(
    code: str,
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    fixes: List[Dict[str, Any]] = []
    usages = extract_pin_usages(code)
    connected = connected_mcu_signal_pins(components, wires)
    if not code or not connected or not usages:
        return issues, fixes

    used = {usage["pin"] for usage in usages}
    mismatches = sorted(used - connected)
    if not mismatches:
        return issues, fixes

    suggested = sorted(connected)[0]
    mcu = find_mcu(components, "")
    mcu_id = str(mcu.get("id")) if mcu else "code"
    for pin in mismatches[:3]:
        issue = {
            "severity": "WARNING",
            "componentId": mcu_id,
            "message": f"Code uses {display_pin(pin)}, but that MCU pin is not connected on the canvas.",
            "suggestedFix": f"Use the wired pin {display_pin(suggested)} in firmware or move the canvas wire to {display_pin(pin)}.",
        }
        issues.append(issue)
        for usage in usages:
            if usage["pin"] == pin:
                fixes.append(
                    {
                        "type": "replace",
                        "from": str(usage["raw"]),
                        "to": code_pin(suggested),
                        "line": usage["line"],
                        "description": f"Replace {usage['raw']} with {code_pin(suggested)} for the canvas-wired pin.",
                    }
                )
                break
    return issues, fixes


def generate_wiring_suggestions(payload: ValidateRequest) -> List[Dict[str, str]]:
    components = payload.components or []
    wires = payload.wires or []
    board_type = payload.boardType or "ARDUINO_UNO"
    mcu = find_mcu(components, board_type)
    if not mcu:
        return []

    mcu_id = str(mcu.get("id"))
    mcu_gnd = mcu_ground_pin(mcu)
    mcu_power = mcu_power_pin(mcu, board_type)
    sda_pin, scl_pin = expected_i2c_pins(board_type)
    used = connected_mcu_signal_pins(components, wires)
    connections = build_connections(wires)
    suggestions: List[Dict[str, str]] = []

    def add_if_missing(to_comp: Dict[str, Any], to_pin: Optional[str], from_pin: str, color: str, description: str) -> None:
        if not to_pin:
            return
        comp_id = str(to_comp.get("id"))
        if pin_is_connected(connections, comp_id, to_pin):
            return
        if wire_exists(wires, mcu_id, from_pin, comp_id, to_pin):
            return
        suggestions.append(make_wire_suggestion(mcu_id, from_pin, comp_id, to_pin, color, description))

    for component in components:
        if is_mcu(component):
            continue
        comp_id = str(component.get("id"))
        kind = component_type(component)

        vcc = find_role_pin(component, ["vcc", "vdd", "vin", "5v", "3v3", "+"])
        gnd = find_role_pin(component, ["gnd", "ground", "vss", "neg", "-"])
        if vcc:
            add_if_missing(component, vcc, mcu_power, "#ef4444", f"Power {component.get('name', kind)} from {display_pin(mcu_power)}")
        if gnd:
            add_if_missing(component, gnd, mcu_gnd, "#555555", f"Connect {component.get('name', kind)} ground to MCU GND")

        if is_led(component):
            anode = find_role_pin(component, ["anode", "a", "+", "red", "green", "blue"], "anode")
            cathode = find_role_pin(component, ["cathode", "k", "gnd", "-", "neg"], "cathode")
            add_if_missing(component, anode, default_signal_pin(board_type, used, "d9"), "#3b82f6", "Drive LED through a 220 Ohm series resistor")
            add_if_missing(component, cathode, mcu_gnd, "#555555", "Return LED cathode to GND")
        elif is_dht(component):
            data_pin = find_role_pin(component, ["data", "dat", "out", "sig"], "data")
            add_if_missing(component, data_pin, default_signal_pin(board_type, used, "d2"), "#22c55e", "DHT data signal")
        elif is_i2c_display(component):
            add_if_missing(component, find_role_pin(component, ["sda"], "sda"), sda_pin, "#a855f7", "I2C SDA")
            add_if_missing(component, find_role_pin(component, ["scl"], "scl"), scl_pin, "#a855f7", "I2C SCL")
        elif is_servo(component):
            sig = find_role_pin(component, ["sig", "signal", "pwm", "s"], "sig")
            add_if_missing(component, sig, default_signal_pin(board_type, used, "d9"), "#f59e0b", "Servo PWM signal")
        elif is_relay(component):
            control = find_role_pin(component, ["in1", "in", "coil1", "sig"], "in")
            add_if_missing(component, control, default_signal_pin(board_type, used, "d7"), "#3b82f6", "Relay control signal")
        elif is_motor(component):
            # Bare motors should not be wired directly to GPIO. The validator explains this.
            continue
        elif "BUTTON" in kind or "PUSH" in kind:
            sig = find_role_pin(component, ["p1", "a", "1"], "p1")
            ret = find_role_pin(component, ["p2", "b", "2"], "p2")
            add_if_missing(component, sig, default_signal_pin(board_type, used, "d2"), "#22c55e", "Button input using INPUT_PULLUP")
            add_if_missing(component, ret, mcu_gnd, "#555555", "Button return to GND")

    return suggestions


def validate_project(payload: ValidateRequest) -> Dict[str, Any]:
    components = payload.components or []
    wires = payload.wires or []
    board_type = payload.boardType or "ARDUINO_UNO"
    by_id = component_by_id(components)
    connections = build_connections(wires)
    issues: List[Dict[str, Any]] = []
    additions: List[Dict[str, Any]] = []
    removals: List[Dict[str, Any]] = []
    code_fixes: List[Dict[str, Any]] = []

    mcu = find_mcu(components, board_type)
    mcu_id = str(mcu.get("id")) if mcu else ""

    for wire in wires:
        left = by_id.get(wire_from_component(wire), {})
        right = by_id.get(wire_to_component(wire), {})
        left_pin = wire_from_pin(wire)
        right_pin = wire_to_pin(wire)
        if not left or not right:
            continue
        if (is_power_pin(left, left_pin) and is_ground_pin(right, right_pin)) or (
            is_ground_pin(left, left_pin) and is_power_pin(right, right_pin)
        ):
            issues.append(
                {
                    "severity": "CRITICAL",
                    "componentId": wire_from_component(wire),
                    "message": "Power is wired directly to ground.",
                    "suggestedFix": "Remove the direct power-to-ground wire before powering or simulating the circuit.",
                }
            )
            removals.append(
                {
                    "type": "wire",
                    "wireId": wire.get("id"),
                    "between": [f"{wire_from_component(wire)}/{left_pin}", f"{wire_to_component(wire)}/{right_pin}"],
                    "reason": "Direct power-to-ground short.",
                }
            )

    for component in components:
        comp_id = str(component.get("id"))
        if is_led(component):
            anode = find_role_pin(component, ["anode", "a", "+", "red", "green", "blue"], "anode")
            cathode = find_role_pin(component, ["cathode", "k", "gnd", "-", "neg"], "cathode")
            if not pin_is_connected(connections, comp_id, anode):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "LED anode is not connected.",
                        "suggestedFix": "Connect the anode through a 220 Ohm resistor to an MCU output pin.",
                    }
                )
            if not pin_is_connected(connections, comp_id, cathode):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "LED cathode is not connected to ground.",
                        "suggestedFix": "Connect the LED cathode to MCU GND.",
                    }
                )

            has_resistor = False
            direct_drive_wire: Optional[Dict[str, Any]] = None
            direct_drive_endpoint: Optional[Tuple[str, str]] = None
            for pin in (anode, cathode):
                for neighbor in connections.get(comp_id, {}).get(pin or "", []):
                    neighbor_comp = by_id.get(neighbor["component"])
                    if not neighbor_comp:
                        continue
                    if is_resistor(neighbor_comp):
                        has_resistor = True
                    if is_mcu(neighbor_comp) and is_signal_pin(neighbor_comp, neighbor["pin"]):
                        direct_drive_wire = neighbor["wire"]
                        direct_drive_endpoint = (neighbor["component"], neighbor["pin"])
            if direct_drive_wire and not has_resistor:
                from_ref = f"{direct_drive_endpoint[0]}/{direct_drive_endpoint[1]}" if direct_drive_endpoint else "MCU/GPIO"
                led_ref = f"{comp_id}/{anode or 'anode'}"
                issues.append(
                    {
                        "severity": "CRITICAL",
                        "componentId": comp_id,
                        "message": "LED is connected directly to a GPIO pin without a current-limiting resistor.",
                        "suggestedFix": "Add a 220 Ohm resistor in series between the MCU output and LED anode.",
                    }
                )
                additions.append(
                    {
                        "type": "component",
                        "componentType": "RESISTOR",
                        "value": "220 Ohm",
                        "between": [from_ref, led_ref],
                        "reason": "Limit LED current to protect the LED and MCU pin.",
                    }
                )
                removals.append(
                    {
                        "type": "wire",
                        "wireId": direct_drive_wire.get("id"),
                        "between": [
                            f"{wire_from_component(direct_drive_wire)}/{wire_from_pin(direct_drive_wire)}",
                            f"{wire_to_component(direct_drive_wire)}/{wire_to_pin(direct_drive_wire)}",
                        ],
                        "reason": "Replace the direct GPIO-to-LED wire with a resistor path.",
                    }
                )

        if is_motor(component) and not is_servo(component):
            for pin, neighbors in connections.get(comp_id, {}).items():
                for neighbor in neighbors:
                    neighbor_comp = by_id.get(neighbor["component"])
                    if neighbor_comp and is_mcu(neighbor_comp) and is_signal_pin(neighbor_comp, neighbor["pin"]):
                        issues.append(
                            {
                                "severity": "CRITICAL",
                                "componentId": comp_id,
                                "message": "DC motor is wired directly to a microcontroller GPIO pin.",
                                "suggestedFix": "Use a transistor, MOSFET, relay module, or motor driver with a flyback diode and shared ground.",
                            }
                        )
                        removals.append(
                            {
                                "type": "wire",
                                "wireId": neighbor["wire"].get("id"),
                                "between": [
                                    f"{wire_from_component(neighbor['wire'])}/{wire_from_pin(neighbor['wire'])}",
                                    f"{wire_to_component(neighbor['wire'])}/{wire_to_pin(neighbor['wire'])}",
                                ],
                                "reason": "A GPIO pin cannot safely drive motor current.",
                            }
                        )

        if is_servo(component):
            sig = find_role_pin(component, ["sig", "signal", "pwm", "s"], "sig")
            vcc = find_role_pin(component, ["vcc", "5v", "vin", "+"])
            gnd = find_role_pin(component, ["gnd", "ground", "-"])
            if sig and not pin_is_connected(connections, comp_id, sig):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "Servo signal pin is not connected.",
                        "suggestedFix": "Connect servo signal to a PWM-capable digital pin such as D9.",
                    }
                )
            if vcc and not pin_is_connected(connections, comp_id, vcc):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "Servo power is missing.",
                        "suggestedFix": "Power the servo from a suitable 5V supply and share ground with the MCU.",
                    }
                )
            if gnd and not pin_is_connected(connections, comp_id, gnd):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "Servo ground is missing.",
                        "suggestedFix": "Connect servo GND to the MCU/common ground.",
                    }
                )

        if is_i2c_display(component):
            expected_sda, expected_scl = expected_i2c_pins(board_type)
            for role, expected in (("sda", expected_sda), ("scl", expected_scl)):
                comp_pin = find_role_pin(component, [role], role)
                for neighbor in connections.get(comp_id, {}).get(comp_pin or "", []):
                    neighbor_comp = by_id.get(neighbor["component"])
                    if neighbor_comp and is_mcu(neighbor_comp) and normalize_pin(neighbor["pin"]) != expected:
                        issues.append(
                            {
                                "severity": "WARNING",
                                "componentId": comp_id,
                                "message": f"I2C {role.upper()} is connected to {display_pin(neighbor['pin'])}, expected {display_pin(expected)} for {board_type}.",
                                "suggestedFix": f"Move {role.upper()} to {display_pin(expected)} or configure custom I2C pins in firmware if the board supports it.",
                            }
                        )

        if not is_mcu(component) and not is_resistor(component):
            has_signal = any(
                is_signal_pin(component, pin) and bool(neighbors)
                for pin, neighbors in connections.get(comp_id, {}).items()
            )
            gnd_pin = find_role_pin(component, ["gnd", "ground", "-"])
            if has_signal and gnd_pin and not pin_is_connected(connections, comp_id, gnd_pin):
                issues.append(
                    {
                        "severity": "WARNING",
                        "componentId": comp_id,
                        "message": "Component has a signal connection but no ground reference.",
                        "suggestedFix": "Connect component GND to the same ground as the microcontroller.",
                    }
                )

    code_issues, detected_code_fixes = analyze_code_against_context(payload.code or "", components, wires)
    issues.extend(code_issues)
    code_fixes.extend(detected_code_fixes)

    critical = sum(1 for issue in issues if issue["severity"] == "CRITICAL")
    warnings = sum(1 for issue in issues if issue["severity"] == "WARNING")
    safety_score = max(0, 100 - critical * 25 - warnings * 10)
    is_valid = critical == 0 and safety_score >= 75
    if not issues:
        feedback = "No unsafe wiring or obvious code/canvas pin mismatches detected."
    else:
        feedback = f"Found {critical} critical issue(s) and {warnings} warning(s). Review the suggested changes before simulation."

    return {
        "isValid": is_valid,
        "safetyScore": safety_score,
        "generalFeedback": feedback,
        "issues": issues,
        "additions": additions,
        "removals": removals,
        "wireSuggestions": generate_wiring_suggestions(payload),
        "codeFixes": code_fixes,
        "confidence": 0.92 if components else 0.65,
    }


def sanitize_name(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value or fallback)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"c_{text}"
    return text.lower()


def generate_code_from_context(
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
    board_type: str,
    instructions: str = "",
) -> str:
    includes: List[str] = []
    globals_: List[str] = []
    setup: List[str] = []
    loop: List[str] = []
    used_any = False

    def add_include(line: str) -> None:
        if line not in includes:
            includes.append(line)

    def add_setup(line: str) -> None:
        if line not in setup:
            setup.append(line)

    for component in components:
        comp_id = str(component.get("id") or component.get("name") or "component")
        name = sanitize_name(comp_id, "component")

        if is_dht(component):
            pin = mcu_pin_connected_to_component(components, wires, component, ["data", "dat", "out", "sig"]) or "d2"
            add_include("#include <DHT.h>")
            globals_.append(f"#define {name.upper()}_PIN {code_pin(pin)}")
            dht_type = "DHT22" if "DHT22" in component_type(component) else "DHT11"
            globals_.append(f"#define {name.upper()}_TYPE {dht_type}")
            globals_.append(f"DHT {name}({name.upper()}_PIN, {name.upper()}_TYPE);")
            add_setup("Serial.begin(9600);")
            add_setup(f"{name}.begin();")
            loop.extend(
                [
                    f"float {name}_humidity = {name}.readHumidity();",
                    f"float {name}_temperature = {name}.readTemperature();",
                    f"Serial.print(\"{component.get('name', 'DHT')} T: \");",
                    f"Serial.print({name}_temperature);",
                    "Serial.print(\" C  H: \");",
                    f"Serial.println({name}_humidity);",
                    "delay(2000);",
                ]
            )
            used_any = True

        elif is_led(component):
            pin = mcu_pin_connected_to_component(components, wires, component, ["anode", "a", "+", "red", "green", "blue"]) or "d13"
            const_name = f"{name}_pin"
            globals_.append(f"const int {const_name} = {code_pin(pin)};")
            add_setup(f"pinMode({const_name}, OUTPUT);")
            loop.extend(
                [
                    f"digitalWrite({const_name}, HIGH);",
                    "delay(500);",
                    f"digitalWrite({const_name}, LOW);",
                    "delay(500);",
                ]
            )
            used_any = True

        elif is_i2c_display(component):
            add_include("#include <Wire.h>")
            add_include("#include <LiquidCrystal_I2C.h>")
            address = str((component.get("properties") or {}).get("address", "0x27"))
            globals_.append(f"LiquidCrystal_I2C {name}({address}, 16, 2);")
            add_setup(f"{name}.init();")
            add_setup(f"{name}.backlight();")
            add_setup(f"{name}.setCursor(0, 0);")
            add_setup(f"{name}.print(\"VoltForge Ready\");")
            if not loop:
                loop.extend(["delay(1000);"])
            used_any = True

        elif is_servo(component):
            pin = mcu_pin_connected_to_component(components, wires, component, ["sig", "signal", "pwm", "s"]) or "d9"
            add_include("#include <Servo.h>")
            globals_.append(f"Servo {name};")
            globals_.append(f"const int {name}_pin = {code_pin(pin)};")
            add_setup(f"{name}.attach({name}_pin);")
            loop.extend(
                [
                    f"{name}.write(0);",
                    "delay(700);",
                    f"{name}.write(90);",
                    "delay(700);",
                ]
            )
            used_any = True

        elif is_relay(component):
            pin = mcu_pin_connected_to_component(components, wires, component, ["in", "in1", "coil1", "sig"]) or "d7"
            const_name = f"{name}_pin"
            globals_.append(f"const int {const_name} = {code_pin(pin)};")
            add_setup(f"pinMode({const_name}, OUTPUT);")
            loop.extend(
                [
                    f"digitalWrite({const_name}, HIGH);",
                    "delay(1000);",
                    f"digitalWrite({const_name}, LOW);",
                    "delay(1000);",
                ]
            )
            used_any = True

    if not used_any:
        globals_.append("const int ledPin = 13;")
        add_setup("pinMode(ledPin, OUTPUT);")
        loop.extend(
            [
                "digitalWrite(ledPin, HIGH);",
                "delay(1000);",
                "digitalWrite(ledPin, LOW);",
                "delay(1000);",
            ]
        )

    header = [
        "// Generated by VoltForge AI",
        f"// Board: {board_type or 'ARDUINO_UNO'}",
    ]
    if instructions:
        header.append(f"// Request: {instructions.strip()[:120]}")

    lines: List[str] = []
    lines.extend(header)
    if includes:
        lines.extend(includes)
    if globals_:
        lines.append("")
        lines.extend(globals_)
    lines.append("")
    lines.append("void setup() {")
    for line in setup or ["Serial.begin(9600);"]:
        lines.append(f"  {line}")
    lines.append("}")
    lines.append("")
    lines.append("void loop() {")
    for line in loop or ["delay(1000);"]:
        lines.append(f"  {line}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def review_code_payload(payload: CodeReviewRequest) -> Dict[str, Any]:
    code = payload.code or ""
    issues: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    score = 100

    if "void setup" not in code:
        issues.append(
            {
                "severity": "ERROR",
                "line": 1,
                "message": "Missing Arduino setup() function.",
                "fix": "Add `void setup() { ... }` for initialization.",
            }
        )
        score -= 35
    if "void loop" not in code:
        issues.append(
            {
                "severity": "ERROR",
                "line": 1,
                "message": "Missing Arduino loop() function.",
                "fix": "Add `void loop() { ... }` for repeated firmware logic.",
            }
        )
        score -= 35

    components = payload.components or [
        {"id": f"type_{index}", "type": comp_type, "name": comp_type}
        for index, comp_type in enumerate(payload.componentTypes)
    ]
    code_issues, code_fixes = analyze_code_against_context(code, components, payload.wires or [])
    for issue in code_issues:
        issues.append(
            {
                "severity": issue["severity"],
                "line": 1,
                "message": issue["message"],
                "fix": issue["suggestedFix"],
            }
        )
        score -= 10

    if payload.compilerDiagnostics:
        issues.append(
            {
                "severity": "ERROR",
                "line": 1,
                "message": "Compiler diagnostics are present.",
                "fix": "Review the compiler output and apply the first syntax/type fix before retrying.",
            }
        )
        suggestions.extend(payload.compilerDiagnostics[:3])
        score -= 15

    if not issues:
        suggestions.append("Sketch structure is valid. Keep pin constants aligned with the canvas wiring.")
    else:
        suggestions.append("Run validation after applying fixes so wiring and firmware stay in sync.")

    improved = code
    for fix in code_fixes[:1]:
        if fix.get("type") == "replace" and fix.get("from") and fix.get("to"):
            improved = improved.replace(str(fix["from"]), str(fix["to"]), 1)

    return {
        "summary": "Code review completed with project-aware checks.",
        "issues": issues,
        "suggestions": suggestions,
        "improvedCode": improved,
        "score": max(0, score),
        "confidence": 0.9,
    }


def parse_context(context_text: Optional[str]) -> Dict[str, Any]:
    if not context_text:
        return {}
    try:
        parsed = json.loads(context_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"notes": context_text}


def context_from_chat(payload: ChatRequest) -> Dict[str, Any]:
    return parse_context(payload.context or payload.canvasContext)


def is_out_of_domain(message: str) -> bool:
    lower = message.lower()
    if any(term in lower for term in DOMAIN_TERMS):
        return False
    return any(term in lower for term in OUT_OF_DOMAIN_TERMS)


def find_best_match(query: str) -> Optional[Dict[str, str]]:
    query_tokens = set(re.findall(r"[a-zA-Z0-9_+.#]+", query.lower()))
    if not query_tokens:
        return None
    best_score = 0.0
    best_match: Optional[Dict[str, str]] = None
    for item in qa_dataset:
        target_tokens = set(re.findall(r"[a-zA-Z0-9_+.#]+", item["question"].lower()))
        if not target_tokens:
            continue
        score = len(query_tokens & target_tokens) / ((len(query_tokens) * len(target_tokens)) ** 0.5)
        if score > best_score:
            best_score = score
            best_match = item
    return best_match if best_score >= 0.32 else None


def search_web(query: str, limit: int = 2) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        logger.info("Performing web search: %s", query)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        encoded = urllib.parse.quote(f"{query} Arduino electronics documentation")
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code != 200:
            return results
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup.select(".result__body")[:limit]:
            title = element.select_one(".result__title .result__a")
            snippet = element.select_one(".result__snippet")
            if title and snippet:
                results.append(
                    {
                        "title": title.get_text(strip=True),
                        "url": title.get("href", ""),
                        "snippet": snippet.get_text(" ", strip=True),
                    }
                )
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
    return results


def validation_markdown(result: Dict[str, Any]) -> str:
    if not result["issues"]:
        return f"Your circuit looks clean. Safety score: {result['safetyScore']}/100."
    lines = [
        f"Safety score: {result['safetyScore']}/100.",
        result["generalFeedback"],
        "",
        "Highest priority fixes:",
    ]
    for issue in result["issues"][:5]:
        lines.append(f"- {issue['severity']}: {issue['message']} Fix: {issue['suggestedFix']}")
    if result.get("additions") or result.get("removals") or result.get("codeFixes"):
        lines.append("")
        lines.append("I also returned structured additions, removals, and code fixes for the UI to apply.")
    return "\n".join(lines)


def suggestions_markdown(suggestions: List[Dict[str, str]]) -> str:
    if not suggestions:
        return "I do not see missing standard wires for the current canvas. Add a board and components, then ask again."
    lines = ["Suggested wiring:"]
    for item in suggestions[:8]:
        lines.append(
            f"- {item['fromComponentId']}/{display_pin(item['fromPin'])} -> "
            f"{item['toComponentId']}/{item['toPin']}: {item['description']}"
        )
    return "\n".join(lines)


def answer_common_question(message: str, context: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    board_type = context.get("boardType", "ARDUINO_UNO")
    if "resistor" in lower and "led" in lower:
        return "Use a current-limiting resistor in series with the LED. For a 5V Arduino and a common red LED, 220 Ohm is a good default; 330 Ohm is also safe and dimmer."
    if "lcd" in lower and "i2c" in lower:
        sda, scl = expected_i2c_pins(str(board_type))
        return f"For {board_type}, wire LCD VCC to power, GND to GND, SDA to {display_pin(sda)}, and SCL to {display_pin(scl)}. In code, include Wire.h and LiquidCrystal_I2C.h."
    if "dht" in lower:
        return "For DHT11/DHT22, connect VCC, GND, and DATA to a digital pin such as D2. Add a 10k pull-up from DATA to VCC if the sensor module does not already include one."
    if "motor" in lower:
        return "Do not drive a DC motor directly from an MCU GPIO. Use a transistor, MOSFET, relay module, or motor driver, add flyback protection, and connect grounds together."
    if "servo" in lower:
        return "A servo uses VCC, GND, and one PWM/control signal. Use D9 as a safe default signal pin on Arduino Uno, and power larger servos from a separate 5V supply with common ground."
    return None


@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "UP",
        "engine": "VoltForge-Hybrid-Assistant",
        "modelLoaded": False,
        "rulesLoaded": True,
        "ragEnabled": True,
        "datasetItems": len(qa_dataset),
        "version": "0.2.0",
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    logger.info("Processing chat request: %s", payload.message)
    context = context_from_chat(payload)
    message = payload.message.strip()
    lower = message.lower()

    if is_out_of_domain(message):
        return ChatResponse(
            reply=(
                "I am VoltForge AI, so I stay focused on electronics, circuit design, "
                "microcontrollers, firmware, wiring, and simulation. I cannot help with that outside-domain request."
            ),
            confidence=0.96,
        )

    components = context.get("components") or []
    wires = context.get("wires") or []
    code = context.get("code") or context.get("activeCode") or ""
    board_type = context.get("boardType", "ARDUINO_UNO")

    if any(term in lower for term in ("who are you", "what can you do", "help", "what are you doing")):
        project = context.get("projectName", "this project")
        return ChatResponse(
            reply=(
                f"I am VoltForge AI, a project-aware electronics assistant for {project}. "
                f"I can validate wiring, find code/canvas pin mismatches, suggest wires, review Arduino code, "
                f"and generate firmware for the active {board_type} layout."
            ),
            confidence=0.95,
        )

    if any(term in lower for term in ("validate", "safe", "short", "error", "fix my circuit", "check circuit")):
        result = validate_project(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(reply=validation_markdown(result), confidence=result["confidence"])

    if any(term in lower for term in ("suggest wire", "suggest wiring", "how to wire", "connections", "connect this")):
        suggestions = generate_wiring_suggestions(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(reply=suggestions_markdown(suggestions), confidence=0.9 if suggestions else 0.7)

    if any(term in lower for term in ("generate code", "write code", "schematic to code", "make firmware")):
        generated = generate_code_from_context(components, wires, board_type, message)
        return ChatResponse(
            reply=f"Here is firmware matched to your current {board_type} canvas:\n\n```cpp\n{generated}```",
            hasCode=True,
            generatedCode=generated,
            confidence=0.88 if components else 0.72,
        )

    if any(term in lower for term in ("review code", "is my code", "code correct", "compile error")):
        review = review_code_payload(
            CodeReviewRequest(boardType=board_type, code=code, components=components, wires=wires)
        )
        lines = [review["summary"], f"Score: {review['score']}/100"]
        for issue in review["issues"][:5]:
            lines.append(f"- {issue['severity']}: {issue['message']} Fix: {issue['fix']}")
        if not review["issues"]:
            lines.append("- No structural Arduino issues found.")
        return ChatResponse(reply="\n".join(lines), confidence=review["confidence"])

    common = answer_common_question(message, context)
    if common:
        return ChatResponse(reply=common, confidence=0.9)

    match = find_best_match(message)
    if match:
        answer = match["answer"]
        has_code = "```" in answer
        code_text = None
        if has_code:
            code_match = re.search(r"```(?:cpp|c\+\+|arduino)?\s*([\s\S]*?)```", answer)
            code_text = code_match.group(1).strip() if code_match else None
        return ChatResponse(reply=answer, hasCode=has_code, generatedCode=code_text, confidence=0.82)

    if any(term in lower for term in DOMAIN_TERMS):
        results = search_web(message)
        if results:
            reply_lines = [
                f"I did not have enough local certainty, so I checked technical references for: {message}",
                "",
            ]
            for result in results:
                reply_lines.append(f"- {result['title']}: {result['snippet']}")
            reply_lines.append("")
            reply_lines.append("Apply this against your canvas pins and verify with the firmware compiler before simulation.")
            return ChatResponse(reply="\n".join(reply_lines), confidence=0.68, citations=results)

    return ChatResponse(
        reply=(
            "I need a little more circuit-specific detail. Mention the component, board pin, wire, or code error you want me to analyze."
        ),
        confidence=0.55,
    )


@router.post("/validate-circuit")
def validate_circuit(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing circuit validation")
    return validate_project(payload)


@router.post("/suggest-wiring")
def suggest_wiring(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing wiring suggestions")
    suggestions = generate_wiring_suggestions(payload)
    return {
        "suggestions": suggestions,
        "wireSuggestions": suggestions,
        "confidence": 0.9 if suggestions else 0.7,
    }


@router.post("/review-code")
def review_code(payload: CodeReviewRequest) -> Dict[str, Any]:
    logger.info("Processing code review")
    return review_code_payload(payload)


@router.post("/schematic-to-code")
def schematic_to_code(payload: SchematicToCodeRequest) -> Dict[str, Any]:
    logger.info("Processing schematic-to-code")
    code = generate_code_from_context(
        payload.components,
        payload.wires,
        payload.boardType or "ARDUINO_UNO",
        payload.additionalInstructions or "",
    )
    return {
        "status": "SUCCESS",
        "message": "Code generated from the active schematic.",
        "generatedCode": code,
        "confidence": 0.88 if payload.components else 0.72,
    }


@router.post("/generate-code")
def generate_code_api(payload: GenerateCodeRequest) -> Dict[str, Any]:
    logger.info("Processing generate-code")
    code = generate_code_from_context(
        payload.components,
        payload.wires,
        payload.boardType or "ARDUINO_UNO",
        payload.prompt or "",
    )
    return {
        "status": "SUCCESS",
        "message": "Code generated from VoltForge project context.",
        "generatedCode": code,
        "confidence": 0.88 if payload.components else 0.72,
    }


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2002)
