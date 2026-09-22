"""Build fail-closed AI coverage for the UI component catalog.

Canvas component labels are presentation and simulation identities, not
manufacturer part numbers. This report inventories every non-board UI type and
keeps physical components variant-required until an exact evidenced part or
module revision is selected.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from typing import Any, Literal

from electronics_corpus.schema import CORPUS_SOURCE_REVISION
from electronics_corpus.store import ElectronicsCorpus


ComponentCoverageStatus = Literal[
    "verified", "variant-required", "simulation-only", "unsupported"
]
COMPONENT_COVERAGE_REPORT_VERSION = "1.0.0"
UI_COMPONENT_CATALOG_VERSION = "component-catalog-v1"

_UI_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("RESISTOR", "Resistor", "PASSIVE"),
    ("CAPACITOR", "Capacitor", "PASSIVE"),
    ("CERAMIC_CAPACITOR", "Ceramic Capacitor", "PASSIVE"),
    ("ELECTROLYTIC_CAPACITOR", "Electrolytic Capacitor", "PASSIVE"),
    ("DIODE", "Diode", "PASSIVE"),
    ("ZENER_DIODE", "Zener Diode", "PASSIVE"),
    ("SCHOTTKY_DIODE", "Schottky Diode", "PASSIVE"),
    ("INDUCTOR", "Inductor", "PASSIVE"),
    ("TRANSFORMER", "Transformer", "POWER"),
    ("VARIABLE_CAPACITOR", "Variable Capacitor", "PASSIVE"),
    ("THERMISTOR_NTC", "NTC Thermistor", "PASSIVE"),
    ("NMOS", "N-Channel MOSFET", "PASSIVE"),
    ("PMOS", "P-Channel MOSFET", "PASSIVE"),
    ("BRIDGE_RECTIFIER", "Bridge Rectifier", "POWER"),
    ("OPAMP_IDEAL", "Ideal Op-Amp", "PASSIVE"),
    ("OPAMP_LM358", "LM358 Op-Amp", "PASSIVE"),
    ("LED_STANDARD", "Standard LED", "LED"),
    ("LED_RGB", "RGB LED", "LED"),
    ("VOLTAGE_REGULATOR_7805", "7805 Voltage Regulator", "POWER"),
    ("BATTERY_9V", "9V Battery", "POWER"),
    ("BATTERY_AA", "AA Battery", "POWER"),
    ("DC_SOURCE_3V3", "3.3V DC Supply", "POWER"),
    ("DC_SOURCE_5V", "5V DC Supply", "POWER"),
    ("DC_SOURCE_12V", "12V DC Supply", "POWER"),
    ("POWER_SUPPLY", "Configurable DC Supply", "POWER"),
    ("AC_FUNCTION_GENERATOR", "AC Function Generator", "POWER"),
    ("GROUND", "Ground", "POWER"),
    ("MOTOR_DC", "DC Motor", "MOTOR"),
    ("SERVO_MOTOR", "Servo Motor", "MOTOR"),
    ("STEPPER_MOTOR", "Stepper Motor", "MOTOR"),
    ("MOTOR_STEPPER", "ULN2003 Stepper Driver", "MOTOR"),
    ("ESC_MODULE", "Brushless ESC", "MOTOR"),
    ("MOTOR_BLDC", "BLDC Motor", "MOTOR"),
    ("BUZZER", "Buzzer", "MOTOR"),
    ("PUSH_BUTTON", "Push Button", "PASSIVE"),
    ("SWITCH_SPST", "SPST Switch", "PASSIVE"),
    ("POTENTIOMETER", "Potentiometer", "PASSIVE"),
    ("RELAY_SINGLE", "Single Relay Module", "RELAY"),
    ("RELAY_SPDT", "SPDT Relay", "RELAY"),
    ("RELAY_2CH", "2-Channel Relay Module", "RELAY"),
    ("RELAY_4CH", "4-Channel Relay Module", "RELAY"),
    ("PIR_SENSOR", "PIR Motion Sensor", "SENSOR"),
    ("SENSOR_PIR", "PIR Motion Sensor", "SENSOR"),
    ("LDR", "Photoresistor", "SENSOR"),
    ("SENSOR_LDR", "Light Sensor", "SENSOR"),
    ("SOIL_MOISTURE", "Soil Moisture Sensor", "SENSOR"),
    ("LCD_16X2", "LCD 16x2", "DISPLAY"),
    ("DISPLAY_LCD_I2C", "I2C LCD 16x2", "DISPLAY"),
    ("OLED_DISPLAY", "OLED Display", "DISPLAY"),
    ("DISPLAY_OLED", "I2C OLED Display", "DISPLAY"),
    ("DISPLAY_7SEG", "7-Segment Display", "DISPLAY"),
    ("BREADBOARD", "Breadboard", "PASSIVE"),
    ("IC_555_TIMER", "NE555 Timer", "LOGIC"),
    ("IC_74HC595", "74HC595 Shift Register", "LOGIC"),
    ("IC_74HC165", "74HC165 Shift Register", "LOGIC"),
    ("IC_74HC138", "74HC138 3-to-8 Decoder", "LOGIC"),
    ("IC_74HC151", "74HC151 8-Ch Multiplexer", "LOGIC"),
    ("IC_CD4017", "CD4017 Decade Counter", "LOGIC"),
    ("MULTIMETER", "Digital Multimeter", "INSTRUMENT"),
    ("OSCILLOSCOPE", "Digital Oscilloscope", "INSTRUMENT"),
)

_SIMULATION_ONLY = frozenset(
    {
        "OPAMP_IDEAL",
        "DC_SOURCE_3V3",
        "DC_SOURCE_5V",
        "DC_SOURCE_12V",
        "POWER_SUPPLY",
        "AC_FUNCTION_GENERATOR",
        "GROUND",
        "MULTIMETER",
        "OSCILLOSCOPE",
    }
)

_SELECTION_GROUPS = {
    "PIR_SENSOR": "pir-motion-sensor-module",
    "SENSOR_PIR": "pir-motion-sensor-module",
    "LDR": "photoresistor",
    "SENSOR_LDR": "photoresistor",
    "OLED_DISPLAY": "oled-display-module",
    "DISPLAY_OLED": "oled-display-module",
    "RELAY_SINGLE": "relay-module",
    "RELAY_2CH": "relay-module",
    "RELAY_4CH": "relay-module",
}

_GATE_RECORDS = {
    "LED_STANDARD": "vf-knowledge-v1-component.generic.led",
    "RELAY_SINGLE": "vf-knowledge-v1-component.generic.relay-module",
    "RELAY_2CH": "vf-knowledge-v1-component.generic.relay-module",
    "RELAY_4CH": "vf-knowledge-v1-component.generic.relay-module",
}

_OLED_CANDIDATES = (
    "vf-knowledge-v1-component.adafruit.ssd1306-128x64-stemma",
    "vf-knowledge-v1-component.adafruit.ssd1306-128x64-v1",
)

_SELECTION_REQUIREMENTS = (
    "manufacturer",
    "orderable-part-number",
    "package-or-module-revision",
    "terminal-or-connector-pinout",
    "electrical-ratings-and-polarity",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _catalog_sha256(corpus: ElectronicsCorpus) -> str:
    return hashlib.sha256(corpus.catalog_path.read_bytes()).hexdigest()


def _record(corpus: ElectronicsCorpus, record_id: str) -> dict[str, Any]:
    record = corpus.by_id.get(record_id)
    if record is None or record.get("recordType") != "component":
        raise RuntimeError(f"Component coverage record is unavailable: {record_id}")
    return record


def _build_entries(corpus: ElectronicsCorpus) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for component_type, display_name, category in _UI_COMPONENTS:
        if component_type in _SIMULATION_ONLY:
            entries.append(
                {
                    "componentType": component_type,
                    "displayName": display_name,
                    "category": category,
                    "status": "simulation-only",
                    "reasonCode": "canvas-simulation-primitive",
                    "reason": "This UI type is a simulator source, reference, ideal model, or instrument rather than a selected physical part.",
                    "selectionGroup": None,
                    "selectionRequirements": [],
                    "physicalIdentitySelected": False,
                    "aiElectricalClaimsAllowed": False,
                }
            )
            continue
        entry: dict[str, Any] = {
            "componentType": component_type,
            "displayName": display_name,
            "category": category,
            "status": "variant-required",
            "reasonCode": "exact-component-variant-required",
            "reason": "The UI label does not select an exact manufacturer part, package or module revision, connector pinout, and bounded electrical ratings.",
            "selectionGroup": _SELECTION_GROUPS.get(
                component_type, component_type.casefold().replace("_", "-")
            ),
            "selectionRequirements": list(_SELECTION_REQUIREMENTS),
            "physicalIdentitySelected": False,
            "aiElectricalClaimsAllowed": False,
        }
        gate_id = _GATE_RECORDS.get(component_type)
        if gate_id is not None:
            gate = _record(corpus, gate_id)
            if gate.get("supportStatus") != "variant-required":
                raise RuntimeError(f"Component variant gate drifted: {gate_id}")
            entry["variantGateRecordId"] = gate_id
        if component_type in {"OLED_DISPLAY", "DISPLAY_OLED"}:
            candidates = []
            for record_id in _OLED_CANDIDATES:
                candidate = _record(corpus, record_id)
                candidates.append(
                    {
                        "recordId": record_id,
                        "supportStatus": candidate["supportStatus"],
                        "variant": candidate["subject"]["variant"],
                    }
                )
            entry["curatedExactCandidates"] = candidates
            entry["candidateSelectionRequired"] = True
        entries.append(entry)
    return entries


@lru_cache(maxsize=1)
def validate_component_coverage() -> dict[str, Any]:
    corpus = ElectronicsCorpus()
    entries = _build_entries(corpus)
    if len(entries) != len(_UI_COMPONENTS) or len(
        {item["componentType"] for item in entries}
    ) != len(entries):
        raise RuntimeError(
            "Component coverage must contain one unique entry per non-board UI component type."
        )
    counts = Counter(item["status"] for item in entries)
    if counts["verified"] != 0:
        raise RuntimeError("A generic UI component was silently promoted to verified.")
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-011-ui-component-coverage",
        "reportVersion": COMPONENT_COVERAGE_REPORT_VERSION,
        "uiCatalogVersion": UI_COMPONENT_CATALOG_VERSION,
        "corpusVersion": CORPUS_SOURCE_REVISION,
        "corpusCatalogSha256": _catalog_sha256(corpus),
        "asOfDate": "2026-09-01",
        "entryCount": len(entries),
        "summary": {
            "verified": counts["verified"],
            "variantRequired": counts["variant-required"],
            "simulationOnly": counts["simulation-only"],
            "unsupported": counts["unsupported"],
            "distinctCuratedExactCandidates": len(_OLED_CANDIDATES),
        },
        "genericLabelsMaySelectCandidate": False,
        "entries": entries,
    }
    report["reportSha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def get_component_coverage() -> dict[str, Any]:
    """Return a defensive copy of the validated component coverage report."""

    return json.loads(json.dumps(validate_component_coverage()))
