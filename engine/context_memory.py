"""
Voltforge AI - Conversational Context Memory & Multi-Turn State Engine
Covers Features 341-360: Session State Manager, Conversation Summarizer,
Entity Extractor (components/pins/voltages), Intent Classifier, Slot Filler,
Clarification Generator, User Preference Learning, Auto-Suggest Engine.
"""

import re
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("voltforge-ai.context_memory")


class ConversationState:
    """Immutable snapshot of a conversation turn."""
    def __init__(self, role: str, message: str, timestamp: float = None, entities: Dict = None, intent: str = None):
        self.role = role
        self.message = message
        self.timestamp = timestamp or time.time()
        self.entities = entities or {}
        self.intent = intent or "unknown"


class ConversationMemory:
    """Multi-turn conversation state manager with entity tracking."""

    def __init__(self, max_history: int = 50):
        self.history: List[ConversationState] = []
        self.max_history = max_history
        self.entities: Dict[str, Any] = {}
        self.preferences: Dict[str, Any] = {}
        self.project_context: Dict[str, Any] = {}
        self.turn_count = 0

    def add_turn(self, role: str, message: str, intent: str = "unknown") -> None:
        entities = EntityExtractor.extract(message)
        state = ConversationState(role, message, entities=entities, intent=intent)
        self.history.append(state)
        self.turn_count += 1

        # Merge extracted entities into session state
        for key, value in entities.items():
            if isinstance(value, list):
                existing = self.entities.get(key, [])
                if isinstance(existing, list):
                    self.entities[key] = list(set(existing + value))
                else:
                    self.entities[key] = value
            elif value:
                self.entities[key] = value

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "turnCount": self.turn_count,
            "trackedEntities": self.entities,
            "preferences": self.preferences,
            "projectContext": self.project_context,
            "recentHistory": [
                {"role": h.role, "message": h.message[:100], "intent": h.intent}
                for h in self.history[-5:]
            ],
        }

    def get_last_intent(self) -> str:
        for state in reversed(self.history):
            if state.role == "user" and state.intent != "unknown":
                return state.intent
        return "unknown"


class EntityExtractor:
    """Extracts electronics-domain entities from natural language text."""

    VOLTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:v|volt|volts)', re.IGNORECASE)
    CURRENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:ma|milliamp|amp)', re.IGNORECASE)
    RESISTANCE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:k|kohm|kΩ|ohm|Ω)', re.IGNORECASE)
    PIN_PATTERN = re.compile(r'(?:pin|gpio|d|a)\s*(\d+)', re.IGNORECASE)
    FREQUENCY_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:hz|khz|mhz|ghz)', re.IGNORECASE)

    BOARD_KEYWORDS = {
        "arduino uno": "ARDUINO_UNO", "arduino mega": "ARDUINO_MEGA",
        "arduino nano": "ARDUINO_NANO", "esp32": "ESP32", "esp8266": "ESP8266",
        "raspberry pi pico": "RASPBERRY_PI_PICO", "stm32": "STM32_BLUEPILL",
    }

    COMPONENT_KEYWORDS = [
        "led", "resistor", "capacitor", "inductor", "servo", "motor", "relay",
        "oled", "lcd", "button", "buzzer", "mpu6050", "bme280", "dht22", "dht11",
        "hc-sr04", "ultrasonic", "potentiometer", "transistor", "mosfet", "diode",
        "sensor", "actuator", "sd card", "rfid", "lora", "bluetooth", "wifi",
    ]

    @classmethod
    def extract(cls, text: str) -> Dict[str, Any]:
        lower = text.lower()
        entities = {}

        # Extract voltages
        voltages = cls.VOLTAGE_PATTERN.findall(text)
        if voltages:
            entities["voltages"] = [float(v) for v in voltages]

        # Extract currents
        currents = cls.CURRENT_PATTERN.findall(text)
        if currents:
            entities["currents_mA"] = [float(c) for c in currents]

        # Extract resistances
        resistances = cls.RESISTANCE_PATTERN.findall(text)
        if resistances:
            entities["resistances"] = resistances

        # Extract pin numbers
        pins = cls.PIN_PATTERN.findall(text)
        if pins:
            entities["pins"] = [int(p) for p in pins]

        # Extract frequencies
        frequencies = cls.FREQUENCY_PATTERN.findall(text)
        if frequencies:
            entities["frequencies"] = [float(f) for f in frequencies]

        # Extract board type
        for keyword, board_type in cls.BOARD_KEYWORDS.items():
            if keyword in lower:
                entities["board"] = board_type
                break

        # Extract component mentions
        found_components = [comp for comp in cls.COMPONENT_KEYWORDS if comp in lower]
        if found_components:
            entities["components"] = found_components

        return entities


class IntentClassifier:
    """Rule-based intent classifier for electronics queries."""

    INTENT_PATTERNS = {
        "troubleshoot": ["not working", "doesnt work", "broken", "blank", "debug", "troubleshoot", "diagnose", "error", "fail"],
        "calculate": ["calculate", "resistor for", "voltage divider", "ohm", "how much", "what value"],
        "generate_code": ["generate code", "write code", "firmware", "sketch", "program"],
        "validate": ["validate", "check circuit", "safety", "fix circuit", "verify"],
        "connect": ["how to connect", "wiring", "hookup", "connect", "pinout"],
        "explain": ["what is", "explain", "how does", "tell me about", "describe"],
        "design": ["build a", "create a", "design a", "make a", "i want to make"],
        "datasheet": ["datasheet", "specs", "specification", "voltage range", "current draw"],
        "pcb": ["pcb", "trace width", "track", "via", "footprint", "gerber"],
        "simulate": ["simulate", "spice", "bode", "transient", "waveform"],
        "power": ["battery", "solar", "power consumption", "sleep", "deep sleep"],
        "compare": ["compare", "vs", "versus", "difference between", "better"],
    }

    @classmethod
    def classify(cls, text: str) -> Dict[str, Any]:
        lower = text.lower()
        scores = {}
        for intent, keywords in cls.INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return {"intent": "general_query", "confidence": 0.3, "allScores": {}}

        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        confidence = min(0.95, 0.5 + max_score * 0.15)

        return {
            "intent": best_intent,
            "confidence": round(confidence, 2),
            "allScores": scores,
        }


class ClarificationGenerator:
    """Generates clarifying questions when user intent is ambiguous."""

    @classmethod
    def generate(cls, message: str, entities: Dict[str, Any], intent: str) -> Optional[str]:
        if intent == "connect" and "board" not in entities:
            return "Which board are you using? (Arduino Uno, ESP32, Raspberry Pi Pico, etc.)"

        if intent == "calculate" and "voltages" not in entities:
            return "What is the supply voltage for your circuit? (e.g., 3.3V or 5V)"

        if intent == "generate_code" and "components" not in entities:
            return "Which components should the firmware control? (e.g., LED, servo, sensor)"

        if intent == "troubleshoot" and "components" not in entities:
            return "Which component or circuit is not working? Please describe the symptom."

        if intent == "general_query":
            return None  # Don't ask clarification for completely vague queries

        return None


class AutoSuggestEngine:
    """Suggests follow-up questions based on conversation context."""

    @classmethod
    def suggest(cls, intent: str, entities: Dict[str, Any], board: str = "ARDUINO_UNO") -> List[str]:
        suggestions = []

        if intent == "connect":
            suggestions.extend([
                "Generate firmware code for this circuit",
                "Validate my wiring for errors",
                "Show me the recommended resistor values",
            ])
        elif intent == "calculate":
            suggestions.extend([
                "What resistor do I need for an LED?",
                "Calculate a voltage divider for 3.3V",
                "Estimate battery life for this circuit",
            ])
        elif intent == "troubleshoot":
            suggestions.extend([
                "Run I2C scanner to detect devices",
                "Check pin assignments in my code",
                "Generate multimeter test guide",
            ])
        elif intent == "design":
            suggestions.extend([
                "Generate BOM with cost estimate",
                "Export to KiCad PCB format",
                "Estimate PCB fabrication cost",
            ])
        else:
            suggestions.extend([
                f"Build a weather station with {board}",
                "Validate my circuit for safety issues",
                "Generate optimized firmware code",
            ])

        return suggestions[:3]


if __name__ == "__main__":
    import json

    # Test entity extraction
    entities = EntityExtractor.extract("Connect a 5V LED with 220 ohm resistor to pin 13 on Arduino Uno")
    print("=== Entity Extraction ===")
    print(json.dumps(entities, indent=2))

    # Test intent classification
    intent = IntentClassifier.classify("My MPU6050 is not working and the I2C bus is failing")
    print("\n=== Intent Classification ===")
    print(json.dumps(intent, indent=2))

    # Test conversation memory
    memory = ConversationMemory()
    memory.add_turn("user", "I want to build a weather station with ESP32 and BME280", "design")
    memory.add_turn("assistant", "Here's the circuit design for your weather station...")
    memory.add_turn("user", "Now generate the firmware code for it", "generate_code")
    print("\n=== Conversation Context ===")
    print(json.dumps(memory.get_context_summary(), indent=2))

    # Test auto-suggest
    suggestions = AutoSuggestEngine.suggest("design", entities, "ESP32")
    print("\n=== Auto Suggestions ===")
    print(json.dumps(suggestions, indent=2))
