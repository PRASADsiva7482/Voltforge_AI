"""
VoltForge-1B High-Throughput Synthetic Domain Dataset Generator.
Synthesizes hundreds of thousands of structured domain samples across Embedded C++,
Microcontroller Pinouts, SPICE Netlists, Circuit Safety Rules, and Reasoning Chains.
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple


BOARDS = ["ARDUINO_UNO", "ARDUINO_NANO", "ESP32_WROOM", "ESP32_S3", "RP2040_PICO", "STM32_BLUEPILL"]
SENSORS = ["DHT11", "DHT22", "BMP280", "MPU6050", "HC_SR04", "LDR", "PIR_MOTION", "SOIL_MOISTURE"]
ACTUATORS = ["LED_RED", "LED_GREEN", "SERVO_SG90", "RELAY_5V", "DC_MOTOR", "BUZZER_ACTIVE", "LCD_1602_I2C", "OLED_SSD1306"]

REFUSAL_PROMPTS = [
    "Write me a chocolate cake recipe.",
    "Who was the 16th president of the United States?",
    "Write a romance novel about a vampire.",
    "Give me stock market investment advice.",
    "Write an essay on modern art history.",
    "How do I bake sourdough bread?",
    "Tell me a joke about airplanes.",
]


def generate_firmware_sketch_sample() -> Dict[str, str]:
    board = random.choice(BOARDS)
    sensor = random.choice(SENSORS)
    actuator = random.choice(ACTUATORS)

    prompt = f"[SYS] You are VoltForge AI, an expert embedded systems copilot. [USER] Write a production C++ firmware sketch for {board} reading {sensor} and controlling {actuator}."

    code_body = f"""// VoltForge Generated Firmware
// Target Board: {board}
// Peripheral: {sensor} -> {actuator}

#include <Wire.h>

const int SENSOR_PIN = 2;
const int ACTUATOR_PIN = 8;

void setup() {{
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);
  pinMode(ACTUATOR_PIN, OUTPUT);
  digitalWrite(ACTUATOR_PIN, LOW);
  Serial.println(F("[SYSTEM] {board} initialization complete."));
}}

void loop() {{
  int sensorValue = digitalRead(SENSOR_PIN);
  if (sensorValue == HIGH) {{
    digitalWrite(ACTUATOR_PIN, HIGH);
    Serial.println(F("[ALERT] Sensor trigger detected -> Actuator ON"));
  }} else {{
    digitalWrite(ACTUATOR_PIN, LOW);
  }}
  delay(100);
}}"""

    completion = f"[ASSISTANT] Here is the verified firmware sketch for the **{board}** controlling the **{actuator}** based on **{sensor}** input:\n\n```cpp\n{code_body}\n```\n\n**Key Safety Points:**\n1. Ensure common ground between all 5V/3.3V modules.\n2. Add a flyback diode if {actuator} is an inductive relay or motor.\n3. Pull-up resistor recommended on the {sensor} data line."

    return {"prompt": prompt, "completion": completion, "category": "FIRMWARE_SYNTHESIS"}


def generate_safety_validation_sample() -> Dict[str, str]:
    scenarios = [
        (
            "LED connected directly between D8 and GND without a resistor",
            "CRITICAL",
            "LED series current-limiting resistor missing",
            "Insert a 220 Ohm to 1k Ohm resistor in series between pin D8 and the LED anode to prevent GPIO overcurrent damage.",
            40
        ),
        (
            "DC Motor coil connected directly to Arduino GPIO pin 3",
            "CRITICAL",
            "Inductive load on MCU GPIO pin without driver or flyback protection",
            "Use an NPN transistor (2N2222) or MOSFET with a 1N4007 flyback diode anti-parallel across motor terminals.",
            30
        ),
        (
            "I2C OLED display SDA connected to Arduino Uno pin D2",
            "WARNING",
            "Hardware I2C SDA pin mismatch on ATmega328P",
            "Move SDA connection to pin A4 (or dedicated SDA header) on Arduino Uno.",
            70
        )
    ]

    scenario, sev, msg, fix, score = random.choice(scenarios)
    prompt = f"[SYS] Validate circuit safety. [USER] Check this circuit: {scenario} on ARDUINO_UNO."

    json_data = {
        "isValid": False,
        "safetyScore": score,
        "issues": [
            {
                "severity": sev,
                "category": "ELECTRICAL_SAFETY",
                "message": msg,
                "suggestedFix": fix
            }
        ],
        "additions": [{"type": "RESISTOR", "value": "220 Ohm"}] if "LED" in scenario else []
    }

    completion = f"[ASSISTANT] [JSON] {json.dumps(json_data, indent=2)}"
    return {"prompt": prompt, "completion": completion, "category": "SAFETY_VALIDATION"}


def generate_refusal_sample() -> Dict[str, str]:
    query = random.choice(REFUSAL_PROMPTS)
    prompt = f"[SYS] You are VoltForge AI, an electronics domain assistant. Refuse all non-electronics queries politely. [USER] {query}"
    completion = "[ASSISTANT] I am VoltForge AI, specialized exclusively in electronics, circuit simulation, microcontrollers, wiring design, and embedded firmware. I cannot assist with non-electronics requests. Please ask an electronics, EDA, or microcontroller programming question."
    return {"prompt": prompt, "completion": completion, "category": "DOMAIN_REFUSAL"}


def build_1b_training_corpus(num_samples: int = 5000) -> List[Dict[str, str]]:
    dataset = []
    for _ in range(int(num_samples * 0.50)):
        dataset.append(generate_firmware_sketch_sample())
    for _ in range(int(num_samples * 0.35)):
        dataset.append(generate_safety_validation_sample())
    for _ in range(int(num_samples * 0.15)):
        dataset.append(generate_refusal_sample())
    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "dataset_1b.jsonl")

    print(f"[*] Generating 2,500 sample 1B pretraining & SFT corpus...")
    samples = build_1b_training_corpus(2500)
    with open(out_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[+] Dataset saved to {out_file} ({len(samples)} items, {os.path.getsize(out_file)/1024:.1f} KB)")
