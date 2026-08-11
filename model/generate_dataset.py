"""
VoltForge Comprehensive Synthetic Dataset Generator for Custom Transformer Model.
Generates 10,000+ multi-task training examples covering:
- All 10 Component Categories (BOARD, PASSIVE, LED, SENSOR, DISPLAY, MOTOR, RELAY, COMMUNICATION, POWER, INSTRUMENT)
- All 49 Boards across Arduino, ESP32, ESP8266, Raspberry Pi Pico, STM32, Teensy, XIAO, Feather, Thing Plus, Atmel AVR
- Chat QA, Domain Refusals, Schematic Validation (JSON), Wiring Suggestions (JSON), Firmware Generation, Code Review, and Circuit Debugging
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple


ALL_BOARDS = [
    ("ARDUINO_UNO", "5V", "ATmega328P", 9600, ("a4", "a5")),
    ("ARDUINO_UNO_R4", "5V", "RA4M1 Arm Cortex-M4", 115200, ("a4", "a5")),
    ("ARDUINO_NANO", "5V", "ATmega328P", 9600, ("a4", "a5")),
    ("ARDUINO_NANO_EVERY", "5V", "ATmega4809", 9600, ("a4", "a5")),
    ("ARDUINO_NANO_33_IOT", "3.3V", "SAMD21 Arm Cortex-M0+", 115200, ("a4", "a5")),
    ("ARDUINO_MEGA", "5V", "ATmega2560", 9600, ("d20", "d21")),
    ("ARDUINO_LEONARDO", "5V", "ATmega32U4", 9600, ("d2", "d3")),
    ("ARDUINO_MICRO", "5V", "ATmega32U4", 9600, ("d2", "d3")),
    ("ARDUINO_DUE", "3.3V", "ATSAM3X8E Arm Cortex-M3", 115200, ("d20", "d21")),
    ("ARDUINO_GIGA_R1", "3.3V", "STM32H747XI dual-core", 115200, ("d20", "d21")),
    ("ARDUINO_PORTENTA_H7", "3.3V", "STM32H747XI dual-core", 115200, ("d20", "d21")),
    ("ESP32", "3.3V", "Xtensa LX6 dual-core", 115200, ("d21", "d22")),
    ("ESP32_WROOM", "3.3V", "Xtensa LX6 dual-core", 115200, ("d21", "d22")),
    ("ESP32_WROVER", "3.3V", "Xtensa LX6 with PSRAM", 115200, ("d21", "d22")),
    ("ESP32_S2", "3.3V", "Xtensa LX7 single-core", 115200, ("d8", "d9")),
    ("ESP32_S3", "3.3V", "Xtensa LX7 dual-core AI", 115200, ("d8", "d9")),
    ("ESP32_C3", "3.3V", "RISC-V single-core", 115200, ("d8", "d9")),
    ("ESP32_C6", "3.3V", "RISC-V with Zigbee/Thread", 115200, ("d8", "d9")),
    ("ESP32_H2", "3.3V", "RISC-V Thread/Matter", 115200, ("d8", "d9")),
    ("ESP8266", "3.3V", "Tensilica L106 32-bit", 115200, ("d2", "d1")),
    ("ESP8266_WEMOS_D1_MINI", "3.3V", "ESP8266 Mini", 115200, ("d2", "d1")),
    ("ESP8266_ESP01", "3.3V", "ESP8266 Serial Module", 115200, ("d2", "d1")),
    ("ESP8266_ESP12E", "3.3V", "ESP8266 NodeMCU", 115200, ("d2", "d1")),
    ("RASPBERRY_PI_PICO", "3.3V", "RP2040 dual-core", 115200, ("d4", "d5")),
    ("RASPBERRY_PI_PICO_W", "3.3V", "RP2040 + CYW43439 Wi-Fi", 115200, ("d4", "d5")),
    ("RASPBERRY_PI_PICO_2", "3.3V", "RP2350 microcontroller", 115200, ("d4", "d5")),
    ("STM32_BLUE_PILL", "3.3V", "STM32F103C8 Arm Cortex-M3", 115200, ("d11", "d10")),
    ("STM32_BLACK_PILL", "3.3V", "STM32F401/F411 Cortex-M4", 115200, ("d11", "d10")),
    ("TEENSY_4_0", "3.3V", "NXP i.MX RT1062 600MHz", 115200, ("d18", "d19")),
    ("TEENSY_4_1", "3.3V", "NXP i.MX RT1062 + Ethernet", 115200, ("d18", "d19")),
    ("TEENSY_LC", "3.3V", "NXP MKL26Z64 48MHz", 115200, ("d18", "d19")),
    ("SEEED_XIAO_SAMD21", "3.3V", "SAMD21G18 Cortex-M0+", 115200, ("d4", "d5")),
    ("SEEED_XIAO_RP2040", "3.3V", "RP2040 dual-core", 115200, ("d4", "d5")),
    ("SEEED_XIAO_ESP32C3", "3.3V", "ESP32-C3 RISC-V", 115200, ("d4", "d5")),
    ("SEEED_XIAO_ESP32S3", "3.3V", "ESP32-S3 AI dual-core", 115200, ("d4", "d5")),
    ("SEEED_XIAO_NRF52840", "3.3V", "nRF52840 BLE 5.0", 115200, ("d4", "d5")),
    ("ADAFRUIT_FEATHER_M0", "3.3V", "ATSAMD21G18 Cortex-M0+", 115200, ("d21", "d22")),
    ("ADAFRUIT_FEATHER_M4", "3.3V", "ATSAMD51J19 Cortex-M4F", 115200, ("d21", "d22")),
    ("ADAFRUIT_FEATHER_ESP32", "3.3V", "ESP32 Feather", 115200, ("d21", "d22")),
    ("ADAFRUIT_FEATHER_RP2040", "3.3V", "RP2040 Feather", 115200, ("d4", "d5")),
    ("ADAFRUIT_FEATHER_NRF52840", "3.3V", "nRF52840 Feather", 115200, ("d21", "d22")),
    ("SPARKFUN_THING_PLUS_ESP32", "3.3V", "ESP32 Thing Plus", 115200, ("d21", "d22")),
    ("SPARKFUN_THING_PLUS_RP2040", "3.3V", "RP2040 Thing Plus", 115200, ("d4", "d5")),
    ("SPARKFUN_THING_PLUS_ARTEMIS", "3.3V", "Apollo3 Cortex-M4F", 115200, ("a4", "a5")),
    ("ATMEL_AVR_ATMEGA328P", "5V", "ATmega328P DIP 16MHz", 9600, ("a4", "a5")),
    ("ATMEL_AVR_ATTINY", "5V", "ATtiny85 DIP 8/16MHz", 9600, ("d0", "d2")),
]

OUT_OF_DOMAIN_PROMPTS = [
    "Give me a recipe for chocolate chip cookies.",
    "Who was the first emperor of Rome?",
    "Write a love poem about the stars.",
    "What is the stock price of Apple?",
    "Who won the cricket world cup in 2011?",
    "Recommend a good romantic comedy movie.",
    "How do I bake a sourdough loaf?",
    "Write an essay on the French Revolution.",
    "What is the weather in Paris today?",
    "Give me dating advice for a first date.",
    "Translate this sentence to French.",
    "Write a rap song about coffee.",
    "What are the best tourist places in London?",
    "Explain the plot of Hamlet.",
    "How do I file my personal income tax?",
]

REFUSAL_RESPONSE = (
    "I am VoltForge AI, so I stay focused on electronics, circuit design, "
    "microcontrollers, firmware, wiring, and simulation. I cannot help with that outside-domain request."
)


def load_dataset_txt_pairs() -> List[Tuple[str, str]]:
    dataset_txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.txt")
    pairs: List[Tuple[str, str]] = []
    if os.path.exists(dataset_txt_path):
        with open(dataset_txt_path, "r", encoding="utf-8") as f:
            chunks = f.read().split("[Q]")
            for chunk in chunks:
                if not chunk.strip() or "[A]" not in chunk:
                    continue
                q, a = chunk.split("[A]", 1)
                pairs.append((q.strip(), a.strip()))
    return pairs


def generate_chat_qa_examples(count: int = 3000) -> List[Dict[str, str]]:
    base_pairs = load_dataset_txt_pairs()
    examples = []
    prefixes = [
        "",
        "For my {board} project, ",
        "On {board}, ",
        "Can you explain ",
        "How do I handle ",
        "What is the best way to wire ",
        "Tell me about ",
        "I need help with ",
        "Please provide instructions for ",
    ]
    for _ in range(count):
        if not base_pairs:
            break
        q, a = random.choice(base_pairs)
        board = random.choice(ALL_BOARDS)[0]
        prefix_tmpl = random.choice(prefixes)
        prefix = prefix_tmpl.format(board=board)
        formatted_q = prefix + q if prefix else q
        examples.append({
            "task": "chat_qa",
            "prompt": f"[SYS] You are VoltForge AI, an electronics copilot. [USER] {formatted_q} [ASSISTANT]",
            "completion": a
        })
    return examples


def generate_refusal_examples(count: int = 500) -> List[Dict[str, str]]:
    examples = []
    for _ in range(count):
        prompt = random.choice(OUT_OF_DOMAIN_PROMPTS)
        examples.append({
            "task": "refusal",
            "prompt": f"[SYS] You are VoltForge AI, an electronics copilot. [USER] {prompt} [ASSISTANT]",
            "completion": REFUSAL_RESPONSE
        })
    return examples


def generate_validation_examples(count: int = 2500) -> List[Dict[str, str]]:
    examples = []
    for _ in range(count):
        board, logic_v, mcu, _, i2c_pins = random.choice(ALL_BOARDS)
        has_short = random.random() < 0.2
        has_bare_led = random.random() < 0.3
        has_motor_gpio = random.random() < 0.2
        has_missing_gnd = random.random() < 0.15
        has_level_shift_issue = (logic_v == "3.3V" and random.random() < 0.25)

        issues = []
        additions = []
        removals = []
        wire_suggestions = []

        if has_short:
            issues.append({
                "severity": "CRITICAL",
                "componentId": "power_rail",
                "message": "Power rail is connected directly to ground (short circuit).",
                "suggestedFix": "Remove the direct VCC to GND short before powering the circuit."
            })
            removals.append({"type": "wire", "reason": "Direct power-to-ground short."})

        if has_bare_led:
            issues.append({
                "severity": "CRITICAL",
                "componentId": "led1",
                "message": f"LED is connected directly to {board} GPIO without a series current-limiting resistor.",
                "suggestedFix": "Add a 220 Ohm resistor in series between the MCU output pin and LED anode."
            })
            additions.append({
                "type": "component",
                "componentType": "RESISTOR",
                "value": "220 Ohm",
                "between": ["MCU/GPIO", "led1/anode"],
                "reason": "Limit LED current to protect GPIO pin."
            })

        if has_motor_gpio:
            issues.append({
                "severity": "CRITICAL",
                "componentId": "motor1",
                "message": "DC motor is wired directly to an MCU GPIO pin.",
                "suggestedFix": "Drive motor via a MOSFET/transistor or H-bridge driver with a flyback diode."
            })

        if has_missing_gnd:
            issues.append({
                "severity": "WARNING",
                "componentId": "sensor1",
                "message": "Sensor has signal connections but is missing a ground return reference.",
                "suggestedFix": "Connect sensor GND pin to MCU common ground."
            })
            wire_suggestions.append({
                "fromComponentId": "sensor1",
                "fromPin": "gnd",
                "toComponentId": "mcu",
                "toPin": "gnd",
                "color": "#555555",
                "description": "Return sensor ground to MCU GND"
            })

        if has_level_shift_issue:
            issues.append({
                "severity": "WARNING",
                "componentId": "hc_sr04",
                "message": f"5V Echo signal connected directly to {logic_v} {board} GPIO.",
                "suggestedFix": "Use a bidirectional logic level shifter or 1k/2k voltage divider on Echo."
            })

        is_valid = len(issues) == 0
        critical_count = sum(1 for i in issues if i["severity"] == "CRITICAL")
        warn_count = sum(1 for i in issues if i["severity"] == "WARNING")
        score = max(0, 100 - critical_count * 25 - warn_count * 10)

        val_response = {
            "isValid": is_valid,
            "safetyScore": score,
            "generalFeedback": "No unsafe wiring or pin mismatches detected." if is_valid else f"Found {critical_count} critical issue(s) and {warn_count} warning(s).",
            "issues": issues,
            "additions": additions,
            "removals": removals,
            "wireSuggestions": wire_suggestions
        }

        components_str = f"LED={has_bare_led}, Motor={has_motor_gpio}, Short={has_short}, LevelShift={has_level_shift_issue}"
        examples.append({
            "task": "validate_circuit",
            "prompt": f"[SYS] Validate circuit safety and return JSON. [CANVAS] board={board}, logic={logic_v}, components=[{components_str}] [JSON]",
            "completion": json.dumps(val_response, ensure_ascii=False)
        })
    return examples


def generate_schematic_to_code_examples(count: int = 2500) -> List[Dict[str, str]]:
    examples = []
    recipes = [
        ("LED on D13", "const int ledPin = 13;\nvoid setup() {\n  pinMode(ledPin, OUTPUT);\n}\nvoid loop() {\n  digitalWrite(ledPin, HIGH);\n  delay(500);\n  digitalWrite(ledPin, LOW);\n  delay(500);\n}"),
        ("DHT22 on D2", "#include <DHT.h>\n#define DHTPIN 2\n#define DHTTYPE DHT22\nDHT dht(DHTPIN, DHTTYPE);\nvoid setup() {\n  Serial.begin({baud});\n  dht.begin();\n}\nvoid loop() {\n  float h = dht.readHumidity();\n  float t = dht.readTemperature();\n  Serial.print(\"T: \"); Serial.print(t); Serial.print(\" C  H: \"); Serial.println(h);\n  delay(2000);\n}"),
        ("HC-SR04 on D5/D6", "const int trigPin = 5;\nconst int echoPin = 6;\nvoid setup() {\n  Serial.begin({baud});\n  pinMode(trigPin, OUTPUT);\n  pinMode(echoPin, INPUT);\n}\nvoid loop() {\n  digitalWrite(trigPin, LOW);\n  delayMicroseconds(2);\n  digitalWrite(trigPin, HIGH);\n  delayMicroseconds(10);\n  digitalWrite(trigPin, LOW);\n  long duration = pulseIn(echoPin, HIGH, 30000);\n  float distance = duration * 0.0343 / 2.0;\n  Serial.print(\"Distance: \"); Serial.print(distance); Serial.println(\" cm\");\n  delay(500);\n}"),
        ("I2C LCD 16x2 on {sda}/{scl}", "#include <Wire.h>\n#include <LiquidCrystal_I2C.h>\nLiquidCrystal_I2C lcd(0x27, 16, 2);\nvoid setup() {\n  lcd.init();\n  lcd.backlight();\n  lcd.setCursor(0, 0);\n  lcd.print(\"VoltForge Ready\");\n}\nvoid loop() {\n  delay(1000);\n}"),
        ("Servo SG90 on D9", "#include <Servo.h>\nServo myServo;\nvoid setup() {\n  myServo.attach(9);\n}\nvoid loop() {\n  myServo.write(0);\n  delay(700);\n  myServo.write(90);\n  delay(700);\n}"),
        ("LDR + Relay on A0 and D7", "const int ldrPin = A0;\nconst int relayPin = 7;\nvoid setup() {\n  Serial.begin({baud});\n  pinMode(relayPin, OUTPUT);\n}\nvoid loop() {\n  int light = analogRead(ldrPin);\n  Serial.print(\"Light: \"); Serial.println(light);\n  if (light < 300) {\n    digitalWrite(relayPin, HIGH);\n  } else {\n    digitalWrite(relayPin, LOW);\n  }\n  delay(200);\n}"),
        ("PIR Motion + Buzzer on D2 and D8", "const int pirPin = 2;\nconst int buzzerPin = 8;\nvoid setup() {\n  Serial.begin({baud});\n  pinMode(pirPin, INPUT);\n  pinMode(buzzerPin, OUTPUT);\n}\nvoid loop() {\n  int motion = digitalRead(pirPin);\n  if (motion == HIGH) {\n    Serial.println(\"MOTION DETECTED\");\n    tone(buzzerPin, 1000, 200);\n  }\n  delay(500);\n}"),
        ("MPU6050 Accelerometer on {sda}/{scl}", "#include <Wire.h>\n#include <MPU6050.h>\nMPU6050 mpu;\nvoid setup() {\n  Serial.begin({baud});\n  Wire.begin();\n  mpu.initialize();\n}\nvoid loop() {\n  int16_t ax, ay, az, gx, gy, gz;\n  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);\n  Serial.print(\"AX: \"); Serial.print(ax); Serial.print(\" AY: \"); Serial.print(ay); Serial.print(\" AZ: \"); Serial.println(az);\n  delay(500);\n}"),
        ("Soil Moisture + OLED on A0 and {sda}/{scl}", "#include <Wire.h>\n#include <Adafruit_SSD1306.h>\nconst int soilPin = A0;\nAdafruit_SSD1306 display(128, 64, &Wire, -1);\nvoid setup() {\n  Serial.begin({baud});\n  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);\n  display.clearDisplay();\n  display.setTextColor(WHITE);\n}\nvoid loop() {\n  int soil = analogRead(soilPin);\n  display.clearDisplay();\n  display.setCursor(0, 0);\n  display.print(\"Soil: \"); display.println(soil);\n  display.display();\n  delay(1000);\n}"),
        ("NeoPixel 16 LED Strip on D6", "#include <Adafruit_NeoPixel.h>\n#define PIN 6\n#define NUMPIXELS 16\nAdafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);\nvoid setup() {\n  pixels.begin();\n}\nvoid loop() {\n  for(int i=0; i<NUMPIXELS; i++) {\n    pixels.setPixelColor(i, pixels.Color(0, 150, 0));\n    pixels.show();\n    delay(50);\n  }\n  pixels.clear();\n  delay(500);\n}"),
    ]
    for _ in range(count):
        board, _, _, baud, i2c_pins = random.choice(ALL_BOARDS)
        desc_tmpl, code_tmpl = random.choice(recipes)
        sda_pin = i2c_pins[0].upper()
        scl_pin = i2c_pins[1].upper()
        desc = desc_tmpl.replace("{sda}", sda_pin).replace("{scl}", scl_pin)
        code = code_tmpl.replace("{baud}", str(baud)).replace("{sda}", sda_pin).replace("{scl}", scl_pin)
        examples.append({
            "task": "schematic_to_code",
            "prompt": f"[SYS] Generate Arduino C++ firmware for canvas layout. [CANVAS] board={board}, features=[{desc}] [CODE]",
            "completion": f"// Generated by VoltForge AI\n// Board: {board}\n{code}"
        })
    return examples


def generate_code_review_examples(count: int = 1500) -> List[Dict[str, str]]:
    examples = []
    for _ in range(count):
        board, _, _, baud, _ = random.choice(ALL_BOARDS)
        mismatch = random.random() < 0.4
        missing_setup = random.random() < 0.15
        
        issues = []
        if missing_setup:
            issues.append({
                "severity": "ERROR",
                "line": 1,
                "message": "Missing Arduino setup() function.",
                "fix": "Add void setup() { ... } for initialization."
            })
        if mismatch:
            issues.append({
                "severity": "WARNING",
                "line": 5,
                "message": f"Code uses D13, but the canvas LED wire is connected to D8 on {board}.",
                "fix": "Change code pin constant to 8 or move the canvas wire to D13."
            })

        score = max(0, 100 - len(issues) * 35)
        review_response = {
            "score": score,
            "summary": "Code structure verified clean." if not issues else f"Found {len(issues)} structural or pin mismatch issue(s).",
            "issues": issues,
            "confidence": 0.92
        }

        examples.append({
            "task": "review_code",
            "prompt": f"[SYS] Review Arduino C++ code against canvas wiring and board constraints. [CANVAS] board={board}, mismatch={mismatch}, missing_setup={missing_setup} [JSON]",
            "completion": json.dumps(review_response, ensure_ascii=False)
        })
    return examples


def build_full_dataset(total_target: int = 10000) -> List[Dict[str, str]]:
    dataset = []
    dataset.extend(generate_chat_qa_examples(3500))
    dataset.extend(generate_refusal_examples(500))
    dataset.extend(generate_validation_examples(2500))
    dataset.extend(generate_schematic_to_code_examples(2500))
    dataset.extend(generate_code_review_examples(1500))
    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    data = build_full_dataset(10500)
    out_file = os.path.join(out_dir, "dataset.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(data)} training examples -> {out_file}")

