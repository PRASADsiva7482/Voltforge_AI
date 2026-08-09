import asyncio
import json
import logging
import os
import re
import urllib.parse
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    netlist: Optional[Dict[str, Any]] = None
    code: Optional[str] = ""
    canvasData: Optional[Dict[str, Any]] = None
    simulationState: Optional[Dict[str, Any]] = None
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    hasCode: bool = False
    generatedCode: Optional[str] = None
    confidence: float = 0.8
    citations: List[Dict[str, str]] = Field(default_factory=list)
    wireSuggestions: List[Dict[str, str]] = Field(default_factory=list)
    additions: List[Dict[str, Any]] = Field(default_factory=list)
    removals: List[Dict[str, Any]] = Field(default_factory=list)
    valueChanges: List[Dict[str, Any]] = Field(default_factory=list)
    codeFixes: List[Dict[str, Any]] = Field(default_factory=list)


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
    "raspberry",
    "pi",
    "rp2040",
    "pico",
    "rp2350",
    "stm32",
    "teensy",
    "xiao",
    "feather",
    "atmega",
    "attiny",
    "microcontroller",
    "mcu",
    "processor",
    "cpu",
    "gpio",
    "pin",
    "pins",
    "pinout",
    "circuit",
    "schematic",
    "wire",
    "wiring",
    "ground",
    "gnd",
    "vcc",
    "vdd",
    "vin",
    "vsys",
    "vbus",
    "5v",
    "3.3v",
    "3v3",
    "voltage",
    "current",
    "amp",
    "milliamp",
    "resistance",
    "ohm",
    "power",
    "watt",
    "sensor",
    "led",
    "rgb",
    "neopixel",
    "ws2812",
    "resistor",
    "capacitor",
    "diode",
    "transistor",
    "mosfet",
    "potentiometer",
    "button",
    "switch",
    "breadboard",
    "buzzer",
    "motor",
    "stepper",
    "dc motor",
    "relay",
    "servo",
    "lcd",
    "oled",
    "display",
    "seven segment",
    "7-segment",
    "ultrasonic",
    "hc-sr04",
    "pir",
    "ldr",
    "photoresistor",
    "imu",
    "mpu6050",
    "i2c",
    "spi",
    "uart",
    "serial",
    "pwm",
    "adc",
    "analog",
    "digital",
    "dht",
    "dht11",
    "dht22",
    "firmware",
    "sketch",
    "cpp",
    "c++",
    "compile",
    "analogread",
    "digitalwrite",
    "pinmode",
    "ammeter",
    "oscilloscope",
    "scope",
    "regulator",
    "7805",
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
    "RASPBERRY_PI",
    "STM32",
    "TEENSY",
    "BBC_MICROBIT",
    "SEEED_XIAO",
    "ADAFRUIT_FEATHER",
    "SPARKFUN_THING_PLUS",
    "ATMEL_AVR",
    "BEAGLEBONE",
    "ORANGE_PI",
    "ODROID",
    "JETSON",
)

VOLTForge_SYSTEM_PROMPT = """
You are VoltForge AI, an expert embedded systems and circuit design engineering copilot.
You deeply analyze the active canvas context before answering: components, pin labels, wire netlist,
board type, firmware code, and live simulation state.

Core responsibilities:
1. Always reference the active circuit context when the user asks about electronics, firmware, wiring,
   simulation, voltages, currents, or debugging.
2. Enforce electrical safety and physics with deterministic calculations: Ohm's Law, resistor power,
   voltage dividers, GPIO current budgets, inductive flyback protection, level shifting, floating inputs,
   and power/ground integrity.
3. Keep firmware synchronized with canvas wiring by checking pin constants, pinMode/digitalWrite/
   analogRead/analogWrite/tone usages, and board-specific pin capabilities.
4. Respond concisely with exact component IDs, pin names, and calculations. Return structured actions
   for wire suggestions, additions, removals, value changes, and code fixes whenever a circuit fix can
   be applied by the UI.
5. Stay focused on electronics, firmware, microcontrollers, and circuit simulation.
""".strip()

BOARD_RULES: Dict[str, Dict[str, Any]] = {
    "ARDUINO_UNO": {
        "logicVoltage": 5.0,
        "recommendedPinMa": 20.0,
        "absolutePinMa": 40.0,
        "totalPackageMa": 200.0,
        "pwmPins": {"d3", "d5", "d6", "d9", "d10", "d11"},
        "adcPins": {"a0", "a1", "a2", "a3", "a4", "a5"},
        "i2c": ("a4", "a5"),
    },
    "ARDUINO_NANO": {
        "logicVoltage": 5.0,
        "recommendedPinMa": 20.0,
        "absolutePinMa": 40.0,
        "totalPackageMa": 200.0,
        "pwmPins": {"d3", "d5", "d6", "d9", "d10", "d11"},
        "adcPins": {"a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"},
        "i2c": ("a4", "a5"),
    },
    "ARDUINO_MEGA": {
        "logicVoltage": 5.0,
        "recommendedPinMa": 20.0,
        "absolutePinMa": 40.0,
        "totalPackageMa": 200.0,
        "pwmPins": {"d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d44", "d45", "d46"},
        "adcPins": {f"a{index}" for index in range(16)},
        "i2c": ("d20", "d21"),
    },
    "ESP32": {
        "logicVoltage": 3.3,
        "recommendedPinMa": 12.0,
        "absolutePinMa": 40.0,
        "totalPackageMa": 200.0,
        "inputOnlyPins": {"d34", "d35", "d36", "d37", "d38", "d39", "vp", "vn"},
        "strapPins": {"d0", "d2", "d5", "d12", "d15"},
        "adc2Pins": {"d0", "d2", "d4", "d12", "d13", "d14", "d15", "d25", "d26", "d27"},
        "pwmPins": {f"d{index}" for index in range(0, 34)} - {"d6", "d7", "d8", "d9", "d10", "d11"},
        "adcPins": {"d32", "d33", "d34", "d35", "d36", "d39", "vp", "vn", "d0", "d2", "d4", "d12", "d13", "d14", "d15", "d25", "d26", "d27"},
        "i2c": ("d21", "d22"),
    },
    "RASPBERRY_PI_PICO": {
        "logicVoltage": 3.3,
        "recommendedPinMa": 12.0,
        "absolutePinMa": 16.0,
        "totalPackageMa": 50.0,
        "pwmPins": {f"d{index}" for index in range(0, 29)},
        "adcPins": {"a0", "a1", "a2", "d26", "d27", "d28"},
        "i2c": ("d4", "d5"),
    },
    "RASPBERRY_PI_SBC": {
        "logicVoltage": 3.3,
        "recommendedPinMa": 8.0,
        "absolutePinMa": 16.0,
        "totalPackageMa": 50.0,
        "pwmPins": {"d12", "d13", "d18", "d19"},
        "adcPins": set(),
        "i2c": ("d2", "d3"),
        "spi": ("d10", "d9", "d11"),
        "uart": ("d14", "d15"),
    },
    "STM32_BLUE_PILL": {
        "logicVoltage": 3.3,
        "recommendedPinMa": 20.0,
        "absolutePinMa": 25.0,
        "totalPackageMa": 150.0,
        "strapPins": {"boot0"},
        "pwmPins": {"d0", "d1", "d2", "d3", "d6", "d7", "d8", "d9", "d10", "d11"},
        "adcPins": {f"a{index}" for index in range(8)},
        "i2c": ("d11", "d10"),
    },
}

ELECTRONICS_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "board",
    "component",
    "components",
    "connect",
    "does",
    "for",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "model",
    "my",
    "of",
    "on",
    "pin",
    "pins",
    "please",
    "tell",
    "the",
    "this",
    "to",
    "use",
    "what",
    "will",
    "with",
    "work",
    "working",
}

COMPONENT_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "title": "LED",
        "keywords": ("led", "led_standard", "red led", "green led", "blue led"),
        "summary": "An LED is a polarized diode that emits light when forward biased.",
        "wiring": "Wire GPIO -> series resistor -> LED anode, and LED cathode -> GND. Size the resistor with R = (supply - forward voltage) / target current; 220 Ohm to 330 Ohm is a safe 5V starter range.",
        "code": "Use pinMode(pin, OUTPUT) and digitalWrite/analogWrite on a GPIO that is safe for the selected board.",
        "checks": "Never connect an LED directly across a supply or GPIO without a current-limiting resistor.",
    },
    {
        "title": "RGB LED",
        "keywords": ("rgb led", "led_rgb", "common cathode", "common anode"),
        "summary": "An RGB LED contains red, green, and blue LED dies in one package.",
        "wiring": "Each color channel needs its own series resistor. Common cathode goes to GND; common anode goes to the positive rail and the GPIOs sink current.",
        "code": "Use three PWM-capable pins for smooth color mixing.",
        "checks": "Check common-anode/common-cathode orientation before deciding whether HIGH or LOW turns a channel on.",
    },
    {
        "title": "NeoPixel / WS2812B",
        "keywords": ("neopixel", "ws2812", "ws2812b", "led_neopixel", "addressable led"),
        "summary": "A NeoPixel is an addressable RGB LED with a one-wire timing protocol.",
        "wiring": "Connect VCC, GND, and DIN to one GPIO. For 5V strips from a 3.3V board, add a logic-level shifter and a large bulk capacitor across the LED supply.",
        "code": "Use an Adafruit_NeoPixel or FastLED style library and keep the data pin constant aligned with the canvas wire.",
        "checks": "Large LED strips need an external 5V supply; do not power them from a GPIO or weak board regulator.",
    },
    {
        "title": "Resistor",
        "keywords": ("resistor", "resistance", "ohm", "pullup", "pull-up", "pulldown", "pull-down"),
        "summary": "A resistor limits current, forms voltage dividers, and creates pull-up or pull-down bias paths.",
        "wiring": "Use series placement for LEDs and current limiting; use a divider pair when a sensor output must be reduced for a lower-voltage GPIO.",
        "code": "No library is required, but code assumptions should match the electrical use, such as INPUT_PULLUP for a button to GND.",
        "checks": "Check both resistance and power: P = I^2 * R or P = V * I.",
    },
    {
        "title": "Capacitor",
        "keywords": ("capacitor", "capacitance", "decoupling", "ceramic capacitor", "electrolytic capacitor"),
        "summary": "A capacitor stores charge and filters supply noise or timing signals.",
        "wiring": "Place 100 nF ceramic capacitors close to IC power pins; add larger electrolytic capacitors near motors, relays, servos, and LED strips.",
        "code": "No firmware driver is needed.",
        "checks": "Electrolytic capacitors are polarized; match voltage rating and polarity before simulation or hardware build.",
    },
    {
        "title": "Diode / Flyback Diode",
        "keywords": ("diode", "flyback", "1n4007", "rectifier"),
        "summary": "A diode conducts mainly one way and is often used for polarity protection or inductive kickback suppression.",
        "wiring": "For relay coils and DC motors, place the flyback diode across the inductive load, reverse-biased during normal operation.",
        "code": "No firmware driver is needed.",
        "checks": "Without flyback protection, relay and motor coils can damage transistors or MCU pins.",
    },
    {
        "title": "Transistor / MOSFET Driver",
        "keywords": ("transistor", "mosfet", "driver", "npn", "pnp", "logic level mosfet"),
        "summary": "A transistor or MOSFET lets a GPIO control a higher-current load without sourcing the load current itself.",
        "wiring": "Use a base/gate resistor, common ground, and a flyback diode for inductive loads. Put the load current through the transistor path, not the GPIO.",
        "code": "Drive the base/gate pin with digitalWrite or PWM depending on the load.",
        "checks": "Choose a logic-level MOSFET for 3.3V boards and check current, heat, and gate threshold.",
    },
    {
        "title": "Potentiometer",
        "keywords": ("potentiometer", "pot", "variable resistor"),
        "summary": "A potentiometer is an adjustable voltage divider.",
        "wiring": "Connect the two outer pins to power and GND, then connect the wiper to an analog input.",
        "code": "Use analogRead on the wiper pin and map the ADC value to the needed range.",
        "checks": "Raspberry Pi SBC GPIO has no built-in analog input; use an external ADC for potentiometers on Pi 3/4/5.",
    },
    {
        "title": "Push Button / Switch",
        "keywords": ("button", "push button", "switch", "momentary", "tactile"),
        "summary": "A button or switch changes a GPIO input between two logic states.",
        "wiring": "The simplest MCU wiring is one side to GPIO and the other to GND, with firmware using INPUT_PULLUP.",
        "code": "Use pinMode(pin, INPUT_PULLUP); pressed usually reads LOW in that wiring.",
        "checks": "Avoid floating inputs; add a pull-up/pull-down or use the MCU internal pull-up.",
    },
    {
        "title": "Buzzer",
        "keywords": ("buzzer", "piezo", "speaker"),
        "summary": "A buzzer converts an electrical signal into sound. Active buzzers only need on/off; passive buzzers need a tone frequency.",
        "wiring": "Connect positive to a GPIO or driver and negative to GND. Use a transistor driver if the current is more than the GPIO budget.",
        "code": "Use digitalWrite for active buzzers or tone(pin, frequency) for passive buzzers on supported boards.",
        "checks": "Check current draw and board support for tone/PWM on the selected pin.",
    },
    {
        "title": "DHT11 / DHT22",
        "keywords": ("dht", "dht11", "dht22", "temperature humidity", "sensor_dht11", "sensor_dht22"),
        "summary": "DHT sensors provide digital temperature and humidity over a single data line.",
        "wiring": "Connect VCC, GND, and DATA to a digital GPIO. Add a 4.7 kOhm to 10 kOhm pull-up from DATA to VCC unless the module already has one.",
        "code": "Use a DHT library and match DHT11 versus DHT22 in the type constant.",
        "checks": "Do not poll too quickly; DHT sensors need slow sampling intervals.",
    },
    {
        "title": "HC-SR04 Ultrasonic Sensor",
        "keywords": ("ultrasonic", "hc-sr04", "distance sensor", "sensor_ultrasonic"),
        "summary": "The HC-SR04 measures distance by timing an ultrasonic echo pulse.",
        "wiring": "Connect VCC, GND, TRIG to an output GPIO, and ECHO to an input GPIO. On 3.3V boards, level-shift or divide the 5V ECHO signal.",
        "code": "Pulse TRIG, measure ECHO pulse width, then convert time to distance.",
        "checks": "ECHO is the dangerous pin for Pi/ESP32/Pico if the module is powered at 5V.",
    },
    {
        "title": "PIR Motion Sensor",
        "keywords": ("pir", "motion sensor", "sensor_pir"),
        "summary": "A PIR sensor outputs a digital signal when it detects infrared motion changes.",
        "wiring": "Connect VCC, GND, and OUT to a digital input. Most modules work from 5V, but check whether OUT is 3.3V-safe for the selected board.",
        "code": "Use digitalRead on OUT, usually with simple debounce or delay logic.",
        "checks": "PIR modules need a warm-up time before readings are stable.",
    },
    {
        "title": "LDR / Photoresistor",
        "keywords": ("ldr", "photoresistor", "light sensor", "sensor_ldr"),
        "summary": "An LDR changes resistance with light level.",
        "wiring": "Use it with a fixed resistor as a voltage divider; connect the divider midpoint to an analog input.",
        "code": "Use analogRead and calibrate thresholds for your lighting.",
        "checks": "Pi SBC boards need an external ADC because their GPIO pins are digital only.",
    },
    {
        "title": "MPU6050 / IMU",
        "keywords": ("imu", "mpu6050", "accelerometer", "gyro", "gyroscope", "sensor_imu"),
        "summary": "An IMU measures acceleration and rotation, commonly over I2C.",
        "wiring": "Connect VCC, GND, SDA to {sda}, and SCL to {scl} for the active board.",
        "code": "Use Wire.h plus an MPU6050/Adafruit sensor library, and confirm the I2C address.",
        "checks": "Keep I2C pull-ups to the board logic voltage, not a higher sensor supply.",
    },
    {
        "title": "I2C LCD / OLED Display",
        "keywords": ("lcd", "oled", "display_lcd_i2c", "display_oled", "ssd1306", "i2c display"),
        "summary": "I2C displays use two shared signal lines, SDA and SCL, plus power and ground.",
        "wiring": "Connect VCC, GND, SDA to {sda}, and SCL to {scl} for the active board.",
        "code": "Use LiquidCrystal_I2C for character LCDs or an SSD1306 display library for OLED modules.",
        "checks": "Check I2C address conflicts and make sure pull-ups go to the board logic voltage.",
    },
    {
        "title": "7-Segment Display",
        "keywords": ("7-segment", "seven segment", "display_7seg"),
        "summary": "A 7-segment display is a set of LEDs arranged as numeric segments.",
        "wiring": "Each segment needs current limiting. Use common cathode/common anode wiring correctly, or use a driver IC for cleaner projects.",
        "code": "Drive segment pins directly for one digit, or use a library/driver for multiplexed displays.",
        "checks": "A display can exceed GPIO current limits if many segments are lit without resistors.",
    },
    {
        "title": "Relay Module",
        "keywords": ("relay", "relay_single", "relay_2ch", "relay_4ch"),
        "summary": "A relay lets a low-voltage control signal switch an isolated higher-power circuit.",
        "wiring": "Connect VCC, GND, and IN to a GPIO. Wire the load through COM and NO/NC according to whether it should be normally off or normally on.",
        "code": "Use digitalWrite on the IN pin; many modules are active-low.",
        "checks": "Keep mains/high-voltage wiring physically separate and do not simulate it as if it were a safe GPIO load.",
    },
    {
        "title": "DC Motor",
        "keywords": ("dc motor", "motor_dc", "motor"),
        "summary": "A DC motor is an inductive high-current load.",
        "wiring": "Use a motor driver, MOSFET, or H-bridge with an external supply, shared ground, and flyback protection.",
        "code": "Use PWM through the driver input for speed control and direction pins for H-bridge modules.",
        "checks": "Never connect a motor directly to an MCU GPIO.",
    },
    {
        "title": "Servo Motor",
        "keywords": ("servo", "servo motor", "sg90", "motor_servo"),
        "summary": "A hobby servo uses a power pair and a timing signal to move to an angle.",
        "wiring": "Connect signal to a PWM/timer-capable GPIO, VCC to a suitable 5V/6V supply, and GND to common ground.",
        "code": "Use a Servo library and write angles or pulse widths.",
        "checks": "Servos can brown out boards; use an external supply sized for stall current.",
    },
    {
        "title": "Stepper Motor",
        "keywords": ("stepper", "28byj", "motor_stepper", "stepper motor"),
        "summary": "A stepper moves in controlled increments by energizing coils in sequence.",
        "wiring": "Use a driver board such as ULN2003, A4988, or DRV8825. The MCU should drive logic inputs, not motor coils directly.",
        "code": "Use Stepper or AccelStepper style control with the driver input pins.",
        "checks": "Match coil wiring, driver current, and external supply voltage.",
    },
    {
        "title": "Breadboard",
        "keywords": ("breadboard", "solderless breadboard"),
        "summary": "A breadboard is a temporary wiring matrix with connected rows and power rails.",
        "wiring": "Rows are internally connected in groups; power rails often need jumpers across breaks.",
        "code": "No firmware driver is needed.",
        "checks": "Many wiring mistakes come from assuming all rail segments are continuous.",
    },
    {
        "title": "Ammeter",
        "keywords": ("ammeter", "current meter"),
        "summary": "An ammeter measures branch current and must be placed in series with the load.",
        "wiring": "Break the branch and insert IN/OUT in the current path.",
        "code": "No firmware driver is needed for the simulated instrument.",
        "checks": "Never place an ammeter directly across power and ground.",
    },
    {
        "title": "Oscilloscope",
        "keywords": ("oscilloscope", "scope", "waveform"),
        "summary": "An oscilloscope shows voltage versus time at one or more nodes.",
        "wiring": "Connect CH1/CH2 to signal nodes and scope GND to circuit ground.",
        "code": "No firmware driver is needed for the instrument, but firmware timing affects the waveform.",
        "checks": "A floating scope ground gives misleading readings.",
    },
    {
        "title": "7805 Voltage Regulator",
        "keywords": ("7805", "voltage regulator", "regulator"),
        "summary": "A 7805-style linear regulator makes a 5V rail from a higher input voltage.",
        "wiring": "Connect VIN, GND, and VOUT with input/output capacitors close to the regulator.",
        "code": "No firmware driver is needed.",
        "checks": "Linear regulators dissipate heat: P = (VIN - VOUT) * current.",
    },
]

BOARD_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "ARDUINO_UNO": {
        "name": "Arduino Uno R3",
        "processor": "ATmega328P AVR at 16 MHz",
        "logic": "5V GPIO logic",
        "pins": "14 digital pins, 6 analog inputs, PWM on D3/D5/D6/D9/D10/D11, I2C on A4/A5",
        "note": "Good for beginner 5V circuits, but each GPIO still needs current limiting.",
    },
    "ARDUINO_UNO_R4": {
        "name": "Arduino Uno R4",
        "processor": "Renesas RA4M1 Arm Cortex-M4 at 48 MHz",
        "logic": "5V board I/O with modern peripherals",
        "pins": "Uno-style headers, analog inputs, PWM pins, I2C/SPI/UART",
        "note": "Keep shield wiring compatible with the Uno footprint while checking library support.",
    },
    "ARDUINO_NANO": {
        "name": "Arduino Nano",
        "processor": "ATmega328P AVR at 16 MHz",
        "logic": "5V GPIO logic",
        "pins": "D0-D13, A0-A7, PWM on D3/D5/D6/D9/D10/D11, I2C on A4/A5",
        "note": "Breadboard-friendly Uno-class board.",
    },
    "ARDUINO_MEGA": {
        "name": "Arduino Mega 2560",
        "processor": "ATmega2560 AVR at 16 MHz",
        "logic": "5V GPIO logic",
        "pins": "54 digital pins, 16 analog inputs, hardware serial ports, I2C on D20/D21",
        "note": "Use it when the project needs many pins.",
    },
    "ESP32": {
        "name": "ESP32 DevKit",
        "processor": "dual-core Tensilica Xtensa LX6 class MCU, commonly up to 240 MHz",
        "logic": "3.3V GPIO only",
        "pins": "GPIO with ADC, PWM, I2C, SPI, UART; default I2C is usually SDA D21 and SCL D22",
        "note": "Avoid boot strapping pins and remember ADC2 conflicts with Wi-Fi on classic ESP32.",
    },
    "ESP8266": {
        "name": "ESP8266 NodeMCU",
        "processor": "Tensilica L106 32-bit MCU at 80 MHz",
        "logic": "3.3V GPIO only",
        "pins": "Limited GPIO, one ADC input, I2C commonly SDA D2 and SCL D1",
        "note": "Boot pins D3/D4/D8 need safe reset levels.",
    },
    "RASPBERRY_PI_PICO": {
        "name": "Raspberry Pi Pico",
        "processor": "RP2040 dual-core Arm Cortex-M0+ at 133 MHz",
        "logic": "3.3V GPIO only",
        "pins": "26 usable GPIO, PWM on most pins, ADC on GP26-GP28, I2C/SPI/UART on flexible pins",
        "note": "It is a microcontroller board, not a Linux Raspberry Pi SBC.",
    },
    "RASPBERRY_PI_PICO_2": {
        "name": "Raspberry Pi Pico 2",
        "processor": "RP2350 microcontroller, typically 150 MHz class",
        "logic": "3.3V GPIO only",
        "pins": "Pico-style GPIO with ADC and flexible serial buses",
        "note": "Treat GPIO as 3.3V-only and verify library/core support for RP2350.",
    },
    "RASPBERRY_PI_ZERO_2_W": {
        "name": "Raspberry Pi Zero 2 W",
        "processor": "Broadcom BCM2710A1 quad-core Arm Cortex-A53 at 1 GHz",
        "logic": "3.3V GPIO only",
        "pins": "40-pin header with GPIO2/GPIO3 I2C, GPIO10/9/11 SPI, GPIO14/15 UART",
        "note": "It runs Linux; use GPIO libraries and add an external ADC for analog sensors.",
    },
    "RASPBERRY_PI_3": {
        "name": "Raspberry Pi 3",
        "processor": "Broadcom BCM2837 quad-core Arm Cortex-A53 class SoC",
        "logic": "3.3V GPIO only",
        "pins": "40-pin header with I2C, SPI, UART, PWM-capable GPIO",
        "note": "No analog inputs; use an ADC for LDRs, potentiometers, and analog sensors.",
    },
    "RASPBERRY_PI_4": {
        "name": "Raspberry Pi 4",
        "processor": "Broadcom BCM2711 quad-core Arm Cortex-A72 class SoC",
        "logic": "3.3V GPIO only",
        "pins": "40-pin header with GPIO2/GPIO3 I2C, GPIO10/9/11 SPI, GPIO14/15 UART, PWM GPIO12/13/18/19",
        "note": "Never feed 5V into GPIO. Use level shifting for 5V sensors such as HC-SR04 echo.",
    },
    "RASPBERRY_PI_5": {
        "name": "Raspberry Pi 5",
        "processor": "Broadcom BCM2712 quad-core Arm Cortex-A76 at 2.4 GHz",
        "logic": "3.3V GPIO only",
        "pins": "40-pin header with I2C, SPI, UART, and PWM-capable GPIO",
        "note": "It is a Linux SBC; use external ADC hardware for analog readings.",
    },
    "RASPBERRY_PI_COMPUTE_MODULE": {
        "name": "Raspberry Pi Compute Module",
        "processor": "varies by generation; CM4 uses BCM2711 and CM5 uses BCM2712 class silicon",
        "logic": "3.3V GPIO only",
        "pins": "I/O depends on the carrier board, not only the module.",
        "note": "Always check the carrier board pinout before wiring.",
    },
    "STM32_BLUE_PILL": {
        "name": "STM32 Blue Pill",
        "processor": "STM32F103C8 Arm Cortex-M3 at 72 MHz",
        "logic": "3.3V GPIO, many pins are 5V tolerant only in specific modes",
        "pins": "GPIO, ADC, timers/PWM, I2C/SPI/UART",
        "note": "Check boot pins and voltage tolerance before connecting 5V modules.",
    },
    "STM32_BLACK_PILL": {
        "name": "STM32 Black Pill",
        "processor": "STM32F401/F411 Arm Cortex-M4 class MCU",
        "logic": "3.3V GPIO",
        "pins": "Compact GPIO, ADC, timers/PWM, I2C/SPI/UART",
        "note": "Board variants differ, so verify the exact pinout.",
    },
    "TEENSY_4_0": {
        "name": "Teensy 4.0",
        "processor": "NXP i.MX RT1062 Arm Cortex-M7 at 600 MHz",
        "logic": "3.3V GPIO only",
        "pins": "High-speed GPIO, PWM, ADC, I2C/SPI/UART, USB",
        "note": "Very fast MCU, but GPIO is not 5V tolerant.",
    },
    "TEENSY_4_1": {
        "name": "Teensy 4.1",
        "processor": "NXP i.MX RT1062 Arm Cortex-M7 at 600 MHz",
        "logic": "3.3V GPIO only",
        "pins": "Expanded Teensy 4.x I/O with Ethernet-capable pins",
        "note": "Use external drivers for heavy loads despite the fast processor.",
    },
    "TEENSY_LC": {
        "name": "Teensy LC",
        "processor": "NXP MKL26Z64 Arm Cortex-M0+ at 48 MHz",
        "logic": "3.3V GPIO",
        "pins": "Compact GPIO, ADC, PWM, I2C/SPI/UART",
        "note": "Good low-cost board, but with tighter memory/peripheral limits.",
    },
}

BOARD_FAMILY_KNOWLEDGE: List[Tuple[Tuple[str, ...], Dict[str, str]]] = [
    (("ARDUINO_GIGA", "ARDUINO_PORTENTA"), {
        "name": "High-end Arduino STM32 board",
        "processor": "STM32H747 dual-core Arm Cortex-M7/M4 class MCU",
        "logic": "3.3V logic",
        "pins": "advanced GPIO, ADC/DAC, I2C/SPI/UART, USB and wireless features by model",
        "note": "Use board-specific pin maps because these are not simple Uno-class AVRs.",
    }),
    (("ARDUINO_DUE",), {
        "name": "Arduino Due",
        "processor": "ATSAM3X8E Arm Cortex-M3 at 84 MHz",
        "logic": "3.3V GPIO only",
        "pins": "54 digital I/O, analog inputs, DAC outputs, I2C/SPI/UART",
        "note": "Do not connect 5V signals directly to Due GPIO.",
    }),
    (("ESP32_C3", "ESP32_C6", "ESP32_H2"), {
        "name": "ESP32 RISC-V family board",
        "processor": "Espressif RISC-V MCU, speed depends on exact variant",
        "logic": "3.3V GPIO only",
        "pins": "GPIO with I2C/SPI/UART/PWM and variant-specific ADC/radio features",
        "note": "Check strapping pins and exact module pinout.",
    }),
    (("ESP32_",), BOARD_KNOWLEDGE["ESP32"]),
    (("ESP8266_",), BOARD_KNOWLEDGE["ESP8266"]),
    (("RASPBERRY_PI_PICO_W",), BOARD_KNOWLEDGE["RASPBERRY_PI_PICO"]),
    (("RASPBERRY_PI_",), BOARD_KNOWLEDGE["RASPBERRY_PI_4"]),
    (("SEEED_XIAO",), {
        "name": "Seeed XIAO board",
        "processor": "depends on variant: SAMD21, RP2040, ESP32, or nRF52840",
        "logic": "mostly 3.3V GPIO",
        "pins": "small GPIO set with I2C/SPI/UART and variant-specific analog/PWM",
        "note": "Use the exact XIAO variant before assigning pins.",
    }),
    (("ADAFRUIT_FEATHER",), {
        "name": "Adafruit Feather board",
        "processor": "depends on Feather variant",
        "logic": "usually 3.3V GPIO",
        "pins": "Feather-style GPIO with I2C/SPI/UART and battery support on many boards",
        "note": "Check the exact Feather processor and pinout.",
    }),
    (("SPARKFUN_THING_PLUS",), {
        "name": "SparkFun Thing Plus board",
        "processor": "depends on variant, commonly ESP32, RP2040, or Artemis",
        "logic": "usually 3.3V GPIO",
        "pins": "Thing Plus/Feather-style GPIO with Qwiic/I2C support on many boards",
        "note": "Check exact variant before choosing analog/PWM pins.",
    }),
    (("ATMEL_AVR", "ARDUINO_LEONARDO", "ARDUINO_MICRO"), {
        "name": "AVR Arduino-class board",
        "processor": "8-bit AVR MCU, commonly ATmega328P or ATmega32U4",
        "logic": "usually 5V GPIO",
        "pins": "digital GPIO, PWM timers, ADC inputs, I2C/SPI/UART",
        "note": "GPIO current is limited even on 5V boards.",
    }),
]


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


def is_ultrasonic(component: Dict[str, Any]) -> bool:
    label = component_label(component)
    return "ULTRASONIC" in component_type(component) or "hc-sr04" in label


def is_pir(component: Dict[str, Any]) -> bool:
    return "PIR" in component_type(component) or "pir" in component_label(component)


def is_ldr(component: Dict[str, Any]) -> bool:
    label = component_label(component)
    return "LDR" in component_type(component) or "photoresistor" in label or "light dependent" in label


def is_imu(component: Dict[str, Any]) -> bool:
    label = component_label(component)
    kind = component_type(component)
    return "IMU" in kind or "MPU6050" in kind or "accelerometer" in label or "gyro" in label


def is_neopixel(component: Dict[str, Any]) -> bool:
    label = component_label(component)
    kind = component_type(component)
    return "NEOPIXEL" in kind or "WS2812" in kind or "ws2812" in label


def is_buzzer(component: Dict[str, Any]) -> bool:
    return "BUZZER" in component_type(component) or "buzzer" in component_label(component)


def is_stepper(component: Dict[str, Any]) -> bool:
    return "STEPPER" in component_type(component) or "stepper" in component_label(component)


def is_seven_segment(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    label = component_label(component)
    return "7SEG" in kind or "7-segment" in label or "seven segment" in label


def is_instrument(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    return "AMMETER" in kind or "OSCILLOSCOPE" in kind or "MULTIMETER" in kind


def is_voltage_regulator(component: Dict[str, Any]) -> bool:
    kind = component_type(component)
    label = component_label(component)
    return "REGULATOR" in kind or "7805" in kind or "voltage regulator" in label


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
    if board.startswith("RASPBERRY_PI") and "PICO" not in board:
        return "d2", "d3"
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
    elif board.startswith("RASPBERRY_PI") and "PICO" not in board:
        candidates.extend(["d17", "d27", "d22", "d23", "d24", "d25", "d5", "d6", "d12", "d13", "d18", "d19", "d20", "d21"])
    elif "PICO" in board or "RP2040" in board or "RP2350" in board:
        candidates.extend([f"d{index}" for index in range(0, 23)])
    else:
        candidates.extend(["d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13"])
    for candidate in candidates:
        if normalize_pin(candidate) not in used:
            used.add(normalize_pin(candidate))
            return candidate
    return candidates[-1]


def default_analog_pin(board_type: str, used: Set[str], preferred: Optional[str] = "a0") -> Optional[str]:
    profile = board_rule_profile(board_type)
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(sorted(str(pin) for pin in profile.get("adcPins", set())))
    for candidate in candidates:
        normalized = normalize_pin(candidate, "analog")
        if normalized and normalized not in used:
            used.add(normalized)
            return candidate
    return candidates[-1] if candidates else None


def board_rule_profile(board_type: str) -> Dict[str, Any]:
    board = (board_type or "ARDUINO_UNO").upper()
    if board in BOARD_RULES:
        return BOARD_RULES[board]
    if board.startswith("ESP32"):
        return BOARD_RULES["ESP32"]
    if "PICO" in board or "RP2040" in board or "RP2350" in board:
        return BOARD_RULES["RASPBERRY_PI_PICO"]
    if board.startswith("RASPBERRY_PI"):
        return BOARD_RULES["RASPBERRY_PI_SBC"]
    if board.startswith("STM32"):
        return BOARD_RULES["STM32_BLUE_PILL"]
    if "MEGA" in board:
        return BOARD_RULES["ARDUINO_MEGA"]
    if "NANO" in board:
        return BOARD_RULES["ARDUINO_NANO"]
    if board.startswith("ARDUINO") or board.startswith("ATMEL_AVR"):
        return BOARD_RULES["ARDUINO_UNO"]
    return {
        "logicVoltage": 3.3,
        "recommendedPinMa": 12.0,
        "absolutePinMa": 20.0,
        "totalPackageMa": 100.0,
        "pwmPins": set(),
        "adcPins": set(),
        "i2c": expected_i2c_pins(board_type),
    }


def parse_numeric(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return fallback
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else fallback


def parse_ohms(value: Any, fallback: float = 220.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.001, float(value))
    text = safe_lower(value)
    if not text:
        return fallback
    numeric = parse_numeric(text, fallback)
    if re.search(r"\bmeg\b|mohm|mega", text):
        numeric *= 1_000_000
    elif "k" in text:
        numeric *= 1_000
    return max(0.001, numeric)


def parse_voltage(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = safe_lower(value).replace("3v3", "3.3v")
    if not text:
        return fallback
    numeric = parse_numeric(text, fallback)
    if "mv" in text:
        numeric /= 1000
    return numeric


def parse_current_ma(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = safe_lower(value)
    if not text:
        return fallback
    numeric = parse_numeric(text, fallback)
    if "ma" not in text and re.search(r"\ba\b|amp", text):
        numeric *= 1000
    return numeric


def parse_power_watts(value: Any, fallback: float = 0.25) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = safe_lower(value)
    if not text:
        return fallback
    numeric = parse_numeric(text, fallback)
    if "mw" in text:
        numeric /= 1000
    return numeric


def format_ohms(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g} M\u03a9"
    if value >= 1000:
        return f"{value / 1000:.3g} k\u03a9"
    return f"{value:.3g} \u03a9"


def format_ma(value: float) -> str:
    return f"{value:.2f} mA"


def format_voltage(value: float) -> str:
    return f"{value:.2f} V"


def format_watts(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.1f} mW"
    return f"{value:.2f} W"


def resistor_value(component: Dict[str, Any]) -> float:
    props = component.get("properties") or {}
    return parse_ohms(props.get("resistance") or props.get("value") or props.get("ohms"), 220.0)


def resistor_power_rating(component: Dict[str, Any]) -> float:
    props = component.get("properties") or {}
    return parse_power_watts(
        props.get("maxPower")
        or props.get("maxPowerWatts")
        or props.get("power")
        or props.get("wattage"),
        0.25,
    )


def led_forward_voltage(component: Dict[str, Any]) -> float:
    props = component.get("properties") or {}
    color = safe_lower(props.get("color") or component.get("name"))
    fallback = 3.1 if any(token in color for token in ("blue", "white")) else 2.0
    return parse_voltage(props.get("forwardVoltage") or props.get("vf"), fallback)


def led_max_current_ma(component: Dict[str, Any]) -> float:
    props = component.get("properties") or {}
    return parse_current_ma(props.get("maxCurrent") or props.get("currentLimit"), 20.0)


def extract_code_constants(code: str) -> Dict[str, str]:
    constants: Dict[str, str] = {}
    if not code:
        return constants
    for match in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)\s+([A-Za-z_]*\d+|[ADad]\d+)\b", code, re.MULTILINE):
        constants[match.group(1)] = match.group(2)
    for match in re.finditer(
        r"\b(?:const\s+)?(?:int|byte|uint8_t|uint16_t|long)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]*\d+|[ADad]\d+)\b",
        code,
    ):
        constants[match.group(1)] = match.group(2)
    return constants


def extract_pin_modes(code: str) -> Dict[str, str]:
    constants = extract_code_constants(code)
    modes: Dict[str, str] = {}
    if not code:
        return modes
    for match in re.finditer(r"\bpinMode\s*\(\s*([A-Za-z_]\w*|[ADad]?\d+)\s*,\s*([A-Za-z_]\w*)", code):
        raw_pin = match.group(1)
        mode = match.group(2).upper()
        normalized = normalize_pin(constants.get(raw_pin, raw_pin))
        if normalized:
            modes[normalized] = mode
    return modes


def code_uses_wifi(code: str) -> bool:
    return bool(re.search(r"\b(WiFi|ESP8266WiFi|WiFiClient|WiFiServer)\b", code or ""))


def pin_ref(component_id: str, pin_id: str) -> str:
    return f"{component_id}/{pin_id}"


def split_pin_ref(ref: str) -> Tuple[str, str]:
    if "/" in ref:
        component_id, pin_id = ref.split("/", 1)
        return component_id, pin_id
    if ":" in ref:
        component_id, pin_id = ref.split(":", 1)
        return component_id, pin_id
    return ref, ""


def wire_pin_refs(wire: Dict[str, Any]) -> Tuple[str, str]:
    return (
        pin_ref(wire_from_component(wire), wire_from_pin(wire)),
        pin_ref(wire_to_component(wire), wire_to_pin(wire)),
    )


def add_breadboard_aliases(parent: Dict[str, str], component: Dict[str, Any]) -> None:
    if component_type(component) != "BREADBOARD":
        return
    comp_id = str(component.get("id"))
    available = {str(pin.get("id")) for pin in pins_for(component)}

    def add(key: str) -> None:
        parent.setdefault(key, key)

    def find(key: str) -> str:
        add(key)
        current = parent[key]
        if current != key:
            parent[key] = find(current)
        return parent[key]

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    def connect(pin_ids: List[str]) -> None:
        existing = [pin_id for pin_id in pin_ids if pin_id in available]
        for index in range(1, len(existing)):
            union(pin_ref(comp_id, existing[0]), pin_ref(comp_id, existing[index]))

    for column in range(1, 31):
        connect([f"{row}{column}" for row in ("a", "b", "c", "d", "e")])
        connect([f"{row}{column}" for row in ("f", "g", "h", "i", "j")])
    connect([pin_id for pin_id in available if re.fullmatch(r"vcc_top_\d+", pin_id, re.IGNORECASE)])
    connect([pin_id for pin_id in available if re.fullmatch(r"gnd_top_\d+", pin_id, re.IGNORECASE)])
    connect([pin_id for pin_id in available if re.fullmatch(r"vcc_bottom_\d+", pin_id, re.IGNORECASE)])
    connect([pin_id for pin_id in available if re.fullmatch(r"gnd_bottom_\d+", pin_id, re.IGNORECASE)])


def supplied_netlist_to_maps(netlist: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    pin_to_net: Dict[str, str] = {}
    net_to_pins: Dict[str, List[str]] = {}
    if not isinstance(netlist, dict):
        return pin_to_net, net_to_pins

    raw_pin_to_net = netlist.get("pinToNet")
    if isinstance(raw_pin_to_net, dict):
        for ref, net_id in raw_pin_to_net.items():
            comp_id, pin_id = split_pin_ref(str(ref))
            if comp_id and pin_id:
                canonical = pin_ref(comp_id, pin_id)
                pin_to_net[canonical] = str(net_id)
                net_to_pins.setdefault(str(net_id), []).append(canonical)

    raw_nets = netlist.get("nets") or netlist.get("nodes") or []
    if isinstance(raw_nets, list):
        for index, net in enumerate(raw_nets):
            if not isinstance(net, dict):
                continue
            net_id = str(net.get("id") or f"N{index + 1}")
            pins = net.get("pins") or []
            if not isinstance(pins, list):
                continue
            for item in pins:
                if isinstance(item, str):
                    comp_id, pin_id = split_pin_ref(item)
                elif isinstance(item, dict):
                    comp_id = str(item.get("nodeId") or item.get("componentId") or item.get("component") or "")
                    pin_id = str(item.get("pinId") or item.get("pin") or "")
                else:
                    continue
                if not comp_id or not pin_id:
                    continue
                canonical = pin_ref(comp_id, pin_id)
                pin_to_net[canonical] = net_id
                net_to_pins.setdefault(net_id, []).append(canonical)

    return pin_to_net, {net_id: sorted(set(pins)) for net_id, pins in net_to_pins.items()}


def build_circuit_graph(
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = context or {}
    by_id = component_by_id(components)
    supplied_netlist = context.get("netlist") if isinstance(context.get("netlist"), dict) else None
    pin_to_net, net_to_pins = supplied_netlist_to_maps(supplied_netlist)

    if not pin_to_net:
        parent: Dict[str, str] = {}

        def add(key: str) -> None:
            parent.setdefault(key, key)

        def find(key: str) -> str:
            add(key)
            current = parent[key]
            if current != key:
                parent[key] = find(current)
            return parent[key]

        def union(a: str, b: str) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for component in components:
            comp_id = str(component.get("id"))
            for pin in pins_for(component):
                add(pin_ref(comp_id, str(pin.get("id"))))

            if is_mcu(component):
                ground_pins = [str(pin.get("id")) for pin in pins_for(component) if is_ground_pin(component, str(pin.get("id")))]
                for index in range(1, len(ground_pins)):
                    union(pin_ref(comp_id, ground_pins[0]), pin_ref(comp_id, ground_pins[index]))
            add_breadboard_aliases(parent, component)

        for wire in wires:
            a, b = wire_pin_refs(wire)
            if a != "/" and b != "/":
                union(a, b)

        groups: Dict[str, List[str]] = {}
        for key in list(parent.keys()):
            groups.setdefault(find(key), []).append(key)

        for index, pins in enumerate(groups.values(), 1):
            net_id = f"N{index}"
            for ref in pins:
                pin_to_net[ref] = net_id
            net_to_pins[net_id] = sorted(set(pins))

    next_index = len(net_to_pins) + 1
    for component in components:
        comp_id = str(component.get("id"))
        for pin in pins_for(component):
            ref = pin_ref(comp_id, str(pin.get("id")))
            if ref in pin_to_net:
                continue
            net_id = f"N{next_index}"
            next_index += 1
            pin_to_net[ref] = net_id
            net_to_pins[net_id] = [ref]

    return {
        "components": components,
        "componentsById": by_id,
        "wires": wires,
        "connections": build_connections(wires),
        "pinToNet": pin_to_net,
        "netToPins": net_to_pins,
    }


def component_for_ref(graph: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
    comp_id, _ = split_pin_ref(ref)
    return graph["componentsById"].get(comp_id)


def net_for_pin(graph: Dict[str, Any], component_id: str, pin_id: Optional[str]) -> Optional[str]:
    if not pin_id:
        return None
    return graph["pinToNet"].get(pin_ref(component_id, pin_id))


def pin_ids_on_net(graph: Dict[str, Any], net_id: Optional[str]) -> List[str]:
    if not net_id:
        return []
    return graph["netToPins"].get(net_id, [])


def rail_voltage_from_pin(component: Dict[str, Any], pin_id: str, board_type: str) -> float:
    label = pin_label(component, pin_id)
    if "3v3" in label or "3.3v" in label or re.search(r"\b3v\b", label):
        return 3.3
    if "5v" in label or "vbus" in label or "usb" in label:
        return 5.0
    if "vin" in label:
        return parse_voltage((component.get("properties") or {}).get("vin"), 7.0)
    if "vsys" in label or "bat" in label:
        return parse_voltage((component.get("properties") or {}).get("voltage"), 3.7)
    if is_mcu(component) and is_signal_pin(component, pin_id):
        return float(board_rule_profile(board_type).get("logicVoltage", 3.3))
    props = component.get("properties") or {}
    if component_type(component) in {"POWER_SUPPLY", "BATTERY_9V"} and is_power_pin(component, pin_id):
        return parse_voltage(props.get("voltage"), 9.0 if component_type(component) == "BATTERY_9V" else 5.0)
    if component_type(component) == "VOLTAGE_REGULATOR_7805" and safe_lower(pin_id) in {"vout", "out"}:
        return 5.0
    return 0.0


def net_voltage(graph: Dict[str, Any], net_id: Optional[str], board_type: str, include_gpio: bool = True) -> float:
    voltage = 0.0
    for ref in pin_ids_on_net(graph, net_id):
        comp_id, pin_id = split_pin_ref(ref)
        component = graph["componentsById"].get(comp_id)
        if not component:
            continue
        if is_mcu(component) and is_signal_pin(component, pin_id) and not include_gpio:
            continue
        voltage = max(voltage, rail_voltage_from_pin(component, pin_id, board_type))
    return voltage


def net_has_ground(graph: Dict[str, Any], net_id: Optional[str]) -> bool:
    return any(
        (component := component_for_ref(graph, ref)) is not None and is_ground_pin(component, split_pin_ref(ref)[1])
        for ref in pin_ids_on_net(graph, net_id)
    )


def net_has_power_source(graph: Dict[str, Any], net_id: Optional[str], board_type: str, include_gpio: bool = True) -> bool:
    return net_voltage(graph, net_id, board_type, include_gpio=include_gpio) > 0


def mcu_gpios_on_net(graph: Dict[str, Any], net_id: Optional[str]) -> List[Tuple[str, str]]:
    gpios: List[Tuple[str, str]] = []
    for ref in pin_ids_on_net(graph, net_id):
        comp_id, pin_id = split_pin_ref(ref)
        component = graph["componentsById"].get(comp_id)
        if component and is_mcu(component) and is_signal_pin(component, pin_id):
            gpios.append((comp_id, normalize_pin(pin_id)))
    return gpios


def other_net_for_two_pin_component(graph: Dict[str, Any], component: Dict[str, Any], net_id: str) -> Optional[str]:
    comp_id = str(component.get("id"))
    nets = []
    for pin in pins_for(component):
        current = net_for_pin(graph, comp_id, str(pin.get("id")))
        if current:
            nets.append(current)
    unique = [item for item in dict.fromkeys(nets) if item != net_id]
    return unique[0] if unique else None


def resistors_touching_net(graph: Dict[str, Any], net_id: Optional[str]) -> List[Tuple[Dict[str, Any], str]]:
    matches: List[Tuple[Dict[str, Any], str]] = []
    if not net_id:
        return matches
    for component in graph["components"]:
        if not is_resistor(component):
            continue
        other = other_net_for_two_pin_component(graph, component, net_id)
        comp_id = str(component.get("id"))
        if other and any(net_for_pin(graph, comp_id, str(pin.get("id"))) == net_id for pin in pins_for(component)):
            matches.append((component, other))
    return matches


def has_pull_resistor(graph: Dict[str, Any], net_id: Optional[str], board_type: str) -> bool:
    for resistor, other_net in resistors_touching_net(graph, net_id):
        value = resistor_value(resistor)
        if 1_000 <= value <= 100_000 and (net_has_ground(graph, other_net) or net_has_power_source(graph, other_net, board_type, include_gpio=False)):
            return True
    return False


def make_issue(severity: str, component_id: str, message: str, suggested_fix: str) -> Dict[str, Any]:
    return {
        "severity": severity,
        "componentId": component_id,
        "message": message,
        "suggestedFix": suggested_fix,
    }


def add_unique_issue(issues: List[Dict[str, Any]], issue: Dict[str, Any]) -> None:
    key = (issue.get("severity"), issue.get("componentId"), issue.get("message"))
    if not any((item.get("severity"), item.get("componentId"), item.get("message")) == key for item in issues):
        issues.append(issue)


def add_unique_action(actions: List[Dict[str, Any]], action: Dict[str, Any]) -> None:
    key = json.dumps(action, sort_keys=True, default=str)
    if not any(json.dumps(item, sort_keys=True, default=str) == key for item in actions):
        actions.append(action)


def find_resistor_path_to_role(
    graph: Dict[str, Any],
    net_id: Optional[str],
    board_type: str,
    role: str,
) -> Optional[Tuple[Dict[str, Any], str, float]]:
    if not net_id:
        return None
    for resistor, other_net in resistors_touching_net(graph, net_id):
        if role == "source" and net_has_power_source(graph, other_net, board_type):
            return resistor, other_net, net_voltage(graph, other_net, board_type)
        if role == "ground" and net_has_ground(graph, other_net):
            return resistor, other_net, 0.0
    return None


def analyze_leds(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
    value_changes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    calculations: List[Dict[str, Any]] = []
    gpio_loads: Dict[str, float] = {}

    for component in graph["components"]:
        if not is_led(component) or "NEOPIXEL" in component_type(component):
            continue
        comp_id = str(component.get("id"))
        anode = find_role_pin(component, ["anode", "a", "+", "red", "green", "blue"], "anode")
        cathode = find_role_pin(component, ["cathode", "k", "gnd", "-", "neg"], "cathode")
        anode_net = net_for_pin(graph, comp_id, anode)
        cathode_net = net_for_pin(graph, comp_id, cathode)
        if not anode_net or not cathode_net:
            continue

        source_net = anode_net if net_has_power_source(graph, anode_net, board_type) else None
        source_voltage = net_voltage(graph, anode_net, board_type) if source_net else 0.0
        series_resistor: Optional[Dict[str, Any]] = None
        series_resistance = 0.0

        source_path = find_resistor_path_to_role(graph, anode_net, board_type, "source")
        if source_path:
            series_resistor, source_net, source_voltage = source_path
            series_resistance += resistor_value(series_resistor)

        ground_ok = net_has_ground(graph, cathode_net)
        ground_path = find_resistor_path_to_role(graph, cathode_net, board_type, "ground")
        if ground_path:
            ground_resistor, _, _ = ground_path
            if not series_resistor or ground_resistor.get("id") != series_resistor.get("id"):
                series_resistance += resistor_value(ground_resistor)
            series_resistor = series_resistor or ground_resistor
            ground_ok = True

        direct_gpio = bool(mcu_gpios_on_net(graph, anode_net) or mcu_gpios_on_net(graph, cathode_net))
        if source_net and ground_ok and not series_resistor:
            severity = "CRITICAL" if direct_gpio else "WARNING"
            add_unique_issue(
                issues,
                make_issue(
                    severity,
                    comp_id,
                    f"{component.get('name', comp_id)} has no current-limiting resistor in the LED branch.",
                    "Insert a calculated series resistor between the source/GPIO and LED anode or between cathode and ground.",
                ),
            )
            source_ref = f"{mcu_gpios_on_net(graph, anode_net)[0][0]}/{mcu_gpios_on_net(graph, anode_net)[0][1]}" if mcu_gpios_on_net(graph, anode_net) else f"{comp_id}/{anode}"
            add_unique_action(
                additions,
                {
                    "type": "component",
                    "componentType": "RESISTOR",
                    "value": "220 Ohm",
                    "between": [source_ref, f"{comp_id}/{anode or 'anode'}"],
                    "reason": "Add a current-limiting resistor for the LED branch.",
                },
            )
            continue

        if not source_net or not ground_ok or series_resistance <= 0:
            continue

        vf = led_forward_voltage(component)
        current_a = max(0.0, (source_voltage - vf) / series_resistance)
        current_ma = current_a * 1000
        resistor_power = current_a * current_a * series_resistance
        max_current = led_max_current_ma(component)
        calculation = {
            "type": "LED_BRANCH",
            "componentId": comp_id,
            "sourceVoltage": source_voltage,
            "forwardVoltage": vf,
            "seriesResistance": series_resistance,
            "currentMa": current_ma,
            "resistorPowerW": resistor_power,
            "formula": "I = (V_supply - V_f) / R",
        }
        calculations.append(calculation)

        for pin_id in {pin_id for _, pin_id in mcu_gpios_on_net(graph, source_net)} | {pin_id for _, pin_id in mcu_gpios_on_net(graph, anode_net)}:
            gpio_loads[pin_id] = gpio_loads.get(pin_id, 0.0) + current_ma

        if current_ma > max_current:
            suggested_resistance = max(1.0, (source_voltage - vf) / (max_current / 1000.0))
            rounded = round(suggested_resistance / 10) * 10
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL" if current_ma > max_current * 2 else "WARNING",
                    comp_id,
                    f"{component.get('name', comp_id)} current is {format_ma(current_ma)}; limit is {format_ma(max_current)}.",
                    f"Increase the series resistance to at least {format_ohms(rounded)}.",
                ),
            )
            if series_resistor:
                add_unique_action(
                    value_changes,
                    {
                        "type": "valueChange",
                        "componentId": str(series_resistor.get("id")),
                        "property": "resistance",
                        "newValue": rounded,
                        "value": format_ohms(rounded),
                        "reason": f"Reduce {component.get('name', comp_id)} current to {format_ma(max_current)} or below.",
                    },
                )

        if series_resistor and resistor_power > resistor_power_rating(series_resistor):
            rating = resistor_power_rating(series_resistor)
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL" if resistor_power > rating * 2 else "WARNING",
                    str(series_resistor.get("id")),
                    f"{series_resistor.get('name', 'Resistor')} dissipates {format_watts(resistor_power)}; rating is {format_watts(rating)}.",
                    "Use a higher wattage resistor or reduce the branch current.",
                ),
            )

    return calculations, gpio_loads


def diode_across_nets(graph: Dict[str, Any], net_a: Optional[str], net_b: Optional[str]) -> bool:
    if not net_a or not net_b:
        return False
    wanted = {net_a, net_b}
    for component in graph["components"]:
        kind = component_type(component)
        if "DIODE" not in kind and "1N400" not in kind and "1N4148" not in kind:
            continue
        comp_id = str(component.get("id"))
        anode = find_role_pin(component, ["anode", "a"], "anode")
        cathode = find_role_pin(component, ["cathode", "k"], "cathode")
        diode_nets = {net_for_pin(graph, comp_id, anode), net_for_pin(graph, comp_id, cathode)}
        if diode_nets == wanted:
            return True
    return False


def capacitor_across_power(graph: Dict[str, Any], supply_net: Optional[str], ground_net: Optional[str]) -> bool:
    if not supply_net or not ground_net:
        return False
    wanted = {supply_net, ground_net}
    for component in graph["components"]:
        if "CAPACITOR" not in component_type(component):
            continue
        comp_id = str(component.get("id"))
        nets = {
            net_for_pin(graph, comp_id, str(pin.get("id")))
            for pin in pins_for(component)
        }
        if wanted.issubset(nets):
            return True
    return False


def has_switching_device_near_load(graph: Dict[str, Any], load_nets: Set[str]) -> bool:
    for component in graph["components"]:
        kind = component_type(component)
        if not any(token in kind for token in ("MOSFET", "TRANSISTOR", "DRIVER", "L293", "L298", "ULN", "ESC")):
            continue
        comp_id = str(component.get("id"))
        nets = {
            net_for_pin(graph, comp_id, str(pin.get("id")))
            for pin in pins_for(component)
        }
        if load_nets & {net for net in nets if net}:
            return True
    return False


def analyze_inductive_loads(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
) -> None:
    for component in graph["components"]:
        kind = component_type(component)
        if not (is_motor(component) or is_relay(component) or "SOLENOID" in kind):
            continue
        if is_servo(component) or "BLDC" in kind or "STEPPER" in kind:
            continue
        comp_id = str(component.get("id"))
        pins = pins_for(component)
        terminal_nets = [
            net_for_pin(graph, comp_id, str(pin.get("id")))
            for pin in pins
            if is_signal_pin(component, str(pin.get("id"))) or "coil" in pin_label(component, str(pin.get("id"))) or "m" in safe_lower(pin.get("id"))
        ]
        terminal_nets = [net for net in terminal_nets if net]
        if len(set(terminal_nets)) < 2:
            continue
        net_a, net_b = list(dict.fromkeys(terminal_nets))[:2]
        load_name = component.get("name", comp_id)
        if not diode_across_nets(graph, net_a, net_b):
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    comp_id,
                    f"{load_name} is an inductive load without a flyback diode across its terminals.",
                    "Add a 1N4007 or 1N4148 diode across the load, cathode to supply side and anode to low-side/ground side.",
                ),
            )
            add_unique_action(
                additions,
                {
                    "type": "component",
                    "componentType": "DIODE",
                    "value": "1N4007",
                    "between": [f"{comp_id}/{pins[0].get('id')}", f"{comp_id}/{pins[1].get('id')}"],
                    "reason": f"Clamp {load_name} back-EMF.",
                },
            )
        if any(mcu_gpios_on_net(graph, net) for net in (net_a, net_b)) and not has_switching_device_near_load(graph, {net_a, net_b}):
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    comp_id,
                    "DC motor is wired directly to a microcontroller GPIO pin." if is_motor(component) else f"{load_name} appears to be driven directly from an MCU GPIO.",
                    "Use a MOSFET/transistor or motor driver, an external supply sized for stall current, and a shared ground.",
                ),
            )
        supply_net = net_a if net_voltage(graph, net_a, board_type, include_gpio=False) >= net_voltage(graph, net_b, board_type, include_gpio=False) else net_b
        ground_net = net_b if supply_net == net_a else net_a
        if not capacitor_across_power(graph, supply_net, ground_net):
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    comp_id,
                    f"{load_name} power path has no visible decoupling capacitor.",
                    "Add local decoupling near the load, for example 100 nF plus a bulk capacitor sized for motor/relay current.",
                ),
            )


def component_supply_voltage(graph: Dict[str, Any], component: Dict[str, Any], board_type: str) -> float:
    comp_id = str(component.get("id"))
    power_pin = find_role_pin(component, ["vcc", "vdd", "vin", "5v", "3v3", "+"])
    return net_voltage(graph, net_for_pin(graph, comp_id, power_pin), board_type, include_gpio=False)


def analyze_logic_levels(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
) -> None:
    profile = board_rule_profile(board_type)
    logic_voltage = float(profile.get("logicVoltage", 3.3))
    if logic_voltage >= 5.0:
        return

    for component in graph["components"]:
        if is_mcu(component):
            continue
        comp_id = str(component.get("id"))
        supply = component_supply_voltage(graph, component, board_type)
        if supply <= logic_voltage + 0.3:
            continue
        for pin in pins_for(component):
            pin_id = str(pin.get("id"))
            label = pin_label(component, pin_id)
            is_output_like = (
                pin.get("type") == "output"
                or any(token in label for token in ("echo", "tx", "out", "sda", "scl", "data", "sig"))
            )
            if not is_output_like:
                continue
            net_id = net_for_pin(graph, comp_id, pin_id)
            gpios = mcu_gpios_on_net(graph, net_id)
            if not gpios:
                continue
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    comp_id,
                    f"{component.get('name', comp_id)} drives a {format_voltage(supply)} signal into a {format_voltage(logic_voltage)} MCU input on {display_pin(gpios[0][1])}.",
                    "Add a voltage divider or bidirectional level shifter before connecting this signal to the MCU.",
                ),
            )


def analyze_gpio_budget(
    graph: Dict[str, Any],
    board_type: str,
    gpio_loads: Dict[str, float],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    profile = board_rule_profile(board_type)
    recommended = float(profile.get("recommendedPinMa", 12.0))
    absolute = float(profile.get("absolutePinMa", recommended))
    total_limit = float(profile.get("totalPackageMa", 100.0))
    total = sum(gpio_loads.values())
    mcu = find_mcu(graph["components"], board_type)
    mcu_id = str(mcu.get("id")) if mcu else "MCU"

    for pin, current in sorted(gpio_loads.items()):
        if current > absolute:
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    mcu_id,
                    f"{display_pin(pin)} is estimated at {format_ma(current)}, above the {format_ma(absolute)} absolute pin limit.",
                    "Use a driver transistor/MOSFET or increase load resistance so the GPIO only supplies a safe signal current.",
                ),
            )
        elif current > recommended:
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    mcu_id,
                    f"{display_pin(pin)} is estimated at {format_ma(current)}, above the recommended {format_ma(recommended)} design limit.",
                    "Reduce load current or use a driver stage for better reliability.",
                ),
            )

    if total > total_limit:
        add_unique_issue(
            issues,
            make_issue(
                "CRITICAL",
                mcu_id,
                f"Total estimated GPIO load is {format_ma(total)}, above the package budget of {format_ma(total_limit)}.",
                "Move loads to external drivers and avoid powering actuators from MCU pins.",
            ),
        )

    return {
        "perPinMa": gpio_loads,
        "totalMa": total,
        "recommendedPinMa": recommended,
        "absolutePinMa": absolute,
        "totalPackageMa": total_limit,
    }


def analyze_high_power_actuators(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
) -> None:
    for component in graph["components"]:
        if not (is_servo(component) or (is_motor(component) and not is_servo(component))):
            continue
        comp_id = str(component.get("id"))
        vcc = find_role_pin(component, ["vcc", "vin", "5v", "+"])
        vcc_net = net_for_pin(graph, comp_id, vcc)
        powered_from_board = any(
            (board_component := component_for_ref(graph, ref)) is not None
            and is_mcu(board_component)
            and is_power_pin(board_component, split_pin_ref(ref)[1])
            for ref in pin_ids_on_net(graph, vcc_net)
        )
        if not powered_from_board:
            continue
        stall_current = parse_current_ma((component.get("properties") or {}).get("stallCurrent"), 650.0 if is_servo(component) else 500.0)
        severity = "CRITICAL" if stall_current >= 500 else "WARNING"
        add_unique_issue(
            issues,
            make_issue(
                severity,
                comp_id,
                f"{component.get('name', comp_id)} may draw {format_ma(stall_current)} from the board rail during stall/startup.",
                "Use an external 5V/6V supply sized for stall current and connect supply ground to MCU ground.",
            ),
        )


def analyze_board_pin_rules(
    graph: Dict[str, Any],
    board_type: str,
    code: str,
    issues: List[Dict[str, Any]],
) -> None:
    profile = board_rule_profile(board_type)
    pin_modes = extract_pin_modes(code)
    pin_usages = extract_pin_usages(code)
    used_functions: Dict[str, Set[str]] = {}
    for usage in pin_usages:
        used_functions.setdefault(usage["pin"], set()).add(usage["function"])

    mcu = find_mcu(graph["components"], board_type)
    if not mcu:
        return
    mcu_id = str(mcu.get("id"))
    connected = connected_mcu_signal_pins(graph["components"], graph["wires"])
    input_only = set(profile.get("inputOnlyPins") or set())
    strap_pins = set(profile.get("strapPins") or set())
    pwm_pins = set(profile.get("pwmPins") or set())
    adc2_pins = set(profile.get("adc2Pins") or set())

    for pin in sorted(connected | set(pin_modes.keys()) | set(used_functions.keys())):
        normalized = normalize_pin(pin)
        if normalized in strap_pins:
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    mcu_id,
                    f"{display_pin(normalized)} is a boot strapping pin on {board_type}.",
                    "Avoid external circuits that pull this pin to the wrong level during reset/boot.",
                ),
            )
        if normalized in input_only and (
            pin_modes.get(normalized) == "OUTPUT"
            or used_functions.get(normalized, set()) & {"digitalWrite", "analogWrite", "tone"}
        ):
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    mcu_id,
                    f"{display_pin(normalized)} is input-only on {board_type}, but firmware/canvas uses it as an output.",
                    "Move this signal to an output-capable GPIO.",
                ),
            )
        if normalized in adc2_pins and "analogRead" in used_functions.get(normalized, set()) and code_uses_wifi(code):
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    mcu_id,
                    f"{display_pin(normalized)} uses ESP32 ADC2 while WiFi is enabled.",
                    "Use an ADC1 pin such as GPIO32-GPIO39 for analog readings when WiFi is active.",
                ),
            )

    for component in graph["components"]:
        if not is_servo(component):
            continue
        comp_id = str(component.get("id"))
        sig = find_role_pin(component, ["sig", "signal", "pwm", "s"], "sig")
        sig_net = net_for_pin(graph, comp_id, sig)
        gpios = mcu_gpios_on_net(graph, sig_net)
        if gpios and pwm_pins and gpios[0][1] not in pwm_pins:
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    comp_id,
                    f"{component.get('name', comp_id)} signal is on {display_pin(gpios[0][1])}, which is not listed as a PWM-capable pin for {board_type}.",
                    "Move the servo signal to a PWM-capable pin or verify that the firmware library supports software timing on this board.",
                ),
            )


def analyze_floating_inputs(
    graph: Dict[str, Any],
    board_type: str,
    code: str,
    issues: List[Dict[str, Any]],
) -> None:
    pin_modes = extract_pin_modes(code)
    for component in graph["components"]:
        kind = component_type(component)
        if not ("BUTTON" in kind or "SWITCH" in kind or "PUSH" in kind):
            continue
        comp_id = str(component.get("id"))
        for pin in pins_for(component):
            pin_id = str(pin.get("id"))
            net_id = net_for_pin(graph, comp_id, pin_id)
            gpios = mcu_gpios_on_net(graph, net_id)
            if not gpios:
                continue
            mode = pin_modes.get(gpios[0][1], "")
            if mode in {"INPUT_PULLUP", "INPUT_PULLDOWN"} or has_pull_resistor(graph, net_id, board_type):
                continue
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    comp_id,
                    f"{component.get('name', comp_id)} input on {display_pin(gpios[0][1])} may float.",
                    "Use pinMode(..., INPUT_PULLUP) with the switch to GND, or add an external 10 k\u03a9 pull-up/pull-down resistor.",
                ),
            )


def i2c_component_pins(component: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    sda = find_role_pin(component, ["sda"], None)
    scl = find_role_pin(component, ["scl"], None)
    return sda, scl


def parse_i2c_address(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    try:
        return int(text, 16) if text.startswith("0x") else int(text)
    except Exception:
        return None


def analyze_bus_rules(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
) -> None:
    bus_groups: Dict[str, List[Tuple[Dict[str, Any], Optional[int]]]] = {}

    for component in graph["components"]:
        sda, scl = i2c_component_pins(component)
        if sda and scl:
            comp_id = str(component.get("id"))
            sda_net = net_for_pin(graph, comp_id, sda)
            scl_net = net_for_pin(graph, comp_id, scl)
            bus_id = f"{sda_net}:{scl_net}"
            address = parse_i2c_address((component.get("properties") or {}).get("address"))
            bus_groups.setdefault(bus_id, []).append((component, address))
            has_module_pullups = bool((component.get("properties") or {}).get("hasPullups"))
            if not has_module_pullups and (not has_pull_resistor(graph, sda_net, board_type) or not has_pull_resistor(graph, scl_net, board_type)):
                add_unique_issue(
                    issues,
                    make_issue(
                        "WARNING",
                        comp_id,
                        f"I2C bus for {component.get('name', comp_id)} has no visible SDA/SCL pull-up resistors.",
                        "Add pull-ups from SDA and SCL to the logic rail, typically 4.7 k\u03a9 to 10 k\u03a9, unless the module includes them.",
                    ),
                )

        kind = component_type(component)
        if is_dht(component) or "DS18B20" in kind or "ONEWIRE" in kind:
            comp_id = str(component.get("id"))
            data_pin = find_role_pin(component, ["data", "dat", "dq", "out", "sig"], None)
            data_net = net_for_pin(graph, comp_id, data_pin)
            if data_net and not has_pull_resistor(graph, data_net, board_type) and not bool((component.get("properties") or {}).get("hasPullup")):
                add_unique_issue(
                    issues,
                    make_issue(
                        "WARNING",
                        comp_id,
                        f"{component.get('name', comp_id)} data line has no visible pull-up resistor.",
                        "Add a 4.7 k\u03a9 to 10 k\u03a9 pull-up from DATA/DQ to VCC unless the module already includes one.",
                    ),
                )

    for devices in bus_groups.values():
        by_address: Dict[int, List[Dict[str, Any]]] = {}
        for component, address in devices:
            if address is None:
                continue
            by_address.setdefault(address, []).append(component)
        for address, duplicates in by_address.items():
            if len(duplicates) <= 1:
                continue
            names = ", ".join(str(item.get("name", item.get("id"))) for item in duplicates)
            for component in duplicates:
                add_unique_issue(
                    issues,
                    make_issue(
                        "WARNING",
                        str(component.get("id")),
                        f"I2C address 0x{address:02X} is used by multiple devices on the same bus: {names}.",
                        "Change one module address or move it to a separate bus.",
                    ),
                )


def normalized_property_summary(properties: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        lower = safe_lower(key)
        if value is None or lower in {"svgdata"}:
            continue
        if "resistance" in lower or lower in {"ohms", "value"}:
            result[key] = format_ohms(parse_ohms(value, 0.0))
        elif "capacitance" in lower:
            numeric = parse_numeric(value, 0.0)
            result[key] = f"{numeric:.3g} uF" if numeric >= 1 else f"{numeric * 1000:.3g} nF"
        elif "voltage" in lower or lower in {"vf", "vcc", "vin"}:
            result[key] = format_voltage(parse_voltage(value, 0.0))
        elif "current" in lower:
            result[key] = format_ma(parse_current_ma(value, 0.0))
        elif "frequency" in lower:
            result[key] = f"{parse_numeric(value, 0.0):.3g} Hz"
        else:
            result[key] = value
    return result


def component_context_summary(component: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    comp_id = str(component.get("id"))
    connected_pins: Dict[str, str] = {}
    unconnected_pins: List[str] = []
    for pin in pins_for(component):
        pin_id = str(pin.get("id"))
        net_id = net_for_pin(graph, comp_id, pin_id)
        if net_id and len(pin_ids_on_net(graph, net_id)) > 1:
            connected_pins[pin_id] = net_id
        else:
            unconnected_pins.append(pin_id)
    return {
        "id": comp_id,
        "name": component.get("name") or comp_id,
        "type": component_type(component),
        "properties": normalized_property_summary(component.get("properties") or {}),
        "pins": connected_pins,
        "unconnectedPins": unconnected_pins,
    }


def analyze_power_integrity(
    graph: Dict[str, Any],
    board_type: str,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    isolated_components: List[str] = []
    unconnected_pins: List[str] = []

    for net_id, refs in graph["netToPins"].items():
        has_ground = net_has_ground(graph, net_id)
        powered_refs = [
            ref
            for ref in refs
            if (component := component_for_ref(graph, ref)) is not None
            and is_power_pin(component, split_pin_ref(ref)[1])
        ]
        if has_ground and powered_refs:
            first_comp = split_pin_ref(powered_refs[0])[0]
            add_unique_issue(
                issues,
                make_issue(
                    "CRITICAL",
                    first_comp,
                    f"Net {net_id} connects a power rail directly to ground.",
                    "Remove the power-to-ground connection before powering or simulating the circuit.",
                ),
            )

    for component in graph["components"]:
        if is_mcu(component) or component_type(component) == "BREADBOARD":
            continue
        kind = component_type(component)
        uses_module_power_pins = not (
            is_led(component)
            or is_resistor(component)
            or is_motor(component)
            or any(token in kind for token in ("CAPACITOR", "DIODE", "BUTTON", "SWITCH", "POTENTIOMETER", "LDR"))
        )
        comp_id = str(component.get("id"))
        connected_count = 0
        for pin in pins_for(component):
            pin_id = str(pin.get("id"))
            net_id = net_for_pin(graph, comp_id, pin_id)
            is_connected = bool(net_id and len(pin_ids_on_net(graph, net_id)) > 1)
            connected_count += 1 if is_connected else 0
            if not is_connected and not any(token in pin_label(component, pin_id) for token in ("nc", "not connected")):
                unconnected_pins.append(f"{comp_id}/{pin_id}")
        if connected_count == 0:
            isolated_components.append(comp_id)
        has_signal = any(
            is_signal_pin(component, str(pin.get("id")))
            and (net_id := net_for_pin(graph, comp_id, str(pin.get("id"))))
            and len(pin_ids_on_net(graph, net_id)) > 1
            for pin in pins_for(component)
        )
        ground_pin = find_role_pin(component, ["gnd", "ground", "vss", "-"])
        power_pin = find_role_pin(component, ["vcc", "vdd", "vin", "5v", "3v3", "+"])
        if uses_module_power_pins and has_signal and ground_pin and not net_has_ground(graph, net_for_pin(graph, comp_id, ground_pin)):
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    comp_id,
                    f"{component.get('name', comp_id)} has signal wiring but no common ground reference.",
                    "Connect its GND pin to the MCU/common ground.",
                ),
            )
        if uses_module_power_pins and has_signal and power_pin and not net_has_power_source(graph, net_for_pin(graph, comp_id, power_pin), board_type, include_gpio=False):
            add_unique_issue(
                issues,
                make_issue(
                    "WARNING",
                    comp_id,
                    f"{component.get('name', comp_id)} has signal wiring but no powered VCC/VDD rail.",
                    "Connect its power pin to a rail compatible with the component and MCU logic level.",
                ),
            )

    return {
        "isolatedComponents": isolated_components,
        "unconnectedPins": unconnected_pins[:24],
    }


def divider_resistive_elements(graph: Dict[str, Any], net_id: Optional[str]) -> List[Tuple[Dict[str, Any], str, float]]:
    elements: List[Tuple[Dict[str, Any], str, float]] = []
    if not net_id:
        return elements
    for component in graph["components"]:
        kind = component_type(component)
        if not (is_resistor(component) or "LDR" in kind or "PHOTORESISTOR" in kind):
            continue
        other = other_net_for_two_pin_component(graph, component, net_id)
        if not other:
            continue
        if is_resistor(component):
            resistance = resistor_value(component)
        else:
            props = component.get("properties") or {}
            light = max(0.0, min(100.0, parse_numeric(props.get("lightLevel"), 50.0))) / 100.0
            dark = parse_ohms(props.get("resistanceDark"), 1_000_000.0)
            bright = parse_ohms(props.get("resistanceLight"), 10_000.0)
            resistance = dark + (bright - dark) * light
        elements.append((component, other, resistance))
    return elements


def analyze_voltage_dividers(graph: Dict[str, Any], board_type: str) -> List[Dict[str, Any]]:
    calculations: List[Dict[str, Any]] = []
    analog_nets: Set[str] = set()
    mcu = find_mcu(graph["components"], board_type)
    if mcu:
        mcu_id = str(mcu.get("id"))
        for pin in pins_for(mcu):
            pin_id = str(pin.get("id"))
            if normalize_pin(pin_id).startswith("a") or "adc" in pin_label(mcu, pin_id):
                net_id = net_for_pin(graph, mcu_id, pin_id)
                if net_id:
                    analog_nets.add(net_id)

    for net_id in analog_nets:
        source: Optional[Tuple[Dict[str, Any], str, float]] = None
        sink: Optional[Tuple[Dict[str, Any], str, float]] = None
        for element in divider_resistive_elements(graph, net_id):
            _, other_net, _ = element
            if net_has_power_source(graph, other_net, board_type, include_gpio=False):
                source = element
            elif net_has_ground(graph, other_net):
                sink = element
        if not source or not sink:
            continue
        source_component, source_net, r1 = source
        sink_component, _, r2 = sink
        vin = net_voltage(graph, source_net, board_type, include_gpio=False)
        vout = vin * (r2 / (r1 + r2)) if (r1 + r2) > 0 else 0.0
        calculations.append(
            {
                "type": "VOLTAGE_DIVIDER",
                "netId": net_id,
                "sourceComponentId": str(source_component.get("id")),
                "sinkComponentId": str(sink_component.get("id")),
                "inputVoltage": vin,
                "r1Ohms": r1,
                "r2Ohms": r2,
                "outputVoltage": vout,
                "formula": "Vout = Vin * R2 / (R1 + R2)",
            }
        )
    return calculations


def simulation_context_summary(
    simulation_state: Optional[Dict[str, Any]],
    graph: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(simulation_state, dict):
        return {"isSimulating": False}

    summary = {
        "isSimulating": bool(simulation_state.get("isSimulating")),
        "solverConverged": simulation_state.get("solverConverged", True),
        "pinStates": simulation_state.get("pinStates") or {},
        "nodeVoltages": simulation_state.get("nodeVoltages") or {},
        "branchCurrents": simulation_state.get("branchCurrents") or {},
        "componentPower": simulation_state.get("componentPower") or {},
        "serialBuffer": simulation_state.get("serialBuffer") or [],
        "oscilloscope": simulation_state.get("oscilloscope") or {},
    }
    if summary["isSimulating"] and summary["solverConverged"] is False:
        add_unique_issue(
            issues,
            make_issue(
                "WARNING",
                "simulation",
                "The live solver did not converge on the latest simulation step.",
                "Check for shorts, floating nodes, missing grounds, or ideal source conflicts.",
            ),
        )

    component_power = summary["componentPower"]
    if isinstance(component_power, dict):
        for component in graph["components"]:
            comp_id = str(component.get("id"))
            power = parse_power_watts(component_power.get(comp_id), 0.0)
            if is_resistor(component) and power > resistor_power_rating(component):
                add_unique_issue(
                    issues,
                    make_issue(
                        "WARNING",
                        comp_id,
                        f"Live simulation reports {format_watts(power)} in {component.get('name', comp_id)}.",
                        "Use a higher wattage resistor or reduce current.",
                    ),
                )
    return summary


def analyze_active_circuit(
    board_type: str,
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
    code: str = "",
    context: Optional[Dict[str, Any]] = None,
    simulation_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = context or {}
    graph = build_circuit_graph(components, wires, context)
    issues: List[Dict[str, Any]] = []
    additions: List[Dict[str, Any]] = []
    removals: List[Dict[str, Any]] = []
    value_changes: List[Dict[str, Any]] = []
    code_fixes: List[Dict[str, Any]] = []

    connectivity = analyze_power_integrity(graph, board_type, issues)
    led_calculations, gpio_loads = analyze_leds(graph, board_type, issues, additions, value_changes)
    divider_calculations = analyze_voltage_dividers(graph, board_type)
    analyze_inductive_loads(graph, board_type, issues, additions)
    analyze_logic_levels(graph, board_type, issues)
    gpio_budget = analyze_gpio_budget(graph, board_type, gpio_loads, issues)
    analyze_high_power_actuators(graph, board_type, issues)
    analyze_board_pin_rules(graph, board_type, code, issues)
    analyze_floating_inputs(graph, board_type, code, issues)
    analyze_bus_rules(graph, board_type, issues)
    sim_summary = simulation_context_summary(simulation_state or context.get("simulationState"), graph, issues)

    code_issues, detected_code_fixes = analyze_code_against_context(code or "", components, wires)
    for issue in code_issues:
        add_unique_issue(issues, issue)
    for fix in detected_code_fixes:
        add_unique_action(code_fixes, fix)

    critical = sum(1 for issue in issues if issue.get("severity") == "CRITICAL")
    warnings = sum(1 for issue in issues if issue.get("severity") == "WARNING")
    safety_score = max(0, 100 - critical * 25 - warnings * 10)
    return {
        "systemPrompt": VOLTForge_SYSTEM_PROMPT,
        "boardType": board_type,
        "boardProfile": board_rule_profile(board_type),
        "graph": graph,
        "components": [component_context_summary(component, graph) for component in components],
        "netlist": {
            "nets": [
                {"id": net_id, "pins": pins}
                for net_id, pins in sorted(graph["netToPins"].items())
                if len(pins) > 1
            ],
            "pinToNet": graph["pinToNet"],
        },
        "connectivity": connectivity,
        "calculations": led_calculations + divider_calculations,
        "gpioBudget": gpio_budget,
        "simulationState": sim_summary,
        "issues": issues,
        "additions": additions,
        "removals": removals,
        "valueChanges": value_changes,
        "codeFixes": code_fixes,
        "safetyScore": safety_score,
        "criticalCount": critical,
        "warningCount": warnings,
    }


def mcu_power_pin(mcu: Optional[Dict[str, Any]], board_type: str) -> str:
    if mcu:
        if (board_type or "").upper().startswith(("ESP", "RASPBERRY_PI", "STM32", "TEENSY", "SEEED_XIAO", "ADAFRUIT_FEATHER", "SPARKFUN_THING_PLUS")):
            found = find_role_pin(mcu, ["3v3", "3.3v"], None)
            if found:
                return found
        return find_role_pin(mcu, ["5v", "vbus", "vin", "3v3", "3.3v"], "5v") or "5v"
    return "3v3" if (board_type or "").upper().startswith(("ESP", "RASPBERRY_PI", "STM32", "TEENSY")) else "5v"


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

        if is_neopixel(component):
            data_pin = find_role_pin(component, ["din", "data", "sig", "in"], "din")
            add_if_missing(component, data_pin, default_signal_pin(board_type, used, "d6"), "#22c55e", "NeoPixel data input")
        elif is_led(component):
            anode = find_role_pin(component, ["anode", "a", "+", "red", "green", "blue"], "anode")
            cathode = find_role_pin(component, ["cathode", "k", "gnd", "-", "neg"], "cathode")
            add_if_missing(component, anode, default_signal_pin(board_type, used, "d9"), "#3b82f6", "Drive LED through a 220 Ohm series resistor")
            add_if_missing(component, cathode, mcu_gnd, "#555555", "Return LED cathode to GND")
        elif is_dht(component):
            data_pin = find_role_pin(component, ["data", "dat", "out", "sig"], "data")
            add_if_missing(component, data_pin, default_signal_pin(board_type, used, "d2"), "#22c55e", "DHT data signal")
        elif is_ultrasonic(component):
            add_if_missing(component, find_role_pin(component, ["trig"], "trig"), default_signal_pin(board_type, used, "d5"), "#22c55e", "Ultrasonic trigger output")
            add_if_missing(component, find_role_pin(component, ["echo"], "echo"), default_signal_pin(board_type, used, "d6"), "#3b82f6", "Ultrasonic echo input; level-shift if the module is powered from 5V")
        elif is_pir(component):
            out_pin = find_role_pin(component, ["out", "sig", "signal"], "out")
            add_if_missing(component, out_pin, default_signal_pin(board_type, used, "d2"), "#22c55e", "PIR digital motion output")
        elif is_i2c_display(component) or is_imu(component):
            add_if_missing(component, find_role_pin(component, ["sda"], "sda"), sda_pin, "#a855f7", "I2C SDA")
            add_if_missing(component, find_role_pin(component, ["scl"], "scl"), scl_pin, "#a855f7", "I2C SCL")
        elif is_servo(component):
            sig = find_role_pin(component, ["sig", "signal", "pwm", "s"], "sig")
            add_if_missing(component, sig, default_signal_pin(board_type, used, "d9"), "#f59e0b", "Servo PWM signal")
        elif is_relay(component):
            control = find_role_pin(component, ["in1", "in", "coil1", "sig"], "in")
            add_if_missing(component, control, default_signal_pin(board_type, used, "d7"), "#3b82f6", "Relay control signal")
        elif is_stepper(component):
            for role, preferred in (("in1", "d8"), ("in2", "d9"), ("in3", "d10"), ("in4", "d11")):
                add_if_missing(component, find_role_pin(component, [role], role), default_signal_pin(board_type, used, preferred), "#3b82f6", f"Stepper driver {role.upper()} signal")
        elif is_motor(component):
            # Bare motors should not be wired directly to GPIO. The validator explains this.
            continue
        elif is_buzzer(component):
            pos = find_role_pin(component, ["pos", "+", "vcc", "signal"], "pos")
            neg = find_role_pin(component, ["neg", "-", "gnd"], "neg")
            add_if_missing(component, pos, default_signal_pin(board_type, used, "d8"), "#f59e0b", "Buzzer control signal")
            add_if_missing(component, neg, mcu_gnd, "#555555", "Buzzer return to GND")
        elif "POTENTIOMETER" in kind:
            wiper = find_role_pin(component, ["wiper", "sig", "out"], "wiper")
            analog_pin = default_analog_pin(board_type, used, "a0")
            if analog_pin:
                add_if_missing(component, wiper, analog_pin, "#22c55e", "Potentiometer wiper to analog input")
            add_if_missing(component, find_role_pin(component, ["p1", "pin1", "left"], "p1"), mcu_gnd, "#555555", "Potentiometer low side")
            add_if_missing(component, find_role_pin(component, ["p2", "pin2", "right"], "p2"), mcu_power, "#ef4444", "Potentiometer high side")
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
    value_changes: List[Dict[str, Any]] = []
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

    analysis = analyze_active_circuit(
        board_type,
        components,
        wires,
        payload.code or "",
        parse_context(payload.context),
        None,
    )
    for issue in analysis["issues"]:
        add_unique_issue(issues, issue)
    for action in analysis["additions"]:
        add_unique_action(additions, action)
    for action in analysis["removals"]:
        add_unique_action(removals, action)
    for action in analysis["valueChanges"]:
        add_unique_action(value_changes, action)
    for fix in analysis["codeFixes"]:
        add_unique_action(code_fixes, fix)

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
        "valueChanges": value_changes,
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
    context = parse_context(payload.context or payload.canvasContext)
    if isinstance(payload.canvasData, dict):
        for key in ("components", "wires", "netlist"):
            if payload.canvasData.get(key) is not None:
                context[key] = payload.canvasData[key]
        context["canvasData"] = payload.canvasData
    if payload.boardType:
        context["boardType"] = payload.boardType
    if payload.components:
        context["components"] = payload.components
    if payload.wires:
        context["wires"] = payload.wires
    if payload.netlist:
        context["netlist"] = payload.netlist
    if payload.code:
        context["code"] = payload.code
    if payload.simulationState:
        context["simulationState"] = payload.simulationState
    return context


def keyword_matches(lower: str, keywords: Any) -> bool:
    for keyword in keywords:
        text = str(keyword).lower().strip()
        if not text:
            continue
        if re.search(r"[^\w]", text):
            if text in lower:
                return True
            continue
        if re.search(rf"\b{re.escape(text)}s?\b", lower):
            return True
    return False


def meaningful_tokens(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9_+.#-]+", text.lower()))
    return {token for token in tokens if token not in ELECTRONICS_STOPWORDS and len(token) > 1}


def is_domain_question(message: str) -> bool:
    lower = message.lower()
    if keyword_matches(lower, DOMAIN_TERMS):
        return True
    component_terms: Set[str] = set()
    for item in COMPONENT_KNOWLEDGE:
        component_terms.update(str(keyword).lower() for keyword in item.get("keywords", ()))
    if keyword_matches(lower, component_terms):
        return True
    board_terms = set(BOARD_KNOWLEDGE.keys())
    board_terms.update(key.replace("_", " ").lower() for key in BOARD_KNOWLEDGE.keys())
    return keyword_matches(lower, board_terms)


def active_board_type_from_context(context: Dict[str, Any]) -> str:
    board_type = str(context.get("boardType") or "ARDUINO_UNO")
    components = context.get("components") or []
    selected_id = str(context.get("selectedNodeId") or "")
    if selected_id:
        for component in components:
            if str(component.get("id")) == selected_id and is_mcu(component):
                return component_type(component)
    mcu = find_mcu(components, board_type)
    if mcu:
        return component_type(mcu)
    return board_type


def board_info_for_type(board_type: str) -> Dict[str, str]:
    board = (board_type or "ARDUINO_UNO").upper()
    if board in BOARD_KNOWLEDGE:
        return BOARD_KNOWLEDGE[board]
    for prefixes, info in BOARD_FAMILY_KNOWLEDGE:
        if board.startswith(prefixes):
            return info
    return {
        "name": board.replace("_", " ").title(),
        "processor": "board-specific MCU or SoC",
        "logic": f"{board_rule_profile(board).get('logicVoltage', 3.3)}V logic profile",
        "pins": "GPIO, power, ground, and serial bus pins depend on the exact board package.",
        "note": "Use the exact board pinout before wiring sensors or high-current loads.",
    }


def pi_family_overview() -> str:
    return "\n".join(
        [
            "Raspberry Pi in VoltForge has two useful meanings:",
            "- Raspberry Pi Pico / Pico W / Pico 2: microcontroller boards for firmware-style projects. They use RP2040/RP2350 class processors and 3.3V GPIO.",
            "- Raspberry Pi 3/4/5/Zero/Compute Module: Linux single-board computers with a 40-pin header. Their GPIO is 3.3V-only and they do not have built-in analog inputs.",
            "For sensors: use GPIO2/GPIO3 for I2C, GPIO10/GPIO9/GPIO11 for SPI, GPIO14/GPIO15 for UART, and add level shifting for any 5V signal.",
        ]
    )


def board_answer(message: str, context: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    board_type = active_board_type_from_context(context)
    mentions_pi = bool(re.search(r"\bpi\b", lower) or "raspberry" in lower)
    asks_board = keyword_matches(
        lower,
        (
            "board",
            "processor",
            "cpu",
            "mcu",
            "chip",
            "soc",
            "pinout",
            "gpio",
            "raspberry",
            "pi",
            "pico",
            "esp32",
            "arduino",
            "stm32",
            "teensy",
        ),
    )
    if not asks_board:
        return None

    if mentions_pi and "pico" not in lower and not board_type.upper().startswith("RASPBERRY_PI"):
        return pi_family_overview()

    if "pico" in lower and not board_type.upper().startswith("RASPBERRY_PI_PICO"):
        board_type = "RASPBERRY_PI_PICO"
    elif mentions_pi and board_type.upper().startswith("RASPBERRY_PI_PICO"):
        board_type = active_board_type_from_context(context)
    elif mentions_pi and not board_type.upper().startswith("RASPBERRY_PI"):
        board_type = "RASPBERRY_PI_4"

    info = board_info_for_type(board_type)
    sda, scl = expected_i2c_pins(board_type)
    return "\n".join(
        [
            f"{info['name']} uses {info['processor']}.",
            f"- Logic: {info['logic']}.",
            f"- Pins: {info['pins']}.",
            f"- I2C default in VoltForge: SDA {display_pin(sda)}, SCL {display_pin(scl)}.",
            f"- Validation note: {info['note']}",
        ]
    )


def format_component_knowledge(item: Dict[str, Any], board_type: str) -> str:
    sda, scl = expected_i2c_pins(board_type)
    wiring = str(item["wiring"]).format(
        board=board_type,
        sda=display_pin(sda),
        scl=display_pin(scl),
    )
    return "\n".join(
        [
            f"{item['title']}: {item['summary']}",
            f"- Wiring: {wiring}",
            f"- Code: {item['code']}",
            f"- Validation: {item['checks']}",
        ]
    )


def component_knowledge_for_text(text: str) -> Optional[Dict[str, Any]]:
    lower = text.lower()
    for item in COMPONENT_KNOWLEDGE:
        if keyword_matches(lower, item.get("keywords", ())):
            return item
    return None


def component_knowledge_for_component(component: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = component_label(component)
    kind = component_type(component).lower()
    return component_knowledge_for_text(f"{label} {kind}")


def answer_supported_components() -> str:
    names = [
        "boards",
        "LED/RGB/NeoPixel",
        "resistor/capacitor/diode/transistor/MOSFET",
        "button/potentiometer/breadboard/buzzer",
        "DHT, ultrasonic, PIR, LDR, MPU6050/IMU",
        "I2C LCD/OLED and 7-segment displays",
        "relay, DC motor, servo, stepper",
        "ammeter, multimeter, oscilloscope, 7805 regulator",
    ]
    return "I have local circuit knowledge for: " + "; ".join(names) + ". Ask for wiring, pinout, code, or safety checks for any of these."


def answer_component_knowledge(message: str, context: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    board_type = active_board_type_from_context(context)

    if keyword_matches(lower, ("what components", "which components", "supported components", "component library", "all components")):
        return answer_supported_components()

    direct_item = component_knowledge_for_text(message)
    asks_processor = keyword_matches(lower, ("processor", "cpu", "mcu", "chip", "soc"))
    if direct_item and not asks_processor:
        return format_component_knowledge(direct_item, board_type)

    components = context.get("components") or []
    selected_id = str(context.get("selectedNodeId") or "")
    if selected_id and keyword_matches(lower, ("this", "selected", "model", "component", "how does this work", "processor")):
        selected = next((component for component in components if str(component.get("id")) == selected_id), None)
        if selected:
            if is_mcu(selected):
                return board_answer(f"processor {component_type(selected)}", {**context, "boardType": component_type(selected)})
            item = component_knowledge_for_component(selected)
            if item:
                return format_component_knowledge(item, board_type)

    board_specific = board_answer(message, context)
    if board_specific:
        return board_specific

    for component in components:
        label_text = f"{component.get('id', '')} {component.get('name', '')} {component.get('type', '')}".lower()
        if keyword_matches(lower, meaningful_tokens(label_text)):
            if is_mcu(component):
                return board_answer(f"processor {component_type(component)}", {**context, "boardType": component_type(component)})
            item = component_knowledge_for_component(component)
            if item:
                return format_component_knowledge(item, board_type)

    if direct_item:
        return format_component_knowledge(direct_item, board_type)

    return None


def is_out_of_domain(message: str) -> bool:
    lower = message.lower()
    if is_domain_question(message):
        return False
    return any(term in lower for term in OUT_OF_DOMAIN_TERMS)


def find_best_match(query: str) -> Optional[Dict[str, str]]:
    query_tokens = meaningful_tokens(query)
    if not query_tokens:
        return None
    best_score = 0.0
    best_match: Optional[Dict[str, str]] = None
    for item in qa_dataset:
        target_tokens = meaningful_tokens(item["question"])
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
        encoded = urllib.parse.quote(f"{query} electronics datasheet pinout wiring documentation")
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
    if result.get("additions") or result.get("removals") or result.get("valueChanges") or result.get("codeFixes"):
        lines.append("")
        lines.append("I also returned structured additions, removals, value changes, and code fixes for the UI to apply.")
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


def issue_summary_lines(issues: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    if not issues:
        return ["No critical electrical issues were detected in the active canvas."]
    return [
        f"- {issue.get('severity', 'INFO')}: {issue.get('message')} Fix: {issue.get('suggestedFix')}"
        for issue in issues[:limit]
    ]


def analysis_validation_markdown(analysis: Dict[str, Any]) -> str:
    lines = [
        f"Safety score: {analysis['safetyScore']}/100.",
        f"Checked {len(analysis['components'])} components, {len(analysis['graph']['wires'])} wires, and {len(analysis['netlist']['nets'])} connected nets for {analysis['boardType']}.",
        "",
        "Highest priority findings:",
    ]
    lines.extend(issue_summary_lines(analysis["issues"], 6))
    if analysis.get("calculations"):
        lines.append("")
        lines.append("Key calculations:")
        for calc in analysis["calculations"][:4]:
            if calc["type"] == "LED_BRANCH":
                lines.append(
                    f"- {calc['componentId']}: I = ({format_voltage(calc['sourceVoltage'])} - {format_voltage(calc['forwardVoltage'])}) / "
                    f"{format_ohms(calc['seriesResistance'])} = {format_ma(calc['currentMa'])}; resistor power {format_watts(calc['resistorPowerW'])}."
                )
            elif calc["type"] == "VOLTAGE_DIVIDER":
                lines.append(
                    f"- {calc['netId']}: Vout = {format_voltage(calc['inputVoltage'])} * "
                    f"{format_ohms(calc['r2Ohms'])} / ({format_ohms(calc['r1Ohms'])} + {format_ohms(calc['r2Ohms'])}) = {format_voltage(calc['outputVoltage'])}."
                )
    if analysis.get("additions") or analysis.get("removals") or analysis.get("valueChanges") or analysis.get("codeFixes"):
        lines.append("")
        lines.append("I also returned structured apply actions for the UI.")
    return "\n".join(lines)


def find_component_mentions(message: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    lower = message.lower()
    matches: List[Dict[str, Any]] = []
    graph = analysis["graph"]
    for component in graph["components"]:
        comp_id = str(component.get("id", ""))
        name = str(component.get("name", ""))
        kind = component_type(component)
        if comp_id and comp_id.lower() in lower:
            matches.append(component)
            continue
        if name and name.lower() in lower:
            matches.append(component)
            continue
        generic_tokens = {
            "LED": is_led(component),
            "MOTOR": is_motor(component),
            "SERVO": is_servo(component),
            "RELAY": is_relay(component),
            "RESISTOR": is_resistor(component),
            "DHT": is_dht(component),
            "LCD": "LCD" in kind,
            "OLED": "OLED" in kind,
            "BUTTON": "BUTTON" in kind or "PUSH" in kind,
            "SWITCH": "SWITCH" in kind,
            "SENSOR": "SENSOR" in kind or any(token in kind for token in ("LDR", "PIR", "ULTRASONIC", "SOIL")),
        }
        if any(token.lower() in lower and predicate for token, predicate in generic_tokens.items()):
            matches.append(component)
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for component in matches:
        comp_id = str(component.get("id"))
        if comp_id not in seen:
            seen.add(comp_id)
            unique.append(component)
    return unique


def extract_pin_queries(message: str) -> List[str]:
    pins: List[str] = []
    for match in re.finditer(r"\b(?:gpio\s*)?([ad]\s*\d+|\d{1,2}|vp|vn|sda|scl|rx|tx)\b", message, re.IGNORECASE):
        token = match.group(1).replace(" ", "")
        if token.isdigit() and not re.search(r"\b(?:pin|d|gpio)\s*" + re.escape(token) + r"\b", message, re.IGNORECASE):
            continue
        pins.append(normalize_pin(token))
    return list(dict.fromkeys(pins))


def describe_net_pin(ref: str, analysis: Dict[str, Any]) -> str:
    comp_id, pin_id = split_pin_ref(ref)
    component = analysis["graph"]["componentsById"].get(comp_id, {})
    name = component.get("name") or comp_id
    return f"{name} ({comp_id}) {display_pin(pin_id)}"


def answer_connection_question(message: str, analysis: Dict[str, Any]) -> Optional[str]:
    if not any(term in message.lower() for term in ("connected", "connection", "wired", "net", "goes to", "on d", "on pin")):
        return None
    pins = extract_pin_queries(message)
    graph = analysis["graph"]
    mcu = find_mcu(graph["components"], analysis["boardType"])
    if pins and mcu:
        mcu_id = str(mcu.get("id"))
        lines: List[str] = []
        for pin in pins:
            net_id = net_for_pin(graph, mcu_id, pin)
            refs = [
                ref
                for ref in pin_ids_on_net(graph, net_id)
                if ref != pin_ref(mcu_id, pin)
            ]
            if refs:
                lines.append(f"{display_pin(pin)} is on net {net_id} with:")
                lines.extend(f"- {describe_net_pin(ref, analysis)}" for ref in refs[:8])
            else:
                lines.append(f"{display_pin(pin)} is not connected to any other pin on the active canvas.")
        return "\n".join(lines)

    mentions = find_component_mentions(message, analysis)
    if mentions:
        component = mentions[0]
        comp_id = str(component.get("id"))
        lines = [f"{component.get('name', comp_id)} ({comp_id}) connections:"]
        for pin in pins_for(component):
            pin_id = str(pin.get("id"))
            net_id = net_for_pin(graph, comp_id, pin_id)
            refs = [
                ref
                for ref in pin_ids_on_net(graph, net_id)
                if ref != pin_ref(comp_id, pin_id)
            ]
            if refs:
                lines.append(f"- {display_pin(pin_id)} on {net_id}: " + ", ".join(describe_net_pin(ref, analysis) for ref in refs[:5]))
            else:
                lines.append(f"- {display_pin(pin_id)} is unconnected.")
        return "\n".join(lines)
    return None


def calculation_for_component(component_id: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [calc for calc in analysis["calculations"] if calc.get("componentId") == component_id or calc.get("sourceComponentId") == component_id or calc.get("sinkComponentId") == component_id]


def answer_measurement_question(message: str, analysis: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    wants_measurement = any(term in lower for term in ("voltage", "volt", "current", "amp", "ma", "power", "watt", "resistor", "ohm", "vout"))
    if not wants_measurement:
        return None

    pins = extract_pin_queries(message)
    graph = analysis["graph"]
    if pins:
        mcu = find_mcu(graph["components"], analysis["boardType"])
        if mcu:
            mcu_id = str(mcu.get("id"))
            lines: List[str] = []
            for pin in pins:
                net_id = net_for_pin(graph, mcu_id, pin)
                live_voltage = (analysis["simulationState"].get("nodeVoltages") or {}).get(net_id)
                expected = net_voltage(graph, net_id, analysis["boardType"])
                if live_voltage is not None:
                    lines.append(f"{display_pin(pin)} / {net_id} live voltage is {format_voltage(parse_voltage(live_voltage))}.")
                else:
                    lines.append(f"{display_pin(pin)} / {net_id} expected rail-level voltage is about {format_voltage(expected)} from the current wiring.")
            return "\n".join(lines)

    mentions = find_component_mentions(message, analysis)
    if mentions:
        component = mentions[0]
        comp_id = str(component.get("id"))
        calculations = calculation_for_component(comp_id, analysis)
        if calculations:
            lines = [f"For {component.get('name', comp_id)} ({comp_id}), I used the active canvas values:"]
            for calc in calculations[:3]:
                if calc["type"] == "LED_BRANCH":
                    lines.append(
                        f"- LED branch: I = ({format_voltage(calc['sourceVoltage'])} - {format_voltage(calc['forwardVoltage'])}) / "
                        f"{format_ohms(calc['seriesResistance'])} = {format_ma(calc['currentMa'])}; resistor power is {format_watts(calc['resistorPowerW'])}."
                    )
                elif calc["type"] == "VOLTAGE_DIVIDER":
                    lines.append(
                        f"- Voltage divider on {calc['netId']}: Vout = {format_voltage(calc['outputVoltage'])} from "
                        f"{format_ohms(calc['r1Ohms'])} and {format_ohms(calc['r2Ohms'])}."
                    )
            related_issues = [issue for issue in analysis["issues"] if issue.get("componentId") == comp_id]
            lines.extend(issue_summary_lines(related_issues, 2))
            return "\n".join(lines)

        if is_resistor(component):
            return f"{component.get('name', comp_id)} ({comp_id}) is set to {format_ohms(resistor_value(component))} with a {format_watts(resistor_power_rating(component))} assumed rating."

    if analysis["calculations"]:
        return analysis_validation_markdown(analysis)
    return None


def answer_debug_question(message: str, analysis: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    if not any(term in lower for term in ("why", "won't", "wont", "doesn't", "does not", "not working", "spin", "nan", "0", "debug", "troubleshoot")):
        return None
    mentions = find_component_mentions(message, analysis)
    mentioned_ids = {str(component.get("id")) for component in mentions}
    relevant = [
        issue for issue in analysis["issues"]
        if not mentioned_ids or issue.get("componentId") in mentioned_ids
    ]
    lines = ["I checked the active canvas, firmware pins, and live simulation state."]
    if relevant:
        lines.append("Most likely cause(s):")
        lines.extend(issue_summary_lines(relevant, 5))
    else:
        lines.append("I do not see a direct safety fault for that component. Check the live readings below and verify the firmware logic.")
    sim = analysis.get("simulationState") or {}
    if sim.get("isSimulating"):
        serial = sim.get("serialBuffer") or []
        if serial:
            lines.append("Latest serial output:")
            lines.extend(f"- {line}" for line in serial[-4:])
        if sim.get("pinStates"):
            lines.append(f"Pin states available: {', '.join(list(sim['pinStates'].keys())[:8])}.")
    return "\n".join(lines)


def answer_with_circuit_context(message: str, analysis: Dict[str, Any]) -> Optional[str]:
    if not analysis["graph"]["components"]:
        return None
    for responder in (answer_connection_question, answer_measurement_question, answer_debug_question):
        answer = responder(message, analysis)
        if answer:
            return answer

    lower = message.lower()
    if is_domain_question(message) or any(word in lower for word in ("this", "my", "circuit", "component")):
        lines = [
            f"I checked the active {analysis['boardType']} canvas: {len(analysis['components'])} components, "
            f"{len(analysis['graph']['wires'])} wires, {len(analysis['netlist']['nets'])} connected nets.",
            f"Safety score: {analysis['safetyScore']}/100.",
        ]
        if analysis["issues"]:
            lines.append("Top findings:")
            lines.extend(issue_summary_lines(analysis["issues"], 4))
        elif analysis["calculations"]:
            lines.append("Key operating points:")
            for calc in analysis["calculations"][:3]:
                if calc["type"] == "LED_BRANCH":
                    lines.append(f"- {calc['componentId']} LED current is {format_ma(calc['currentMa'])}.")
                elif calc["type"] == "VOLTAGE_DIVIDER":
                    lines.append(f"- {calc['netId']} divider output is {format_voltage(calc['outputVoltage'])}.")
        else:
            lines.append("No obvious power, ground, GPIO budget, level-shift, flyback, bus, or code/canvas pin mismatch issue was detected.")
        return "\n".join(lines)
    return None


def answer_common_question(message: str, context: Dict[str, Any]) -> Optional[str]:
    lower = message.lower()
    board_type = context.get("boardType", "ARDUINO_UNO")
    trained = answer_component_knowledge(message, context)
    if trained:
        return trained
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
    simulation_state = context.get("simulationState") or {}
    analysis = analyze_active_circuit(board_type, components, wires, code, context, simulation_state) if components else None

    if any(term in lower for term in ("who are you", "what can you do", "help", "what are you doing")):
        project = context.get("projectName", "this project")
        active_summary = (
            f" I am currently seeing {len(components)} canvas component(s), {len(wires)} wire(s), "
            f"and a {analysis['safetyScore']}/100 safety score." if analysis else ""
        )
        return ChatResponse(
            reply=(
                f"I am VoltForge AI, a project-aware electronics assistant for {project}. "
                f"I can validate wiring, find code/canvas pin mismatches, suggest wires, review Arduino code, "
                f"and generate firmware for the active {board_type} layout.{active_summary}"
            ),
            confidence=0.95,
        )

    if any(term in lower for term in ("validate", "safe", "short", "error", "fix my circuit", "check circuit")):
        result = validate_project(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(
            reply=validation_markdown(result),
            confidence=result["confidence"],
            wireSuggestions=result.get("wireSuggestions", []),
            additions=result.get("additions", []),
            removals=result.get("removals", []),
            valueChanges=result.get("valueChanges", []),
            codeFixes=result.get("codeFixes", []),
        )

    if any(term in lower for term in ("suggest wire", "suggest wiring", "how to wire", "connections", "connect this")):
        suggestions = generate_wiring_suggestions(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(
            reply=suggestions_markdown(suggestions),
            confidence=0.9 if suggestions else 0.7,
            wireSuggestions=suggestions,
            additions=analysis.get("additions", []) if analysis else [],
            valueChanges=analysis.get("valueChanges", []) if analysis else [],
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
        )

    if any(term in lower for term in ("generate code", "write code", "schematic to code", "make firmware")):
        generated = generate_code_from_context(components, wires, board_type, message)
        return ChatResponse(
            reply=f"Here is firmware matched to your current {board_type} canvas:\n\n```cpp\n{generated}```",
            hasCode=True,
            generatedCode=generated,
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
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
        return ChatResponse(
            reply="\n".join(lines),
            confidence=review["confidence"],
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
        )

    if analysis:
        contextual = answer_with_circuit_context(message, analysis)
        if contextual:
            return ChatResponse(
                reply=contextual,
                confidence=0.9,
                wireSuggestions=generate_wiring_suggestions(
                    ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
                ),
                additions=analysis.get("additions", []),
                removals=analysis.get("removals", []),
                valueChanges=analysis.get("valueChanges", []),
                codeFixes=analysis.get("codeFixes", []),
            )

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

    if is_domain_question(message):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SSE Streaming Chat Endpoint — Token-by-token with Thinking phase
# ═══════════════════════════════════════════════════════════════════════════════

def _sse_event(data: Dict[str, Any]) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def build_thinking_steps(
    message: str,
    context: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
) -> List[str]:
    """Produce concise reasoning steps based on actual circuit analysis."""
    steps: List[str] = []
    board_type = context.get("boardType", "ARDUINO_UNO")
    components = context.get("components") or []
    wires = context.get("wires") or []
    code = context.get("code") or context.get("activeCode") or ""

    # Step 1: Canvas awareness
    if components:
        comp_types = [str(c.get("type") or c.get("name", "?")) for c in components[:8]]
        steps.append(f"Inspecting canvas: {len(components)} component(s) [{', '.join(comp_types)}], {len(wires)} wire(s), board={board_type}.")
    else:
        steps.append(f"No canvas components detected. Answering from electronics knowledge base for {board_type}.")

    # Step 2: Safety analysis summary
    if analysis:
        score = analysis.get("safetyScore", 100)
        criticals = analysis.get("criticalCount", 0)
        warnings = analysis.get("warningCount", 0)
        steps.append(f"Safety score: {score}/100 ({criticals} critical, {warnings} warning).")

        # Highlight top issues
        for issue in (analysis.get("issues") or [])[:2]:
            steps.append(f"{issue.get('severity', 'INFO')}: {issue.get('message', '?')}")

        # Electrical calculations
        for calc in (analysis.get("calculations") or [])[:2]:
            if calc.get("current_mA"):
                steps.append(f"Calculated {calc.get('component', '?')}: I={calc['current_mA']:.1f}mA, P={calc.get('power_mW', 0):.1f}mW.")

    # Step 3: Code awareness
    if code:
        pin_usages = extract_pin_usages(code)
        if pin_usages:
            pins_used = list({u["pin"] for u in pin_usages})[:5]
            steps.append(f"Firmware uses pins: {', '.join(display_pin(p) for p in pins_used)}.")

    # Step 4: Message intent
    lower = message.lower()
    if any(t in lower for t in ("validate", "safe", "check", "error")):
        steps.append("Intent: circuit validation and safety check.")
    elif any(t in lower for t in ("generate", "write code", "firmware")):
        steps.append("Intent: code generation from canvas layout.")
    elif any(t in lower for t in ("wire", "connect", "how to")):
        steps.append("Intent: wiring guidance and connection help.")
    elif any(t in lower for t in ("why", "not working", "debug", "problem")):
        steps.append("Intent: troubleshooting and debugging.")
    else:
        steps.append("Intent: general electronics question.")

    return steps


async def stream_chat_sse(
    payload: ChatRequest,
) -> AsyncIterator[str]:
    """Generator that yields SSE events: thought steps, then word-by-word tokens, then done."""
    context = context_from_chat(payload)
    message = payload.message.strip()

    # Build circuit analysis if components exist
    components = context.get("components") or []
    wires = context.get("wires") or []
    code = context.get("code") or context.get("activeCode") or ""
    board_type = context.get("boardType", "ARDUINO_UNO")
    simulation_state = context.get("simulationState") or {}
    analysis = analyze_active_circuit(board_type, components, wires, code, context, simulation_state) if components else None

    # ── Phase 1: Emit thinking steps ──
    thinking_steps = build_thinking_steps(message, context, analysis)
    for step in thinking_steps:
        yield _sse_event({"type": "thought", "content": step})
        await asyncio.sleep(0.06)

    # ── Phase 2: Compute the full response using existing chat logic ──
    chat_response = _compute_chat_response(message, context, analysis)

    # ── Phase 3: Stream the reply word by word ──
    reply_text = chat_response.reply
    # Split on word boundaries, keeping whitespace attached
    words = re.split(r'(\s+)', reply_text)
    for word in words:
        if word:  # Skip empty strings from split
            yield _sse_event({"type": "token", "content": word})
            # Faster for whitespace, slight pause for actual words
            if word.strip():
                await asyncio.sleep(0.03)

    # ── Phase 4: Emit final metadata ──
    done_payload: Dict[str, Any] = {
        "type": "done",
        "confidence": chat_response.confidence,
        "hasCode": chat_response.hasCode,
        "generatedCode": chat_response.generatedCode,
        "citations": chat_response.citations,
        "wireSuggestions": chat_response.wireSuggestions,
        "additions": chat_response.additions,
        "removals": chat_response.removals,
        "valueChanges": chat_response.valueChanges,
        "codeFixes": chat_response.codeFixes,
    }
    yield _sse_event(done_payload)


def _compute_chat_response(
    message: str,
    context: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
) -> ChatResponse:
    """Core chat logic extracted from the synchronous chat() handler for reuse."""
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
        active_summary = (
            f" I am currently seeing {len(components)} canvas component(s), {len(wires)} wire(s), "
            f"and a {analysis['safetyScore']}/100 safety score." if analysis else ""
        )
        return ChatResponse(
            reply=(
                f"I am VoltForge AI, a project-aware electronics assistant for {project}. "
                f"I can validate wiring, find code/canvas pin mismatches, suggest wires, review Arduino code, "
                f"and generate firmware for the active {board_type} layout.{active_summary}"
            ),
            confidence=0.95,
        )

    if any(term in lower for term in ("validate", "safe", "short", "error", "fix my circuit", "check circuit")):
        result = validate_project(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(
            reply=validation_markdown(result),
            confidence=result["confidence"],
            wireSuggestions=result.get("wireSuggestions", []),
            additions=result.get("additions", []),
            removals=result.get("removals", []),
            valueChanges=result.get("valueChanges", []),
            codeFixes=result.get("codeFixes", []),
        )

    if any(term in lower for term in ("suggest wire", "suggest wiring", "how to wire", "connections", "connect this")):
        suggestions = generate_wiring_suggestions(
            ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
        )
        return ChatResponse(
            reply=suggestions_markdown(suggestions),
            confidence=0.9 if suggestions else 0.7,
            wireSuggestions=suggestions,
            additions=analysis.get("additions", []) if analysis else [],
            valueChanges=analysis.get("valueChanges", []) if analysis else [],
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
        )

    if any(term in lower for term in ("generate code", "write code", "schematic to code", "make firmware")):
        generated = generate_code_from_context(components, wires, board_type, message)
        return ChatResponse(
            reply=f"Here is firmware matched to your current {board_type} canvas:\n\n```cpp\n{generated}```",
            hasCode=True,
            generatedCode=generated,
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
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
        return ChatResponse(
            reply="\n".join(lines),
            confidence=review["confidence"],
            codeFixes=analysis.get("codeFixes", []) if analysis else [],
        )

    if analysis:
        contextual = answer_with_circuit_context(message, analysis)
        if contextual:
            return ChatResponse(
                reply=contextual,
                confidence=0.9,
                wireSuggestions=generate_wiring_suggestions(
                    ValidateRequest(boardType=board_type, components=components, wires=wires, code=code)
                ),
                additions=analysis.get("additions", []),
                removals=analysis.get("removals", []),
                valueChanges=analysis.get("valueChanges", []),
                codeFixes=analysis.get("codeFixes", []),
            )

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

    if is_domain_question(message):
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


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    """SSE streaming chat — emits thought steps, then word-by-word tokens, then final metadata."""
    logger.info("Processing streaming chat request: %s", payload.message)
    return StreamingResponse(
        stream_chat_sse(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
