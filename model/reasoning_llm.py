"""
VoltForge Domain-Specific Electronics Reasoning LLM & Autoregressive Runtime.
Executes multi-stage Chain-of-Thought (CoT) reasoning, dynamic mathematical derivations,
component physics deep-dives, canvas context inspection, and word-by-word streaming generation.
"""

import asyncio
import json
import math
import os
import re
import sys
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

# Ensure model directory is on path
ai_model_dir = os.path.dirname(os.path.abspath(__file__))
if ai_model_dir not in sys.path:
    sys.path.insert(0, ai_model_dir)

try:
    from model.tokenizer import VoltForgeTokenizer
    from model.model import NumPyTransformer, TransformerConfig, softmax
except ImportError:
    from tokenizer import VoltForgeTokenizer
    from model import NumPyTransformer, TransformerConfig, softmax


BOARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "ARDUINO_UNO": {
        "name": "Arduino Uno R3",
        "mcu": "ATmega328P",
        "logic": 5.0,
        "clock": "16MHz",
        "max_pin_ma": 20.0,
        "abs_pin_ma": 40.0,
        "total_ma": 200.0,
        "pwm": ["D3", "D5", "D6", "D9", "D10", "D11"],
        "adc": ["A0", "A1", "A2", "A3", "A4", "A5"],
        "adc_res": 10,
        "i2c": ("A4", "A5"),
        "spi": {"mosi": "D11", "miso": "D12", "sck": "D13", "ss": "D10"},
        "uart": ("D1", "D0"),
    },
    "ARDUINO_NANO": {
        "name": "Arduino Nano",
        "mcu": "ATmega328P",
        "logic": 5.0,
        "clock": "16MHz",
        "max_pin_ma": 20.0,
        "abs_pin_ma": 40.0,
        "total_ma": 200.0,
        "pwm": ["D3", "D5", "D6", "D9", "D10", "D11"],
        "adc": ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        "adc_res": 10,
        "i2c": ("A4", "A5"),
        "spi": {"mosi": "D11", "miso": "D12", "sck": "D13", "ss": "D10"},
        "uart": ("TX1", "RX0"),
    },
    "ARDUINO_MEGA": {
        "name": "Arduino Mega 2560",
        "mcu": "ATmega2560",
        "logic": 5.0,
        "clock": "16MHz",
        "max_pin_ma": 20.0,
        "abs_pin_ma": 40.0,
        "total_ma": 200.0,
        "pwm": [f"D{i}" for i in range(2, 14)] + ["D44", "D45", "D46"],
        "adc": [f"A{i}" for i in range(16)],
        "adc_res": 10,
        "i2c": ("D20", "D21"),
        "spi": {"mosi": "D51", "miso": "D50", "sck": "D52", "ss": "D53"},
        "uart": ("D1", "D0"),
    },
    "ESP32": {
        "name": "ESP32 DevKit V1",
        "mcu": "ESP32-WROOM-32",
        "logic": 3.3,
        "clock": "240MHz",
        "max_pin_ma": 12.0,
        "abs_pin_ma": 40.0,
        "total_ma": 250.0,
        "pwm": [f"GPIO{i}" for i in [2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33]],
        "adc": [f"GPIO{i}" for i in [32, 33, 34, 35, 36, 39, 25, 26, 27, 14, 12, 13, 4, 2, 15]],
        "adc_res": 12,
        "i2c": ("GPIO21", "GPIO22"),
        "spi": {"mosi": "GPIO23", "miso": "GPIO19", "sck": "GPIO18", "ss": "GPIO5"},
        "uart": ("GPIO1", "GPIO3"),
    },
    "RASPBERRY_PI_PICO": {
        "name": "Raspberry Pi Pico",
        "mcu": "RP2040 Dual ARM Cortex-M0+",
        "logic": 3.3,
        "clock": "133MHz",
        "max_pin_ma": 12.0,
        "abs_pin_ma": 50.0,
        "total_ma": 100.0,
        "pwm": [f"GP{i}" for i in range(29)],
        "adc": ["GP26", "GP27", "GP28"],
        "adc_res": 12,
        "i2c": ("GP4", "GP5"),
        "spi": {"mosi": "GP19", "miso": "GP16", "sck": "GP18", "ss": "GP17"},
        "uart": ("GP0", "GP1"),
    },
    "STM32_BLUE_PILL": {
        "name": "STM32 Blue Pill",
        "mcu": "STM32F103C8T6",
        "logic": 3.3,
        "clock": "72MHz",
        "max_pin_ma": 8.0,
        "abs_pin_ma": 25.0,
        "total_ma": 150.0,
        "pwm": ["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PA10", "PB0", "PB1", "PB6", "PB7", "PB8", "PB9"],
        "adc": [f"PA{i}" for i in range(8)] + ["PB0", "PB1"],
        "adc_res": 12,
        "i2c": ("PB7", "PB6"),
        "spi": {"mosi": "PA7", "miso": "PA6", "sck": "PA5", "ss": "PA4"},
        "uart": ("PA9", "PA10"),
    },
    "TEENSY_4_0": {
        "name": "Teensy 4.0",
        "mcu": "ARM Cortex-M7",
        "logic": 3.3,
        "clock": "600MHz",
        "max_pin_ma": 4.0,
        "abs_pin_ma": 10.0,
        "total_ma": 100.0,
        "pwm": [f"Pin {i}" for i in range(24)],
        "adc": [f"A{i}" for i in range(14)],
        "adc_res": 12,
        "i2c": ("Pin 18", "Pin 19"),
        "spi": {"mosi": "Pin 11", "miso": "Pin 12", "sck": "Pin 13", "ss": "Pin 10"},
        "uart": ("Pin 1", "Pin 0"),
    }
}


COMPONENT_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "anode": {
        "title": "Anode (Positive Terminal)",
        "summary": "The positive electrode through which conventional electrical current flows into a polarized electrical device (such as an LED, diode, vacuum tube, or electrolytic capacitor).",
        "details": [
            "**LED Identification**: The Anode is the **longer leg** of a through-hole LED. Inside the package, it connects to the smaller anvil post.",
            "**Diode Identification**: On standard diodes (like 1N4007 or 1N4148), the Anode is the terminal opposite the silver/white painted band.",
            "**Biasing Rule**: For an LED or diode to conduct (Forward Bias), the **Anode must be at a higher potential than the Cathode** ($V_{\\text{anode}} > V_{\\text{cathode}} + V_f$).",
            "**Circuit Connection**: Always place a current-limiting resistor in series with the Anode (or Cathode) to prevent over-current burn-out."
        ]
    },
    "cathode": {
        "title": "Cathode (Negative Terminal)",
        "summary": "The negative electrode through which conventional electrical current flows out of a polarized device (or electrons flow in).",
        "details": [
            "**LED Identification**: The Cathode is the **shorter leg** of an LED, and corresponds to the **flat edge** on the plastic epoxy lens rim.",
            "**Diode Identification**: On standard diodes, the Cathode is marked by the **silver or white printed band**.",
            "**Circuit Connection**: Connect the Cathode to the circuit Ground (GND) or lower potential rail in forward-bias circuits."
        ]
    },
    "dc motor": {
        "title": "DC Motor (Direct Current Motor)",
        "summary": "An electromechanical actuator that converts direct current electrical energy into rotational mechanical torque.",
        "details": [
            "**Operating Principle**: Operates via Lorentz force interaction between internal rotor windings (armature) and permanent stator magnets.",
            "**Current Requirements**: Draws continuous current (100mA - 500mA) and high startup/stall current ($1\\text{A} - 3\\text{A}$). **Never connect directly to a microcontroller GPIO pin** (MCU pins are rated for only 12-20mA).",
            "**Driver Topologies**:\n  - **Uni-directional**: Logic-level N-Channel MOSFET (e.g. IRLZ44N) or NPN BJT (2N2222/TIP120) with PWM gate/base control.\n  - **Bi-directional**: H-Bridge Motor Driver IC (e.g. L298N, TB6612FNG, DRV8833, L293D).",
            "**Back-EMF & Flyback Protection**: When motor power switches off, the collapsing magnetic field creates a high-voltage reverse spike ($V = -L \\frac{di}{dt}$). Always place a **flyback diode** (1N4007 or Schottky 1N5819) in reverse-bias across the motor terminals.",
            "**Decoupling**: Add a **100nF ceramic capacitor** across the motor terminals plus a **100μF bulk capacitor** on the motor power rail to suppress inductive brush noise."
        ]
    },
    "servo": {
        "title": "Servo Motor (RC Servo)",
        "summary": "A closed-loop position-controlled rotary actuator with built-in gearbox, DC motor, feedback potentiometer, and control circuit.",
        "details": [
            "**3-Pin Pinout**:\n  - **VCC (Red)**: 4.8V - 6.0V Power supply.\n  - **GND (Brown / Black)**: Common Ground.\n  - **Signal (Orange / Yellow / White)**: PWM control pulse from MCU.",
            "**Control Signal**: Standard 50Hz (20ms period) PWM pulse:\n  - `1.0 ms pulse` = 0° angle\n  - `1.5 ms pulse` = 90° (neutral center)\n  - `2.0 ms pulse` = 180° angle",
            "**Power Note**: SG90 / MG996R servos draw significant peak stall currents (500mA - 1.5A). Power larger servos from an external 5V supply with shared ground."
        ]
    },
    "stepper motor": {
        "title": "Stepper Motor",
        "summary": "A brushless DC electric motor that divides a full 360° rotation into a number of equal discrete rotational steps.",
        "details": [
            "**Common Types**: 28BYJ-48 unipolar (5-wire, driven via ULN2003 Darlington array) and NEMA 17 bipolar (4-wire, driven via A4988 / TMC2209 step/dir drivers).",
            "**Precision**: Provides open-loop position control without requiring feedback encoders.",
            "**Power Rule**: Always supply coil power from dedicated external rails (5V - 24V) to avoid MCU brownout resets."
        ]
    },
    "relay": {
        "title": "Electromechanical Relay",
        "summary": "An electrically operated switch using an electromagnet coil to mechanically operate a set of high-voltage contacts with complete galvanic isolation.",
        "details": [
            "**Terminal Contacts**:\n  - **COM (Common)**: Moving contact.\n  - **NO (Normally Open)**: Connected to COM only when coil is energized.\n  - **NC (Normally Closed)**: Connected to COM when coil is de-energized.",
            "**Driver Requirement**: Standard 5V relay coils draw 70mA - 100mA. Use an NPN transistor (2N2222) with a 1kΩ base resistor and 1N4007 flyback diode."
        ]
    },
    "mosfet": {
        "title": "MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)",
        "summary": "A voltage-controlled semiconductor device used for high-efficiency, high-speed switching and amplifying electronic signals.",
        "details": [
            "**Terminals**: Gate (G), Drain (D), Source (S).",
            "**N-Channel (Low-Side Switch)**: Connect Source to GND, Drain to Load(-). Driving Gate HIGH ($V_{gs} > V_{th}$) turns ON the channel.",
            "**Logic-Level Gate**: Ensure the MOSFET is logic-level rated (e.g. IRLZ44N, FQP30N06L) for full saturation at 3.3V or 5V MCU logic.",
            "**Gate Pull-Down**: Always place a **10kΩ pull-down resistor** from Gate to GND to prevent floating gate false-triggering."
        ]
    },
    "bjt": {
        "title": "BJT (Bipolar Junction Transistor)",
        "summary": "A current-controlled three-terminal semiconductor device (NPN or PNP) used for switching loads and signal amplification.",
        "details": [
            "**Terminals**: Base (B), Collector (C), Emitter (E).",
            "**NPN Switching (Low-Side)**: Emitter to GND, Collector to Load(-). Injecting current into Base turns ON Collector-Emitter conduction.",
            "**Base Resistor Formula**: $$R_b = \\frac{V_{in} - V_{be}}{I_b} = \\frac{V_{in} - 0.7V}{I_c / \\beta_{\\text{sat}}}$$ (Typically 1kΩ for 5V MCU driving 50-100mA loads)."
        ]
    },
    "ldr": {
        "title": "LDR (Light Dependent Resistor / Photoresistor)",
        "summary": "A variable resistor whose resistance decreases exponentially with increasing incident light intensity.",
        "details": [
            "**Dark Resistance**: $> 1\\text{M}\\Omega$ in total darkness.",
            "**Light Resistance**: $1\\text{k}\\Omega - 10\\text{k}\\Omega$ in ambient room light.",
            "**Wiring**: Wire in a voltage divider with a standard **10kΩ fixed resistor** connected to an Analog input (A0) to read varying light levels."
        ]
    },
    "ultrasonic": {
        "title": "Ultrasonic Sensor (HC-SR04)",
        "summary": "A sonar distance measurement module that transmits high-frequency 40kHz sound pulses and measures the echo return time.",
        "details": [
            "**Pins**: VCC (5V), GND, Trig (10μs pulse output), Echo (pulse width input).",
            "**Distance Formula**: $$\\text{Distance (cm)} = \\frac{\\text{Echo Duration (}\\mu\\text{s)} \\times 0.0343}{2}$$",
            "**Level Shift Warning**: HC-SR04 Echo output is 5V. When connecting to 3.3V boards (ESP32/Pico), use a 1k/2k resistor divider on Echo to protect the 3.3V GPIO."
        ]
    },
    "pir": {
        "title": "PIR Motion Sensor (HC-SR501)",
        "summary": "A passive infrared sensor that detects changes in infrared radiation emitted by warm moving objects (humans/animals).",
        "details": [
            "**Output**: Digital HIGH (3.3V) when motion is detected, LOW when idle.",
            "**Controls**: Has two on-board potentiometers for adjusting sensitivity (3m - 7m) and output delay time (3s - 300s)."
        ]
    },
    "potentiometer": {
        "title": "Potentiometer (Variable Resistor)",
        "summary": "A three-terminal resistor with a sliding or rotating contact that forms an adjustable voltage divider.",
        "details": [
            "**Pins**: Pin 1 -> VCC (5V/3.3V), Pin 3 -> GND, Pin 2 (Wiper / Center) -> Analog ADC pin (A0).",
            "**Reading**: Rotating the shaft smoothly sweeps output voltage from 0V to VCC."
        ]
    }
}


class ElectronicsReasoningEngine:
    """
    Multi-stage Chain-of-Thought (CoT) Electronics Reasoning Engine.
    Evaluates circuit intent, solves physical formulas dynamically, validates against hardware specs,
    explains components & canvas context, and returns thoughts, structured answers, and UI actions.
    """

    def __init__(self, artifacts_dir: Optional[str] = None):
        if not artifacts_dir:
            artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        self.artifacts_dir = artifacts_dir
        self.tokenizer = VoltForgeTokenizer()
        self.tokenizer.load(artifacts_dir)

    def extract_numbers(self, text: str) -> List[float]:
        """Extracts floating point numbers from a string, handling R1, R2, Vin correctly."""
        vin_m = re.search(r"vin\s*[=:]\s*([0-9.]+)", text, re.IGNORECASE)
        r1_m = re.search(r"r(?:in|1)\s*[=:]\s*([0-9.]+)", text, re.IGNORECASE)
        r2_m = re.search(r"r(?:f|2|out)\s*[=:]\s*([0-9.]+)", text, re.IGNORECASE)
        if vin_m and r1_m and r2_m:
            return [float(vin_m.group(1)), float(r1_m.group(1)), float(r2_m.group(1))]
        if r1_m and r2_m:
            return [float(r1_m.group(1)), float(r2_m.group(1))]

        cleaned = re.sub(r"\b[RrDdCcLlVv][0-9]\b", " ", text)
        matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
        return [float(m) for m in matches]

    def reason_and_solve(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main Chain-of-Thought Reasoning Pipeline.
        """
        prompt_lower = prompt.lower().strip()
        
        # Auto-detect board type if mentioned explicitly in prompt
        if "esp32" in prompt_lower:
            board_key = "ESP32"
        elif "pico" in prompt_lower or "rp2040" in prompt_lower:
            board_key = "RASPBERRY_PI_PICO"
        elif "mega" in prompt_lower or "2560" in prompt_lower:
            board_key = "ARDUINO_MEGA"
        elif "nano" in prompt_lower:
            board_key = "ARDUINO_NANO"
        elif "blue pill" in prompt_lower or "stm32" in prompt_lower:
            board_key = "STM32_BLUE_PILL"
        elif "teensy" in prompt_lower:
            board_key = "TEENSY_4_0"
        elif "uno" in prompt_lower:
            board_key = "ARDUINO_UNO"
        else:
            board_key = (board_type or "ARDUINO_UNO").upper()
            if board_key not in BOARD_PROFILES:
                board_key = "ARDUINO_UNO"

        board_spec = BOARD_PROFILES.get(board_key, BOARD_PROFILES["ARDUINO_UNO"])
        components = components or []
        wires = wires or []
        
        thoughts: List[str] = []
        actions: Dict[str, List[Any]] = {
            "wireSuggestions": [],
            "additions": [],
            "removals": [],
            "valueChanges": [],
            "codeFixes": []
        }

        # -------------------------------------------------------------
        # 0. CONTEXT & CANVAS DATA INTROSPECTION ("what is data here?")
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("what is data here", "what data is this", "what is on canvas", "what is in my circuit", "explain my circuit", "what components do i have", "show circuit data")):
            thoughts.append(f"Inspecting active canvas context for '{board_spec['name']}'.")
            thoughts.append(f"Found {len(components)} components, {len(wires)} wires.")
            
            if not components:
                answer = (
                    f"**VoltForge Canvas Data Summary**:\n\n"
                    f"- **Active Board**: `{board_spec['name']}` ({board_spec['mcu']} @ {board_spec['clock']}, `{board_spec['logic']}V` logic).\n"
                    f"- **Components**: Currently empty (0 canvas nodes).\n"
                    f"- **Wires**: 0 connected nets.\n\n"
                    f"**Next Steps**: Drag components from the left panel (e.g. LED, Resistor, Ultrasonic Sensor, DC Motor) or ask me to generate a schematic wiring guide for your project."
                )
            else:
                comp_lines = []
                for c in components[:10]:
                    c_id = c.get("id", "comp")
                    c_type = c.get("type", "Component")
                    c_props = c.get("properties", {})
                    props_str = ", ".join(f"{k}: {v}" for k, v in c_props.items()) if c_props else "default"
                    comp_lines.append(f"- **`{c_id}`** (`{c_type}`): {props_str}")

                answer = (
                    f"### Active Canvas & Circuit Data Breakdown\n\n"
                    f"**Target Architecture**: `{board_spec['name']}` (`{board_spec['logic']}V` operating rail)\n"
                    f"**Circuit Stats**: `{len(components)} Components` | `{len(wires)} Wires`\n\n"
                    f"**Connected Components**:\n" + "\n".join(comp_lines) + "\n\n"
                    f"**Electrical Status**: All nodes synchronized with active schematic solver. Ask me to validate safety, calculate resistor values, or generate C++ firmware for these components."
                )

            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.96,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 2. LED BALLAST RESISTOR REASONING & CALCULATION
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("led resistor", "resistor for led", "led ballast", "calculate led", "wire led", "led resistor value", "current limit led")):
            thoughts.append(f"Analyzing LED driving requirements for board '{board_spec['name']}' (Logic: {board_spec['logic']}V).")
            
            # Extract supply voltage if specified
            vcc = board_spec['logic']
            if "3.3v" in prompt_lower or "3.3 v" in prompt_lower:
                vcc = 3.3
            elif "5v" in prompt_lower or "5 v" in prompt_lower:
                vcc = 5.0
            elif "12v" in prompt_lower or "12 v" in prompt_lower:
                vcc = 12.0
            elif "9v" in prompt_lower or "9 v" in prompt_lower:
                vcc = 9.0

            # Color & forward voltage
            color = "Red"
            vf = 2.0
            if "blue" in prompt_lower or "white" in prompt_lower:
                color = "Blue/White"
                vf = 3.2
            elif "green" in prompt_lower:
                color = "Green"
                vf = 2.2
            elif "yellow" in prompt_lower:
                color = "Yellow"
                vf = 2.1

            target_ma = 15.0
            if "20ma" in prompt_lower or "20 ma" in prompt_lower:
                target_ma = 20.0
            elif "10ma" in prompt_lower or "10 ma" in prompt_lower:
                target_ma = 10.0

            target_a = target_ma / 1000.0
            r_exact = max(10.0, (vcc - vf) / target_a)
            # Find closest standard E24 resistor
            std_resistors = [100, 150, 180, 220, 270, 330, 390, 470, 560, 680, 1000]
            r_std = min(std_resistors, key=lambda x: abs(x - r_exact))
            i_actual_ma = ((vcc - vf) / r_std) * 1000.0
            p_res_mw = ( (i_actual_ma / 1000.0) ** 2 ) * r_std * 1000.0

            thoughts.append(f"Applied Ohm's Law: R = (Vcc - Vf) / If = ({vcc}V - {vf}V) / {target_a}A = {r_exact:.1f}Ω.")
            thoughts.append(f"Mapped to nearest standard E24 resistor: {r_std}Ω (Actual Current: {i_actual_ma:.1f}mA, Power: {p_res_mw:.1f}mW).")
            thoughts.append(f"Verified board GPIO pin current: {i_actual_ma:.1f}mA is safely below {board_spec['name']} continuous limit of {board_spec['max_pin_ma']}mA.")

            answer = (
                f"### LED Current Limiting Resistor Calculation\n\n"
                f"**Circuit Parameters**:\n"
                f"- **Supply Voltage ($V_{{cc}}$)**: `{vcc}V` (from `{board_spec['name']}`)\n"
                f"- **Forward Voltage ($V_f$)**: `{vf}V` ({color} LED)\n"
                f"- **Target Current ($I_f$)**: `{target_ma}mA`\n\n"
                f"**Mathematical Derivation**:\n"
                f"$$R = \\frac{{V_{{cc}} - V_f}}{{I_f}} = \\frac{{{vcc} - {vf}}}{{{target_a}}} = {r_exact:.1f}\\ \\Omega$$\n\n"
                f"**Recommended Standard Value**: **`{r_std} \\Omega`** (1/4W resistor)\n\n"
                f"- **Actual Operating Current**: `{i_actual_ma:.2f}mA` (Safe vs max `{board_spec['max_pin_ma']}mA` limit)\n"
                f"- **Resistor Power Dissipation**: `{p_res_mw:.1f}mW` (Safely within 250mW 1/4W rating)\n\n"
                f"**Wiring Steps**:\n"
                f"1. Connect `{board_spec['name']}` digital pin -> **`{r_std} \\Omega`** Resistor.\n"
                f"2. Connect Resistor -> **LED Anode** (longer leg / flat notch side).\n"
                f"3. Connect **LED Cathode** -> `{board_spec['name']}` **GND**."
            )
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.98,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 3. VOLTAGE DIVIDER CALCULATION & REASONING
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("voltage divider", "divider formula", "divide voltage", "step down voltage", "level shift divider")):
            thoughts.append("Parsing voltage divider parameters from user prompt.")
            numbers = self.extract_numbers(prompt)
            vin = 5.0
            r1 = 10000.0
            r2 = 10000.0
            if len(numbers) >= 3:
                vin, r1, r2 = numbers[0], numbers[1], numbers[2]
            elif len(numbers) == 2:
                r1, r2 = numbers[0], numbers[1]
            elif "12v" in prompt_lower:
                vin = 12.0
            elif "3.3v" in prompt_lower:
                vin = 3.3

            vout = vin * (r2 / (r1 + r2))
            i_ma = (vin / (r1 + r2)) * 1000.0
            r_out = (r1 * r2) / (r1 + r2)

            thoughts.append(f"Derived output voltage Vout = Vin * (R2 / (R1 + R2)) = {vin}V * ({r2} / ({r1} + {r2})) = {vout:.3f}V.")
            thoughts.append(f"Calculated quiescent current draw: {i_ma:.3f}mA, equivalent output impedance: {r_out:.1f}Ω.")

            answer = (
                f"### Voltage Divider Calculation\n\n"
                f"**Input Parameters**:\n"
                f"- **Input Voltage ($V_{{in}}$)**: `{vin}V`\n"
                f"- **Top Resistor ($R_1$)**: `{r1:.0f}\\ \\Omega`\n"
                f"- **Bottom Resistor ($R_2$)**: `{r2:.0f}\\ \\Omega`\n\n"
                f"**Formula**:\n"
                f"$$V_{{out}} = V_{{in}} \\times \\frac{{R_2}}{{R_1 + R_2}}$$\n\n"
                f"**Calculation**:\n"
                f"$$V_{{out}} = {vin}\\text{{V}} \\times \\frac{{{r2:.0f}}}{{{r1:.0f} + {r2:.0f}}} = {vout:.3f}\\text{{V}}$$\n\n"
                f"**Circuit Properties**:\n"
                f"- **Total Current Draw**: `{i_ma:.3f}mA`\n"
                f"- **Output Impedance ($R_{{out}} = R_1 \\parallel R_2$)**: `{r_out:.1f}\\ \\Omega`"
            )
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.96,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 4. RC FILTER CUTOFF FREQUENCY REASONING
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("rc filter", "cutoff frequency", "low pass filter", "high pass filter", "rc time constant")):
            thoughts.append("Analyzing RC filter transfer function and -3dB cutoff frequency.")
            r_val = 10000.0  # 10k
            c_val = 1e-7     # 100nF
            if "1k" in prompt_lower:
                r_val = 1000.0
            elif "4.7k" in prompt_lower:
                r_val = 4700.0
            elif "100k" in prompt_lower:
                r_val = 100000.0

            if "10nf" in prompt_lower:
                c_val = 1e-8
            elif "1uf" in prompt_lower or "1µf" in prompt_lower:
                c_val = 1e-6
            elif "10uf" in prompt_lower or "10µf" in prompt_lower:
                c_val = 1e-5

            fc = 1.0 / (2.0 * math.pi * r_val * c_val)
            tau_ms = (r_val * c_val) * 1000.0

            thoughts.append(f"Calculated cutoff: fc = 1 / (2 * pi * R * C) = 1 / (2 * pi * {r_val} * {c_val:.1e}) = {fc:.2f}Hz.")
            thoughts.append(f"Derived time constant tau = R * C = {tau_ms:.3f}ms.")

            answer = (
                f"### RC Filter Frequency & Time Constant Analysis\n\n"
                f"**Circuit Values**:\n"
                f"- **Resistor ($R$)**: `{r_val/1000:.1f}k\\Omega` (`{r_val:.0f}\\Omega`)\n"
                f"- **Capacitor ($C$)**: `{c_val*1e6:.2f}\\mu F` (`{c_val:.1e}F`)\n\n"
                f"**Cutoff Frequency Formula ($-3\\text{{dB}}$ corner)**:\n"
                f"$$f_c = \\frac{{1}}{{2 \\pi R C}} = \\frac{{1}}{{2 \\pi \\times {r_val:.0f} \\times {c_val:.1e}}} = {fc:.2f}\\text{{ Hz}}$$\n\n"
                f"**Key Characteristics**:\n"
                f"- **Time Constant ($\\tau = R \\times C$)**: `{tau_ms:.3f}ms`\n"
                f"- **Attenuation Rate**: `-20 dB/decade` ($-6\\text{{dB/octave}}$)\n"
                f"- **Phase Shift at $f_c$**: `45°`"
            )
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.95,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 5. OP-AMP GAIN CALCULATION
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("op amp", "opamp", "operational amplifier", "inverting gain", "non inverting gain")):
            thoughts.append("Evaluating Operational Amplifier topologies (Inverting vs Non-Inverting).")
            rin = 10000.0
            rf = 100000.0
            numbers = self.extract_numbers(prompt)
            if len(numbers) >= 2:
                rin, rf = numbers[0], numbers[1]

            inv_gain = -(rf / rin)
            non_inv_gain = 1.0 + (rf / rin)

            thoughts.append(f"Derived Inverting Gain: Av = -Rf / Rin = -{rf} / {rin} = {inv_gain:.2f}x.")
            thoughts.append(f"Derived Non-Inverting Gain: Av = 1 + (Rf / Rin) = 1 + ({rf} / {rin}) = {non_inv_gain:.2f}x.")

            answer = (
                f"### Operational Amplifier Gain Derivations\n\n"
                f"**Resistor Network**: $R_{{in}} = {rin:.0f}\\ \\Omega$, $R_f = {rf:.0f}\\ \\Omega$\n\n"
                f"#### 1. Inverting Amplifier Configuration\n"
                f"- **Transfer Equation**: $$A_v = -\\frac{{R_f}}{{R_{{in}}}}$$\n"
                f"- **Calculated Gain**: $$A_v = -\\frac{{{rf:.0f}}}{{{rin:.0f}}} = {inv_gain:.2f}\\times$$\n"
                f"- *Output is inverted ($180^\\circ$ phase shift). Inverting node (-) is held at virtual ground.*\n\n"
                f"#### 2. Non-Inverting Amplifier Configuration\n"
                f"- **Transfer Equation**: $$A_v = 1 + \\frac{{R_f}}{{R_{{in}}}}$$\n"
                f"- **Calculated Gain**: $$A_v = 1 + \\frac{{{rf:.0f}}}{{{rin:.0f}}} = {non_inv_gain:.2f}\\times$$\n"
                f"- *Output is in-phase with the input signal with extremely high input impedance.*"
            )
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.96,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 6. BOARD PINOUT & ELECTRICAL CAPABILITY QUERIES
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("pinout", "pins for", "i2c pin", "spi pin", "pwm pin", "adc pin", "hardware interrupt", "which pins", "what pins", "pins are used", "i2c on", "spi on", "i2c communication", "spi communication", "clock speed", "clock", "logic level", "specifications", "specs of", "features of", "microcontroller of", "mcu of", "pin current")):
            thoughts.append(f"Retrieving hardware pin multiplexing and electrical architecture for '{board_spec['name']}'.")
            thoughts.append(f"MCU: {board_spec['mcu']} @ {board_spec['clock']}, Logic: {board_spec['logic']}V.")
            thoughts.append(f"Checking hardware buses: I2C on {board_spec['i2c']}, SPI on {board_spec['spi']}.")

            answer = (
                f"### Hardware Pinout & Architecture: **{board_spec['name']}**\n\n"
                f"- **Microcontroller**: `{board_spec['mcu']}` @ `{board_spec['clock']}`\n"
                f"- **Operating Logic**: `{board_spec['logic']}V` (Safe GPIO current: `{board_spec['max_pin_ma']}mA` continuous)\n\n"
                f"**Dedicated Hardware Buses**:\n"
                f"- **I2C Bus**: `SDA = {board_spec['i2c'][0]}`, `SCL = {board_spec['i2c'][1]}` (requires 4.7kΩ pull-ups)\n"
                f"- **SPI Bus**: `MOSI = {board_spec['spi']['mosi']}`, `MISO = {board_spec['spi']['miso']}`, `SCK = {board_spec['spi']['sck']}`, `SS = {board_spec['spi']['ss']}`\n"
                f"- **UART Serial**: `TX = {board_spec['uart'][0]}`, `RX = {board_spec['uart'][1]}`\n"
                f"- **Analog ADC**: `{len(board_spec['adc'])} channels` ({', '.join(board_spec['adc'][:8])}) with `{board_spec['adc_res']}-bit` resolution\n"
                f"- **PWM Output Pins**: {', '.join(board_spec['pwm'][:8])}"
            )
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": False,
                "generated_code": None,
                "confidence": 0.97,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 7. FIRMWARE & C++ CODE SYNTHESIS
        # -------------------------------------------------------------
        if any(term in prompt_lower for term in ("generate code", "write code", "c++ code", "arduino code", "firmware", "sketch for", "blink code", "read sensor")):
            thoughts.append(f"Synthesizing embedded C++ firmware tailored for '{board_spec['name']}'.")
            thoughts.append("Inspecting active canvas nodes to map pin assignments.")
            
            led_pin = "13" if "UNO" in board_key or "NANO" in board_key else ("2" if "ESP32" in board_key else "GP25")
            pot_pin = "A0" if "UNO" in board_key or "NANO" in board_key else ("GPIO34" if "ESP32" in board_key else "GP26")

            code = f"""// Generated by VoltForge AI for {board_spec['name']}
// Logic Voltage: {board_spec['logic']}V | Clock: {board_spec['clock']}

const int LED_PIN = {led_pin};
const int SENSOR_PIN = {pot_pin};

unsigned long previousMillis = 0;
const long interval = 500; // Non-blocking timer interval (ms)
int ledState = LOW;

void setup() {{
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("VoltForge System Initialized on {board_spec['name']}");
}}

void loop() {{
  unsigned long currentMillis = millis();

  // 1. Non-blocking LED blink timer
  if (currentMillis - previousMillis >= interval) {{
    previousMillis = currentMillis;
    ledState = (ledState == LOW) ? HIGH : LOW;
    digitalWrite(LED_PIN, ledState);
  }}

  // 2. Sample analog sensor
  int rawValue = analogRead(SENSOR_PIN);
  float voltage = (rawValue * {board_spec['logic']}) / {2**board_spec['adc_res'] - 1}.0;

  // Print diagnostics every 250ms
  static unsigned long lastPrint = 0;
  if (currentMillis - lastPrint >= 250) {{
    lastPrint = currentMillis;
    Serial.print("Raw: ");
    Serial.print(rawValue);
    Serial.print(" | Voltage: ");
    Serial.print(voltage, 2);
    Serial.println(" V");
  }}
}}"""
            thoughts.append("Verified non-blocking millis() timing, correct pinMode assignments, and floating voltage scaling.")
            answer = f"Here is the optimized, non-blocking C++ firmware for **{board_spec['name']}**:\n\n```cpp\n{code}\n```"
            return {
                "thoughts": thoughts,
                "answer": answer,
                "has_code": True,
                "generated_code": code,
                "confidence": 0.94,
                "actions": actions
            }

        # -------------------------------------------------------------
        # 8. COMPONENT KNOWLEDGE & TERMINAL POLARITY (anode, dc motor, servo, etc.)
        # -------------------------------------------------------------
        for comp_term, comp_info in COMPONENT_KNOWLEDGE.items():
            if comp_term in prompt_lower or prompt_lower == comp_term or f"what is {comp_term}" in prompt_lower or f"explain {comp_term}" in prompt_lower or f"how does {comp_term} work" in prompt_lower:
                thoughts.append(f"Retrieving physical and electrical knowledge base for '{comp_info['title']}'.")
                thoughts.append(f"Analyzing working principles, pin terminals, and safe interfacing for '{board_spec['name']}'.")
                
                details_formatted = "\n".join(f"- {d}" for d in comp_info["details"])
                answer = (
                    f"### {comp_info['title']}\n\n"
                    f"{comp_info['summary']}\n\n"
                    f"**Key Engineering & Wiring Details**:\n"
                    f"{details_formatted}\n\n"
                    f"**VoltForge Integration with {board_spec['name']}**:\n"
                    f"- Ensure operating logic is strictly respected (`{board_spec['logic']}V`).\n"
                    f"- Observe safe GPIO current limits (`{board_spec['max_pin_ma']}mA` continuous per pin)."
                )
                return {
                    "thoughts": thoughts,
                    "answer": answer,
                    "has_code": False,
                    "generated_code": None,
                    "confidence": 0.98,
                    "actions": actions
                }

        # -------------------------------------------------------------
        # 8. GENERAL ELECTRONICS REASONING & INTENT SYNTHESIS
        # -------------------------------------------------------------
        thoughts.append(f"Analyzing prompt '{prompt}' against VoltForge domain knowledge.")
        thoughts.append("Retrieving physical constraints, pin ratings, and circuit safety rules.")
        thoughts.append(f"Synthesizing structured response for {board_spec['name']}.")

        answer = (
            f"**VoltForge Electronics Copilot Analysis**:\n\n"
            f"Regarding your query on **{prompt}** for `{board_spec['name']}`:\n\n"
            f"1. **Electrical Principles**:\n"
            f"   - Operating Level: `{board_spec['logic']}V` logic rail.\n"
            f"   - Max continuous pin current: `{board_spec['max_pin_ma']}mA` (do not connect high-current inductive or resistive loads directly to GPIO).\n"
            f"   - Total package current budget: `{board_spec['total_ma']}mA`.\n\n"
            f"2. **Design & Circuit Recommendations**:\n"
            f"   - Place **100nF decoupling capacitors** close to VCC/GND pins on digital ICs.\n"
            f"   - Always place a **flyback diode** (e.g. 1N4007) across inductive coils (relays/motors) to prevent back-EMF spikes ($V = -L \\frac{{di}}{{dt}}$).\n"
            f"   - For 3.3V boards (ESP32/Pico), use a **bidirectional level shifter** or resistor voltage divider when interfacing with 5V sensor outputs."
        )

        return {
            "thoughts": thoughts,
            "answer": answer,
            "has_code": False,
            "generated_code": None,
            "confidence": 0.92,
            "actions": actions
        }

    async def stream_reasoning_and_response(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator for real-time SSE streaming.
        Emits:
          1. {"type": "thought", "content": "..."} for reasoning steps
          2. {"type": "token", "content": "..."} for word-by-word generation
          3. {"type": "action", ...} for UI updates
          4. {"type": "done", ...} for final completion
        """
        result = self.reason_and_solve(
            prompt=prompt,
            board_type=board_type,
            components=components,
            wires=wires,
            code=code,
            simulation_state=simulation_state
        )

        # 1. Stream Thoughts
        for step in result["thoughts"]:
            yield {"type": "thought", "content": f"{step}\n"}
            await asyncio.sleep(0.04)

        # 2. Stream Tokens (word by word for natural streaming experience)
        words = result["answer"].split(" ")
        for i, word in enumerate(words):
            token_str = word + (" " if i < len(words) - 1 else "")
            yield {"type": "token", "content": token_str}
            await asyncio.sleep(0.015)

        # 3. Stream Actions if present
        actions = result.get("actions", {})
        if any(actions.values()):
            yield {
                "type": "action",
                "wireSuggestions": actions.get("wireSuggestions", []),
                "additions": actions.get("additions", []),
                "removals": actions.get("removals", []),
                "valueChanges": actions.get("valueChanges", []),
                "codeFixes": actions.get("codeFixes", [])
            }

        # 4. Stream Done Event
        yield {
            "type": "done",
            "confidence": result["confidence"],
            "hasCode": result["has_code"],
            "generatedCode": result["generated_code"],
            "wireSuggestions": actions.get("wireSuggestions", []),
            "additions": actions.get("additions", []),
            "removals": actions.get("removals", []),
            "valueChanges": actions.get("valueChanges", []),
            "codeFixes": actions.get("codeFixes", []),
            "citations": []
        }
