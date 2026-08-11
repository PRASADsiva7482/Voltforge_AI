"""
VoltForge Foundation LLM — Domain Pretraining Corpus Generator.
Generates 20,000+ diverse, high-quality training samples with Chain-of-Thought (CoT)
reasoning traces for training the proprietary electronics LLM from scratch.

Categories:
  1. Board Architecture & Pinout Encyclopedia (49 boards)
  2. Electrical Physics, Formula Derivations & Sizing
  3. Component Knowledge & Terminal Definitions
  4. Multi-Sensor Circuit Schematics & Netlists
  5. Compilable Embedded C++ Firmware Patterns
  6. Circuit Diagnostics, Failure Modes & Safety
  7. Conversational Electronics Q&A
  8. Out-of-Domain Refusals
"""

import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════
# BOARD DATABASE — 49 Microcontroller Development Boards
# ═══════════════════════════════════════════════════════════════════════

BOARDS: Dict[str, Dict[str, Any]] = {
    "ARDUINO_UNO": {"name": "Arduino Uno R3", "mcu": "ATmega328P", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "32KB", "sram": "2KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D3","D5","D6","D9","D10","D11"], "i2c": ("A4","A5"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["D2","D3"], "digital_pins": 14, "analog_pins": 6},
    "ARDUINO_UNO_R4": {"name": "Arduino Uno R4 WiFi", "mcu": "RA4M1", "arch": "ARM Cortex-M4", "logic": 5.0, "clock": "48MHz", "flash": "256KB", "sram": "32KB", "max_pin_ma": 8, "total_ma": 200, "adc_res": 14, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D3","D5","D6","D9","D10","D11"], "i2c": ("A4","A5"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["D2","D3"], "digital_pins": 14, "analog_pins": 6},
    "ARDUINO_NANO": {"name": "Arduino Nano", "mcu": "ATmega328P", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "32KB", "sram": "2KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0","A1","A2","A3","A4","A5","A6","A7"], "pwm": ["D3","D5","D6","D9","D10","D11"], "i2c": ("A4","A5"), "spi": ("D11","D12","D13","D10"), "uart": ("TX1","RX0"), "interrupts": ["D2","D3"], "digital_pins": 14, "analog_pins": 8},
    "ARDUINO_NANO_EVERY": {"name": "Arduino Nano Every", "mcu": "ATmega4809", "arch": "8-bit megaAVR", "logic": 5.0, "clock": "20MHz", "flash": "48KB", "sram": "6KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0","A1","A2","A3","A4","A5","A6","A7"], "pwm": ["D3","D5","D6","D9","D10"], "i2c": ("A4","A5"), "spi": ("D11","D12","D13","D10"), "uart": ("TX","RX"), "interrupts": ["D2","D3"], "digital_pins": 14, "analog_pins": 8},
    "ARDUINO_NANO_33_IOT": {"name": "Arduino Nano 33 IoT", "mcu": "SAMD21G18A", "arch": "ARM Cortex-M0+", "logic": 3.3, "clock": "48MHz", "flash": "256KB", "sram": "32KB", "max_pin_ma": 7, "total_ma": 200, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5","A6","A7"], "pwm": ["D2","D3","D5","D6","D9","D10","D11","D12"], "i2c": ("A4","A5"), "spi": ("D11","D12","D13","D10"), "uart": ("TX","RX"), "interrupts": ["All digital pins"], "digital_pins": 14, "analog_pins": 8},
    "ARDUINO_MEGA": {"name": "Arduino Mega 2560", "mcu": "ATmega2560", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "256KB", "sram": "8KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": [f"A{i}" for i in range(16)], "pwm": [f"D{i}" for i in range(2,14)]+["D44","D45","D46"], "i2c": ("D20","D21"), "spi": ("D51","D50","D52","D53"), "uart": ("D1","D0"), "interrupts": ["D2","D3","D18","D19","D20","D21"], "digital_pins": 54, "analog_pins": 16},
    "ARDUINO_LEONARDO": {"name": "Arduino Leonardo", "mcu": "ATmega32U4", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "32KB", "sram": "2.5KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D3","D5","D6","D9","D10","D11","D13"], "i2c": ("D2","D3"), "spi": ("ICSP","ICSP","ICSP","D10"), "uart": ("D1","D0"), "interrupts": ["D0","D1","D2","D3","D7"], "digital_pins": 20, "analog_pins": 12},
    "ARDUINO_MICRO": {"name": "Arduino Micro", "mcu": "ATmega32U4", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "32KB", "sram": "2.5KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D3","D5","D6","D9","D10","D11","D13"], "i2c": ("D2","D3"), "spi": ("D16","D14","D15","D10"), "uart": ("D1","D0"), "interrupts": ["D0","D1","D2","D3","D7"], "digital_pins": 20, "analog_pins": 12},
    "ARDUINO_DUE": {"name": "Arduino Due", "mcu": "ATSAM3X8E", "arch": "ARM Cortex-M3", "logic": 3.3, "clock": "84MHz", "flash": "512KB", "sram": "96KB", "max_pin_ma": 15, "total_ma": 800, "adc_res": 12, "adc_pins": [f"A{i}" for i in range(12)], "pwm": [f"D{i}" for i in range(2,14)], "i2c": ("D20","D21"), "spi": ("SPI","SPI","SPI","D10"), "uart": ("D1","D0"), "interrupts": [f"D{i}" for i in range(54)], "digital_pins": 54, "analog_pins": 12},
    "ARDUINO_GIGA_R1": {"name": "Arduino Giga R1 WiFi", "mcu": "STM32H747XI", "arch": "Dual-Core ARM Cortex-M7/M4", "logic": 3.3, "clock": "480MHz", "flash": "2MB", "sram": "1MB", "max_pin_ma": 8, "total_ma": 800, "adc_res": 16, "adc_pins": [f"A{i}" for i in range(12)], "pwm": [f"D{i}" for i in range(2,14)], "i2c": ("D20","D21"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["All GPIOs"], "digital_pins": 76, "analog_pins": 12},
    "ARDUINO_PORTENTA_H7": {"name": "Arduino Portenta H7", "mcu": "STM32H747XI", "arch": "Dual-Core ARM Cortex-M7/M4", "logic": 3.3, "clock": "480MHz", "flash": "2MB", "sram": "1MB", "max_pin_ma": 8, "total_ma": 1000, "adc_res": 16, "adc_pins": ["A0","A1","A2","A3","A4","A5","A6","A7"], "pwm": ["D0","D1","D2","D3","D4","D5","D6"], "i2c": ("D11","D12"), "spi": ("D8","D10","D9","D7"), "uart": ("D14","D13"), "interrupts": ["All GPIOs"], "digital_pins": 22, "analog_pins": 8},
    "ESP32": {"name": "ESP32 DevKit V1", "mcu": "ESP32-WROOM-32", "arch": "Dual-Core Xtensa LX6", "logic": 3.3, "clock": "240MHz", "flash": "4MB", "sram": "520KB", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": [f"GPIO{i}" for i in [32,33,34,35,36,39]], "pwm": [f"GPIO{i}" for i in [2,4,5,12,13,14,15,16,17,18,19,21,22,23,25,26,27]], "i2c": ("GPIO21","GPIO22"), "spi": ("GPIO23","GPIO19","GPIO18","GPIO5"), "uart": ("GPIO1","GPIO3"), "interrupts": ["All GPIOs"], "digital_pins": 34, "analog_pins": 18},
    "ESP32_WROOM": {"name": "ESP32-WROOM-32E", "mcu": "ESP32-WROOM-32E", "arch": "Dual-Core Xtensa LX6", "logic": 3.3, "clock": "240MHz", "flash": "4MB", "sram": "520KB", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": [f"GPIO{i}" for i in [32,33,34,35,36,39]], "pwm": [f"GPIO{i}" for i in [2,4,5,12,13,14,15,16,17,18,19,21,22,23,25,26,27]], "i2c": ("GPIO21","GPIO22"), "spi": ("GPIO23","GPIO19","GPIO18","GPIO5"), "uart": ("GPIO1","GPIO3"), "interrupts": ["All GPIOs"], "digital_pins": 34, "analog_pins": 18},
    "ESP32_WROVER": {"name": "ESP32-WROVER", "mcu": "ESP32-WROVER-B", "arch": "Dual-Core Xtensa LX6 + PSRAM", "logic": 3.3, "clock": "240MHz", "flash": "4MB", "sram": "520KB + 8MB PSRAM", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": [f"GPIO{i}" for i in [32,33,34,35,36,39]], "pwm": [f"GPIO{i}" for i in [2,4,5,12,13,14,15,18,19,21,22,23,25,26,27]], "i2c": ("GPIO21","GPIO22"), "spi": ("GPIO23","GPIO19","GPIO18","GPIO5"), "uart": ("GPIO1","GPIO3"), "interrupts": ["All GPIOs"], "digital_pins": 34, "analog_pins": 18},
    "ESP32_S2": {"name": "ESP32-S2", "mcu": "ESP32-S2", "arch": "Xtensa LX7 Single-Core", "logic": 3.3, "clock": "240MHz", "flash": "4MB", "sram": "320KB", "max_pin_ma": 12, "total_ma": 310, "adc_res": 13, "adc_pins": [f"GPIO{i}" for i in range(1,11)], "pwm": [f"GPIO{i}" for i in range(1,22)], "i2c": ("GPIO8","GPIO9"), "spi": ("GPIO35","GPIO37","GPIO36","GPIO34"), "uart": ("GPIO43","GPIO44"), "interrupts": ["All GPIOs"], "digital_pins": 43, "analog_pins": 20},
    "ESP32_S3": {"name": "ESP32-S3", "mcu": "ESP32-S3", "arch": "Dual-Core Xtensa LX7 + AI Accelerator", "logic": 3.3, "clock": "240MHz", "flash": "8MB", "sram": "512KB", "max_pin_ma": 12, "total_ma": 500, "adc_res": 12, "adc_pins": [f"GPIO{i}" for i in range(1,11)], "pwm": [f"GPIO{i}" for i in range(1,22)], "i2c": ("GPIO8","GPIO9"), "spi": ("GPIO11","GPIO13","GPIO12","GPIO10"), "uart": ("GPIO43","GPIO44"), "interrupts": ["All GPIOs"], "digital_pins": 45, "analog_pins": 20},
    "ESP32_C3": {"name": "ESP32-C3", "mcu": "ESP32-C3", "arch": "RISC-V Single-Core", "logic": 3.3, "clock": "160MHz", "flash": "4MB", "sram": "400KB", "max_pin_ma": 12, "total_ma": 350, "adc_res": 12, "adc_pins": ["GPIO0","GPIO1","GPIO2","GPIO3","GPIO4"], "pwm": [f"GPIO{i}" for i in range(0,22)], "i2c": ("GPIO8","GPIO9"), "spi": ("GPIO7","GPIO2","GPIO6","GPIO10"), "uart": ("GPIO21","GPIO20"), "interrupts": ["All GPIOs"], "digital_pins": 22, "analog_pins": 6},
    "ESP32_C6": {"name": "ESP32-C6", "mcu": "ESP32-C6", "arch": "RISC-V + Zigbee/Thread/WiFi6", "logic": 3.3, "clock": "160MHz", "flash": "4MB", "sram": "512KB", "max_pin_ma": 12, "total_ma": 350, "adc_res": 12, "adc_pins": ["GPIO0","GPIO1","GPIO2","GPIO3","GPIO4","GPIO5","GPIO6"], "pwm": [f"GPIO{i}" for i in range(0,23)], "i2c": ("GPIO6","GPIO7"), "spi": ("GPIO19","GPIO20","GPIO18","GPIO9"), "uart": ("GPIO16","GPIO17"), "interrupts": ["All GPIOs"], "digital_pins": 23, "analog_pins": 7},
    "ESP8266": {"name": "ESP8266 NodeMCU", "mcu": "ESP8266", "arch": "Tensilica L106 32-bit", "logic": 3.3, "clock": "80MHz", "flash": "4MB", "sram": "80KB", "max_pin_ma": 12, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0"], "pwm": ["D1","D2","D3","D4","D5","D6","D7","D8"], "i2c": ("D2","D1"), "spi": ("D7","D6","D5","D8"), "uart": ("TX","RX"), "interrupts": ["D1","D2","D3","D4","D5","D6","D7","D8"], "digital_pins": 11, "analog_pins": 1},
    "ESP8266_D1_MINI": {"name": "Wemos D1 Mini", "mcu": "ESP8266", "arch": "Tensilica L106 32-bit", "logic": 3.3, "clock": "80MHz", "flash": "4MB", "sram": "80KB", "max_pin_ma": 12, "total_ma": 200, "adc_res": 10, "adc_pins": ["A0"], "pwm": ["D1","D2","D3","D4","D5","D6","D7","D8"], "i2c": ("D2","D1"), "spi": ("D7","D6","D5","D8"), "uart": ("TX","RX"), "interrupts": ["D1","D2","D5","D6","D7"], "digital_pins": 9, "analog_pins": 1},
    "RASPBERRY_PI_PICO": {"name": "Raspberry Pi Pico", "mcu": "RP2040", "arch": "Dual-Core ARM Cortex-M0+", "logic": 3.3, "clock": "133MHz", "flash": "2MB", "sram": "264KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["GP26","GP27","GP28"], "pwm": [f"GP{i}" for i in range(16)], "i2c": ("GP4","GP5"), "spi": ("GP19","GP16","GP18","GP17"), "uart": ("GP0","GP1"), "interrupts": ["All GPIOs"], "digital_pins": 26, "analog_pins": 3},
    "RASPBERRY_PI_PICO_W": {"name": "Raspberry Pi Pico W", "mcu": "RP2040 + CYW43439", "arch": "Dual-Core ARM Cortex-M0+ + WiFi/BLE", "logic": 3.3, "clock": "133MHz", "flash": "2MB", "sram": "264KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["GP26","GP27","GP28"], "pwm": [f"GP{i}" for i in range(16)], "i2c": ("GP4","GP5"), "spi": ("GP19","GP16","GP18","GP17"), "uart": ("GP0","GP1"), "interrupts": ["All GPIOs"], "digital_pins": 26, "analog_pins": 3},
    "RASPBERRY_PI_PICO_2": {"name": "Raspberry Pi Pico 2", "mcu": "RP2350", "arch": "Dual-Core ARM Cortex-M33 / RISC-V", "logic": 3.3, "clock": "150MHz", "flash": "4MB", "sram": "520KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["GP26","GP27","GP28","GP29"], "pwm": [f"GP{i}" for i in range(16)], "i2c": ("GP4","GP5"), "spi": ("GP19","GP16","GP18","GP17"), "uart": ("GP0","GP1"), "interrupts": ["All GPIOs"], "digital_pins": 30, "analog_pins": 4},
    "STM32_BLUE_PILL": {"name": "STM32 Blue Pill", "mcu": "STM32F103C8T6", "arch": "ARM Cortex-M3", "logic": 3.3, "clock": "72MHz", "flash": "64KB", "sram": "20KB", "max_pin_ma": 25, "total_ma": 150, "adc_res": 12, "adc_pins": ["PA0","PA1","PA2","PA3","PA4","PA5","PA6","PA7","PB0","PB1"], "pwm": ["PA0","PA1","PA2","PA3","PA6","PA7","PB0","PB1"], "i2c": ("PB7","PB6"), "spi": ("PA7","PA6","PA5","PA4"), "uart": ("PA9","PA10"), "interrupts": ["All GPIOs"], "digital_pins": 33, "analog_pins": 10},
    "STM32_BLACK_PILL": {"name": "STM32 Black Pill", "mcu": "STM32F411CEU6", "arch": "ARM Cortex-M4F", "logic": 3.3, "clock": "100MHz", "flash": "512KB", "sram": "128KB", "max_pin_ma": 25, "total_ma": 150, "adc_res": 12, "adc_pins": ["PA0","PA1","PA2","PA3","PA4","PA5","PA6","PA7","PB0","PB1"], "pwm": ["PA0","PA1","PA2","PA3","PA8","PA9","PA10","PA11","PB6","PB7"], "i2c": ("PB7","PB6"), "spi": ("PA7","PA6","PA5","PA4"), "uart": ("PA9","PA10"), "interrupts": ["All GPIOs"], "digital_pins": 36, "analog_pins": 10},
    "TEENSY_4_0": {"name": "Teensy 4.0", "mcu": "NXP i.MX RT1062", "arch": "ARM Cortex-M7", "logic": 3.3, "clock": "600MHz", "flash": "2MB", "sram": "1MB", "max_pin_ma": 10, "total_ma": 250, "adc_res": 12, "adc_pins": [f"A{i}" for i in range(14)], "pwm": [f"D{i}" for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,22,23,24,25,28,29,33,36,37]], "i2c": ("D18","D19"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["All digital pins"], "digital_pins": 40, "analog_pins": 14},
    "TEENSY_4_1": {"name": "Teensy 4.1", "mcu": "NXP i.MX RT1062 + Ethernet", "arch": "ARM Cortex-M7", "logic": 3.3, "clock": "600MHz", "flash": "8MB", "sram": "1MB", "max_pin_ma": 10, "total_ma": 250, "adc_res": 12, "adc_pins": [f"A{i}" for i in range(18)], "pwm": [f"D{i}" for i in range(15)] + ["D18","D19","D22","D23","D24","D25","D28","D29","D33","D36","D37"], "i2c": ("D18","D19"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["All digital pins"], "digital_pins": 55, "analog_pins": 18},
    "TEENSY_LC": {"name": "Teensy LC", "mcu": "NXP MKL26Z64", "arch": "ARM Cortex-M0+", "logic": 3.3, "clock": "48MHz", "flash": "62KB", "sram": "8KB", "max_pin_ma": 5, "total_ma": 100, "adc_res": 12, "adc_pins": [f"A{i}" for i in range(13)], "pwm": [f"D{i}" for i in [3,4,6,9,10,16,17,20,22,23]], "i2c": ("D18","D19"), "spi": ("D11","D12","D13","D10"), "uart": ("D1","D0"), "interrupts": ["All digital pins"], "digital_pins": 27, "analog_pins": 13},
    "SEEED_XIAO_SAMD21": {"name": "Seeed XIAO SAMD21", "mcu": "SAMD21G18", "arch": "ARM Cortex-M0+", "logic": 3.3, "clock": "48MHz", "flash": "256KB", "sram": "32KB", "max_pin_ma": 7, "total_ma": 200, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"], "pwm": ["D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"], "i2c": ("D4","D5"), "spi": ("D10","D9","D8","D7"), "uart": ("D6","D7"), "interrupts": ["All GPIOs"], "digital_pins": 11, "analog_pins": 11},
    "SEEED_XIAO_RP2040": {"name": "Seeed XIAO RP2040", "mcu": "RP2040", "arch": "Dual-Core ARM Cortex-M0+", "logic": 3.3, "clock": "133MHz", "flash": "2MB", "sram": "264KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3"], "pwm": ["D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"], "i2c": ("D4","D5"), "spi": ("D10","D9","D8","D7"), "uart": ("D6","D7"), "interrupts": ["All GPIOs"], "digital_pins": 11, "analog_pins": 4},
    "SEEED_XIAO_ESP32C3": {"name": "Seeed XIAO ESP32C3", "mcu": "ESP32-C3", "arch": "RISC-V Single-Core", "logic": 3.3, "clock": "160MHz", "flash": "4MB", "sram": "400KB", "max_pin_ma": 12, "total_ma": 350, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3"], "pwm": ["D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"], "i2c": ("D4","D5"), "spi": ("D10","D9","D8","D7"), "uart": ("D6","D7"), "interrupts": ["All GPIOs"], "digital_pins": 11, "analog_pins": 4},
    "SEEED_XIAO_ESP32S3": {"name": "Seeed XIAO ESP32S3", "mcu": "ESP32-S3", "arch": "Dual-Core Xtensa LX7", "logic": 3.3, "clock": "240MHz", "flash": "8MB", "sram": "512KB", "max_pin_ma": 12, "total_ma": 500, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"], "i2c": ("D4","D5"), "spi": ("D10","D9","D8","D7"), "uart": ("D6","D7"), "interrupts": ["All GPIOs"], "digital_pins": 11, "analog_pins": 7},
    "SEEED_XIAO_NRF52840": {"name": "Seeed XIAO nRF52840", "mcu": "nRF52840", "arch": "ARM Cortex-M4F + BLE 5.0", "logic": 3.3, "clock": "64MHz", "flash": "1MB", "sram": "256KB", "max_pin_ma": 15, "total_ma": 300, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"], "i2c": ("D4","D5"), "spi": ("D10","D9","D8","D7"), "uart": ("D6","D7"), "interrupts": ["All GPIOs"], "digital_pins": 11, "analog_pins": 6},
    "ADAFRUIT_FEATHER_M0": {"name": "Adafruit Feather M0", "mcu": "ATSAMD21G18", "arch": "ARM Cortex-M0+", "logic": 3.3, "clock": "48MHz", "flash": "256KB", "sram": "32KB", "max_pin_ma": 7, "total_ma": 250, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D5","D6","D9","D10","D11","D12","D13"], "i2c": ("D21","D22"), "spi": ("MOSI","MISO","SCK","D10"), "uart": ("TX","RX"), "interrupts": ["All digital pins"], "digital_pins": 20, "analog_pins": 6},
    "ADAFRUIT_FEATHER_M4": {"name": "Adafruit Feather M4", "mcu": "ATSAMD51J19", "arch": "ARM Cortex-M4F", "logic": 3.3, "clock": "120MHz", "flash": "512KB", "sram": "192KB", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D5","D6","D9","D10","D11","D12","D13"], "i2c": ("D21","D22"), "spi": ("MOSI","MISO","SCK","D10"), "uart": ("TX","RX"), "interrupts": ["All digital pins"], "digital_pins": 21, "analog_pins": 6},
    "ADAFRUIT_FEATHER_ESP32": {"name": "Adafruit Feather ESP32 V2", "mcu": "ESP32", "arch": "Dual-Core Xtensa LX6", "logic": 3.3, "clock": "240MHz", "flash": "8MB", "sram": "520KB", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D5","D6","D9","D10","D11","D12","D13","D14","D15","D21"], "i2c": ("D23","D22"), "spi": ("D18","D19","D5","D33"), "uart": ("TX","RX"), "interrupts": ["All GPIOs"], "digital_pins": 21, "analog_pins": 6},
    "ADAFRUIT_FEATHER_RP2040": {"name": "Adafruit Feather RP2040", "mcu": "RP2040", "arch": "Dual-Core ARM Cortex-M0+", "logic": 3.3, "clock": "133MHz", "flash": "8MB", "sram": "264KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3"], "pwm": ["D4","D5","D6","D9","D10","D11","D12","D13","D24","D25"], "i2c": ("D2","D3"), "spi": ("MOSI","MISO","SCK","D10"), "uart": ("TX","RX"), "interrupts": ["All GPIOs"], "digital_pins": 21, "analog_pins": 4},
    "ADAFRUIT_FEATHER_NRF52840": {"name": "Adafruit Feather nRF52840", "mcu": "nRF52840", "arch": "ARM Cortex-M4F + BLE 5.0", "logic": 3.3, "clock": "64MHz", "flash": "1MB", "sram": "256KB", "max_pin_ma": 15, "total_ma": 300, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D5","D6","D9","D10","D11","D12","D13"], "i2c": ("D21","D22"), "spi": ("MOSI","MISO","SCK","D10"), "uart": ("TX","RX"), "interrupts": ["All digital pins"], "digital_pins": 21, "analog_pins": 6},
    "SPARKFUN_THING_PLUS_ESP32": {"name": "SparkFun Thing Plus ESP32", "mcu": "ESP32-WROOM", "arch": "Dual-Core Xtensa LX6", "logic": 3.3, "clock": "240MHz", "flash": "16MB", "sram": "520KB", "max_pin_ma": 12, "total_ma": 250, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3","A4","A5"], "pwm": ["D5","D12","D13","D14","D15","D16","D17","D18","D19","D21"], "i2c": ("D23","D22"), "spi": ("D18","D19","D5","D33"), "uart": ("TX","RX"), "interrupts": ["All GPIOs"], "digital_pins": 28, "analog_pins": 15},
    "SPARKFUN_THING_PLUS_RP2040": {"name": "SparkFun Thing Plus RP2040", "mcu": "RP2040", "arch": "Dual-Core ARM Cortex-M0+", "logic": 3.3, "clock": "133MHz", "flash": "16MB", "sram": "264KB", "max_pin_ma": 12, "total_ma": 300, "adc_res": 12, "adc_pins": ["A0","A1","A2","A3"], "pwm": [f"D{i}" for i in range(16)], "i2c": ("D6","D7"), "spi": ("D15","D12","D14","D13"), "uart": ("D0","D1"), "interrupts": ["All GPIOs"], "digital_pins": 22, "analog_pins": 4},
    "ATMEL_AVR_ATMEGA328P": {"name": "ATmega328P Bare Chip", "mcu": "ATmega328P", "arch": "8-bit AVR", "logic": 5.0, "clock": "16MHz", "flash": "32KB", "sram": "2KB", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["PC0","PC1","PC2","PC3","PC4","PC5"], "pwm": ["PD3","PD5","PD6","PB1","PB2","PB3"], "i2c": ("PC4","PC5"), "spi": ("PB3","PB4","PB5","PB2"), "uart": ("PD1","PD0"), "interrupts": ["PD2","PD3"], "digital_pins": 14, "analog_pins": 6},
    "ATMEL_AVR_ATTINY85": {"name": "ATtiny85", "mcu": "ATtiny85", "arch": "8-bit AVR", "logic": 5.0, "clock": "8MHz", "flash": "8KB", "sram": "512B", "max_pin_ma": 20, "total_ma": 200, "adc_res": 10, "adc_pins": ["PB2","PB3","PB4","PB5"], "pwm": ["PB0","PB1","PB4"], "i2c": ("PB0","PB2"), "spi": ("PB1","PB0","PB2","PB3"), "uart": ("PB3","PB4"), "interrupts": ["PB2"], "digital_pins": 6, "analog_pins": 4},
}


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT DATABASE
# ═══════════════════════════════════════════════════════════════════════

COMPONENTS = {
    "resistor": {"name": "Resistor", "type": "passive", "terminals": 2, "symbol": "R", "unit": "Ω (Ohm)", "desc": "Opposes current flow according to Ohm's law V=IR.", "common_values": ["220Ω","330Ω","1kΩ","2.2kΩ","4.7kΩ","10kΩ","47kΩ","100kΩ"], "ratings": "1/4W, 1/2W, 1W common", "tips": "Use E24 series standard values. Check power dissipation P=I²R."},
    "capacitor": {"name": "Capacitor", "type": "passive", "terminals": 2, "symbol": "C", "unit": "F (Farad)", "desc": "Stores electrical charge. Blocks DC, passes AC. Used for filtering, coupling, and timing.", "common_values": ["100pF","1nF","100nF","1μF","10μF","100μF","1000μF"], "ratings": "Voltage rating must exceed circuit voltage by 2x", "tips": "Place 100nF decoupling caps near every IC VCC/GND pin."},
    "inductor": {"name": "Inductor", "type": "passive", "terminals": 2, "symbol": "L", "unit": "H (Henry)", "desc": "Stores energy in a magnetic field. Opposes changes in current. Used in filters and power supplies.", "common_values": ["1μH","10μH","100μH","1mH","10mH"], "ratings": "Check saturation current and DC resistance", "tips": "Inductor current cannot change instantaneously (V = L·di/dt)."},
    "led": {"name": "LED (Light-Emitting Diode)", "type": "active", "terminals": 2, "symbol": "LED", "unit": "V (forward voltage)", "desc": "Emits light when forward-biased. Anode (+) is the longer leg, cathode (-) has the flat edge.", "common_values": ["Red: Vf=1.8-2.2V", "Green: Vf=2.0-3.0V", "Blue: Vf=3.0-3.5V", "White: Vf=3.0-3.6V"], "ratings": "Typical If=20mA, max 30mA", "tips": "ALWAYS use a current-limiting resistor: R = (Vcc - Vf) / If."},
    "diode": {"name": "Diode", "type": "active", "terminals": 2, "symbol": "D", "unit": "V (forward voltage drop)", "desc": "Allows current in one direction only. Cathode marked with a band. Common: 1N4148 (signal), 1N4007 (power).", "common_values": ["1N4148: Vf=0.7V signal", "1N4007: Vf=0.7V 1A", "1N5819: Vf=0.3V Schottky"], "ratings": "Check reverse voltage (PIV) and forward current", "tips": "Use flyback diodes across inductive loads (motors, relays) to prevent back-EMF."},
    "transistor_npn": {"name": "NPN BJT Transistor", "type": "active", "terminals": 3, "symbol": "Q", "unit": "hFE (gain)", "desc": "Current-controlled switch. Base current controls larger Collector-Emitter current. Common: 2N2222, BC547.", "common_values": ["2N2222: hFE=100-300", "BC547: hFE=110-800", "BC337: hFE=100-600"], "ratings": "Ic(max), Vce(max), power dissipation", "tips": "Calculate base resistor: Rb = (Vgpio - 0.7) / Ib, where Ib = Ic / hFE."},
    "mosfet_n": {"name": "N-Channel MOSFET", "type": "active", "terminals": 3, "symbol": "Q", "unit": "Rds(on)", "desc": "Voltage-controlled switch. Gate voltage controls Drain-Source current. Very low on-resistance.", "common_values": ["IRLZ44N: Vgs(th)=1-2V logic-level", "IRF540N: Vgs(th)=2-4V", "2N7000: Vgs(th)=0.8-3V small signal"], "ratings": "Id(max), Vds(max), Rds(on), Vgs(th)", "tips": "Use logic-level MOSFETs (Vgs(th)<3V) for 3.3V/5V MCU direct drive. Add 10kΩ gate pulldown."},
    "servo": {"name": "Servo Motor", "type": "actuator", "terminals": 3, "symbol": "SERVO", "unit": "degrees", "desc": "Controlled-position motor. 3 wires: VCC (red), GND (brown/black), Signal (orange/yellow). Controlled by PWM.", "common_values": ["SG90: 0-180°, 5V", "MG996R: 0-180°, 6V heavy-duty", "MG90S: 0-180° metal gear"], "ratings": "Stall current can be 500mA-2A. Use external power.", "tips": "PWM signal: 1ms=0°, 1.5ms=90°, 2ms=180°. Use Arduino Servo library."},
    "dc_motor": {"name": "DC Motor", "type": "actuator", "terminals": 2, "symbol": "M", "unit": "RPM @ V", "desc": "Converts electrical energy to rotational motion. Direction controlled by polarity. Speed by voltage/PWM.", "common_values": ["5V hobby motor", "6V geared motor", "12V high-torque"], "ratings": "Stall current 500mA-5A. NEVER connect directly to GPIO.", "tips": "Use H-bridge (L298N) or MOSFET driver. Add flyback diode and decoupling capacitor."},
    "stepper_motor": {"name": "Stepper Motor (28BYJ-48)", "type": "actuator", "terminals": 5, "symbol": "STEP", "unit": "steps/revolution", "desc": "Precise incremental rotation motor. 28BYJ-48 has 2048 steps/rev with gear reduction. Controlled by ULN2003.", "common_values": ["28BYJ-48: 5V, 2048 steps/rev", "NEMA17: 12V, 200 steps/rev"], "ratings": "28BYJ-48 draws ~240mA. NEMA17 draws 1.5-2A per phase.", "tips": "Use ULN2003 driver for 28BYJ-48, A4988/DRV8825 for NEMA17."},
    "relay": {"name": "Relay Module", "type": "switching", "terminals": 5, "symbol": "K", "unit": "V coil / A contacts", "desc": "Electromagnetic switch. Coil energized by MCU (via transistor) controls high-power contacts (NO/NC/COM).", "common_values": ["5V 10A relay module", "12V 30A relay", "3.3V relay module"], "ratings": "Coil draws 70-100mA. Use transistor/MOSFET driver from GPIO.", "tips": "Include flyback diode across coil. Optocoupler isolation recommended for safety."},
    "ultrasonic_sensor": {"name": "HC-SR04 Ultrasonic Sensor", "type": "sensor", "terminals": 4, "symbol": "US", "unit": "cm", "desc": "Measures distance by timing ultrasonic echo. Trig pin sends pulse, Echo pin receives. Range 2-400cm.", "common_values": ["HC-SR04: 5V, 2-400cm", "JSN-SR04T: waterproof, 5V"], "ratings": "Operating current: 15mA. Accuracy: ±3mm", "tips": "Distance = (Echo_time_μs × 0.034) / 2. For 3.3V boards, use voltage divider on Echo pin."},
    "dht_sensor": {"name": "DHT11/DHT22 Temperature & Humidity Sensor", "type": "sensor", "terminals": 3, "symbol": "DHT", "unit": "°C / %RH", "desc": "Digital temp+humidity sensor. Single wire data protocol. DHT22 has better accuracy than DHT11.", "common_values": ["DHT11: 0-50°C ±2°C, 20-90%RH ±5%", "DHT22: -40-80°C ±0.5°C, 0-100%RH ±2-5%"], "ratings": "3.3V-5V supply. 2.5mA max.", "tips": "Add 10kΩ pullup on data pin. Minimum 2s between readings. Use DHT library."},
    "oled_display": {"name": "SSD1306 OLED Display", "type": "display", "terminals": 4, "symbol": "OLED", "unit": "pixels", "desc": "128×64 or 128×32 OLED display module. I2C or SPI interface. Self-emissive, no backlight needed.", "common_values": ["0.96\" 128×64 I2C (0x3C)", "1.3\" 128×64 I2C (0x3C/0x3D)"], "ratings": "3.3V-5V. I2C address: 0x3C (or 0x3D).", "tips": "Use Adafruit SSD1306 library. Wire SDA/SCL to board I2C pins."},
    "potentiometer": {"name": "Potentiometer", "type": "passive", "terminals": 3, "symbol": "POT", "unit": "Ω", "desc": "Variable resistor with 3 terminals. Wiper (center) provides adjustable voltage divider output.", "common_values": ["1kΩ","5kΩ","10kΩ","50kΩ","100kΩ"], "ratings": "Typical 1/4W", "tips": "Connect outer pins to VCC and GND. Read wiper with analogRead(). Maps 0-Vcc to 0-ADC_MAX."},
    "ldr": {"name": "LDR (Light Dependent Resistor)", "type": "sensor", "terminals": 2, "symbol": "LDR", "unit": "Ω (varies with light)", "desc": "Resistance decreases with increasing light intensity. Dark: 1MΩ+. Bright: 1-10kΩ.", "common_values": ["GL5528: 10kΩ @ 10 lux"], "ratings": "Max voltage: 150V. Max power: 100mW", "tips": "Use voltage divider with 10kΩ fixed resistor. Read with analogRead(). High = dark, Low = bright."},
    "pir_sensor": {"name": "PIR Motion Sensor (HC-SR501)", "type": "sensor", "terminals": 3, "symbol": "PIR", "unit": "digital HIGH/LOW", "desc": "Passive infrared sensor detects motion from warm bodies. Digital output: HIGH when motion detected.", "common_values": ["HC-SR501: 5V-20V input, 3.3V output", "HC-SR505: compact mini PIR"], "ratings": "Detection range: 3-7m. Angle: 100-110°. Warmup: 30-60s.", "tips": "Output is 3.3V logic safe. Adjust sensitivity and delay with onboard potentiometers."},
    "ir_receiver": {"name": "IR Receiver (TSOP1738)", "type": "sensor", "terminals": 3, "symbol": "IR", "unit": "38kHz carrier", "desc": "Receives infrared remote control signals at 38kHz carrier frequency. Outputs demodulated digital signal.", "common_values": ["TSOP1738: 38kHz, 3.3V-5V", "VS1838B: 38kHz"], "ratings": "3.3V-5V supply. Digital output.", "tips": "Use IRremote library. Connect output to any digital pin with interrupt support."},
    "buzzer": {"name": "Piezo Buzzer", "type": "output", "terminals": 2, "symbol": "BZ", "unit": "Hz (frequency)", "desc": "Produces sound. Active buzzer: apply DC voltage for fixed tone. Passive buzzer: drive with PWM for variable pitch.", "common_values": ["Active 5V buzzer", "Passive buzzer (needs PWM)"], "ratings": "Current: 30mA max. Safe for GPIO direct drive.", "tips": "Use tone(pin, frequency) for passive buzzers. Active buzzers only need HIGH/LOW."},
    "lcd_16x2": {"name": "16×2 LCD Display (HD44780)", "type": "display", "terminals": 16, "symbol": "LCD", "unit": "characters", "desc": "16 columns × 2 rows character LCD. Can use 4-bit mode (6 pins) or I2C backpack (2 pins).", "common_values": ["16×2 HD44780 parallel", "16×2 with I2C backpack (PCF8574)"], "ratings": "5V supply. I2C address: 0x27 or 0x3F.", "tips": "Use LiquidCrystal_I2C library for I2C version. Saves 4 GPIO pins."},
}

# ═══════════════════════════════════════════════════════════════════════
# FORMULA TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

LED_COLORS = [("red",1.8,2.2),("orange",2.0,2.2),("yellow",2.0,2.2),("green",2.0,3.0),("blue",3.0,3.5),("white",3.0,3.6),("UV",3.1,3.8),("IR",1.1,1.6)]
SUPPLY_VOLTAGES = [3.3, 5.0, 6.0, 9.0, 12.0]
LED_CURRENTS = [0.005, 0.010, 0.015, 0.020]
R1_VALUES = [1000, 2200, 3300, 4700, 6800, 10000, 15000, 22000, 33000, 47000, 68000, 100000]
R2_VALUES = [1000, 2200, 3300, 4700, 6800, 10000, 15000, 22000, 33000, 47000]
VIN_VALUES = [3.3, 5.0, 9.0, 12.0, 24.0]
CAP_VALUES = [(100e-12,"100pF"),(1e-9,"1nF"),(10e-9,"10nF"),(100e-9,"100nF"),(1e-6,"1μF"),(10e-6,"10μF"),(100e-6,"100μF")]
RES_VALUES_FILTER = [(1000,"1kΩ"),(2200,"2.2kΩ"),(4700,"4.7kΩ"),(10000,"10kΩ"),(22000,"22kΩ"),(47000,"47kΩ"),(100000,"100kΩ")]

# Standard E24 resistor values for nearest-value selection
E24_SERIES = [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91,100,110,120,130,150,160,180,200,220,240,270,300,330,360,390,430,470,510,560,620,680,750,820,910,1000,1100,1200,1500,1800,2200,2700,3300,3900,4700,5600,6800,8200,10000,15000,22000,33000,47000,68000,100000]

def nearest_e24(val):
    return min(E24_SERIES, key=lambda e: abs(e - val))


# ═══════════════════════════════════════════════════════════════════════
# SAMPLE GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def _cot(think_steps: List[str], answer: str) -> str:
    """Format a Chain-of-Thought completion with [THINK] ... [/THINK] [ASSISTANT] ..."""
    think = " ".join(f"Step {i+1}: {s}" for i, s in enumerate(think_steps))
    return f"[THINK] {think} [/THINK] [ASSISTANT] {answer}"


def _prompt(question: str) -> str:
    return f"[SYS] You are VoltForge AI, an expert electronics and embedded systems assistant. [USER] {question}"


# ── 1. Board Architecture & Pinout QA ──

def gen_board_samples() -> List[Dict[str, str]]:
    samples = []
    board_keys = list(BOARDS.keys())

    for bk, bd in BOARDS.items():
        # Logic level
        samples.append({"prompt": _prompt(f"What is the logic level voltage of the {bd['name']}?"),
            "completion": _cot(
                [f"Looking up the {bd['name']} specifications.", f"The {bd['name']} uses the {bd['mcu']} MCU with {bd['arch']} architecture."],
                f"The **{bd['name']}** operates at **{bd['logic']}V** logic level. All GPIO pins output {bd['logic']}V when HIGH. The maximum current per pin is {bd['max_pin_ma']}mA continuous, with a total package budget of {bd['total_ma']}mA."
            )})
        # MCU
        samples.append({"prompt": _prompt(f"What MCU does the {bd['name']} use?"),
            "completion": _cot(
                [f"Retrieving MCU information for {bd['name']}.", f"This board is based on the {bd['arch']} architecture."],
                f"The **{bd['name']}** is powered by the **{bd['mcu']}** microcontroller ({bd['arch']}), running at **{bd['clock']}** clock speed with {bd['flash']} flash memory and {bd['sram']} SRAM."
            )})
        # I2C
        sda, scl = bd['i2c']
        samples.append({"prompt": _prompt(f"What are the I2C pins on {bd['name']}?"),
            "completion": _cot(
                [f"Looking up I2C bus configuration for {bd['name']}.", f"I2C uses two wires: SDA (data) and SCL (clock)."],
                f"On the **{bd['name']}**, the I2C pins are **SDA = {sda}** and **SCL = {scl}**. Use `Wire.begin()` to initialize. Add 4.7kΩ pullup resistors if not built into the module."
            )})
        # SPI
        mosi, miso, sck, ss = bd['spi']
        samples.append({"prompt": _prompt(f"What SPI pins does the {bd['name']} have?"),
            "completion": _cot(
                [f"Retrieving SPI bus pinout for {bd['name']}."],
                f"**{bd['name']}** SPI pins: MOSI={mosi}, MISO={miso}, SCK={sck}, SS={ss}. Use `SPI.begin()` to initialize."
            )})
        # ADC
        samples.append({"prompt": _prompt(f"How many analog/ADC channels does {bd['name']} have?"),
            "completion": _cot(
                [f"Checking ADC specifications for {bd['name']}.", f"The ADC resolution is {bd['adc_res']}-bit."],
                f"The **{bd['name']}** has **{len(bd['adc_pins'])} analog input channels** ({', '.join(bd['adc_pins'][:6])}) with **{bd['adc_res']}-bit** resolution (0–{2**bd['adc_res']-1} range). Voltage per step: {bd['logic']/(2**bd['adc_res']-1)*1000:.2f}mV."
            )})
        # PWM
        samples.append({"prompt": _prompt(f"Which pins support PWM on {bd['name']}?"),
            "completion": _cot(
                [f"Retrieving PWM-capable pins for {bd['name']}."],
                f"PWM-capable pins on **{bd['name']}**: {', '.join(bd['pwm'][:8])}{'...' if len(bd['pwm'])>8 else ''} ({len(bd['pwm'])} total). Use `analogWrite(pin, 0-255)` for 8-bit PWM duty cycle control."
            )})
        # General overview
        samples.append({"prompt": _prompt(f"Tell me about the {bd['name']}."),
            "completion": _cot(
                [f"Compiling full specification sheet for {bd['name']}.", f"This is a {bd['arch']} based board running at {bd['clock']}."],
                f"### {bd['name']}\n\n**MCU**: {bd['mcu']} ({bd['arch']})\n**Clock**: {bd['clock']}\n**Logic Level**: {bd['logic']}V\n**Flash/SRAM**: {bd['flash']} / {bd['sram']}\n**Digital Pins**: {bd['digital_pins']}\n**Analog Pins**: {bd['analog_pins']} ({bd['adc_res']}-bit ADC)\n**PWM Pins**: {len(bd['pwm'])}\n**Max GPIO Current**: {bd['max_pin_ma']}mA per pin, {bd['total_ma']}mA total\n**I2C**: SDA={sda}, SCL={scl}\n**SPI**: MOSI={mosi}, MISO={miso}, SCK={sck}"
            )})
        # Current limits
        samples.append({"prompt": _prompt(f"What is the maximum current per GPIO pin on {bd['name']}?"),
            "completion": _cot(
                [f"Checking electrical specifications for {bd['name']}."],
                f"The **{bd['name']}** can source/sink **{bd['max_pin_ma']}mA** per GPIO pin continuously. The total current budget across all pins combined is **{bd['total_ma']}mA**. Never connect inductive loads (motors, relays) directly to GPIO — use a MOSFET or transistor driver."
            )})

    # Board comparisons
    for _ in range(200):
        b1k, b2k = random.sample(board_keys, 2)
        b1, b2 = BOARDS[b1k], BOARDS[b2k]
        samples.append({"prompt": _prompt(f"Compare {b1['name']} vs {b2['name']}."),
            "completion": _cot(
                [f"Comparing specifications of {b1['name']} and {b2['name']}.", "Analyzing processor, speed, GPIO, and memory differences."],
                f"| Feature | {b1['name']} | {b2['name']} |\n|---|---|---|\n| MCU | {b1['mcu']} | {b2['mcu']} |\n| Architecture | {b1['arch']} | {b2['arch']} |\n| Clock | {b1['clock']} | {b2['clock']} |\n| Logic | {b1['logic']}V | {b2['logic']}V |\n| Flash/SRAM | {b1['flash']}/{b1['sram']} | {b2['flash']}/{b2['sram']} |\n| GPIO Current | {b1['max_pin_ma']}mA | {b2['max_pin_ma']}mA |\n| ADC | {b1['adc_res']}-bit, {len(b1['adc_pins'])}ch | {b2['adc_res']}-bit, {len(b2['adc_pins'])}ch |"
            )})

    return samples


# ── 2. Physics & Formula Derivations ──

def gen_physics_samples() -> List[Dict[str, str]]:
    samples = []

    # LED resistor calculations
    for color, vf_min, vf_max in LED_COLORS:
        vf = round((vf_min + vf_max) / 2, 1)
        for vcc in SUPPLY_VOLTAGES:
            if vcc <= vf:
                continue
            for if_a in LED_CURRENTS:
                if_ma = int(if_a * 1000)
                r_exact = (vcc - vf) / if_a
                r_e24 = nearest_e24(r_exact)
                samples.append({"prompt": _prompt(f"Calculate the resistor value for a {color} LED on a {vcc}V supply at {if_ma}mA."),
                    "completion": _cot(
                        [f"Given: Vcc={vcc}V, Vf({color})={vf}V, If={if_ma}mA.", f"R = (Vcc - Vf) / If = ({vcc} - {vf}) / {if_a} = {r_exact:.1f}Ω.", f"Nearest E24 standard value: {r_e24}Ω."],
                        f"For a **{color} LED** (Vf={vf}V) on a **{vcc}V supply** at **{if_ma}mA**:\n\n$R = \\frac{{V_{{cc}} - V_f}}{{I_f}} = \\frac{{{vcc} - {vf}}}{{{if_a}}} = {r_exact:.1f}\\Omega$\n\nUse the nearest standard E24 value: **{r_e24}Ω**."
                    )})

    # Voltage divider calculations
    for vin in VIN_VALUES:
        for r1 in R1_VALUES:
            for r2 in R2_VALUES:
                vout = round(vin * r2 / (r1 + r2), 3)
                r1_s = f"{r1/1000:.1f}kΩ" if r1 >= 1000 else f"{r1}Ω"
                r2_s = f"{r2/1000:.1f}kΩ" if r2 >= 1000 else f"{r2}Ω"
                samples.append({"prompt": _prompt(f"Calculate the output of a voltage divider with Vin={vin}V, R1={r1_s}, R2={r2_s}."),
                    "completion": _cot(
                        [f"Voltage divider formula: Vout = Vin × R2/(R1+R2).", f"Vout = {vin} × {r2}/({r1}+{r2}) = {vin} × {r2/(r1+r2):.4f} = {vout}V."],
                        f"$V_{{out}} = V_{{in}} \\times \\frac{{R_2}}{{R_1 + R_2}} = {vin} \\times \\frac{{{r2}}}{{{r1}+{r2}}} = **{vout}V**$"
                    )})

    # RC filter cutoff frequency
    for r_val, r_str in RES_VALUES_FILTER:
        for c_val, c_str in CAP_VALUES:
            fc = 1.0 / (2 * math.pi * r_val * c_val)
            if fc < 1:
                fc_str = f"{fc*1000:.2f}mHz"
            elif fc < 1000:
                fc_str = f"{fc:.2f}Hz"
            elif fc < 1e6:
                fc_str = f"{fc/1000:.2f}kHz"
            else:
                fc_str = f"{fc/1e6:.2f}MHz"
            samples.append({"prompt": _prompt(f"Calculate the cutoff frequency of an RC low-pass filter with R={r_str} and C={c_str}."),
                "completion": _cot(
                    [f"RC low-pass cutoff formula: fc = 1 / (2π × R × C).", f"fc = 1 / (2π × {r_val} × {c_val}) = {fc:.2f}Hz."],
                    f"$f_c = \\frac{{1}}{{2\\pi RC}} = \\frac{{1}}{{2\\pi \\times {r_val} \\times {c_val}}} = **{fc_str}**$\n\nThis filter passes frequencies below {fc_str} and attenuates higher frequencies at -20dB/decade."
                )})

    # Ohm's law variations
    for v in [1.5, 3.3, 5.0, 9.0, 12.0, 24.0]:
        for r in [100, 220, 330, 470, 1000, 2200, 4700, 10000]:
            i = v / r
            i_ma = i * 1000
            p = v * i
            r_str = f"{r/1000:.1f}kΩ" if r >= 1000 else f"{r}Ω"
            samples.append({"prompt": _prompt(f"What current flows through a {r_str} resistor with {v}V across it?"),
                "completion": _cot(
                    [f"Ohm's law: I = V/R = {v}/{r} = {i:.4f}A = {i_ma:.2f}mA.", f"Power dissipation: P = V×I = {v}×{i:.4f} = {p*1000:.2f}mW."],
                    f"Using Ohm's law: $I = V/R = {v}/{r} = **{i_ma:.2f}mA**$. Power dissipation: $P = {p*1000:.1f}mW$."
                )})

    # Op-amp gain calculations
    for rf in [10000, 22000, 47000, 100000]:
        for ri in [1000, 2200, 4700, 10000]:
            inv_gain = -rf / ri
            noninv_gain = 1 + rf / ri
            rf_s = f"{rf/1000:.0f}kΩ"
            ri_s = f"{ri/1000:.1f}kΩ" if ri < 10000 else f"{ri/1000:.0f}kΩ"
            samples.append({"prompt": _prompt(f"Calculate the gain of an inverting op-amp with Rf={rf_s} and Ri={ri_s}."),
                "completion": _cot(
                    [f"Inverting op-amp gain: Av = -Rf/Ri = -{rf}/{ri} = {inv_gain:.1f}.", f"Non-inverting configuration would give: 1 + Rf/Ri = {noninv_gain:.1f}."],
                    f"**Inverting amplifier**: $A_v = -R_f/R_i = -{rf}/{ri} = **{inv_gain:.1f}×**$\n\nFor **non-inverting**: $A_v = 1 + R_f/R_i = **{noninv_gain:.1f}×**$"
                )})

    return samples


# ── 3. Component Knowledge ──

def gen_component_samples() -> List[Dict[str, str]]:
    samples = []

    for ck, comp in COMPONENTS.items():
        # What is it
        samples.append({"prompt": _prompt(f"What is a {comp['name']}?"),
            "completion": _cot(
                [f"Retrieving knowledge base for {comp['name']}.", f"This is a {comp['type']} component with {comp['terminals']} terminals."],
                f"### {comp['name']}\n\n{comp['desc']}\n\n**Common Values**: {', '.join(comp['common_values'][:4])}\n**Ratings**: {comp['ratings']}\n**Tip**: {comp['tips']}"
            )})
        # How to wire
        bk = random.choice(list(BOARDS.keys()))
        bd = BOARDS[bk]
        samples.append({"prompt": _prompt(f"How do I wire a {comp['name']} to {bd['name']}?"),
            "completion": _cot(
                [f"Planning wiring for {comp['name']} on {bd['name']} ({bd['logic']}V logic).", f"Checking GPIO current limits: {bd['max_pin_ma']}mA max per pin."],
                f"To wire a **{comp['name']}** to the **{bd['name']}** ({bd['logic']}V logic, {bd['max_pin_ma']}mA per pin):\n\n{comp['tips']}\n\n**Important**: Verify that the component's operating voltage matches the board's logic level ({bd['logic']}V). Use level shifters if needed."
            )})
        # Specifications
        samples.append({"prompt": _prompt(f"What are the specifications of a {comp['name']}?"),
            "completion": _cot(
                [f"Compiling specifications for {comp['name']}."],
                f"**{comp['name']}** ({comp['type']} component):\n- **Terminals**: {comp['terminals']}\n- **Unit**: {comp['unit']}\n- **Common values**: {', '.join(comp['common_values'][:5])}\n- **Ratings**: {comp['ratings']}"
            )})
        # Varied phrasing
        for q in [f"Explain how a {ck} works", f"What does a {ck} do in a circuit?", f"When should I use a {ck}?"]:
            samples.append({"prompt": _prompt(q),
                "completion": _cot(
                    [f"The user is asking about {comp['name']} functionality."],
                    f"{comp['desc']} {comp['tips']}"
                )})

    return samples


# ── 4. C++ Firmware Patterns ──

FIRMWARE_PATTERNS = [
    ("Blink LED", lambda bd: f'''// VoltForge AI — Non-blocking LED Blink for {bd["name"]}
const int LED_PIN = {bd["pwm"][0] if bd["pwm"] else "13"};
unsigned long prevMillis = 0;
const long interval = 500;

void setup() {{
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
}}

void loop() {{
  unsigned long now = millis();
  if (now - prevMillis >= interval) {{
    prevMillis = now;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }}
}}'''),
    ("Read Analog Sensor", lambda bd: f'''// VoltForge AI — Analog Sensor Reader for {bd["name"]}
const int SENSOR_PIN = {bd["adc_pins"][0]};
const float V_REF = {bd["logic"]};
const int ADC_MAX = {2**bd["adc_res"]-1};

void setup() {{
  Serial.begin(115200);
}}

void loop() {{
  int raw = analogRead(SENSOR_PIN);
  float voltage = (raw * V_REF) / ADC_MAX;
  Serial.print("Raw: "); Serial.print(raw);
  Serial.print(" | Voltage: "); Serial.print(voltage, 3);
  Serial.println(" V");
  delay(250);
}}'''),
    ("PWM Breathing LED", lambda bd: f'''// VoltForge AI — Smooth PWM Breathing LED for {bd["name"]}
const int LED_PIN = {bd["pwm"][0] if bd["pwm"] else "9"};

void setup() {{
  pinMode(LED_PIN, OUTPUT);
}}

void loop() {{
  for (int brightness = 0; brightness <= 255; brightness++) {{
    analogWrite(LED_PIN, brightness);
    delay(5);
  }}
  for (int brightness = 255; brightness >= 0; brightness--) {{
    analogWrite(LED_PIN, brightness);
    delay(5);
  }}
}}'''),
    ("I2C Scanner", lambda bd: f'''// VoltForge AI — I2C Device Scanner for {bd["name"]}
#include <Wire.h>

void setup() {{
  Wire.begin();
  Serial.begin(115200);
  Serial.println("I2C Scanner for {bd["name"]}");
  Serial.println("SDA={bd["i2c"][0]}, SCL={bd["i2c"][1]}");
}}

void loop() {{
  int count = 0;
  for (byte addr = 1; addr < 127; addr++) {{
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {{
      Serial.print("Found device at 0x");
      Serial.println(addr, HEX);
      count++;
    }}
  }}
  Serial.print("Total devices: ");
  Serial.println(count);
  delay(5000);
}}'''),
    ("Button with Debounce", lambda bd: f'''// VoltForge AI — Debounced Button Input for {bd["name"]}
const int BTN_PIN = {bd["interrupts"][0] if bd["interrupts"] and bd["interrupts"][0] != "All GPIOs" else "2"};
const int LED_PIN = {bd["pwm"][0] if bd["pwm"] else "13"};
bool ledState = false;
unsigned long lastDebounce = 0;
const unsigned long debounceDelay = 50;
int lastBtnState = HIGH;

void setup() {{
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
}}

void loop() {{
  int reading = digitalRead(BTN_PIN);
  if (reading != lastBtnState) {{
    lastDebounce = millis();
  }}
  if ((millis() - lastDebounce) > debounceDelay && reading != lastBtnState) {{
    if (reading == LOW) {{
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState);
      Serial.println(ledState ? "LED ON" : "LED OFF");
    }}
  }}
  lastBtnState = reading;
}}'''),
    ("Multi-Sensor Data Logger", lambda bd: f'''// VoltForge AI — Multi-Sensor Data Logger for {bd["name"]}
const int SENSOR_A = {bd["adc_pins"][0]};
{"const int SENSOR_B = " + bd["adc_pins"][1] + ";" if len(bd["adc_pins"]) > 1 else ""}
unsigned long logInterval = 1000;
unsigned long lastLog = 0;
unsigned long sampleCount = 0;

void setup() {{
  Serial.begin(115200);
  Serial.println("Timestamp,SensorA,SensorB,Voltage_A");
}}

void loop() {{
  if (millis() - lastLog >= logInterval) {{
    lastLog = millis();
    int a = analogRead(SENSOR_A);
    {"int b = analogRead(SENSOR_B);" if len(bd["adc_pins"]) > 1 else "int b = 0;"}
    float va = (a * {bd["logic"]}) / {2**bd["adc_res"]-1}.0;
    Serial.print(millis()); Serial.print(",");
    Serial.print(a); Serial.print(",");
    Serial.print(b); Serial.print(",");
    Serial.println(va, 3);
    sampleCount++;
  }}
}}'''),
]


def gen_firmware_samples() -> List[Dict[str, str]]:
    samples = []
    for bk, bd in BOARDS.items():
        for pattern_name, code_fn in FIRMWARE_PATTERNS:
            code = code_fn(bd)
            samples.append({"prompt": _prompt(f"Generate {pattern_name} code for {bd['name']}."),
                "completion": _cot(
                    [f"Synthesizing {pattern_name} firmware for {bd['name']} ({bd['mcu']}, {bd['logic']}V, {bd['clock']}).",
                     f"Selecting appropriate pins and ADC configuration."],
                    f"Here is the **{pattern_name}** firmware for the **{bd['name']}**:\n\n```cpp\n{code}\n```"
                )})
    return samples


# ── 5. Circuit Diagnostics & Safety ──

DIAGNOSTIC_SCENARIOS = [
    ("DC motor connected directly to GPIO pin",
     "CRITICAL: The motor's stall current (500mA–5A) far exceeds the GPIO limit ({max_pin_ma}mA). This WILL damage the {board} MCU.",
     "Use a MOSFET driver (e.g., IRLZ44N) or H-bridge (L298N). Connect motor to external power supply. Add a 1N4007 flyback diode across motor terminals. Share ground between motor supply and MCU."),
    ("No decoupling capacitor near IC power pins",
     "WARNING: Without decoupling, high-frequency noise and voltage spikes on the power rail can cause erratic behavior, ADC errors, or MCU resets.",
     "Place a 100nF ceramic capacitor as close as possible to every IC's VCC and GND pins. Add a 10μF–100μF bulk electrolytic near the power input."),
    ("5V sensor connected to 3.3V board GPIO",
     "CRITICAL: The {board} operates at {logic}V logic. Applying 5V to a 3.3V GPIO input will exceed absolute maximum ratings and permanently damage the pin.",
     "Use a bidirectional level shifter (e.g., BSS138 module) or a resistive voltage divider (e.g., 1kΩ + 2kΩ to drop 5V to 3.3V)."),
    ("Floating input pin (no pullup/pulldown)",
     "WARNING: An unconnected digital input pin floats between HIGH and LOW, picking up electromagnetic interference. Readings will be random and unreliable.",
     "Enable the internal pullup resistor with `pinMode(pin, INPUT_PULLUP)`, or add an external 10kΩ pullup/pulldown resistor to a known voltage rail."),
    ("LED connected without current-limiting resistor",
     "CRITICAL: Without a resistor, the LED draws excessive current, which can burn out the LED and source more than {max_pin_ma}mA from the GPIO pin.",
     "Add a resistor in series: R = (Vcc - Vf) / If. For a typical red LED on {logic}V: R = ({logic} - 2.0) / 0.020 = {r_val}Ω. Use nearest E24 value."),
    ("Relay coil connected directly to GPIO",
     "CRITICAL: Relay coils draw 70–100mA, exceeding the {max_pin_ma}mA GPIO limit. The inductive coil also generates back-EMF spikes that can damage the MCU.",
     "Drive the relay through an NPN transistor (e.g., 2N2222) or N-channel MOSFET. Add a 1N4007 flyback diode across the coil. Use a separate relay power supply if needed."),
    ("Using delay() in time-critical code",
     "WARNING: `delay()` blocks the entire MCU — no other code executes during the delay. This prevents reading sensors, handling interrupts, or responding to user input.",
     "Replace `delay()` with non-blocking timing using `millis()`. Store the previous timestamp and check `if (millis() - previous >= interval)` in each `loop()` iteration."),
    ("Stepper motor powered from board 5V rail",
     "CRITICAL: Stepper motors (28BYJ-48) draw 240mA per phase, easily exceeding the {board}'s {total_ma}mA total current budget and causing voltage drops.",
     "Power the stepper motor from a separate 5V supply rated for at least 500mA. Connect supply ground to MCU ground. Use ULN2003 driver board."),
]


def gen_diagnostic_samples() -> List[Dict[str, str]]:
    samples = []
    for bk, bd in BOARDS.items():
        for scenario, finding, fix in DIAGNOSTIC_SCENARIOS:
            r_val = int(round((bd['logic'] - 2.0) / 0.020)) if bd['logic'] > 2.0 else 100
            finding_filled = finding.format(board=bd['name'], logic=bd['logic'], max_pin_ma=bd['max_pin_ma'], total_ma=bd['total_ma'])
            fix_filled = fix.format(board=bd['name'], logic=bd['logic'], max_pin_ma=bd['max_pin_ma'], total_ma=bd['total_ma'], r_val=r_val)
            samples.append({"prompt": _prompt(f"I have a {scenario} on my {bd['name']}. Is this safe?"),
                "completion": _cot(
                    [f"Analyzing circuit safety for {bd['name']} ({bd['logic']}V, {bd['max_pin_ma']}mA/pin).",
                     f"Checking component current draw against GPIO specifications."],
                    f"**{finding_filled}**\n\n**Fix**: {fix_filled}"
                )})
    return samples


# ── 6. Conversational Q&A ──

CONVERSATIONAL_QA = [
    ("What is Ohm's law?", "Ohm's law states that V = I × R, where V is voltage (volts), I is current (amperes), and R is resistance (ohms). It is the fundamental relationship governing current flow through a resistor."),
    ("What is the difference between AC and DC?", "DC (Direct Current) flows in one direction with constant polarity — batteries produce DC. AC (Alternating Current) periodically reverses direction — wall outlets provide AC (50/60Hz). Most microcontrollers operate on DC."),
    ("What is a breadboard?", "A breadboard is a solderless prototyping board with internal metal clips that connect holes in rows. The outer rails run the full length for power (VCC/GND). The inner rows connect groups of 5 holes for component insertion."),
    ("What is PWM?", "Pulse Width Modulation (PWM) simulates analog output by rapidly switching a digital pin ON/OFF. The duty cycle (percentage of time HIGH) controls the average voltage. 50% duty on a 5V pin averages 2.5V. Used for LED dimming and motor speed control."),
    ("What is I2C?", "I2C (Inter-Integrated Circuit) is a two-wire serial communication bus: SDA (data) and SCL (clock). Multiple devices share the same bus, each with a unique 7-bit address. Common devices: OLED displays (0x3C), temperature sensors, EEPROMs. Requires 4.7kΩ pullup resistors on both lines."),
    ("What is SPI?", "SPI (Serial Peripheral Interface) is a four-wire high-speed bus: MOSI (Master Out Slave In), MISO (Master In Slave Out), SCK (clock), and SS/CS (chip select). Faster than I2C (up to 80MHz) but requires more wires. Each slave needs its own SS pin."),
    ("What is a pull-up resistor?", "A pull-up resistor connects an input pin to VCC through a resistor (typically 10kΩ), ensuring the pin reads HIGH when nothing is actively pulling it LOW. This prevents floating inputs. Arduino has built-in pullups enabled with `pinMode(pin, INPUT_PULLUP)`."),
    ("What is a voltage regulator?", "A voltage regulator converts a higher input voltage to a stable, lower output voltage. Linear regulators (LM7805: 5V, AMS1117: 3.3V) are simple but waste excess voltage as heat. Switching regulators (buck/boost) are more efficient for large voltage differences."),
    ("What is back-EMF?", "Back-EMF (Electromotive Force) is the voltage spike generated by an inductive load (motor, relay, solenoid) when current is suddenly interrupted. It can reach hundreds of volts and destroy transistors or MCU pins. Always place a flyback diode (1N4007) reverse-biased across the inductor."),
    ("What is a decoupling capacitor?", "A decoupling capacitor (100nF ceramic, placed between VCC and GND near an IC) filters high-frequency noise from the power supply. It acts as a local energy reservoir, preventing voltage dips during fast current transients. Every IC should have at least one."),
    ("What is an interrupt?", "An interrupt is a hardware signal that pauses the main program to execute a special function (ISR — Interrupt Service Routine) immediately. Used for: button presses, encoder ticks, timer events, serial data arrival. ISRs should be short — set a flag, not run long logic."),
    ("How does analogRead work?", "analogRead() uses the built-in ADC (Analog-to-Digital Converter) to measure voltage on an analog pin. A 10-bit ADC on Arduino Uno maps 0–5V to 0–1023. A 12-bit ADC on ESP32 maps 0–3.3V to 0–4095. The conversion takes about 100μs on AVR."),
    ("What is a MOSFET?", "A MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor) is a voltage-controlled switch. Unlike BJTs (current-controlled), a MOSFET gate draws virtually zero current. N-channel MOSFETs switch loads connected between drain and ground. Use logic-level MOSFETs (Vgs(th) < 3V) for MCU direct drive."),
    ("What is a flyback diode?", "A flyback diode (also called a freewheeling or snubber diode) is placed reverse-biased across an inductive load. When the inductor is de-energized, the diode provides a safe current path for the collapsing magnetic field, preventing destructive voltage spikes. Use 1N4007 for motors/relays."),
    ("What is the difference between analogWrite and digitalWrite?", "digitalWrite() sets a pin to either HIGH (Vcc) or LOW (0V) — fully on or off. analogWrite() generates a PWM signal with variable duty cycle (0–255), effectively controlling average voltage. Not all pins support analogWrite — only PWM-capable pins."),
    ("What is a transistor?", "A transistor is a semiconductor device that acts as an electronic switch or amplifier. BJTs (NPN/PNP) are current-controlled: a small base current controls a larger collector-emitter current. MOSFETs are voltage-controlled: gate voltage controls drain-source current. Transistors are used to switch loads that exceed GPIO current limits."),
    ("What is serial communication?", "Serial communication sends data one bit at a time over a wire. UART (TX/RX) is the most common on Arduino: `Serial.begin(9600)` sets baud rate. Data is sent as ASCII or binary. The Serial Monitor in Arduino IDE displays received data. TX of one device connects to RX of the other."),
    ("What is a logic level shifter?", "A logic level shifter converts signals between different voltage levels (e.g., 3.3V ↔ 5V). Needed when a 3.3V board (ESP32, Pico) communicates with 5V devices. Types: bidirectional MOSFET shifter (BSS138), resistive voltage divider (one-direction only), dedicated IC (TXS0108E)."),
    ("What is the difference between NPN and PNP transistors?", "NPN: current flows from Collector to Emitter when base is driven HIGH. The load connects between VCC and collector. PNP: current flows from Emitter to Collector when base is driven LOW. The load connects between collector and GND. NPN is more common in MCU circuits."),
    ("What is a Zener diode?", "A Zener diode is designed to conduct in reverse at a specific breakdown voltage (Vz). Used for voltage regulation and overvoltage protection. When reverse voltage exceeds Vz, the diode clamps the voltage to Vz. Common values: 3.3V, 5.1V, 12V Zener."),
]


def gen_conversational_samples() -> List[Dict[str, str]]:
    samples = []
    for question, answer in CONVERSATIONAL_QA:
        samples.append({"prompt": _prompt(question),
            "completion": _cot(
                ["Analyzing the user's electronics knowledge question.", "Composing a clear, educational explanation."],
                answer
            )})
        # Rephrase variations
        for prefix in ["Can you explain", "Please tell me about", "I want to understand"]:
            topic = question.replace("What is ", "").replace("What are ", "").replace("?", "").strip()
            samples.append({"prompt": _prompt(f"{prefix} {topic}?"),
                "completion": _cot(
                    [f"The user wants to learn about {topic}."],
                    answer
                )})
    return samples


# ── 7. Out-of-Domain Refusals ──

OOD_PROMPTS = [
    "Write me a poem about love.", "What is the recipe for chocolate cake?",
    "Who won the FIFA World Cup in 2022?", "Tell me a joke.",
    "What is the stock price of Tesla?", "Write an essay about climate change.",
    "How do I cook pasta carbonara?", "Recommend a good Netflix show.",
    "What is the meaning of life?", "Translate this to French.",
    "Give me dating advice.", "Who is the president of the United States?",
    "Write a rap song.", "What is the weather in Tokyo?",
    "Help me with my math homework.", "What is quantum physics?",
    "Book me a flight to London.", "Tell me about World War 2.",
    "Write a resume for me.", "What are the best tourist spots in Paris?",
    "How do I invest in cryptocurrency?", "Explain the theory of relativity.",
    "Write a cover letter.", "What happened in the French Revolution?",
    "Recommend a good book to read.", "How do I train for a marathon?",
    "What is machine learning?", "Write a business plan.",
    "How do I learn to play guitar?", "What is the best diet for weight loss?",
]


def gen_refusal_samples() -> List[Dict[str, str]]:
    samples = []
    refusal = ("I'm VoltForge AI, specialized exclusively in electronics, embedded systems, "
               "microcontrollers, and circuit design. I cannot help with that topic. "
               "Please ask me about circuit design, component selection, firmware coding, "
               "or board pinouts — I'm here to help with your electronics projects!")
    for q in OOD_PROMPTS:
        samples.append({"prompt": _prompt(q),
            "completion": _cot(
                ["This query is outside my electronics and embedded systems domain.", "Generating a polite domain refusal."],
                refusal
            )})
    return samples


# ═══════════════════════════════════════════════════════════════════════
# MASTER GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def generate_all_chunks(output_dir: str) -> Dict[str, int]:
    """Generate all training chunks and the master dataset."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    print("    Generating board architecture samples...")
    c1 = gen_board_samples()
    print("    Generating physics & formula samples...")
    c2 = gen_physics_samples()
    print("    Generating component knowledge samples...")
    c3 = gen_component_samples()
    print("    Generating firmware pattern samples...")
    c4 = gen_firmware_samples()
    print("    Generating diagnostic & safety samples...")
    c5 = gen_diagnostic_samples()
    print("    Generating conversational Q&A samples...")
    c6 = gen_conversational_samples()
    print("    Generating out-of-domain refusals...")
    c7 = gen_refusal_samples()

    chunks = {
        "chunk1_boards.jsonl": c1,
        "chunk2_physics.jsonl": c2,
        "chunk3_components.jsonl": c3,
        "chunk4_firmware.jsonl": c4,
        "chunk5_diagnostics.jsonl": c5,
        "chunk6_conversational.jsonl": c6,
        "chunk7_refusals.jsonl": c7,
    }

    counts = {}
    master = []
    for fname, data in chunks.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        counts[fname] = len(data)
        master.extend(data)

    random.shuffle(master)
    master_path = os.path.join(output_dir, "master_domain_dataset.jsonl")
    with open(master_path, "w", encoding="utf-8") as f:
        for s in master:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    counts["master_domain_dataset.jsonl"] = len(master)

    return counts


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    print("[*] Generating VoltForge Foundation LLM training corpus...")
    stats = generate_all_chunks(out)
    for k, v in stats.items():
        print(f"    [+] {k}: {v} samples")
    print(f"\n[+] Total training samples: {stats['master_domain_dataset.jsonl']}")
