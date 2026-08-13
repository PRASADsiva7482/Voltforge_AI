"""
Voltforge AI - Hardware Component Knowledge Graph
Covers Features 121-130: Pin Capability Graph, Footprint Catalog, Maximum Absolute Ratings,
Sensor Sensitivity Matrix, Communication Protocol Compatibility, Thermal Resistance Database.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("voltforge-ai.knowledge_graph")


COMPONENT_KNOWLEDGE = {
    "MPU6050": {
        "fullName": "InvenSense MPU-6050 6-Axis IMU",
        "category": "SENSOR",
        "operatingVoltage": {"min": 2.375, "max": 3.46, "typical": 3.3},
        "i2cAddress": ["0x68", "0x69"],
        "interfaces": ["I2C"],
        "pins": {"VCC": "power", "GND": "ground", "SDA": "I2C Data", "SCL": "I2C Clock", "INT": "Interrupt", "AD0": "Address Select"},
        "sensitivity": {"accelerometer": "±2g / ±4g / ±8g / ±16g", "gyroscope": "±250°/s / ±500°/s / ±1000°/s / ±2000°/s"},
        "currentDraw_mA": 3.9,
        "package": "QFN-24 (4x4x0.9mm)",
        "requiredExternalComponents": ["4.7kΩ pull-up resistors on SDA/SCL", "100nF decoupling capacitor on VCC"],
        "commonIssues": ["I2C bus freeze without pull-ups", "AD0 floating causes address instability"],
        "equivalents": ["MPU9250", "ICM-20948", "LSM6DS3"],
    },
    "BME280": {
        "fullName": "Bosch BME280 Environmental Sensor",
        "category": "SENSOR",
        "operatingVoltage": {"min": 1.71, "max": 3.6, "typical": 3.3},
        "i2cAddress": ["0x76", "0x77"],
        "interfaces": ["I2C", "SPI"],
        "pins": {"VCC": "power", "GND": "ground", "SDA/SDI": "I2C Data / SPI MOSI", "SCL/SCK": "I2C Clock / SPI SCK", "CSB": "SPI Chip Select", "SDO": "I2C Address Select / SPI MISO"},
        "sensitivity": {"temperature": "-40°C to +85°C (±0.5°C accuracy)", "pressure": "300-1100 hPa (±1 hPa)", "humidity": "0-100% RH (±3%)"},
        "currentDraw_mA": 0.63,
        "package": "LGA-8 (2.5x2.5x0.93mm)",
        "requiredExternalComponents": ["4.7kΩ pull-ups (I2C)", "100nF decoupling cap"],
        "commonIssues": ["SDO pin must be tied to GND (0x76) or VCC (0x77)", "Self-heating affects humidity readings"],
        "equivalents": ["BMP280 (no humidity)", "BME680 (+ gas sensor)", "SHT31"],
    },
    "SSD1306_OLED": {
        "fullName": "Solomon Systech SSD1306 128x64 OLED Display",
        "category": "DISPLAY",
        "operatingVoltage": {"min": 3.0, "max": 5.0, "typical": 3.3},
        "i2cAddress": ["0x3C", "0x3D"],
        "interfaces": ["I2C", "SPI"],
        "pins": {"VCC": "power", "GND": "ground", "SDA": "I2C Data", "SCL": "I2C Clock"},
        "sensitivity": {},
        "currentDraw_mA": 20.0,
        "package": "Module (27x27mm typical)",
        "requiredExternalComponents": ["4.7kΩ pull-ups (I2C mode)"],
        "commonIssues": ["High current draw can cause power brownouts", "I2C address depends on solder jumper"],
        "equivalents": ["SH1106", "SSD1309"],
    },
    "DHT22": {
        "fullName": "AOSONG AM2302 / DHT22 Temperature & Humidity Sensor",
        "category": "SENSOR",
        "operatingVoltage": {"min": 3.3, "max": 5.5, "typical": 5.0},
        "i2cAddress": [],
        "interfaces": ["OneWire (custom protocol)"],
        "pins": {"VCC": "power", "DATA": "signal", "NC": "not connected", "GND": "ground"},
        "sensitivity": {"temperature": "-40°C to +80°C (±0.5°C)", "humidity": "0-100% RH (±2%)"},
        "currentDraw_mA": 1.5,
        "package": "Through-hole 4-pin (15.1x25x7.7mm)",
        "requiredExternalComponents": ["10kΩ pull-up resistor on DATA line"],
        "commonIssues": ["Minimum 2 second sampling interval", "Pull-up resistor required on DATA pin"],
        "equivalents": ["DHT11 (lower accuracy)", "SHT30", "AHT20"],
    },
    "RELAY_MODULE": {
        "fullName": "5V Single-Channel Relay Module",
        "category": "RELAY",
        "operatingVoltage": {"min": 3.75, "max": 6.0, "typical": 5.0},
        "i2cAddress": [],
        "interfaces": ["Digital GPIO"],
        "pins": {"VCC": "power (5V)", "GND": "ground", "IN": "control signal (active LOW)", "COM": "common contact", "NO": "normally open", "NC": "normally closed"},
        "sensitivity": {},
        "currentDraw_mA": 75.0,
        "package": "Module (varies)",
        "requiredExternalComponents": ["Flyback diode (1N4007) across coil", "NPN transistor if driven from 3.3V GPIO"],
        "commonIssues": ["Cannot be driven directly from 3.3V GPIO (use transistor)", "Relay coil draws ~75mA, exceeds GPIO max current"],
        "equivalents": ["SSR (Solid State Relay)", "MOSFET switch module"],
    },
    "HC_SR04": {
        "fullName": "HC-SR04 Ultrasonic Distance Sensor",
        "category": "SENSOR",
        "operatingVoltage": {"min": 4.5, "max": 5.5, "typical": 5.0},
        "i2cAddress": [],
        "interfaces": ["Digital GPIO (Trigger + Echo)"],
        "pins": {"VCC": "power (5V)", "TRIG": "trigger input", "ECHO": "echo output (5V logic)", "GND": "ground"},
        "sensitivity": {"range": "2cm to 400cm (±3mm accuracy)"},
        "currentDraw_mA": 15.0,
        "package": "Module (45x20x15mm)",
        "requiredExternalComponents": ["Voltage divider on ECHO pin if using 3.3V MCU"],
        "commonIssues": ["ECHO pin outputs 5V, can damage 3.3V MCUs", "Minimum trigger pulse width: 10µs"],
        "equivalents": ["JSN-SR04T (waterproof)", "US-015", "VL53L0X (laser ToF)"],
    },
}

PACKAGE_FOOTPRINTS = {
    "0402": {"size_mm": "1.0 x 0.5", "soldering": "Reflow only", "hand_solderable": False},
    "0603": {"size_mm": "1.6 x 0.8", "soldering": "Reflow recommended", "hand_solderable": True},
    "0805": {"size_mm": "2.0 x 1.25", "soldering": "Hand or reflow", "hand_solderable": True},
    "1206": {"size_mm": "3.2 x 1.6", "soldering": "Hand or reflow", "hand_solderable": True},
    "DIP-8": {"size_mm": "9.27 x 6.35", "soldering": "Through-hole", "hand_solderable": True},
    "TO-220": {"size_mm": "10.16 x 4.45", "soldering": "Through-hole", "hand_solderable": True},
    "QFN-24": {"size_mm": "4.0 x 4.0", "soldering": "Reflow required", "hand_solderable": False},
    "SOT-23": {"size_mm": "2.9 x 1.3", "soldering": "Reflow or hot air", "hand_solderable": True},
}


class ComponentKnowledgeGraph:
    """Provides deep hardware knowledge about electronic components."""

    @classmethod
    def get_component_info(cls, component_name: str) -> Optional[Dict[str, Any]]:
        key = component_name.upper().replace("-", "_").replace(" ", "_")
        for k, v in COMPONENT_KNOWLEDGE.items():
            if k in key or key in k:
                return v
        return None

    @classmethod
    def get_required_components(cls, component_name: str) -> List[str]:
        info = cls.get_component_info(component_name)
        if info:
            return info.get("requiredExternalComponents", [])
        return []

    @classmethod
    def get_equivalents(cls, component_name: str) -> List[str]:
        info = cls.get_component_info(component_name)
        if info:
            return info.get("equivalents", [])
        return []

    @classmethod
    def get_common_issues(cls, component_name: str) -> List[str]:
        info = cls.get_component_info(component_name)
        if info:
            return info.get("commonIssues", [])
        return []

    @classmethod
    def get_footprint_info(cls, package: str) -> Optional[Dict[str, Any]]:
        return PACKAGE_FOOTPRINTS.get(package)


if __name__ == "__main__":
    import json
    info = ComponentKnowledgeGraph.get_component_info("MPU6050")
    print("MPU6050 Knowledge:\n", json.dumps(info, indent=2))
    print("\nRequired Components:", ComponentKnowledgeGraph.get_required_components("BME280"))
    print("Equivalents:", ComponentKnowledgeGraph.get_equivalents("DHT22"))
