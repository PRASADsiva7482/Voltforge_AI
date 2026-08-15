import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("voltforge-ai.pin_router")

DEFAULT_BOARD_PINOUTS = {
    "ARDUINO_UNO": {"i2c_sda": "A4", "i2c_scl": "A5", "spi_mosi": "11", "spi_miso": "12", "spi_sck": "13", "uart_tx": "1", "uart_rx": "0", "vcc_5v": "5V", "gnd": "GND"},
    "ARDUINO_MEGA": {"i2c_sda": "20", "i2c_scl": "21", "spi_mosi": "51", "spi_miso": "50", "spi_sck": "52", "uart_tx": "1", "uart_rx": "0", "vcc_5v": "5V", "gnd": "GND"},
    "ESP32": {"i2c_sda": "21", "i2c_scl": "22", "spi_mosi": "23", "spi_miso": "19", "spi_sck": "18", "uart_tx": "1", "uart_rx": "3", "vcc_3v3": "3V3", "gnd": "GND"},
    "ESP32_S3": {"i2c_sda": "8", "i2c_scl": "9", "spi_mosi": "11", "spi_miso": "13", "spi_sck": "12", "uart_tx": "43", "uart_rx": "44", "vcc_3v3": "3V3", "gnd": "GND"},
    "RASPBERRY_PI_PICO": {"i2c_sda": "GP4", "i2c_scl": "GP5", "spi_mosi": "GP19", "spi_miso": "GP16", "spi_sck": "GP18", "uart_tx": "GP0", "uart_rx": "GP1", "vcc_3v3": "3V3", "gnd": "GND"},
    "STM32_BLUE_PILL": {"i2c_sda": "PB7", "i2c_scl": "PB6", "spi_mosi": "PA7", "spi_miso": "PA6", "spi_sck": "PA5", "uart_tx": "PA9", "uart_rx": "PA10", "vcc_3v3": "3V3", "gnd": "GND"},
}


class PinRouter:
    """Hardware pin allocation & netlist topology analyzer."""

    @classmethod
    def get_board_pinout(cls, board_type: str = "ARDUINO_UNO") -> Dict[str, str]:
        board_upper = (board_type or "ARDUINO_UNO").upper()
        for b_key, pinout in DEFAULT_BOARD_PINOUTS.items():
            if b_key in board_upper or board_upper in b_key:
                return pinout
        return DEFAULT_BOARD_PINOUTS["ARDUINO_UNO"]

    @classmethod
    def resolve_connections(cls, component_name: str, board_type: str = "ARDUINO_UNO") -> List[Dict[str, str]]:
        pinout = cls.get_board_pinout(board_type)
        c_upper = component_name.upper()
        vcc_5v = pinout.get("vcc_5v", "5V")
        gnd_pin = pinout.get("gnd", "GND")

        if "LDR" in c_upper or "PHOTORESISTOR" in c_upper or "LIGHT" in c_upper:
            return [
                {"from": f"{component_name}/PIN1", "to": f"{board_type}/{vcc_5v}", "wireColor": "#ef4444", "reason": "5V Supply"},
                {"from": f"{component_name}/PIN2", "to": f"{board_type}/A0", "wireColor": "#3b82f6", "reason": "Analog ADC Input (A0)"},
                {"from": "10k_Resistor/PIN1", "to": f"{component_name}/PIN2", "wireColor": "#10b981", "reason": "Voltage Divider junction"},
                {"from": "10k_Resistor/PIN2", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Pull-Down to GND"},
            ]
        elif "RELAY" in c_upper:
            return [
                {"from": f"{component_name}/VCC", "to": f"{board_type}/{vcc_5v}", "wireColor": "#ef4444", "reason": "5V Relay Coil Power"},
                {"from": f"{component_name}/GND", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Ground rail"},
                {"from": f"{component_name}/IN", "to": f"{board_type}/D8", "wireColor": "#a855f7", "reason": "Digital Control Pin (D8)"},
            ]
        elif "SERVO" in c_upper:
            return [
                {"from": f"{component_name}/VCC", "to": f"{board_type}/{vcc_5v}", "wireColor": "#ef4444", "reason": "5V Motor Power"},
                {"from": f"{component_name}/GND", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Ground rail"},
                {"from": f"{component_name}/PWM", "to": f"{board_type}/D9", "wireColor": "#f97316", "reason": "PWM Signal Pin (D9)"},
            ]
        elif "ENCODER" in c_upper or "ROTARY" in c_upper:
            return [
                {"from": f"{component_name}/VCC", "to": f"{board_type}/{vcc_5v}", "wireColor": "#ef4444", "reason": "Power rail 5V"},
                {"from": f"{component_name}/GND", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Ground rail"},
                {"from": f"{component_name}/CLK", "to": f"{board_type}/D2", "wireColor": "#3b82f6", "reason": "Phase A / Clock (Interrupt D2)"},
                {"from": f"{component_name}/DT", "to": f"{board_type}/D3", "wireColor": "#eab308", "reason": "Phase B / Data (D3)"},
                {"from": f"{component_name}/SW", "to": f"{board_type}/D4", "wireColor": "#10b981", "reason": "Push Button Switch (D4)"},
            ]
        elif "DHT" in c_upper or "TEMP" in c_upper:

            return [
                {"from": f"{component_name}/VCC", "to": f"{board_type}/{vcc_5v}", "wireColor": "#ef4444", "reason": "Power rail"},
                {"from": f"{component_name}/GND", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Ground rail"},
                {"from": f"{component_name}/DATA", "to": f"{board_type}/D2", "wireColor": "#3b82f6", "reason": "OneWire Data (D2)"},
            ]
        else:
            # Default I2C bus wiring
            vcc_pin = pinout.get("vcc_3v3") if "3" in pinout.get("vcc_3v3", "") else vcc_5v
            return [
                {"from": f"{component_name}/VCC", "to": f"{board_type}/{vcc_pin}", "wireColor": "#ef4444", "reason": "Power rail"},
                {"from": f"{component_name}/GND", "to": f"{board_type}/{gnd_pin}", "wireColor": "#111827", "reason": "Ground rail"},
                {"from": f"{component_name}/SDA", "to": f"{board_type}/{pinout.get('i2c_sda', 'A4')}", "wireColor": "#3b82f6", "reason": "I2C Data"},
                {"from": f"{component_name}/SCL", "to": f"{board_type}/{pinout.get('i2c_scl', 'A5')}", "wireColor": "#eab308", "reason": "I2C Clock"},
            ]

    @classmethod
    def resolve_i2c_connections(cls, component_name: str, board_type: str = "ARDUINO_UNO") -> List[Dict[str, str]]:
        return cls.resolve_connections(component_name, board_type)

