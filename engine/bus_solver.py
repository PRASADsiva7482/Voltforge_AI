"""
Voltforge AI - Inter-IC Communication Bus Solvers
Covers Features 151-160: CAN Bus Termination, RS485 Transceiver, Modbus CRC,
SPI Daisy-Chaining, I2C Multiplexer, OneWire Bus, UART Flow Control, Level Shifter Selector.
"""

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger("voltforge-ai.bus_solver")


class CANBusSolver:
    """CAN Bus termination and configuration solver."""

    @classmethod
    def calculate_termination(cls, bus_length_m: float, num_nodes: int) -> Dict[str, Any]:
        needs_termination = bus_length_m > 0.3 or num_nodes > 2
        return {
            "busLength_m": bus_length_m,
            "numNodes": num_nodes,
            "needsTermination": needs_termination,
            "terminationResistance": "120Ω at each end of the bus" if needs_termination else "Not required for short bus",
            "maxBaudRate": "1 Mbps" if bus_length_m < 40 else "500 kbps" if bus_length_m < 100 else "250 kbps",
            "recommendation": "Place 120Ω termination resistors between CAN_H and CAN_L at both physical ends of the bus.",
        }


class I2CMultiplexerSolver:
    """I2C address collision resolver using PCA9548A multiplexer."""

    @classmethod
    def resolve_collisions(cls, devices: List[Dict[str, str]]) -> Dict[str, Any]:
        address_map = {}
        collisions = []
        for dev in devices:
            addr = dev.get("address", "0x00")
            name = dev.get("name", "Unknown")
            if addr in address_map:
                collisions.append({"address": addr, "devices": [address_map[addr], name]})
            else:
                address_map[addr] = name

        solution = None
        if collisions:
            solution = {
                "multiplexer": "PCA9548A (TCA9548A)",
                "description": "8-channel I2C multiplexer. Connect conflicting devices to separate channels.",
                "channels": [],
            }
            for idx, collision in enumerate(collisions):
                solution["channels"].append({
                    "channel": idx,
                    "device": collision["devices"][1],
                    "address": collision["address"],
                })

        return {
            "deviceCount": len(devices),
            "collisionCount": len(collisions),
            "collisions": collisions,
            "solution": solution,
            "recommendation": "Use PCA9548A I2C multiplexer" if collisions else "No address collisions detected.",
        }


class SPIDaisyChainSolver:
    """SPI daisy-chain calculator for cascaded shift registers."""

    @classmethod
    def calculate_cascade(cls, num_registers: int = 2, bits_per_register: int = 8) -> Dict[str, Any]:
        total_bits = num_registers * bits_per_register
        clock_cycles = total_bits
        return {
            "numRegisters": num_registers,
            "bitsPerRegister": bits_per_register,
            "totalBits": total_bits,
            "clockCyclesNeeded": clock_cycles,
            "chipSelect": "Single CS pin shared across all cascaded registers",
            "dataFlow": f"MOSI → Register 1 → Register 2 → ... → Register {num_registers} (QH' daisy-chained to SER)",
            "latchPulse": "Pulse RCLK (latch) pin after all bits are shifted in",
        }


class ModbusCRCSolver:
    """Modbus RTU CRC16 frame checksum calculator."""

    @classmethod
    def calculate_crc16(cls, data_bytes: List[int]) -> Dict[str, Any]:
        crc = 0xFFFF
        for byte in data_bytes:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        crc_low = crc & 0xFF
        crc_high = (crc >> 8) & 0xFF
        return {
            "inputBytes": [f"0x{b:02X}" for b in data_bytes],
            "crc16": f"0x{crc:04X}",
            "crcLowByte": f"0x{crc_low:02X}",
            "crcHighByte": f"0x{crc_high:02X}",
            "fullFrame": [f"0x{b:02X}" for b in data_bytes] + [f"0x{crc_low:02X}", f"0x{crc_high:02X}"],
        }


class BusCapacitanceEstimator:
    """Estimates maximum I2C bus length based on total capacitance load."""

    @classmethod
    def estimate_max_length(cls, num_devices: int, capacitance_per_device_pF: float = 10, cable_pF_per_m: float = 50, max_bus_capacitance_pF: float = 400) -> Dict[str, Any]:
        device_capacitance = num_devices * capacitance_per_device_pF
        remaining = max_bus_capacitance_pF - device_capacitance
        max_length = remaining / cable_pF_per_m if cable_pF_per_m > 0 else 0
        return {
            "numDevices": num_devices,
            "deviceCapacitance_pF": device_capacitance,
            "maxBusCapacitance_pF": max_bus_capacitance_pF,
            "remainingBudget_pF": remaining,
            "estimatedMaxLength_m": round(max(0, max_length), 2),
            "recommendation": f"Maximum estimated bus length: {round(max(0, max_length), 1)}m at standard mode (100 kHz).",
        }


if __name__ == "__main__":
    import json
    print("=== CAN Bus Termination ===")
    print(json.dumps(CANBusSolver.calculate_termination(5.0, 3), indent=2))

    print("\n=== I2C Address Collision ===")
    devices = [{"name": "BME280_1", "address": "0x76"}, {"name": "BME280_2", "address": "0x76"}]
    print(json.dumps(I2CMultiplexerSolver.resolve_collisions(devices), indent=2))

    print("\n=== Modbus CRC16 ===")
    print(json.dumps(ModbusCRCSolver.calculate_crc16([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A]), indent=2))
