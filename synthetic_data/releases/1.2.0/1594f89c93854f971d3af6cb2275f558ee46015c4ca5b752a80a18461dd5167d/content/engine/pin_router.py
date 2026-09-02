import logging
from typing import Any, Dict, List

from electronics_corpus import get_electronics_corpus

logger = logging.getLogger("voltforge-ai.pin_router")

class PinRouter:
    """Exact-variant pin and wiring resolver backed by the curated corpus."""

    @staticmethod
    def _first_bus(buses: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        for name, bus in buses.items():
            if name.startswith(prefix) and isinstance(bus, dict):
                return bus
        return {}

    @classmethod
    def get_board_pinout(cls, board_type: str = "ARDUINO_UNO") -> Dict[str, Any]:
        corpus = get_electronics_corpus()
        board_result = corpus.lookup("board", board_type)
        if board_result.status != "found":
            return {
                "status": board_result.status,
                "reasonCode": board_result.reasonCode,
                "requestedBoard": board_type,
            }
        result = corpus.lookup("pin-map", board_type)
        if result.status != "found":
            return {
                "status": result.status,
                "reasonCode": result.reasonCode,
                "requestedBoard": board_type,
            }
        record = result.records[0]
        claims = corpus.claims(record)
        buses = claims["default-buses"]
        power = claims["power-pins"]
        i2c = cls._first_bus(buses, "i2c")
        spi = cls._first_bus(buses, "spi")
        uart = cls._first_bus(buses, "uart")
        return {
            "status": "found",
            "reasonCode": result.reasonCode,
            "recordId": record["recordId"],
            "effectiveRevision": record["effectiveRevision"]["revision"],
            "i2c_sda": i2c.get("sda"),
            "i2c_scl": i2c.get("scl"),
            "spi_mosi": spi.get("copi"),
            "spi_miso": spi.get("cipo"),
            "spi_sck": spi.get("sck"),
            "spi_cs": spi.get("cs"),
            "uart_tx": uart.get("tx"),
            "uart_rx": uart.get("rx"),
            "vcc_5v": "5V" if "5V" in power else None,
            "vcc_3v3": next((name for name in ("3V3", "3V3_OUT", "QWIIC") if name in power), None),
            "gnd": "GND" if "GND" in power else None,
            "pinCapabilities": claims["pin-capabilities"],
        }

    @classmethod
    def resolve_connections_result(
        cls, component_name: str, board_type: str = "ARDUINO_UNO"
    ) -> Dict[str, Any]:
        result = get_electronics_corpus().lookup_wiring_recipe(board_type, component_name)
        if result.status != "found":
            return {
                "status": result.status,
                "reasonCode": result.reasonCode,
                "requestedBoard": board_type,
                "requestedComponent": component_name,
                "connections": [],
            }
        record = result.records[0]
        claims = get_electronics_corpus().claims(record)
        colors = ["#ef4444", "#111827", "#3b82f6", "#eab308"]
        connections = [
            {
                "from": f"{component_name}/{item['componentTerminal']}",
                "to": f"{board_type}/{item['boardPin']}",
                "wireColor": colors[index % len(colors)],
                "reason": "Curated exact-variant wiring recipe",
                "knowledgeRecordId": record["recordId"],
                "effectiveRevision": record["effectiveRevision"]["revision"],
            }
            for index, item in enumerate(claims["connections"])
        ]
        return {
            "status": "found",
            "reasonCode": result.reasonCode,
            "recordId": record["recordId"],
            "effectiveRevision": record["effectiveRevision"]["revision"],
            "connections": connections,
        }

    @classmethod
    def resolve_connections(
        cls, component_name: str, board_type: str = "ARDUINO_UNO"
    ) -> List[Dict[str, Any]]:
        return list(cls.resolve_connections_result(component_name, board_type)["connections"])

    @classmethod
    def resolve_i2c_connections(
        cls, component_name: str, board_type: str = "ARDUINO_UNO"
    ) -> List[Dict[str, Any]]:
        return cls.resolve_connections(component_name, board_type)
