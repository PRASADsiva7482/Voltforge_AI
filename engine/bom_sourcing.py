"""
Voltforge AI - Bill of Materials (BOM) Cost Estimator & Component Sourcing Engine
Estimates parts cost, volume price breaks, and distributor part numbers (LCSC, DigiKey, Mouser).
"""

from typing import Any, Dict, List


class BomSourcingEngine:
    """Calculates component pricing, supplier part numbers, and batch volume costs."""

    CATALOG_PRICING: Dict[str, Dict[str, Any]] = {
        "RESISTOR": {"unit_price": 0.015, "distributor_pn": "LCSC-C22998", "mfg": "Yageo", "pkg": "0805 / Axial"},
        "CAPACITOR": {"unit_price": 0.025, "distributor_pn": "LCSC-C14663", "mfg": "Samsung", "pkg": "0805 / Radial"},
        "CAPACITOR_CERAMIC": {"unit_price": 0.018, "distributor_pn": "LCSC-C14663", "mfg": "Murata", "pkg": "0805"},
        "LED": {"unit_price": 0.040, "distributor_pn": "LCSC-C72038", "mfg": "Everlight", "pkg": "5mm THT / 0805"},
        "DIODE": {"unit_price": 0.035, "distributor_pn": "LCSC-C81598", "mfg": "ON Semi", "pkg": "DO-35 / SOD-123"},
        "BUZZER": {"unit_price": 0.350, "distributor_pn": "LCSC-C96531", "mfg": "Murata", "pkg": "12mm THT"},
        "ARDUINO_UNO": {"unit_price": 4.500, "distributor_pn": "DIGI-A000066", "mfg": "Arduino", "pkg": "DIP-28 Board"},
        "ARDUINO_NANO": {"unit_price": 2.800, "distributor_pn": "DIGI-A000005", "mfg": "Arduino", "pkg": "TQFP-32 Module"},
        "ESP32_DEVKIT_V1": {"unit_price": 3.200, "distributor_pn": "MOUSER-ESP32-WROOM", "mfg": "Espressif", "pkg": "Module"},
        "RPI_PICO": {"unit_price": 4.000, "distributor_pn": "RPI-SC0915", "mfg": "Raspberry Pi", "pkg": "Castellated Module"},
        "DISPLAY_OLED_SSD1306": {"unit_price": 1.950, "distributor_pn": "LCSC-C285324", "mfg": "Solomon", "pkg": "0.96 inch I2C"},
        "SENSOR_DHT22": {"unit_price": 2.200, "distributor_pn": "LCSC-C43501", "mfg": "Aosong", "pkg": "4-pin Single Bus"},
        "VOLTAGE_REGULATOR_LM7805": {"unit_price": 0.280, "distributor_pn": "LCSC-C69348", "mfg": "TI", "pkg": "TO-220"},
    }

    @classmethod
    def calculate_bom_cost(
        cls,
        components: List[Dict[str, Any]],
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """Rolls up components into line-items with volume price breaks and sourcing info."""
        grouped: Dict[str, Dict[str, Any]] = {}

        for comp in components:
            ctype = str(comp.get("type", "UNKNOWN")).upper()
            cval = str(comp.get("value", "-"))
            key = f"{ctype}::{cval}"

            pricing = cls.CATALOG_PRICING.get(ctype, {
                "unit_price": 0.10,
                "distributor_pn": f"GENERIC-{ctype[:8]}",
                "mfg": "Generic",
                "pkg": comp.get("packageType", "Standard")
            })

            if key not in grouped:
                grouped[key] = {
                    "type": ctype,
                    "value": cval,
                    "package": pricing.get("pkg", "Standard"),
                    "manufacturer": pricing.get("mfg", "Generic"),
                    "partNumber": pricing.get("distributor_pn", "GEN-001"),
                    "unitPrice": pricing.get("unit_price", 0.10),
                    "quantity": 1
                }
            else:
                grouped[key]["quantity"] += 1

        line_items: List[Dict[str, Any]] = []
        total_unit_cost = 0.0

        for item in grouped.values():
            ext_price = round(item["unitPrice"] * item["quantity"], 3)
            total_unit_cost += ext_price
            line_items.append({
                **item,
                "extendedPrice": ext_price
            })

        # Calculate volume discount scale
        volume_pricing = {
            "qty_1": round(total_unit_cost, 2),
            "qty_10": round(total_unit_cost * 10 * 0.90, 2),     # 10% off
            "qty_100": round(total_unit_cost * 100 * 0.78, 2),   # 22% off
            "qty_1000": round(total_unit_cost * 1000 * 0.65, 2)  # 35% off
        }

        return {
            "status": "SUCCESS",
            "currency": currency,
            "totalComponents": len(components),
            "uniqueLineItems": len(line_items),
            "estimatedUnitCost": round(total_unit_cost, 2),
            "lineItems": line_items,
            "volumePricing": volume_pricing
        }
