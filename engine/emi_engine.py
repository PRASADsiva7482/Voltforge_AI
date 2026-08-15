"""
Voltforge AI - Smart Decoupling & EMI Rule Engine
Detects missing bypass/decoupling capacitors and power integrity hazards in EDA schematics.
"""

from typing import Any, Dict, List


class EmiRuleEngine:
    """Evaluates schematics for EMI hazards and generates automatic bypass capacitor insertions."""

    IC_TYPES = {
        "ESP32_DEVKIT_V1", "ARDUINO_UNO", "ARDUINO_NANO", "RPI_PICO", "STM32",
        "DISPLAY_OLED_SSD1306", "SHIFT_REGISTER_74HC595", "OPAMP_LM358", "TIMER_NE555"
    }

    @classmethod
    def evaluate_decoupling(
        cls,
        components: List[Dict[str, Any]],
        wires: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Scans components and proposes auto-inserted 100nF decoupling caps for vulnerable ICs."""
        violations: List[Dict[str, Any]] = []
        proposed_insertions: List[Dict[str, Any]] = []
        cap_counter = 1

        # Find existing capacitors
        existing_caps = [c for c in components if "CAPACITOR" in str(c.get("type", "")).upper()]

        for comp in components:
            ctype = str(comp.get("type", "")).upper()
            cid = str(comp.get("id", ""))
            x = float(comp.get("x", 100))
            y = float(comp.get("y", 100))

            is_ic = any(ic in ctype for ic in cls.IC_TYPES) or comp.get("packageType") in ("DIP", "SOIC", "QFP")

            if is_ic:
                # Check if there is a capacitor placed physically close to this IC (< 100px)
                has_nearby_cap = False
                for cap in existing_caps:
                    cx = float(cap.get("x", 0))
                    cy = float(cap.get("y", 0))
                    if abs(cx - x) < 120 and abs(cy - y) < 120:
                        has_nearby_cap = True
                        break

                if not has_nearby_cap:
                    cap_id = f"cap_bypass_{cap_counter}"
                    violations.append({
                        "componentId": cid,
                        "componentType": ctype,
                        "rule": "EMI_MISSING_DECOUPLING_CAPACITOR",
                        "severity": "WARNING",
                        "message": f"{ctype} is missing a 0.1µF (100nF) ceramic bypass capacitor across VCC/GND."
                    })

                    # Propose auto-insertion
                    proposed_insertions.append({
                        "capacitor": {
                            "id": cap_id,
                            "type": "CAPACITOR_CERAMIC",
                            "name": f"C_dec_{cap_counter}",
                            "value": "100nF",
                            "x": x - 40,
                            "y": y + 20
                        },
                        "wires": [
                            {"fromNodeId": cap_id, "fromPinId": "PIN1", "toNodeId": cid, "toPinId": "VCC", "color": "#ef4444"},
                            {"fromNodeId": cap_id, "fromPinId": "PIN2", "toNodeId": cid, "toPinId": "GND", "color": "#000000"}
                        ]
                    })
                    cap_counter += 1

        return {
            "status": "SUCCESS",
            "totalComponentsEvaluated": len(components),
            "violationCount": len(violations),
            "violations": violations,
            "proposedInsertions": proposed_insertions,
            "passRate": round(1.0 - (len(violations) / max(1, len(components))), 2)
        }
