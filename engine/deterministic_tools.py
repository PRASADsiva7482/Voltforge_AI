"""Deterministic tool adapters, with no model, tokenizer or neural generation.

Hardware facts come only from the checksum-verified exact-variant corpus.
Numerical operations use explicit inputs; missing operands remain unknown.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from electronics_corpus import get_electronics_corpus
from engine.physics_solver import ElectricalPhysicsSolver
from engine.security import ElectronicsSafetyGuard
from engine.spice_engine import Timer555Solver


_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


class DeterministicElectronicsTools:
    generation_source = "deterministic-tools"

    def __init__(self, artifacts_dir: str | None = None):
        # Legacy argument accepted for import compatibility; no artifact is read.
        pass

    @staticmethod
    def lookup_hardware(record_type: str, query: str) -> dict[str, Any]:
        if record_type not in {"board", "component", "pin-map"}:
            raise ValueError("Unsupported hardware lookup type")
        return get_electronics_corpus().lookup(record_type, query).to_dict()

    @staticmethod
    def rc_cutoff(resistance: float, capacitance: float) -> dict[str, Any]:
        if not all(math.isfinite(value) and value > 0 for value in (resistance, capacitance)):
            raise ValueError("Resistance and capacitance must be finite and positive")
        return ElectricalPhysicsSolver.calculate_rc_cutoff(resistance, capacitance)

    @staticmethod
    def timer_astable(r1: float, r2: float, capacitance: float) -> dict[str, Any]:
        if not all(math.isfinite(value) and value > 0 for value in (r1, r2, capacitance)):
            raise ValueError("R1, R2 and capacitance must be finite and positive")
        return Timer555Solver.astable(r1, r2, capacitance)

    @staticmethod
    def capacitive_reactance(capacitance: float, frequency: float) -> float:
        if not all(math.isfinite(value) and value > 0 for value in (capacitance, frequency)):
            raise ValueError("Capacitance and frequency must be finite and positive")
        return 1 / (2 * math.pi * frequency * capacitance)

    @staticmethod
    def _operand(prompt: str, name: str, unit: str) -> float | None:
        match = re.search(
            rf"\b{name}\s*=\s*({_NUMBER})\s*([kMmunpµμ]?)(?:{unit})?(?![A-Za-z0-9.])",
            prompt,
        )
        if match is None:
            return None
        scale = {"": 1, "k": 1e3, "M": 1e6, "m": 1e-3, "u": 1e-6,
                 "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12}[match[2]]
        return float(match[1]) * scale

    def reason_and_solve(
        self, prompt: str, board_type: str = "ARDUINO_UNO", components=None,
        wires=None, code=None, simulation_state=None, engineering_findings=None,
        netlist=None, memory=None,
    ) -> dict[str, Any]:
        result = {"answer": "", "has_code": False, "generated_code": None,
                  "actions": {}, "mode": self.generation_source, "neuralAttempted": False,
                  "evidence": []}
        guard = ElectronicsSafetyGuard.check_prompt_safety(prompt)
        if not guard["safe"]:
            return {**result, "answer": guard["refusal_message"], "safety_violation": guard["hazard_type"]}
        lower = prompt.casefold()
        if "555" in lower and any(term in lower for term in ("pinout", "maximum voltage", "rating")):
            result["answer"] = "555 timer pinout and ratings require an exact evidenced manufacturer part and package; a generic timer label does not establish NE555 ratings."
            return result
        if any(term in lower for term in ("base resistor", "transistor bias", "transistor switch")):
            result["answer"] = "Transistor base resistor calculation requires explicit logic voltage, load current, base-emitter voltage and forced gain. Provide the missing values and exact transistor variant."
            return result
        if "capacitive reactance" in lower:
            cap = re.search(rf"(?<![\w.])({_NUMBER})\s*([munpµμ]?)F\b", prompt)
            freq = re.search(rf"(?<![\w.])({_NUMBER})\s*([kM]?)Hz\b", prompt)
            if cap and freq:
                c = self._operand(f"C={cap[0]}", "C", "F")
                f = self._operand(f"f={freq[0]}", "f", "Hz")
                try:
                    value = self.capacitive_reactance(c, f)
                    result["answer"] = f"Capacitive Reactance (ideal capacitor): {value:.6g} ohm."
                    result["calculation"] = {"capacitanceF": c, "frequencyHz": f, "reactanceOhm": value}
                except ValueError as error:
                    result["answer"] = str(error)
            else:
                result["answer"] = "Capacitive Reactance requires capacitance in farads and frequency in hertz."
            return result
        if any(term in lower for term in ("what board", "board am i", "active board", "board specifications")):
            lookup = self.lookup_hardware("board", board_type)
            if lookup["status"] == "found":
                record = lookup["records"][0]
                result["answer"] = f"Active board: {record['subject']['name']}.\n" + json.dumps(
                    {"variant": record["subject"]["variant"], "claims": record["claims"]},
                    ensure_ascii=False,
                )
                result["evidence"] = [record["recordId"]]
            else:
                result["answer"] = f"Active board identifier: {board_type}. Exact board evidence is unavailable or ambiguous; select an evidenced variant."
            return result
        if any(term in lower for term in ("on my canvas", "on canvas", "components do i have", "what display")):
            lines = ["Active Canvas (reported project data):"]
            for component in components or []:
                identifier = component.get("id", "unknown")
                kind = component.get("type", "unknown")
                lines.append(f"- {identifier}: {component.get('name') or kind} [{kind}]")
                lookup = self.lookup_hardware("component", kind)
                if lookup["status"] == "found":
                    record = lookup["records"][0]
                    lines.append(f"  Exact evidenced part: {record['subject']['name']}")
                    result["evidence"].append(record["recordId"])
                else:
                    lines.append("  Exact manufacturer variant not verified.")
            for wire in wires or []:
                lines.append(f"- Wire {wire.get('id', 'unknown')}: reported connection")
            result["answer"] = "\n".join(lines)
            return result
        if "555" in lower and any(term in lower for term in ("frequency", "astable", "calculate")):
            values = [self._operand(prompt, "R1", "(?:ohm|Ω)"), self._operand(prompt, "R2", "(?:ohm|Ω)"), self._operand(prompt, "C", "F")]
            if any(value is None for value in values):
                result["answer"] = "555 astable calculation requires explicit R1, R2 and C values with units."
            else:
                try:
                    calculated = self.timer_astable(*values)
                    result["answer"] = f"555 ideal astable calculation: {calculated['frequency_Hz']} Hz; duty cycle {calculated['dutyCyclePercent']}%."
                    result["calculation"] = calculated
                except ValueError as error:
                    result["answer"] = str(error)
            return result
        # No general conversation, firmware template, implicit ratings, fixed
        # confidence or pretend model response is manufactured here.
        return result
