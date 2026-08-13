"""
Voltforge AI - Digital Logic & HDL Engine
Covers Features 261-280: Truth Table Generator, Karnaugh Map Simplifier,
74-Series IC Emulator, FSM Designer, Seven-Segment Decoder, Counter Builder.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

logger = logging.getLogger("voltforge-ai.digital_logic")


class TruthTableGenerator:
    """Generates truth tables for combinational logic gate networks."""

    GATES = {
        "AND": lambda a, b: a & b,
        "OR": lambda a, b: a | b,
        "XOR": lambda a, b: a ^ b,
        "NAND": lambda a, b: ~(a & b) & 1,
        "NOR": lambda a, b: ~(a | b) & 1,
        "XNOR": lambda a, b: ~(a ^ b) & 1,
        "NOT": lambda a, _=0: ~a & 1,
        "BUFFER": lambda a, _=0: a,
    }

    @classmethod
    def generate(cls, gate: str, num_inputs: int = 2) -> Dict[str, Any]:
        gate_upper = gate.upper()
        gate_fn = cls.GATES.get(gate_upper)
        if not gate_fn:
            return {"error": f"Unknown gate: {gate}. Supported: {list(cls.GATES.keys())}"}

        if gate_upper in ("NOT", "BUFFER"):
            num_inputs = 1

        rows = []
        for combo in product([0, 1], repeat=num_inputs):
            if num_inputs == 1:
                output = gate_fn(combo[0])
            elif num_inputs == 2:
                output = gate_fn(combo[0], combo[1])
            else:
                result = combo[0]
                for i in range(1, num_inputs):
                    result = gate_fn(result, combo[i])
                output = result
            rows.append({"inputs": list(combo), "output": output})

        return {
            "gate": gate_upper,
            "numInputs": num_inputs,
            "truthTable": rows,
            "totalCombinations": len(rows),
        }


class KarnaughMapSimplifier:
    """Simplifies boolean expressions using 2-variable Karnaugh maps."""

    @classmethod
    def simplify_2var(cls, minterms: List[int]) -> Dict[str, Any]:
        """Simplifies a 2-variable boolean function from minterms (0-3)."""
        variables = ["A", "B"]
        kmap = [[0, 0], [0, 0]]
        for m in minterms:
            row = (m >> 1) & 1
            col = m & 1
            kmap[row][col] = 1

        terms = []
        # Check for all ones
        if all(kmap[r][c] == 1 for r in range(2) for c in range(2)):
            terms = ["1"]
        else:
            # Check rows
            for r in range(2):
                if kmap[r][0] == 1 and kmap[r][1] == 1:
                    terms.append(f"{variables[0]}" if r == 1 else f"{variables[0]}'")
            # Check columns
            for c in range(2):
                if kmap[0][c] == 1 and kmap[1][c] == 1:
                    terms.append(f"{variables[1]}" if c == 1 else f"{variables[1]}'")
            # Individual minterms not covered
            if not terms:
                for m in minterms:
                    a = (m >> 1) & 1
                    b = m & 1
                    a_str = "A" if a else "A'"
                    b_str = "B" if b else "B'"
                    term = f"{a_str} . {b_str}"
                    terms.append(term)

        expression = " + ".join(terms) if terms else "0"
        return {
            "variables": variables,
            "minterms": minterms,
            "kmap": kmap,
            "simplifiedExpression": expression,
        }


class IC74SeriesEmulator:
    """Emulates common 74-series logic ICs."""

    IC_DATABASE = {
        "74HC00": {"name": "Quad 2-Input NAND Gate", "gates": 4, "gate_type": "NAND", "inputs_per_gate": 2, "package": "DIP-14"},
        "74HC02": {"name": "Quad 2-Input NOR Gate", "gates": 4, "gate_type": "NOR", "inputs_per_gate": 2, "package": "DIP-14"},
        "74HC04": {"name": "Hex Inverter", "gates": 6, "gate_type": "NOT", "inputs_per_gate": 1, "package": "DIP-14"},
        "74HC08": {"name": "Quad 2-Input AND Gate", "gates": 4, "gate_type": "AND", "inputs_per_gate": 2, "package": "DIP-14"},
        "74HC32": {"name": "Quad 2-Input OR Gate", "gates": 4, "gate_type": "OR", "inputs_per_gate": 2, "package": "DIP-14"},
        "74HC86": {"name": "Quad 2-Input XOR Gate", "gates": 4, "gate_type": "XOR", "inputs_per_gate": 2, "package": "DIP-14"},
        "74HC138": {"name": "3-to-8 Line Decoder", "gates": 1, "gate_type": "DECODER", "inputs_per_gate": 3, "package": "DIP-16"},
        "74HC595": {"name": "8-Bit Shift Register", "gates": 1, "gate_type": "SHIFT_REGISTER", "inputs_per_gate": 3, "package": "DIP-16"},
        "74HC4511": {"name": "BCD-to-7-Segment Decoder", "gates": 1, "gate_type": "7SEG_DECODER", "inputs_per_gate": 4, "package": "DIP-16"},
    }

    @classmethod
    def get_ic_info(cls, ic_number: str) -> Optional[Dict[str, Any]]:
        key = ic_number.upper().replace("-", "").replace(" ", "")
        for k, v in cls.IC_DATABASE.items():
            if k in key or key in k:
                return {"partNumber": k, **v}
        return None

    @classmethod
    def simulate_gate(cls, ic_number: str, inputs: List[int]) -> Dict[str, Any]:
        info = cls.get_ic_info(ic_number)
        if not info:
            return {"error": f"IC {ic_number} not found in database."}
        gate_type = info["gate_type"]
        gate_fn = TruthTableGenerator.GATES.get(gate_type)
        if gate_fn and len(inputs) >= 2:
            output = gate_fn(inputs[0], inputs[1])
            return {"ic": info["partNumber"], "name": info["name"], "inputs": inputs, "output": output}
        elif gate_fn and len(inputs) == 1:
            output = gate_fn(inputs[0])
            return {"ic": info["partNumber"], "name": info["name"], "inputs": inputs, "output": output}
        return {"ic": info["partNumber"], "name": info["name"], "inputs": inputs, "output": "N/A (complex IC)"}


class SevenSegmentDecoder:
    """Generates BCD-to-7-Segment display pin mappings."""

    SEGMENTS = {
        0: [1, 1, 1, 1, 1, 1, 0],  # a,b,c,d,e,f,g
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1],
    }

    @classmethod
    def decode(cls, digit: int) -> Dict[str, Any]:
        if digit < 0 or digit > 9:
            return {"error": "Digit must be 0-9."}
        segments = cls.SEGMENTS[digit]
        segment_names = ["a", "b", "c", "d", "e", "f", "g"]
        active = [segment_names[i] for i, v in enumerate(segments) if v == 1]
        return {
            "digit": digit,
            "segments": dict(zip(segment_names, segments)),
            "activeSegments": active,
            "binaryPattern": "".join(str(s) for s in segments),
        }


class FSMDesigner:
    """Generates Finite State Machine state transition tables."""

    @classmethod
    def design_moore(cls, states: List[str], transitions: List[Dict[str, str]], outputs: Dict[str, str]) -> Dict[str, Any]:
        return {
            "type": "Moore FSM",
            "stateCount": len(states),
            "states": states,
            "transitions": transitions,
            "outputs": outputs,
            "encodingBits": (len(states) - 1).bit_length() or 1,
            "description": f"Moore FSM with {len(states)} states and {len(transitions)} transitions.",
        }


class SynchronousCounterBuilder:
    """Generates modulo-N binary counter configurations."""

    @classmethod
    def build(cls, modulo: int, direction: str = "up") -> Dict[str, Any]:
        bits = (modulo - 1).bit_length() or 1
        sequence = list(range(modulo)) if direction == "up" else list(range(modulo - 1, -1, -1))
        return {
            "modulo": modulo,
            "direction": direction,
            "bitsRequired": bits,
            "flipFlopsNeeded": bits,
            "countSequence": sequence,
            "maxCount": modulo - 1,
            "overflowAt": modulo,
        }


if __name__ == "__main__":
    import json
    print("=== AND Gate Truth Table ===")
    print(json.dumps(TruthTableGenerator.generate("AND", 2), indent=2))

    print("\n=== K-Map Simplification ===")
    print(json.dumps(KarnaughMapSimplifier.simplify_2var([0, 1, 2, 3]), indent=2))

    print("\n=== 74HC595 Info ===")
    print(json.dumps(IC74SeriesEmulator.get_ic_info("74HC595"), indent=2))

    print("\n=== 7-Segment for digit 5 ===")
    print(json.dumps(SevenSegmentDecoder.decode(5), indent=2))

    print("\n=== Mod-8 Counter ===")
    print(json.dumps(SynchronousCounterBuilder.build(8, "up"), indent=2))
