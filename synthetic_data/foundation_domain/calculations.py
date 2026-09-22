"""Independent answer review: circuit invariants, enumeration and exact arithmetic.

This module must not import the recipe answer function. Bounds deliberately
describe this reference dataset's supported parameter domain, not device limits.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math

UNITS = dict(zip(("weighted-summer", "sensor-bridge", "capacitor-decay", "even-parity", "signed-saturation", "rollover-elapsed", "scheduler-load", "uart-frame-time", "i2c-clock-budget", "linear-regulator-loss", "ideal-buck-ripple", "ideal-adc-bin", "uncorrelated-average", "hysteresis-state", "proportional-clamp", "ring-occupancy", "buffer-budget"), ("V", "V", "s", "bit", "code", "ms", "fraction", "s", "s", "W", "A", "code", "V-rms", "state", "fraction", "slots", "bytes")))


def bounded(p, rules):
    if set(p) != set(rules): raise ValueError("Unexpected or missing parameter")
    for key, (low, high, integer) in rules.items():
        value = p[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or integer and type(value) is not int:
            raise ValueError("Parameter outside reviewed domain: " + key)


POS = (1e-12, 1e9, False)
INT = (1, 1000000, True)


def oracle(family, p):
    q = {key: F(str(value)) for key, value in p.items() if type(value) in (int, float) and math.isfinite(value)}
    if family == "weighted-summer":
        bounded(p, dict(rf=POS, r1=POS, r2=POS, v1=(-12, 12, False), v2=(-12, 12, False)))
        # Exact branch currents and feedback current satisfy KCL.
        out = -(q["v1"] / q["r1"] + q["v2"] / q["r2"]) / (1 / q["rf"])
        if not -12 < out < 12: raise ValueError("Ideal amplifier saturates at supplied rails")
        return float(out), "exact-rational-node-current-balance"
    if family == "sensor-bridge":
        bounded(p, dict(v=POS, lt=POS, lb=POS, rt=POS, rb=POS))
        nodes = [(q["v"] / q[top]) / (1 / q[top] + 1 / q[bottom]) for top, bottom in (("lt", "lb"), ("rt", "rb"))]
        return float(nodes[0] - nodes[1]), "exact-rational-nodal-conductance-solve"
    if family == "capacitor-decay":
        bounded(p, dict(r=POS, c=(1e-12, 1, False), v0=POS, vt=POS))
        if not p["vt"] < p["v0"]: raise ValueError("Threshold must lie below initial voltage")
        low, high = 0.0, 100 * p["r"] * p["c"]
        for _ in range(160):
            mid = (low + high) / 2
            if p["v0"] * math.exp(-mid / (p["r"] * p["c"])) > p["vt"]: low = mid
            else: high = mid
        return (low + high) / 2, "monotone-discharge-bisection-no-logarithm"
    if family == "even-parity":
        bounded(p, dict(payload=(0, 255, True)))
        parity = 0
        for bit in range(8): parity ^= (p["payload"] >> bit) & 1
        return parity, "xor-truth-enumeration"
    if family == "signed-saturation":
        bounded(p, dict(a=(-32768, 32767, True), b=(-32768, 32767, True)))
        exact = q["a"] / 32768 + q["b"] / 32768
        if exact < -1: return -32768, "rational-normalized-saturation"
        if exact > F(32767, 32768): return 32767, "rational-normalized-saturation"
        return int(exact * 32768), "rational-normalized-saturation"
    if family == "rollover-elapsed":
        bounded(p, dict(now=(0, 4294967295, True), then=(0, 4294967295, True)))
        return p["now"] - p["then"] if p["now"] >= p["then"] else 4294967296 - p["then"] + p["now"], "conditional-wrap-distance"
    if family == "scheduler-load":
        bounded(p, dict(c1=(1, 1000, True), t1=(1, 1000, True), c2=(1, 1000, True), t2=(1, 1000, True)))
        if p["c1"] > p["t1"] or p["c2"] > p["t2"]: raise ValueError("Execution exceeds own period")
        hyper = math.lcm(p["t1"], p["t2"])
        work = sum(p["c1"] for _ in range(0, hyper, p["t1"])) + sum(p["c2"] for _ in range(0, hyper, p["t2"]))
        return float(F(work, hyper)), "enumerated-hyperperiod-demand"
    if family == "uart-frame-time":
        bounded(p, dict(n=(1, 10000, True), d=(5, 9, True), p=(0, 1, True), s=(1, 2, True), baud=INT))
        bits = sum(len([0] + [1] * p["d"] + [0] * p["p"] + [1] * p["s"]) for _ in range(p["n"]))
        return float(F(bits, p["baud"])), "enumerated-frame-bits"
    if family == "i2c-clock-budget":
        bounded(p, dict(n=(1, 10000, True), hz=INT))
        clocks = sum(1 for _ in range(p["n"] + 1) for _bit in range(8)) + (p["n"] + 1)
        return float(F(clocks, p["hz"])), "data-clocks-plus-ack-clocks"
    if family == "linear-regulator-loss":
        bounded(p, dict(vin=POS, vout=POS, i=POS))
        if p["vout"] >= p["vin"]: raise ValueError("Positive dropout required")
        return float(q["vin"] * q["i"] - q["vout"] * q["i"]), "exact-input-output-power-balance"
    if family == "ideal-buck-ripple":
        bounded(p, dict(vin=POS, vout=POS, l=POS, hz=POS))
        if p["vout"] >= p["vin"]: raise ValueError("Buck output must be below input")
        off_time = (1 - q["vout"] / q["vin"]) / q["hz"]
        return float(q["vout"] / q["l"] * off_time), "off-interval-volt-second-balance"
    if family == "ideal-adc-bin":
        bounded(p, dict(bits=(1, 16, True), vin=(0, 1e9, False), vref=POS))
        if p["vin"] >= p["vref"]: raise ValueError("Input outside half-open conversion interval")
        low, high = 0, 2 ** p["bits"]
        while low + 1 < high:
            mid = (low + high) // 2
            if F(mid, 2 ** p["bits"]) * q["vref"] <= q["vin"]: low = mid
            else: high = mid
        return low, "exact-rational-bin-boundary-search"
    if family == "uncorrelated-average":
        bounded(p, dict(n=(1, 10000, True), sigma=POS))
        with localcontext() as ctx:
            ctx.prec = 40
            variance = sum(Decimal(str(p["sigma"])) ** 2 for _ in range(p["n"])) / Decimal(p["n"]) ** 2
            return float(variance.sqrt()), "decimal-sum-of-independent-variances"
    if family == "hysteresis-state":
        bounded(p, dict(x=(-273.15, 1000, False), low=(-273.15, 1000, False), high=(-273.15, 1000, False), previous=(0, 1, True)))
        if p["low"] >= p["high"]: raise ValueError("Empty hysteresis band")
        table = {(True, False): 1, (False, True): 0, (False, False): p["previous"]}
        return table[p["x"] <= p["low"], p["x"] >= p["high"]], "exhaustive-threshold-state-table"
    if family == "proportional-clamp":
        bounded(p, dict(gain=POS, error=(-1e6, 1e6, False)))
        candidates = [F(0), q["gain"] * q["error"], F(1)]
        return float(sorted(candidates)[1]), "rational-interval-projection-median"
    if family == "ring-occupancy":
        bounded(p, dict(n=(2, 1024, True), head=(0, 1023, True), tail=(0, 1023, True)))
        if p["n"] & (p["n"] - 1) or max(p["head"], p["tail"]) >= p["n"]: raise ValueError("Invalid ring capacity or index")
        cursor, count = p["tail"], 0
        while cursor != p["head"]:
            cursor += 1
            if cursor == p["n"]: cursor = 0
            count += 1
        return count, "enumerated-ring-distance"
    if family == "buffer-budget":
        bounded(p, dict(n=(1, 10000, True), w=(1, 1024, True), m=(0, 10000, True)))
        return sum(p["w"] for _ in range(p["n"]) for _buffer in range(2)) + p["m"], "enumerated-object-allocation"
    raise ValueError("Unknown independently reviewed formula")


def review(family, parameters, value, unit):
    expected, method = oracle(family, parameters)
    if unit != UNITS[family] or type(value) not in (int, float) or not math.isfinite(value): raise ValueError("Invalid result or unit")
    exact = type(expected) is int
    absolute, relative = (0, 0) if exact else (1e-12, 1e-9)
    if exact and type(value) is not int or not math.isclose(value, expected, abs_tol=absolute, rel_tol=relative): raise ValueError("Independent answer mismatch")
    return {"status": "passed", "method": method, "expected": expected, "value": value, "unit": unit,
            "absoluteTolerance": absolute, "relativeTolerance": relative, "parameterBoundsChecked": True,
            "reviewType": "independent-algorithm-not-human-review"}


def receipt(family, parameters, value, unit):
    result = review(family, parameters, value, unit)
    corruptions = [(parameters, value + max(1, abs(value)), unit), (parameters, value, "wrong-unit"),
                   ({**parameters, next(iter(parameters)): float("nan")}, value, unit)]
    rejected = 0
    for args in corruptions:
        try: review(family, *args)
        except (ValueError, OverflowError): rejected += 1
    if rejected != len(corruptions): raise ValueError("Calculation negative control accepted")
    return result | {"negativeControlsRejected": ["wrong-answer", "wrong-unit", "nonfinite-input"]}
