"""Original reference prose with independently checked, exact rational examples.

Families and validation roles are declared before rendering or guard access.
No evaluation prompts, web text, private projects or model completions are inputs.
"""
from fractions import Fraction as F
from data_governance.splitting.signatures import sha

SOURCE = "vf-owned-pilot-reference-v1"
FAMILIES = {
    "polynomial-area": "math",
    "finite-probability": "math",
    "capacitor-charge-sharing": "electronics",
    "resistive-two-node-network": "electronics",
}
RESERVATIONS = {name: "validation" for name in FAMILIES}


def render():
    integral = sum(F(coefficient, degree + 1) for degree, coefficient in enumerate((3, 2, 1)))
    expectation = F(1, 4) * 1 + F(1, 2) * 4 + F(1, 4) * 9
    final_voltage = (F(2) * 6 + F(4) * 0) / (2 + 4)
    dissipated = F(2, 2) * 6**2 - F(6, 2) * final_voltage**2
    # Conductances in mS; currents are mA and potentials are V.
    a, b, c, d, j, k = F(3), F(-1), F(-1), F(2), F(5), F(0)
    va, vb = (j*d-b*k)/(a*d-b*c), (a*k-j*c)/(a*d-b*c)
    return [
        {"family": "polynomial-area", "domain": "math", "values": {"area": str(integral)}, "text": f"""Integrating a polynomial on a finite interval

Consider the explicitly defined function f(x)=3+2x+x^2 for dimensionless x between zero and one. An antiderivative is 3x+x^2+x^3/3. Differentiating each term recovers the corresponding coefficient of f, so evaluating this antiderivative at the two endpoints gives an area of {integral}. The result is an exact rational number; a decimal expansion is optional presentation, not the stored reference answer.

This example separates three operations: interpreting the coefficients, applying a linear integral, and substituting the limits. The constant contributes three, the linear term contributes one, and the quadratic term contributes one third. Changing the interval would require evaluating both endpoints again. A definite integral is a signed accumulation, although the integrand here is positive throughout the interval.

There is an independent numerical identity for this particular polynomial. Simpson's rule on one interval uses the endpoint values and four times the midpoint value, multiplied by one sixth. It is exact for polynomials of degree at most three. Its weights therefore provide a check without calling the antiderivative implementation. Agreement here does not establish accuracy for arbitrary functions, discontinuities or incorrectly specified integration limits.

Keep exact fractions until reporting a result. Rounding the midpoint before applying the weights can introduce a discrepancy that belongs to arithmetic precision rather than calculus. This reference is a derivation and a checked example, not a template for estimating an unspecified physical area."""},
        {"family": "finite-probability", "domain": "math", "values": {"expectation": str(expectation)}, "text": f"""Expectation over a declared finite probability space

Let a random variable X take the values one, two and three with probabilities one quarter, one half and one quarter respectively. The probabilities are nonnegative and sum to one. For the derived variable Y=X^2, the expectation is the probability weighted sum of one, four and nine. Its exact value is {expectation}. Squaring the expectation of X would give a different result because the squaring function is nonlinear.

A concrete equally likely sample space contains four outcomes labelled a, b, c and d. Assign X values one, two, two and three to those outcomes. Enumerating their squares and taking an ordinary arithmetic mean checks the weighted calculation by a separate representation. The enumeration is valid because these four outcomes have equal probabilities; replacing an unequal distribution with an unweighted list would be an error.

The expectation describes a distribution. It does not promise that any individual observed Y equals the mean, nor that a short empirical sequence exactly matches its theoretical probabilities. Independence would matter for claims about repeated trials, but no repeated-trial assumption is necessary to compute this single-variable expectation.

The mapping, probabilities and outcome set are all explicit assumptions in this example. A program should reject probabilities whose total differs from one rather than silently normalizing unknown measurements. Fractions preserve these small rational values exactly and make both the normalization check and the independent enumeration auditable."""},
        {"family": "capacitor-charge-sharing", "domain": "electronics", "values": {"voltageV": str(final_voltage), "lossMicrojoule": str(dissipated)}, "text": f"""Charge redistribution between two ideal capacitors

Two hypothetical capacitors have capacitances two and four microfarads. Initially their voltages relative to a common reference are six and zero volts. Disconnect the charging supply and join the positive plates through a dissipative connection, leaving the reference plates connected. Assume positive capacitances, no charge escaping the combined positive node, negligible stray capacitance and a settled final state. Charge conservation gives a common final potential of {final_voltage} volts: divide the sum of initial C times V by the sum of capacitances.

Initially the stored energy is thirty-six microjoules. The settled common voltage leaves twelve microjoules stored, so {dissipated} microjoules have left the capacitors. Energy has not vanished: the connection dissipates energy during redistribution. An ideal instantaneous switch with no resistance or inductance does not describe the transient current. This settled-state calculation deliberately avoids such a physically incomplete transient model.

An independent check multiplies the final common potential by the total capacitance and compares that charge with the two original charges. A second energy identity uses one half of the reduced capacitance C1*C2/(C1+C2), multiplied by the square of the initial voltage difference. That identity checks the energy difference without subtracting the two stored-energy calculations.

Units matter. Microfarads times volts give microcoulombs, and microfarads times squared volts give microjoules. These chosen values are mathematical assumptions, not component ratings or wiring instructions. ESR, dielectric absorption, leakage, tolerance and voltage dependence would need additional evidence for a real component model."""},
        {"family": "resistive-two-node-network", "domain": "electronics", "values": {"nodeAV": str(va), "nodeBV": str(vb)}, "text": f"""Solving a fully specified two-node resistive network

Use ground as the reference potential. Node A connects to ground through a two millisiemens conductance and to node B through a one millisiemens conductance. Node B also connects to ground through one millisiemens. An ideal current source injects five milliamperes into A; no independent source injects current into B. Conductance times volts gives milliamperes with this unit convention.

Kirchhoff's current law at A gives 3*VA-VB=5. At B it gives -VA+2*VB=0. The determinant of this conductance matrix is five, so this declared grounded network has a unique solution. Cramer's rule gives VA={va} volts and VB={vb} volt. All voltage references and current injection signs are explicit; changing a source direction would change the right-hand side.

Check the result by evaluating branch currents, independently of the determinant formula. A's ground branch carries four milliamperes. The A-to-B branch carries one milliampere, and B's ground branch also carries one milliampere. Thus A's outgoing current equals the injected five milliamperes and B has zero net injection. The source delivers ten milliwatts and the three conductances dissipate eight, one and one milliwatts respectively, providing an additional power-balance check.

This example models positive, linear, time invariant conductances at a settled operating point. It includes neither reactive storage nor semiconductor behavior. A floating network or contradictory ideal-source constraints can make a different system singular; a solver must report that condition instead of filling missing voltages with arbitrary constants."""},
    ]


def verify_example(row):
    """Different algorithms/identities from the renderer; exact comparisons."""
    family, values = row["family"], {key: F(value) for key, value in row["values"].items()}
    if family == "polynomial-area":
        f = lambda x: 3 + 2*x + x*x
        expected = {"area": (f(F(0)) + 4*f(F(1, 2)) + f(F(1))) / 6}
    elif family == "finite-probability":
        expected = {"expectation": F(sum(x*x for x in (1, 2, 2, 3)), 4)}
    elif family == "capacitor-charge-sharing":
        expected = {"voltageV": F(12, 6), "lossMicrojoule": F(1, 2)*F(2*4, 2+4)*(6-0)**2}
    elif family == "resistive-two-node-network":
        if set(values) != {"nodeAV", "nodeBV"}:
            raise ValueError("Invalid network receipt fields")
        va, vb = values["nodeAV"], values["nodeBV"]
        if 2*va + (va-vb) != 5 or vb + (vb-va) != 0 or 5*va != 2*va**2+(va-vb)**2+vb**2:
            raise ValueError("Network current or power balance failed")
        expected = values
    else:
        raise ValueError("Unregistered reference family")
    if values != expected or row["domain"] != FAMILIES[family]:
        raise ValueError("Independent reference calculation failed")
    return {"family": family, "status": "passed-independent-exact-arithmetic", "textSha256": sha(row["text"]), "values": row["values"]}
