"""Conservative checks on packed excerpts, not a general factual verifier.

Only direct specification questions are eligible. Calculations/explanations go
to the model. A conflict requires an identical explicit label (including any
revision/conditions) in distinct sources and different normalized quantities.
No source commands are interpreted, no inferred circuit values are produced.
"""
from decimal import Decimal, InvalidOperation
import re

UNITS = {
    'current': {'A': '1', 'mA': '.001', 'uA': '.000001', 'µA': '.000001', 'μA': '.000001', 'nA': '.000000001'},
    'voltage': {'V': '1', 'mV': '.001', 'uV': '.000001', 'µV': '.000001', 'kV': '1000'},
    'frequency': {'Hz': '1', 'kHz': '1000', 'MHz': '1000000', 'GHz': '1000000000'},
    'resistance': {'Ω': '1', 'ohm': '1', 'ohms': '1', 'kΩ': '1000', 'MΩ': '1000000'},
}


def clarification_without_context(question):
    """Clarify short references to unavailable artifacts; never guess a circuit.

    Caller must first establish that no history, references or project data is
    supplied. Messages with literal details/code go to ordinary inference.
    """
    if len(question.split()) > 16 or re.search(r'[\d\n`"\':]', question):
        return None
    if re.search(r'\b(this|my|that)\s+(circuit|schematic|wiring)\b', question, re.I):
        return ('Please share the circuit diagram or the board, components and pin connections. '
                'I need those details before I can assess whether it will work.')
    if (re.search(r'\b(this|that|the|an)\s+error\b', question, re.I)
            and re.search(r'\b(firmware|serial|compiler|code)\b', question, re.I)):
        return ('Please paste the exact error or serial output and the relevant firmware. '
                'I do not have those details in this conversation yet.')
    return None


def requested_dimension(question):
    if not re.match(r'\s*what\b', question, re.I):
        return None
    if re.search(r'\b(calculate|compute|explain|why|formula|difference|compare|relationship)\b', question, re.I):
        # "Compare the references" can accompany a direct specification query.
        first = re.split(r'[?.]', question, maxsplit=1)[0]
        if re.search(r'\b(calculate|compute|explain|why|formula|difference|compare|relationship)\b', first, re.I):
            return None
    if not re.search(r'\b(maximum|minimum|max|min|limit|rated|supply|output|input|clock)\b', question, re.I):
        return None
    dimensions = [name for name in UNITS if re.search(r'\b' + name + r'\b', question, re.I)]
    return dimensions[0] if len(dimensions) == 1 else None


def specification_guidance(question, sources):
    """Return qualified guidance only for provable gaps/conflicts in excerpts.

    A missing unit means the *excerpts* do not establish the numerical value,
    not that the full document lacks it. Unit presence alone never proves a
    value is relevant or safe. All other cases retain ordinary native inference.
    """
    dimension = requested_dimension(question)
    if not dimension or not sources:
        return None
    units = '|'.join(re.escape(unit) for unit in sorted(UNITS[dimension], key=len, reverse=True))
    quantity = re.compile(r'(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(' + units + r')(?!\w)')
    texts = [(source, source['snippet']) for source in sources]
    # Written unit names are left to the model: do not incorrectly call "amps"
    # absent merely because it is not an SI symbol. Bare numbers are ambiguous.
    names = {'current': r'(?:milli|micro|nano|kilo)?amp(?:ere)?s?',
             'voltage': r'(?:milli|micro|nano|kilo)?volts?',
             'frequency': r'(?:kilo|mega|giga)?hertz',
             'resistance': r'(?:(?:kilo|mega)?ohms?|kilohms?|megohms?)'}
    if not any(quantity.search(text) or re.search(r'\b' + names[dimension] + r'\b', text, re.I)
               for _, text in texts):
        return (f'The provided excerpts do not specify a numeric {dimension} value for this question. '
                'Please provide the relevant datasheet section, including the part revision and operating conditions.')

    facts = {}
    for source, text in texts:
        for sentence in re.split(r'(?<=[A-Za-zΩ])\.\s+|\n', text):
            match = re.fullmatch(r'\s*([^:\n]{3,240}):\s*([+-]?\d+(?:\.\d+)?)\s*('
                                 + units + r')\s*\.?\s*', sentence)
            if not match:
                continue
            label, value, unit = match.groups()
            if not re.search(r'\b' + dimension + r'\b', label, re.I):
                continue
            try:
                normalized = Decimal(value) * Decimal(UNITS[dimension][unit])
            except InvalidOperation:
                continue
            key = ' '.join(label.casefold().split())
            entries = facts.setdefault(key, [])
            # Only compare labels relevant to the actual request, rather than
            # conflicting unrelated fields that happen to appear in a document.
            terms = set(re.findall(r'[a-z0-9]+', key)) - {'the', 'is', 'of', 'a', 'an'}
            if not terms.issubset(set(re.findall(r'[a-z0-9]+', question.casefold()))):
                continue
            for previous, prior_value, prior_unit, prior_source in entries:
                if normalized != previous and prior_source['sourceId'] != source['sourceId']:
                    return (f'The provided references conflict for the requested {dimension}: '
                            f'{prior_value} {prior_unit} [{prior_source["citationId"]}] versus '
                            f'{value} {unit} [{source["citationId"]}]. '
                            'I cannot select a reliable value from these excerpts. '
                            'Please confirm the authoritative datasheet revision and operating conditions before using either value.')
            entries.append((normalized, value, unit, source))
    return None
