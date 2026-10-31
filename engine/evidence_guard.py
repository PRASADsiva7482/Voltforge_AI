"""Conservative checks on packed excerpts, not a general factual verifier.

Direct specification questions, bounded ideal LED/ADC calculations and explicit
missing-input records and serial configuration comparisons are eligible;
other calculations/explanations go to the model.
A conflict requires an identical explicit label (including any
revision/conditions) in distinct sources and different normalized quantities.
No source commands are interpreted, no inferred circuit values are produced.
"""
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import re

UNITS = {
    'current': {'A': '1', 'mA': '.001', 'uA': '.000001', 'µA': '.000001', 'μA': '.000001', 'nA': '.000000001'},
    'voltage': {'V': '1', 'mV': '.001', 'uV': '.000001', 'µV': '.000001', 'kV': '1000'},
    'frequency': {'Hz': '1', 'kHz': '1000', 'MHz': '1000000', 'GHz': '1000000000'},
    'resistance': {'Ω': '1', 'ohm': '1', 'ohms': '1', 'kΩ': '1000', 'MΩ': '1000000'},
}


def evidence_calculation(question, sources):
    """Return (response mode, text) for supported complete input records only."""
    missing = missing_calculation_inputs(question, sources)
    if missing:
        return 'clarification', missing
    led = led_resistance_calculation(question, sources)
    return ('calculation', led) if led else adc_code_calculation(question, sources)


def serial_configuration_guidance(question, sources):
    """Compare explicit single-port settings, without interpreting firmware.

    This checks supplied configuration, not runtime output or board capability.
    Complete records only; extra facts/instructions and multi-port settings
    defer to native inference. Caller excludes other context and source loss.
    """
    request = (r'\s*(?:Why (?:might|does|is) (?:my |the )?serial (?:text|output) '
               r'(?:look |appear |seem |come out )?(?:garbled|unreadable)'
               r'|(?:Check|Compare) (?:the |my )?serial baud (?:rates|settings))'
               r'[.?]?(?:\s*Use only the provided configuration and suggest one check[.]?)?'
               r'(?:\s*Do not claim you ran the firmware[.]?)?\s*')
    if not re.fullmatch(request, question, re.I) or len(sources) != 1:
        return None
    source = sources[0]
    if not re.fullmatch(r'S[1-9]\d*', source.get('citationId', '')):
        return None
    rate = r'([1-9]\d{0,6})'
    prose = (r'(?:User-shared )?firmware calls Serial\.begin\(' + rate
             + r'\)\. The serial monitor is configured to ' + rate + r' baud\.')
    labels = r'Firmware baud rate:\s*' + rate + r';\s*Serial monitor baud rate:\s*' + rate + r'\.'
    suffix = r'(?: No runtime trace or compiler output is available\.)?'
    text = source.get('snippet', '').strip()
    match = re.fullmatch(prose + suffix, text, re.I) or re.fullmatch(labels + suffix, text)
    if not match:
        return None
    firmware, monitor = map(int, match.groups())
    citation = '[' + source['citationId'] + ']'
    if firmware == monitor:
        return (f'The supplied firmware and monitor settings both specify {firmware} baud {citation}; '
                'there is no baud-rate mismatch in this record. Please share the board/serial-port type '
                'and a sample of the garbled output to investigate further. '
                'Matching settings alone do not establish correct operation; no runtime test was performed.')
    return (f'The supplied firmware setting is {firmware} baud, while the monitor is set to {monitor} baud '
            f'{citation}. This mismatch is a possible cause of garbled text on a baud-dependent serial link: '
            f'check that the monitor matches the firmware at {firmware} baud. '
            'USB CDC ports may ignore the baud setting; the port type is not established here. '
            'This is a configuration check, not a verified runtime diagnosis.')


def missing_calculation_inputs(question, sources):
    """Ask only for explicitly absent inputs in complete single-circuit records.

    This is not absence inference from arbitrary prose. Additional facts,
    conditions or source instructions invalidate these bounded formats. The
    caller also excludes history, project context and shortened evidence.
    """
    if len(sources) != 1 or not re.fullmatch(r'S[1-9]\d*', sources[0].get('citationId', '')):
        return None
    text = sources[0].get('snippet', '').strip()
    citation = '[' + sources[0]['citationId'] + ']'
    number = r'(\d{1,6}(?:\.\d{1,6})?)'
    power_request = (r'\s*(?:Can you )?(?:calculate|determine) the (?:dissipated )?power of '
                     r'(?:this|the) (?:fictional )?resistor[.?]?'
                     r'(?:\s*Ask for the needed input if the data are insufficient[.]?)?\s*')
    if re.fullmatch(power_request, question, re.I):
        resistance = number + r'\s*(ohms?|Ω|kΩ|MΩ)'
        prose = (r'(?:A|The) (?:fictional )?resistor has resistance ' + resistance
                 + r'\. Its voltage drop and current are not (?:given|supplied|specified)\.')
        labels = (r'Resistance:\s*' + resistance
                  + r';\s*Voltage drop: not supplied;\s*Current: not supplied\.')
        match = re.fullmatch(prose, text) or re.fullmatch(labels, text)
        if match and Decimal(match[1]) > 0:
            return (f'Only the resistance is supplied {citation}. Please provide either the voltage across '
                    'the resistor or the current through it; either one is sufficient with the resistance '
                    'to calculate ideal dissipated power using P = V²/R or P = I²R. '
                    'No numeric power can be determined from resistance alone.')
        return None
    led_request = (r'\s*(?:Can you )?(?:determine|calculate) the series resistor for '
                   r'(?:this|the) (?:fictional )?LED circuit(?: from the reference)?[.?]?'
                   r'(?:\s*If not, ask for the missing inputs rather than selecting a value[.]?)?\s*')
    if not re.fullmatch(led_request, question, re.I):
        return None
    voltage = number + r'\s*(mV|V)'
    prefix = r'The LED circuit supply is ' + voltage + r'\. '
    variants = [
        ('both', prefix + r'No LED forward-voltage specification or target LED current is supplied\.'),
        ('current', prefix + r'The LED forward voltage is ' + voltage + r'\. Target LED current is not supplied\.'),
        ('forward', prefix + r'The target LED current is ' + number + r'\s*(uA|µA|μA|mA|A)'
         + r'\. LED forward voltage is not supplied\.'),
    ]
    for missing, pattern in variants:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        supply = Decimal(match[1]) * Decimal(UNITS['voltage'][match[2]])
        if not 0 < supply <= 60:
            return None
        if missing == 'current':
            forward = Decimal(match[3]) * Decimal(UNITS['voltage'][match[4]])
            if not 0 < forward < supply:
                return None
            needed = 'the target LED current'
        elif missing == 'forward':
            current = Decimal(match[3]) * Decimal(UNITS['current'][match[4]])
            if not 0 < current <= Decimal('.1'):
                return None
            needed = 'the LED forward voltage at the intended operating current'
        else:
            needed = 'the LED forward voltage and the target LED current'
        return (f'Please provide {needed}; the reference does not supply that information {citation}. '
                'For a single LED, the ideal series resistance is R = (Vs − Vf) / I. '
                'I cannot select a numeric resistor value from the supplied data.')
    return None


def adc_code_calculation(question, sources):
    """Evaluate an explicit unsigned ADC transfer formula using exact rationals.

    No bit-depth inference, guessed 1023/1024 convention, expression execution,
    saturation assumption or measurement. An unspecified rounding tie needs a
    clarification. The caller excludes other context and shortened excerpts.
    """
    request = (
        r'\s*(?:For (?:this|the) (?:fictional )?ideal ADC, calculate the nearest integer code for the given input'
        r'|(?:Calculate|Compute|Determine) (?:the )?nearest integer code for (?:this|the) (?:fictional )?ideal ADC)'
        r'[.?]?(?:\s*Show one short equation and cite the supplied inputs[.]?)?'
        r'(?:\s*No test has run[.]?)?\s*')
    if not re.fullmatch(request, question, re.I) or len(sources) != 1:
        return None
    item = sources[0]
    if not re.fullmatch(r'S[1-9]\d*', item.get('citationId', '')):
        return None
    voltage = r'(\d{1,6}(?:\.\d{1,6})?)\s*(mV|V)'
    formula = (r'code = \(input/reference\) \* ([1-9]\d{0,7}), rounded to the nearest integer'
               r'(?:, ties (to even|away from zero))?\.')
    prose = (r'The ideal (?:fictional )?ADC has a ' + voltage + r' reference, input voltage '
             + voltage + r', and explicitly defined ' + formula)
    labels = (r'Reference voltage:\s*' + voltage + r';\s*Input voltage:\s*' + voltage
              + r';\s*' + formula)
    match = re.fullmatch(prose, item.get('snippet', '').strip()) or re.fullmatch(labels, item.get('snippet', '').strip())
    if not match:
        return None
    reference, ref_unit, input_voltage, input_unit, multiplier, tie_rule = match.groups()
    reference = Decimal(reference) * Decimal(UNITS['voltage'][ref_unit])
    input_voltage = Decimal(input_voltage) * Decimal(UNITS['voltage'][input_unit])
    multiplier = int(multiplier)
    if not (0 < reference <= 60 and 0 <= input_voltage <= reference and multiplier <= 16777215):
        return None
    raw = Fraction(input_voltage) * multiplier / Fraction(reference)
    integer, remainder = divmod(raw.numerator, raw.denominator)
    tie = 2 * remainder == raw.denominator
    if tie and not tie_rule:
        return ('clarification',
                f'The supplied formula gives exactly {integer}.5 [{item["citationId"]}]. '
                'Please specify the tie-breaking rule (for example, ties to even or away from zero); '
                'nearest integer alone does not select a unique code here. No measurement or simulation was performed.')
    rounded = integer + int(2 * remainder > raw.denominator or
                            tie and (tie_rule == 'away from zero' or integer % 2 == 1))
    # Display rounding is separate from integer rounding; boundary decisions
    # above use exact fractions, even arbitrarily close to a half-code.
    with localcontext() as ctx:
        ctx.prec = 10
        shown = Decimal(raw.numerator) / Decimal(raw.denominator)
    relation = '=' if Fraction(shown) == raw else '≈'
    def plain(value):
        return format(value.normalize(), 'f')
    rounding = f' using ties {tie_rule}' if tie else ''
    return ('calculation',
            f'Using the supplied formula [{item["citationId"]}], code = ({plain(input_voltage)} V / '
            f'{plain(reference)} V) × {multiplier} {relation} {plain(shown)}; '
            f'the nearest integer code is {rounded}{rounding}. '
            'This is ideal arithmetic, not a measured or simulated ADC result.')


def led_resistance_calculation(question, sources):
    """Evaluate a deliberately bounded single-LED calculation, not circuit safety.

    Only a complete, single record in one of two explicit input formats is
    accepted. No free-text quantity guessing, expression evaluation, defaults,
    topology inference or standard resistor selection. Caller must exclude
    truncated evidence and other project/history context before using this.
    """
    request = (r'\s*(?:Calculate|Compute|Determine) (?:the )?(?:ideal )?'
               r'(?:LED series resistance|series resistance for (?:a|one|a single) LED)'
               r'(?: from (?:the )?(?:supplied|given) values)?[.?]?'
               r'(?:\s*Show the arithmetic(?: in at most (?:three|3) sentences)?'
               r'[.;]?(?:\s*do not choose a standard resistor or claim a test[.]?)?)?\s*')
    if not re.fullmatch(request, question, re.I) or len(sources) != 1:
        return None
    item = sources[0]
    if not re.fullmatch(r'S[1-9]\d*', item.get('citationId', '')):
        return None
    # Bounded plain decimals only: reject exponent notation, signs, ranges and
    # tolerances rather than accidentally selecting one of several values.
    number = r'(\d{1,6}(?:\.\d{1,6})?)'
    voltage = number + r'\s*(mV|V)'
    current = number + r'\s*(uA|µA|μA|mA|A)'
    prose = (r'(?:A|The) (?:fictional |ideal )?(?:single-LED |LED )?circuit has (?:a )?'
             + voltage + r' supply, (?:a )?' + voltage
             + r' LED forward (?:drop|voltage) and (?:a )?(?:desired|target) LED current of '
             + current + r'\.')
    labels = (r'Supply voltage:\s*' + voltage + r';\s*LED forward voltage:\s*'
              + voltage + r';\s*Target LED current:\s*' + current + r'\.?')
    text = item.get('snippet', '').strip()
    match = re.fullmatch(prose, text) or re.fullmatch(labels, text)
    if not match:
        return None
    supply, supply_unit, forward, forward_unit, target, current_unit = match.groups()
    # All accepted products are exact at this precision. Non-terminating
    # division is explicitly marked approximate in the answer.
    with localcontext() as ctx:
        ctx.prec = 40
        supply = Decimal(supply) * Decimal(UNITS['voltage'][supply_unit])
        forward = Decimal(forward) * Decimal(UNITS['voltage'][forward_unit])
        target = Decimal(target) * Decimal(UNITS['current'][current_unit])
        if not (0 < forward < supply <= 60 and 0 < target <= Decimal('.1')):
            return None
        resistance = (supply - forward) / target
        # Six significant digits keep small and large derived values readable.
        shown = resistance.quantize(Decimal(1).scaleb(resistance.adjusted() - 5))
        relation = '=' if shown * target == supply - forward else '≈'
        def plain(value):
            return format(value.normalize(), 'f')
        return (f'For the supplied single-LED inputs [{item["citationId"]}], '
                f'R = (Vs − Vf) / I = ({plain(supply)} V − {plain(forward)} V) / {plain(target)} A '
                f'{relation} {plain(shown)} Ω. '
                'This is ideal arithmetic, not a measured or simulated result; resistor ratings and circuit safety are not assessed.')


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


def current_information_without_context(question):
    """Handle direct release lookups only when the caller has no supplied context.

    This does not activate retrieval, interpret links, or answer questions about
    release concepts. History/project/reference data must take the normal path.
    """
    if (len(question.split()) > 40 or re.search(r'[\n`"\':]', question)
            or re.search(r'\b\d+(?:\.\d+)+\b', question)
            or re.search(r'\b(explain|mean|means|compare|calculate|difference|define|provided|supplied|given)\b'
                         r'|\b(according to|based on|notes? says?)\b', question, re.I)):
        return None
    lookup = re.match(r'\s*(?:what|which)\s+(?:(?:is|was|are|were)\s+)?(?:the\s+)?'
                      r'(?:(?:latest|newest|current)\s+)?(?:(?:firmware|software|SDK|driver)\s+)?'
                      r'(?:versions?|releases?)\b', question, re.I)
    release_date = re.match(r'\s*when\b', question, re.I) and re.search(r'\breleased\b', question, re.I)
    if not (lookup or release_date):
        return None
    if (re.search(r'\b(latest|newest|current|currently|today|this week|this month)\b', question, re.I)
            and re.search(r'\b(firmware|software|releases?|released|versions?|drivers?|SDK)\b', question, re.I)):
        return ('I cannot verify the current release from the information provided. '
                'Please provide the official release notes or paste the relevant text from a current official source. '
                'No web search was performed for this answer.')
    return None


def explicit_missing_specification(question, texts, dimension):
    """Recognize explicit absence for a single named part, never infer a value.

    Every mention of the exact part must be in a simple absence statement for
    the requested dimension. Other mentions, headings, aliases and conflicting
    supplied values defer to the model. This is deliberately not entity linking.
    """
    identifiers = {word.casefold(): word for word in re.findall(
        r'\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b', question)
        if len(word) <= 64 and (any(char.isdigit() for char in word)
                              or ('-' in word and word.upper() == word))}
    if len(identifiers) != 1:
        return None
    identifier = next(iter(identifiers.values()))
    target = re.escape(identifier)
    mention = re.compile(r'(?<![\w-])' + target + r'(?![\w-])', re.I)
    # A suffix or alternate punctuation may name a related part. Do not assert
    # absence across that ambiguity, even if another sentence says unavailable.
    canonical = re.sub(r'[^a-z0-9]', '', identifier.casefold())
    for text in texts:
        for word in re.findall(r'\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b', text):
            normalized = re.sub(r'[^a-z0-9]', '', word.casefold())
            if normalized.startswith(canonical) and word.casefold() != identifier.casefold():
                return None
    property_name = r'(?:clock(?: frequency)?|frequency)' if dimension == 'frequency' else dimension
    property_name = r'(?:(?:maximum|minimum|rated|output|input|supply)\s+)*' + property_name
    property_name += r'(?:\s+(?:rating|limit|value|specification))?'
    absence = re.compile(
        r'(?:No\s+' + target + r'\s+' + property_name
        + r'\s+(?:appears|is (?:given|provided|specified|available))'
        + r'|' + target + r'(?:\'s)?\s+' + property_name
        + r'\s+is\s+(?:not (?:given|provided|specified|available)|unavailable))'
        + r'(?:\s+in (?:this|the|these) (?:reference|references|excerpt|excerpts))?[.!]?$', re.I)
    mentioned = False
    for text in texts:
        for sentence in re.split(r'(?<=[.!?])\s+|\n', text):
            if mention.search(sentence):
                mentioned = True
                if not absence.fullmatch(sentence.strip()):
                    return None
                # Missing an input/minimum rating does not establish a gap for
                # an output/maximum question (or an unqualified one).
                qualifiers = {'maximum': r'\b(maximum|max)\b', 'minimum': r'\b(minimum|min)\b',
                              'rated': r'\brated\b', 'output': r'\boutput\b',
                              'input': r'\binput\b', 'supply': r'\bsupply\b'}
                if any(re.search(r'\b' + term + r'\b', sentence, re.I)
                       and not re.search(pattern, question, re.I)
                       for term, pattern in qualifiers.items()):
                    return None
    if mentioned:
        return (f'A numeric {dimension} specification for {identifier} is not provided in the supplied excerpts. '
                f'Please provide the matching {identifier} datasheet section, including the revision and operating conditions. '
                'A value stated for another part cannot establish this specification.')
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
    missing = explicit_missing_specification(question, [text for _, text in texts], dimension)
    if missing:
        return missing
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
