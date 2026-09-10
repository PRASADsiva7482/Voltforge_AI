"""Transparent duplicate signals; numeric/code structure grouping is conservative."""
import hashlib
import json
import re
import unicodedata


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


STOP = set("a an the is are was were be being been to of for and or in on at by with through this that please my your its as it only give answer return explain state describe determine calculate calculation find compute what how which do does can could would should using use".split())
ALIASES = {
    "volts": "voltage", "volt": "voltage", "potential": "voltage", "amps": "current", "amperes": "current",
    "resistors": "resistor", "capacitors": "capacitor", "resistance": "resistor", "ohms": "ohm", "ω": "ohm",
    "repair": "fix", "correct": "fix", "corrected": "fix", "broken": "failure", "error": "failure",
    "sketch": "code", "program": "code", "firmware": "code", "pinout": "pin", "pins": "pin",
    "wires": "wire", "wiring": "wire", "connect": "wire", "connection": "wire", "connections": "wire",
    "subtract": "minus", "minus": "minus", "add": "plus", "sum": "plus",
    "latest": "current", "replace": "change", "changed": "change", "remember": "recall",
}
WORD = re.compile(r"[^\W_]+(?:[._-][^\W_]+)*", re.UNICODE)
CODE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[A-Za-z_]\w*|(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?)(?:[uUlLfF]+)?|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\s]')
KEYWORDS = set("if else while for return break continue switch case default void int float double char bool const static unsigned long short volatile true false class struct def import from pass None True False try except finally lambda in and or not print".split())


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def words(value, *, numbers=True):
    value = unicodedata.normalize("NFKC", value).casefold()
    tokens = [ALIASES.get(token, token) for token in WORD.findall(value) if token not in STOP]
    return tuple("#" if not numbers and re.fullmatch(r"\d+(?:\.\d*)?(?:e[+-]?\d+)?", token) else token for token in tokens)


def code_signature(value, *, numbers=True):
    """Rename identifiers by first occurrence, preserve strings/operators/control flow."""
    names, tokens = {}, []
    for token in CODE.findall(value):
        if token.startswith(("//", "/*")):
            continue
        if token[0] in "\"'":
            tokens.append(token)
        elif re.fullmatch(r"[A-Za-z_]\w*", token) and token not in KEYWORDS:
            names.setdefault(token, "identifier" + str(len(names)))
            tokens.append(names[token])
        elif not numbers and re.fullmatch(r"(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?)(?:[uUlLfF]+)?", token):
            tokens.append("NUMBER")
        else:
            tokens.append(token)
    return tuple(tokens)


def code_blocks(text):
    result = re.findall(r"```[^\n]*\n([\s\S]*?)```", text)
    if not result and (re.search(r"\b(?:void\s+setup|int\s+main|def\s+\w+\s*\(|(?:if|for|while)\s*\()", text)
                       or re.search(r"\b(?:int|void|float|bool)\s+\w+\s*\([^)]*\)\s*\{", text)):
        result = [text]
    return result


def content_segments(value):
    """Inspect content leaves, not shared protocol/system/provenance boilerplate."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {"system", "metadata", "lineage", "sourceProjectRevision", "projectId", "evidenceId", "citationId", "evidenceRefs", "sourceRevision", "contentSha256", "recordId", "type"}:
                yield from content_segments(child)
    elif isinstance(value, list):
        for child in value:
            yield from content_segments(child)


def features(segments):
    text = list(dict.fromkeys(segment for segment in segments if segment.strip()))
    codes = [block for segment in text for block in code_blocks(segment)]
    # Case and whitespace inside code string literals are semantic content. Only
    # prose uses the case-insensitive exact/containment representation.
    exact_text = [unicodedata.normalize("NFC", segment.replace("\r\n", "\n")).strip() if code_blocks(segment)
                  else normalized(segment) for segment in text]
    prose = []
    for segment in text:
        if "```" in segment:
            prose.append(re.sub(r"```[^\n]*\n[\s\S]*?```", " ", segment))
        elif not code_blocks(segment):
            prose.append(segment)
    return {"exact": {sha(segment) for segment in exact_text if len(segment) >= 24},
            "literal": {segment for segment in exact_text if len(segment) >= 24},
            "text": [set(words(segment)) for segment in prose if len(set(words(segment))) >= 6],
            "code": {sha(canonical(code_signature(code))) for code in codes if len(code_signature(code)) >= 12},
            "codeFamily": {sha(canonical(code_signature(code, numbers=False))) for code in codes if len(code_signature(code)) >= 12}}


def compare(left, right, policy, *, protected=False):
    if left["exact"] & right["exact"]:
        return "exact-content"
    if left["code"] & right["code"]:
        return "renamed-code"
    if left["codeFamily"] & right["codeFamily"]:
        return "parameterized-code-family"
    if protected and any(b in a for a in left["literal"] for b in right["literal"]):
        return "protected-exact-containment"
    for a in left["text"]:
        for b in right["text"]:
            common = len(a & b)
            if common / len(a | b) >= policy["nearTextJaccard"]:
                return "lexical-paraphrase"
            if protected and common >= policy["minimumNearWords"] and common / len(b) >= policy["protectedContainment"]:
                return "protected-content-containment"
    return None
