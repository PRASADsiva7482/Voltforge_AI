"""Conservative extraction: retain code whitespace and math, reject uncertain bytes."""
from __future__ import annotations

from collections import Counter
from html import escape, unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))


class Quarantine(ValueError):
    """Only a content-free reason code may cross the builder boundary."""


SECRET_PATTERNS = (
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}\b",
    r"\bAKIA[A-Z0-9]{16}\b",
    r"(?i)[\"']?\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)[\"']?\s*[:=]\s*[\"']?[^\s\"'<>;,}]{8,}",
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}",
    r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@",
)
SECRETS = tuple(re.compile(pattern) for pattern in SECRET_PATTERNS)


def reject_secrets(text):
    if any(pattern.search(text) for pattern in SECRETS):
        raise Quarantine("secret-pattern")


def decode(raw, encoding):
    if encoding not in {"utf-8", "utf-8-sig", "utf-16", "cp1252"}:
        raise Quarantine("unsupported-encoding")
    try:
        text = raw.decode(encoding, errors="strict")
    except UnicodeError:
        raise Quarantine("invalid-encoding") from None
    if "\ufffd" in text or any(unicodedata.category(c) in {"Cc", "Cs", "Cf"} and c not in "\n\r\t" for c in text):
        raise Quarantine("broken-or-hidden-text")
    # Common double-decoding fingerprints; never guess a replacement encoding.
    if any(marker in text for marker in ("Ã©", "â€™", "â€œ", "Âµ", "ï¿½")):
        raise Quarantine("suspected-mojibake")
    return text


class ExtractHTML(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    DROP = {"head", "nav", "footer", "aside", "script", "style", "noscript", "template"}
    BLOCK = {"article", "section", "div", "p", "h1", "h2", "h3", "h4", "li", "ul", "ol", "table", "tr", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack, self.skip = [], [], 0
        self.pre, self.math, self.tex = 0, 0, 0
        self.stats = Counter()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        is_tex = tag == "script" and attributes.get("type", "").startswith("math/tex")
        drop = (tag in self.DROP and not is_tex) or "hidden" in attributes or attributes.get("aria-hidden") == "true"
        if tag not in self.VOID:
            self.stack.append((tag, drop))
        if drop and tag not in self.VOID:
            self.skip += 1
            self.stats["htmlBoilerplateRegions"] += 1
        if self.skip:
            return
        if is_tex:
            self.tex += 1
            self.parts.append("\\[")
        elif tag == "math" or self.math:
            self.math += int(tag not in self.VOID)
            self.parts.append(self.get_starttag_text())
        elif tag == "pre":
            self.pre += 1
            self.parts.append("\n\n<pre>\n")
        elif tag == "code" and not self.pre:
            self.parts.append("`")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK and not self.pre:
            self.parts.append("\n\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")
        elif tag == "sup":
            self.parts.append("^(")
        elif tag == "sub":
            self.parts.append("_(")
        elif tag == "img" and attributes.get("alt"):
            self.parts.append(attributes["alt"])

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        # Accept ordinary HTML omitted paragraph/list ends, but fail on a broken
        # code/math boundary instead of silently losing the rest of a document.
        match = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if match is None:
            raise Quarantine("malformed-html")
        removed = self.stack[match:]
        if (self.math and len(removed) != 1) or any(name in {"pre", "code", "math"} for name, _ in removed[1:]):
            raise Quarantine("malformed-html")
        was_skipped = self.skip > 0
        self.skip -= sum(drop for _, drop in removed)
        del self.stack[match:]
        if was_skipped:
            return
        if tag == "script" and self.tex:
            self.tex -= 1
            self.parts.append("\\]")
        elif self.math:
            self.math -= len(removed)
            self.parts.append("</" + tag + ">")
        elif tag == "pre":
            self.pre -= 1
            self.parts.append("\n</pre>\n\n")
        elif tag == "code" and not self.pre:
            self.parts.append("`")
        elif tag in {"sup", "sub"}:
            self.parts.append(")")
        elif tag in self.BLOCK and not self.pre:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(escape(data, quote=False) if self.math else data if self.pre or self.tex else re.sub(r"\s+", " ", data))

    def result(self):
        self.close()
        if self.skip or self.pre or self.math or self.tex:
            raise Quarantine("malformed-html")
        return "".join(self.parts), dict(self.stats)


def normalize_lines(text, *, code=False):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if code:
        return text, {}
    lines, removed, protected = [], 0, False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")) or line in {"<pre>", "</pre>"}:
            protected = not protected
        if not protected and line.strip() in POLICY["boilerplateLines"]:
            removed += 1
            continue
        # Preserve indentation/tabs and code strings exactly. Empty-line collapse
        # is restricted to prose; no NFKC, unit replacement or code reflow.
        if not protected and not line.strip() and lines and lines[-1] == "":
            continue
        lines.append(line if protected else line.rstrip())
    return "\n".join(lines).strip("\n") + "\n", {"boilerplateLines": removed}


def extract_pdf(raw):
    if len(raw) > POLICY["maxPdfBytes"]:
        raise Quarantine("document-too-large")
    reject_secrets(raw.decode("latin-1"))
    # A disposable offline worker isolates parser failures and imposes a deadline.
    # It emits fixed reason codes, never parser exceptions or source text to logs.
    with tempfile.TemporaryDirectory(prefix="vf-pdf-") as temp:
        source, target = Path(temp) / "input.pdf", Path(temp) / "output.json"
        source.write_bytes(raw)
        try:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "pdf_worker.py"), str(source), str(target)],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=POLICY["pdfTimeoutSeconds"],
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            raise Quarantine("pdf-timeout") from None
        if result.returncode or not target.is_file() or target.stat().st_size > POLICY["maxNormalizedBytes"]:
            raise Quarantine("pdf-extraction-failed")
        value = json.loads(target.read_text(encoding="utf-8"))
        if "reason" in value:
            raise Quarantine(value["reason"])
        return value["text"], value["quality"]


def normalize(raw, media_type, encoding="utf-8"):
    if media_type == "application/pdf":
        text, quality = extract_pdf(raw)
        text = decode(text.encode("utf-8"), "utf-8")
        reject_secrets(text)
        text, counters = normalize_lines(text, code=True)
    else:
        if len(raw) > POLICY["maxDocumentBytes"]:
            raise Quarantine("document-too-large")
        text = decode(raw, encoding)
        reject_secrets(unescape(text))
        quality = {}
        if media_type == "text/html":
            parser = ExtractHTML()
            parser.feed(text)
            text, quality = parser.result()
        elif media_type not in {"text/plain", "text/markdown", "text/x-code", "application/x-vf-task+json"}:
            raise Quarantine("unsupported-media-type")
        text, counters = normalize_lines(text, code=media_type in {"text/x-code", "application/x-vf-task+json"})
    reject_secrets(text)
    if not text.strip():
        raise Quarantine("empty-extraction")
    if len(text.encode("utf-8")) > POLICY["maxNormalizedBytes"]:
        raise Quarantine("normalized-too-large")
    return text, {**quality, **counters}
