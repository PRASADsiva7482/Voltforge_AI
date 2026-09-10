"""Author deterministic extraction regression fixtures; never training data."""
import hashlib
import io
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))


def make_pdf(lines, *, blank=False, encrypted=False, pages=1):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, NumberObject
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        if blank:
            continue
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Courier")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        content = "BT /F1 12 Tf 14 TL 72 720 Td\n"
        for i, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content += ("T* " if i else "") + "(" + escaped + ") Tj\n"
        stream.set_data((content + "ET\n").encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("synthetic-fixture-password")
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def main():
    target = AI / "tests/fixtures/ingestion/v1"
    if target.exists():
        raise ValueError("Fixture version exists; do not overwrite")
    target.mkdir(parents=True)
    code = "def current(voltage, resistance):\r\n\tif resistance > 0:\r\n\t\treturn voltage / resistance\r\n"
    html = '<!doctype html><html><head><title>fixture</title></head><body><nav>Navigation only</nav><article><h1>Owned extraction fixture</h1><p>I = V/R; R = 2.2 kΩ; C = 100 µF; temperature = 25 °C.</p><p>E = mc<sup>2</sup>, V<sub>out</sub> = 3.3 V; \\(X_C=1/(2\\pi fC)\\).</p><math><mfrac><mi>V</mi><mi>R</mi></mfrac></math><pre><code>if (ready) {\n    digitalWrite(13, HIGH);\n}\n</code></pre></article><footer>Footer only</footer></body></html>'
    fixtures = [
        ("01-si.html", html.encode(), "text/html", "utf-8", "accepted"),
        ("02-indent.py.txt", code.encode(), "text/x-code", "utf-8", "accepted"),
        ("03-unicode.txt", "Cafe\u0301; R = 10 kΩ; C = 4.7 µF; x² + y² = z².\r\nAccept all cookies\r\n".encode(), "text/plain", "utf-8", "accepted"),
        ("04-unicode-duplicate.txt", "Café; R = 10 kΩ; C = 4.7 µF; x² + y² = z².\n".encode(), "text/plain", "utf-8", "exact-normalized-duplicate"),
        ("05-utf16.txt", "UTF16 source: 220 Ω and 10 µA.\r\n".encode("utf-16"), "text/plain", "utf-16", "accepted"),
        ("06-cp1252.txt", "Legacy source: 47 µF, 25 °C.\r\n".encode("cp1252"), "text/plain", "cp1252", "accepted"),
        ("07-code-math.pdf", make_pdf(["PDF extraction fixture", "V = I * R; R = 220 ohm; I = 0.01 A", "if (ready) {", "    digitalWrite(13, HIGH);", "}"]), "application/pdf", "binary", "accepted"),
        ("08-empty.pdf", make_pdf([], blank=True), "application/pdf", "binary", "pdf-ocr-or-review-required"),
        ("09-broken.pdf", b"%PDF-1.7\ninvalid fixture", "application/pdf", "binary", "pdf-extraction-failed"),
        ("10-invalid-utf8.txt", b"broken \xff text", "text/plain", "utf-8", "invalid-encoding"),
        ("11-hidden.txt", "Text with \u202ehidden direction".encode(), "text/plain", "utf-8", "broken-or-hidden-text"),
        ("12-secret.txt", ("api_" + "key = '" + "fixture_credential_" + "A" * 24 + "'\n").encode(), "text/plain", "utf-8", "secret-pattern"),
        ("13-secret-hidden.html", ('<nav>password=' + 'fixture_' * 5 + '</nav><p>Otherwise safe text</p>').encode(), "text/html", "utf-8", "secret-pattern"),
        ("14-broken-code.html", b"<pre><code>  unfinished block", "text/html", "utf-8", "malformed-html"),
        ("15-boilerplate.html", b"<nav>Menu</nav><footer>Footer</footer>", "text/html", "utf-8", "empty-extraction"),
        ("16-fenced.md", b"Code sample:\r\n```python\r\n    print('keep indentation')\r\n\r\n\r\n```\r\nEquation: V = I * R.\r\n", "text/markdown", "utf-8", "accepted"),
        ("17-margins.pdf", make_pdf(["Repeated synthetic document header", "Formula: V = I * R", "Repeated synthetic document footer"], pages=3), "application/pdf", "binary", "accepted"),
    ]
    files = []
    for name, data, media, encoding, expected in fixtures:
        (target / name).write_bytes(data)
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "format": "document", "mediaType": media,
                      "encoding": encoding, "expectedOutcome": expected})
    (target / "catalog.json").write_text(json.dumps({"schemaVersion": 1, "fixtureOnly": True, "trainingAllowed": False,
        "authorship": "Authored deterministic parser test material; credentials are fabricated sentinels, never service configuration.",
        "files": files}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"fixtureFiles": len(files), "trainingAllowed": False}))


if __name__ == "__main__":
    main()
