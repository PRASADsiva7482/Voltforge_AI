"""Isolated PDF text extraction; images/uncertain content require later review."""
from collections import Counter
import io
import json
import logging
from pathlib import Path
import sys


def extract(path, policy):
    import pypdf
    from pypdf import filters
    if pypdf.__version__ != policy["pypdfVersion"]:
        return {"reason": "pdf-parser-version-mismatch"}
    # Bound a single decompressed stream; document/page/output bounds are separate.
    filters.ZLIB_MAX_OUTPUT_LENGTH = policy["maxPdfStreamBytes"]
    logging.disable(logging.CRITICAL)
    reader = pypdf.PdfReader(io.BytesIO(path.read_bytes()), strict=True)
    if reader.is_encrypted:
        return {"reason": "pdf-encrypted"}
    if len(reader.pages) > policy["maxPdfPages"]:
        return {"reason": "pdf-too-many-pages"}
    pages, size = [], 0
    for page in reader.pages:
        contents = page.get_contents()
        if contents is None:
            return {"reason": "pdf-ocr-or-review-required"}
        if contents is not None and len(contents.get_data()) > policy["maxPdfStreamBytes"]:
            return {"reason": "pdf-stream-too-large"}
        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
        if not text.strip():
            return {"reason": "pdf-ocr-or-review-required"}
        size += len(text.encode("utf-8"))
        if size > policy["maxNormalizedBytes"] // 2:
            return {"reason": "normalized-too-large"}
        pages.append(text)
    if not pages:
        return {"reason": "empty-extraction"}
    # Only identical prose margins on >=3 pages are treated as boilerplate.
    edges = Counter(line for page in pages for line in (page.splitlines()[0], page.splitlines()[-1]))
    repeated = {line for line, n in edges.items() if len(pages) >= 3 and n >= len(pages) and len(line.strip()) >= 20
                and not any(mark in line for mark in ("=", "{", "}", ";", "\\", "(", ")"))}
    cleaned, removed = [], 0
    for page in pages:
        lines = page.splitlines()
        kept = [line for i, line in enumerate(lines) if not (i in {0, len(lines) - 1} and line in repeated)]
        removed += len(lines) - len(kept)
        cleaned.append("\n".join(kept))
    return {"text": "\n\n".join(cleaned) + "\n", "quality": {"pdfPages": len(pages), "pdfMarginLinesRemoved": removed,
            "needsExtractionReview": True, "ocrPerformed": False}}


if __name__ == "__main__":
    policy = json.loads(Path(__file__).with_name("policy.v1.json").read_text(encoding="utf-8"))
    try:
        value = extract(Path(sys.argv[1]), policy)
    except Exception:
        value = {"reason": "pdf-extraction-failed"}
    Path(sys.argv[2]).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
