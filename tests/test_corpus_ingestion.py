"""Adversarial extraction, lineage, resume and immutable publication contracts."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

import pytest

from data_governance.ingestion import pipeline as p
from data_governance.ingestion.normalization import normalize, Quarantine, POLICY
from tools.build_ingestion_fixtures import make_pdf


def fixture_input(tmp_path, files):
    root = tmp_path / "input"
    root.mkdir()
    rows = []
    for name, raw, media, form in files:
        (root / name).write_bytes(raw)
        rows.append({"path": name, "sha256": p.digest(raw), "format": form, "mediaType": media, "encoding": "utf-8"})
    p.exclusive_json(root / "catalog.json", {"fixtureOnly": True, "trainingAllowed": False, "files": rows})
    return root, p.fixture_plan(root)


def records(release, kind):
    manifest = p.read_json(Path(release) / "manifest.json")
    return [json.loads(line) for shard in manifest["files"] if shard["kind"] == kind
            for line in (Path(release) / shard["path"]).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def reviewed_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("ingestion-fixtures")
    plan = p.fixture_plan()
    with patch("socket.socket.connect", side_effect=AssertionError("Offline build attempted network")):
        result = p.build(plan, p.FIXTURES, root / "out", root / "work")
    return result


def test_fixture_decisions_and_raw_quarantine(reviewed_fixtures):
    accepted = {r["lineage"]["inputPath"]: r for r in records(reviewed_fixtures["releasePath"], "normalized")}
    denied = {r["lineage"]["inputPath"]: r for r in records(reviewed_fixtures["releasePath"], "quarantine")}
    for row in p.read_json(p.FIXTURES / "catalog.json")["files"]:
        if row["expectedOutcome"] == "accepted":
            assert row["path"] in accepted
        else:
            assert denied[row["path"]]["reason"] == row["expectedOutcome"]
    raw = records(reviewed_fixtures["releasePath"], "raw")
    assert len(raw) == len(accepted) == 8
    assert len(denied) == 9
    assert all(row["lineage"]["inputPath"] not in denied for row in raw)
    for row in raw:
        assert base64.b64decode(row["rawBase64"]) == (p.FIXTURES / row["lineage"]["inputPath"]).read_bytes()
    serialized = p.canonical(denied)
    assert "fixture_credential_" not in serialized and "rawBase64" not in serialized


def test_html_math_si_code_and_boilerplate(reviewed_fixtures):
    row = next(r for r in records(reviewed_fixtures["releasePath"], "normalized") if r["lineage"]["inputPath"] == "01-si.html")
    for expected in ("2.2 kΩ", "100 µF", "25 °C", "E = mc^(2)", "V_(out)", r"\(X_C=1/(2\pi fC)\)",
                     "<math><mfrac><mi>V</mi><mi>R</mi></mfrac></math>", "    digitalWrite(13, HIGH);"):
        assert expected in row["text"]
    assert "Navigation only" not in row["text"] and "Footer only" not in row["text"]


def test_pdf_code_math_and_repeated_margins(reviewed_fixtures):
    data = {r["lineage"]["inputPath"]: r for r in records(reviewed_fixtures["releasePath"], "normalized")}
    assert "V = I * R; R = 220 ohm; I = 0.01 A" in data["07-code-math.pdf"]["text"]
    assert "\n    digitalWrite(13, HIGH);\n" in data["07-code-math.pdf"]["text"]
    assert data["07-code-math.pdf"]["quality"]["needsExtractionReview"] is True
    assert "Repeated synthetic document" not in data["17-margins.pdf"]["text"]
    assert data["17-margins.pdf"]["text"].count("Formula: V = I * R") == 3


def test_unicode_nfc_does_not_fold_units_superscripts_or_indentation():
    text, _ = normalize("Cafe\u0301: 3 µA, 10 Ω, x².\r\n".encode(), "text/plain")
    assert text == "Café: 3 µA, 10 Ω, x².\n"
    code = "\tif ok:\r\n\t\tvalue = 'space  '\r\n"
    assert normalize(code.encode(), "text/x-code")[0] == code.replace("\r\n", "\n")


@pytest.mark.parametrize("encoding,text", [("utf-16", "R = 330 Ω"), ("cp1252", "C = 10 µF")])
def test_encoding_is_explicit_and_strict(encoding, text):
    assert normalize(text.encode(encoding), "text/plain", encoding)[0] == text + "\n"


@pytest.mark.parametrize("text", ["\ufffd", "abc\x00def", "a\u200bb", "a\u202eb", "Ã©"])
def test_broken_or_hidden_text_is_quarantined(text):
    with pytest.raises(Quarantine):
        normalize(text.encode(), "text/plain")


@pytest.mark.parametrize("text", ["password = " + "fabricated" * 3, "Bearer " + "a" * 30,
                                   "https://fixture:madeup@host.invalid", "ghp_" + "z" * 25,
                                   "-----BEGIN " + "PRIVATE KEY-----"])
def test_secret_shapes_are_quarantined_without_echo(text):
    with pytest.raises(Quarantine, match="^secret-pattern$"):
        normalize(text.encode(), "text/plain")


def test_entity_escaped_hidden_html_credentials_rejected():
    with pytest.raises(Quarantine, match="secret-pattern"):
        normalize(b"<nav>password&#61;syntheticcredential</nav><p>Safe</p>", "text/html")


@pytest.mark.parametrize("html", [b"<pre><code>never closed", b"<div><pre>x</div>", b"<math><mi>R</math>"])
def test_unclosed_code_and_math_do_not_silently_flatten(html):
    with pytest.raises(Quarantine, match="malformed-html"):
        normalize(html, "text/html")


def test_pdf_limits_and_encryption():
    with pytest.raises(Quarantine, match="document-too-large"):
        normalize(b"x" * (POLICY["maxPdfBytes"] + 1), "application/pdf")
    with pytest.raises(Quarantine, match="pdf-encrypted"):
        normalize(make_pdf(["Encrypted fixture"], encrypted=True), "application/pdf")
    with pytest.raises(Quarantine, match="pdf-too-many-pages"):
        normalize(make_pdf([], blank=True, pages=101), "application/pdf")


def test_pdf_timeout_has_no_parser_excerpt(monkeypatch):
    import subprocess
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("parser", 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(Quarantine, match="^pdf-timeout$"):
        normalize(b"%PDF fixture", "application/pdf")


def test_streamed_oversized_line_does_not_swallow_next_document(tmp_path):
    raw = b"x" * (POLICY["maxDocumentBytes"] * 3) + b"\nValid next document\n"
    root, plan = fixture_input(tmp_path, [("data.jsonl", raw, "text/plain", "jsonl")])
    docs = list(p.documents(root / "data.jsonl", plan["units"][0]))
    assert docs[0][1] is None and docs[0][2] == p.digest(raw.splitlines(keepends=True)[0])
    assert docs[1][0] == "line:2" and docs[1][1] == b"Valid next document\n"


def test_resume_exact_duplicate_handling_matches_fresh_build(tmp_path):
    root, plan = fixture_input(tmp_path, [("data.jsonl", b"First line\nSecond line\nFirst line\nThird line\n", "text/plain", "jsonl")])
    paused = p.build(plan, root, tmp_path / "out", tmp_path / "work", stop_after=2)
    assert paused["status"] == "interrupted-for-resume-test"
    assert not list((tmp_path / "out").glob("*/manifest.json"))
    resumed = p.build(plan, root, tmp_path / "out", tmp_path / "work")
    fresh = p.build(plan, root, tmp_path / "fresh", tmp_path / "fresh-work")
    assert resumed["resumedDocuments"] == 2 and resumed["newDocuments"] == 2
    assert resumed["manifestSha256"] == fresh["manifestSha256"]
    assert resumed["acceptedDocuments"] == 3 and resumed["quarantinedDocuments"] == 1
    assert p.verify(resumed["releasePath"], recompute=True, input_root=root)["recomputed"]


def test_completed_build_is_idempotent_and_tamper_fails(tmp_path):
    root, plan = fixture_input(tmp_path, [("one.txt", b"Independent fixture text", "text/plain", "document")])
    first = p.build(plan, root, tmp_path / "out", tmp_path / "work")
    assert p.build(plan, root, tmp_path / "out", tmp_path / "work")["status"] == "already-complete"
    release = Path(first["releasePath"])
    path = next((release / "normalized").glob("*.jsonl"))
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        p.build(plan, root, tmp_path / "out", tmp_path / "work")


def test_journal_tampering_denied_before_resume(tmp_path):
    root, plan = fixture_input(tmp_path, [("data.jsonl", b"one\ntwo\n", "text/plain", "jsonl")])
    p.build(plan, root, tmp_path / "out", tmp_path / "work", stop_after=1)
    with sqlite3.connect(next((tmp_path / "work").glob("*.sqlite3"))) as db:
        db.execute("UPDATE documents SET raw_sha='tampered'")
    with pytest.raises(ValueError, match="Resume journal checksum"):
        p.build(plan, root, tmp_path / "out", tmp_path / "work")


def test_changed_inputs_denied_before_resume(tmp_path):
    root, plan = fixture_input(tmp_path, [("data.jsonl", b"one\ntwo\n", "text/plain", "jsonl")])
    p.build(plan, root, tmp_path / "out", tmp_path / "work", stop_after=1)
    (root / "data.jsonl").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        p.build(plan, root, tmp_path / "out", tmp_path / "work")


def test_changed_parser_policy_cannot_resume(tmp_path):
    root, plan = fixture_input(tmp_path, [("data.txt", b"one", "text/plain", "document")])
    plan["parser"]["version"] = "unreviewed"
    with pytest.raises(ValueError, match="plan changed"):
        p.build(plan, root, tmp_path / "out", tmp_path / "work")


@pytest.mark.parametrize("name", ["../outside.txt", "C:/outside", "file.txt:stream", "//server/share"])
def test_fixture_manifest_cannot_escape_input_root(tmp_path, name):
    p.exclusive_json(tmp_path / "catalog.json", {"files": [{"path": name, "sha256": "0" * 64}]})
    with pytest.raises(ValueError, match="path"):
        p.fixture_plan(tmp_path)


def test_concurrent_writer_is_rejected_and_lock_released(tmp_path):
    lock = tmp_path / "build.lock"
    with p.build_lock(lock):
        with pytest.raises(ValueError, match="locked"):
            with p.build_lock(lock):
                pass
    with p.build_lock(lock):
        pass


def test_partial_export_can_resume_without_overwriting(tmp_path, monkeypatch):
    root, plan = fixture_input(tmp_path, [("data.txt", b"publication crash fixture", "text/plain", "document")])
    original = p.write_immutable
    calls = []
    def crash(path, data):
        original(path, data)
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("simulated publication crash")
    monkeypatch.setattr(p, "write_immutable", crash)
    with pytest.raises(RuntimeError):
        p.build(plan, root, tmp_path / "out", tmp_path / "work")
    old_bytes = calls[0].read_bytes()
    monkeypatch.setattr(p, "write_immutable", original)
    result = p.build(plan, root, tmp_path / "out", tmp_path / "work")
    assert result["verification"] == "passed" and calls[0].read_bytes() == old_bytes


@pytest.fixture(scope="module")
def approved_build(tmp_path_factory):
    target = tmp_path_factory.mktemp("source-ingestion")
    plan = p.approved_plan()
    result = p.build(plan, p.AI, target / "out", target / "work")
    return plan, result


def test_only_227_inventoried_records_enter_staging(approved_build):
    plan, result = approved_build
    assert len(plan["units"]) == 4
    assert result["acceptedDocuments"] == 227 and result["quarantinedDocuments"] == 23
    normalized = records(result["releasePath"], "normalized")
    assert {row["lineage"]["recordId"] for row in normalized} == {row["recordId"] for row in plan["eligibleRecords"]}
    assert all(row["trainingAllowed"] is False for row in normalized)
    assert all(row["lineage"]["permissionReceipts"] for row in normalized)
    assert all(r["reason"] == "not-in-inventoried-training-pool" for r in records(result["releasePath"], "quarantine"))


def test_source_permission_cannot_admit_substituted_file(tmp_path, approved_build):
    plan = deepcopy(approved_build[0])
    plan["units"][0]["path"] = "dataset.txt"
    with pytest.raises(ValueError, match="admission plan changed"):
        p.build(plan, p.AI, tmp_path / "out", tmp_path / "work")


def test_ingestion_manifest_cannot_grant_training(tmp_path, reviewed_fixtures):
    source = Path(reviewed_fixtures["releasePath"])
    dest = tmp_path / source.name
    shutil.copytree(source, dest)
    manifest = p.read_json(dest / "manifest.json")
    manifest["trainingAllowed"] = True
    (dest / "manifest.json").write_text(p.canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize training"):
        p.verify(dest)


def test_unknown_media_and_ambiguous_encoding_quarantined():
    with pytest.raises(Quarantine, match="unsupported-media-type"):
        normalize(b"fixture", "application/unknown")
    with pytest.raises(Quarantine, match="unsupported-encoding"):
        normalize(b"fixture", "text/plain", "guess")


def test_fenced_code_blank_lines_are_preserved():
    raw = b"Code:\n```python\n    print('x')\n\n\n```\n"
    assert normalize(raw, "text/markdown")[0] == raw.decode()


def test_mathjax_tex_survives_while_executable_script_is_removed():
    raw = br'<p>Impedance:</p><script type="math/tex">X_C=\frac{1}{2\pi fC}</script><script>tracking()</script>'
    text, _ = normalize(raw, "text/html")
    assert r'\[X_C=\frac{1}{2\pi fC}\]' in text
    assert "tracking" not in text
