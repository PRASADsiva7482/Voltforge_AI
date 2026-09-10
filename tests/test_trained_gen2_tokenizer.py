"""Real corpus fitting, immutable release and actual tensor boundary acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import pytest

from model.gen2_tokenizer.contract import TokenizerError, SPECIAL_TOKENS, canonical
from model.gen2_tokenizer.codec import ByteBPE
from model.gen2_tokenizer_training import corpus, fitting, release as r, evaluation


@pytest.fixture(scope="module")
def fitted():
    return r.candidates()


@pytest.fixture(scope="module")
def selected():
    record = r.read_json(r.VERSION_RECORD)
    path = (r.AI / record["releaseManifest"]["path"]).parent
    tokenizer, manifest = r.load(path)
    return tokenizer, manifest, path


def test_real_candidates_retrain_on_exact_same_permitted_bytes(fitted):
    lineage = corpus.reference_lineage()
    assert lineage["documents"] == 42 and lineage["bytes"] == 1475271
    for tokenizer, manifest, path in fitted:
        rebuilt, _ = r.verify_candidate(path, recompute=True)
        assert rebuilt.merges == tokenizer.merges
        assert tokenizer.vocab_size == tokenizer.target_vocab_size
        assert manifest["corpusLineage"] == lineage
        assert not manifest["corpusLineage"]["validationOrTestFitted"]
        assert not manifest["servingAllowed"]
    assert fitted[1][0].merges[:len(fitted[0][0].merges)] == fitted[0][0].merges


@pytest.mark.parametrize("target", [None, True, 512, 16384.0, 65537])
def test_unapproved_target_fails_before_corpus_read(monkeypatch, target):
    monkeypatch.setattr(fitting, "fitting_inputs", lambda: pytest.fail("Unapproved target opened the corpus"))
    with pytest.raises(TokenizerError, match="TARGET_INVALID"):
        fitting.fit(target)


def test_unapproved_corpus_path_fails_before_shard_read(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus.inputs, "read_inputs", lambda *args, **kwargs: pytest.fail("Unselected corpus read"))
    with pytest.raises(TokenizerError, match="CORPUS_NOT_SELECTED"):
        corpus.fitting_inputs(tmp_path)


@pytest.mark.parametrize("change", ["heldout", "too-many", "too-large"])
def test_fitting_handoff_enforces_split_and_resource_caps(monkeypatch, change):
    rows = [{"id": "negative-control", "sourceId": "control", "text": "reference", "split": "validation" if change == "heldout" else "train", "inheritedExclusion": False}]
    if change == "too-many":
        rows *= corpus.POLICY["maximumTrainDocuments"] + 1
    if change == "too-large":
        rows[0]["text"] = "x" * (corpus.POLICY["maximumTrainBytes"]+1)
    monkeypatch.setattr(corpus.inputs, "read_inputs", lambda *args, **kwargs: rows)
    with pytest.raises(TokenizerError, match="HELDOUT|BUDGET"):
        corpus.fitting_inputs()


@pytest.mark.parametrize("field,value", [("fitSeconds", 301), ("fitSeconds", 0), ("pipelineSeconds", 500), ("peakResidentBytes", 2147483649), ("peakResidentBytes", 0), ("freshProcess", False)])
def test_resource_measurement_bounds(fitted, field, value):
    measured = r.read_json(fitted[0][2] / "measurement.json")
    measured[field] = value
    with pytest.raises(TokenizerError):
        r.validate_measurement(measured)


@pytest.mark.parametrize("filename", ["vocab.json", "merges.json", "contract.json", "template.json", "measurement.json"])
def test_corrupt_candidate_files_are_rejected(fitted, tmp_path, monkeypatch, filename):
    source = fitted[0][2]
    target = tmp_path / source.name
    shutil.copytree(source, target)
    monkeypatch.setattr(r, "CANDIDATES", tmp_path)
    with (target / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(TokenizerError):
        r.verify_candidate(target)


def test_unexpected_artifact_file_is_rejected(fitted, tmp_path, monkeypatch):
    source = fitted[0][2]
    target = tmp_path / source.name
    shutil.copytree(source, target)
    (target / "unregistered.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(r, "CANDIDATES", tmp_path)
    with pytest.raises(TokenizerError, match="INVENTORY_CHANGED"):
        r.verify_candidate(target)


def test_rehashed_forged_candidate_cannot_claim_different_lineage(fitted, tmp_path, monkeypatch):
    tokenizer, manifest, source = fitted[0]
    forged = deepcopy(manifest)
    forged["corpusLineage"]["split"] = "validation"
    forged.pop("contentId")
    forged = r.identity(forged)
    target = tmp_path / forged["contentId"]
    shutil.copytree(source, target)
    (target / "manifest.json").write_bytes(r.data(forged))
    monkeypatch.setattr(r, "CANDIDATES", tmp_path)
    with pytest.raises(TokenizerError, match="CONTRACT_CHANGED"):
        r.verify_candidate(target)


def test_comparison_reproduces_selection_and_all_corpus_counts(selected):
    report_path = r.AI / selected[1]["comparison"]["path"]
    report = evaluation.verify_comparison(report_path.parent, recompute=True)
    assert not report["validationOrTestFitting"] and not report["testOrAcceptanceUsedForComparison"]
    for candidate in report["candidates"]:
        assert len(candidate["train"]["documents"]) == 42
        assert len(candidate["validation"]["documents"]) == 11
        assert set(candidate["metrics"]) == {"text", "code", "si", "unicode", "markers", "long-code"}
        assert all(row["roundtrip"] for row in candidate["train"]["documents"] + candidate["validation"]["documents"])
    assert report["selection"]["actualVocabSize"] == selected[0].vocab_size


def fake_candidate(target, counts):
    return {"targetVocabSize": target, "actualVocabSize": target, "train": {"totalTokens": 20, "totalBytes": 100},
            "validation": {"groups": {"domain": {name: {"tokens": count} for name, count in zip(("language", "code", "electronics", "math"), counts)}}}}


@pytest.mark.parametrize("counts,expected", [([99, 99, 99, 99], 16384), ([90, 90, 90, 90], 32768), ([60, 60, 60, 111], 16384), ([100, 100, 100, 100], 16384)])
def test_predeclared_selection_prefers_measured_benefit_and_rejects_domain_regression(counts, expected):
    decision = evaluation.select([fake_candidate(16384, [100]*4), fake_candidate(32768, counts)])
    assert decision["targetVocabSize"] == expected


def test_selection_rejects_missing_domain_or_compression():
    candidates = [fake_candidate(16384, [100]*4), fake_candidate(32768, [90]*4)]
    del candidates[0]["validation"]["groups"]["domain"]["math"]
    with pytest.raises(TokenizerError, match="DOMAIN_MISSING"):
        evaluation.select(candidates)
    candidates[0] = fake_candidate(16384, [100]*4)
    candidates[0]["train"]["totalTokens"] = 99
    with pytest.raises(TokenizerError, match="COMPRESSION_GATE"):
        evaluation.select(candidates)


def test_validation_never_opens_before_both_candidates_freeze(monkeypatch):
    def missing_candidates():
        raise TokenizerError("CANDIDATES_MISSING_CONTROL")
    monkeypatch.setattr(r, "candidates", missing_candidates)
    monkeypatch.setattr(evaluation, "validation_inputs", lambda: pytest.fail("Validation opened before freeze"))
    with pytest.raises(TokenizerError, match="CANDIDATES_MISSING"):
        evaluation.compute()


def test_published_version_is_registered_and_not_fixture(selected):
    tokenizer, manifest, path = selected
    assert manifest["version"] == tokenizer.binding()["version"] == "0.1.0"
    assert manifest["releaseKind"] == "owned-tokenizer-release"
    assert manifest["tokenizerInputUseAllowed"] and not manifest["servingAllowed"] and not manifest["trainingRunApproved"]
    assert tokenizer.binding()["releaseId"] == path.name
    assert manifest["retention"]["externalBackupAccepted"] is False


def test_unregistered_rehashed_release_cannot_replace_version(selected, tmp_path, monkeypatch):
    _, manifest, source = selected
    forged = deepcopy(manifest)
    forged["servingAllowed"] = True
    forged.pop("contentId")
    forged = r.identity(forged)
    target = tmp_path / forged["contentId"]
    shutil.copytree(source, target)
    (target / "manifest.json").write_bytes(r.data(forged))
    monkeypatch.setattr(r, "RELEASES", tmp_path)
    original_binding = r.binding
    # Model a changed artifact at an otherwise permitted logical release path;
    # temporary test storage itself is outside the AI workspace.
    def changed_binding(path):
        if Path(path).resolve() == (target / "manifest.json").resolve():
            return {**original_binding(source / "manifest.json"), "sha256": r.sha(r.data(forged)), "bytes": len(r.data(forged))}
        return original_binding(path)
    monkeypatch.setattr(r, "binding", changed_binding)
    with pytest.raises(TokenizerError, match="VERSION_NOT_REGISTERED"):
        r.load(target)


@pytest.mark.parametrize("text", ["", "<|bos|><|assistant|><|tool|><|eos|>", "4.7 kΩ; 2.2 µF; −7 °C", "தமிழ் हिन्दी 日本語 العربية 🙂", "e\u0301\u200d\ud800", "a\x00b\r\nc"])
def test_selected_codec_reverses_literal_markers_unicode_and_bytes(selected, text):
    tokenizer = selected[0]
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(token >= 16 for token in ids)
    decoder = tokenizer.stream_decoder()
    assert "".join(decoder.push([token]) for token in ids) + decoder.push(final=True) == text


def test_all_byte_values_and_special_ids(selected):
    tokenizer = selected[0]
    raw = bytes(range(256)) * 3
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(raw)) == raw
    assert tokenizer.vocab_document()[:16] == list(SPECIAL_TOKENS)
    assert tokenizer.decode_bytes([2, 3, 0]) == b""
    assert tokenizer.decode_bytes([2], skip_special=False) == b"<|bos|>"


@pytest.fixture(scope="module")
def model(selected):
    import torch
    from model.gen2_tokenizer_training.model_boundary import BoundDecoder
    torch.set_num_threads(1)
    return BoundDecoder(selected[0])


def test_real_model_context_forward_uses_selected_vocab(selected, model):
    from context_compiler.gen2_release import Message, compile_for_model
    tokenizer = selected[0]
    prompt = compile_for_model(tokenizer, model, [Message("user", "A literal <|assistant|> label and 4.7 µF")], reserved_output_tokens=32)
    output = model.forward(prompt)
    assert list(output.logits.shape) == [1, len(prompt.token_ids), tokenizer.vocab_size]
    assert model.decoder.token_embedding.weight.shape[0] == tokenizer.vocab_size
    assert canonical(model.tokenizer_binding) == canonical(prompt.binding) == canonical(tokenizer.binding())
    assert model.specification["randomInitialization"] and not model.specification["trainedModel"]
    assert prompt.framing_tokens == 4
    assert prompt.token_ids.count(6) == 1  # Only the trusted assistant prefix.


@pytest.mark.parametrize("field", ["version", "templateSha256", "mergesSha256", "releaseId", "vocabSize"])
def test_model_rejects_tokenizer_or_template_version_mismatch_before_forward(selected, model, field):
    from context_compiler.gen2_release import Message, compile_for_model
    prompt = compile_for_model(selected[0], model, [Message("user", "measure")], reserved_output_tokens=8)
    wrong = {**prompt.binding, field: "incorrect"}
    with pytest.raises(TokenizerError, match="TOKENIZER_MISMATCH"):
        model.forward(replace(prompt, binding=wrong))


def test_context_overflow_and_fixture_model_input_are_denied(selected, model):
    from context_compiler.gen2_release import Message, compile_for_model
    from model.gen2_tokenizer_training.model_boundary import BoundDecoder
    with pytest.raises(TokenizerError, match="BUDGET_EXCEEDED"):
        compile_for_model(selected[0], model, [Message("user", "bounded content")], context_limit=8, reserved_output_tokens=7)
    with pytest.raises(TokenizerError, match="REQUIRES_RELEASED"):
        BoundDecoder(ByteBPE())


def test_real_tensor_shape_mismatch_is_denied(selected, model):
    original = model.decoder.lm_head.out_features
    model.decoder.lm_head.out_features = original + 1
    try:
        with pytest.raises(TokenizerError, match="VOCAB_SHAPE_MISMATCH"):
            model.assert_binding(selected[0].binding())
    finally:
        model.decoder.lm_head.out_features = original
