"""LLM-TASK-016 containment, compatibility and preserved tool contracts."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from engine.deterministic_tools import DeterministicElectronicsTools
from model.registry_manager import RuntimeContract, RegistryManagerError, validate_compatibility


AI = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("field", ["pythonAbi", "torchVersion", "tokenizerId", "runtime", "schemaVersion"])
def test_environment_flag_cannot_bypass_artifact_compatibility(monkeypatch, field):
    monkeypatch.setenv("VOLTFORGE_AI_ALLOW_CROSS_ENV_COMPATIBILITY", "true")
    contract = RuntimeContract.current()
    incompatible = deepcopy(contract.as_compatibility())
    incompatible[field] = "not-the-verified-runtime"
    with pytest.raises(RegistryManagerError, match="compatibility"):
        validate_compatibility(incompatible, contract)


def test_unvalidated_torch_upgrade_flag_cannot_import_the_native_model(monkeypatch):
    monkeypatch.setenv("VOLTFORGE_AI_ALLOW_TORCH_UPGRADE", "true")
    spec = importlib.util.spec_from_file_location("_unvalidated_gen1", AI / "model/gen1/model.py")
    module = importlib.util.module_from_spec(spec)
    with patch("importlib.metadata.version", return_value="9.0.0"):
        with pytest.raises(RuntimeError, match="requires PyTorch"):
            spec.loader.exec_module(module)


def test_activation_shortcut_import_is_inert_and_execution_fails():
    path = AI / "tools/activate_domain_artifact.py"
    spec = importlib.util.spec_from_file_location("_retired_activation", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(Path, "write_text", side_effect=AssertionError("write attempted")), patch.object(
        Path, "write_bytes", side_effect=AssertionError("write attempted")
    ):
        spec.loader.exec_module(module)
        assert module.main() == 2
    result = subprocess.run([sys.executable, "-B", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["changedFiles"] == 0


@pytest.mark.parametrize("r,c", [(1000, 1e-6), (4700, 100e-9), (22000, 10e-9)])
def test_preserved_rc_tool_matches_independent_formula(r, c):
    assert DeterministicElectronicsTools.rc_cutoff(r, c)["cutoffFrequencyHz"] == pytest.approx(1 / (2 * math.pi * r * c), abs=0.005)


@pytest.mark.parametrize("r1,r2,c", [(1000, 2000, 1e-6), (10000, 47000, 10e-6), (2200, 3300, 100e-9)])
def test_preserved_timer_tool_matches_independent_formula(r1, r2, c):
    result = DeterministicElectronicsTools.timer_astable(r1, r2, c)
    assert result["frequency_Hz"] == pytest.approx(1 / (0.693 * (r1 + 2 * r2) * c), abs=0.00005)


def test_capacitive_reactance_changes_with_inputs():
    tools = DeterministicElectronicsTools()
    first = tools.reason_and_solve("Calculate capacitive reactance for 100nF at 1kHz")
    second = tools.reason_and_solve("Calculate capacitive reactance for 100nF at 2kHz")
    assert first["calculation"]["reactanceOhm"] == pytest.approx(1591.5494309189535)
    assert second["calculation"]["reactanceOhm"] == pytest.approx(first["calculation"]["reactanceOhm"] / 2)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_tool_inputs_cannot_produce_physical_claims(value):
    with pytest.raises(ValueError):
        DeterministicElectronicsTools.rc_cutoff(value, 1e-6)


def test_hardware_lookup_uses_exact_corpus_and_never_invents_model_identity():
    tools = DeterministicElectronicsTools()
    exact = tools.lookup_hardware("board", "ARDUINO_UNO")
    assert exact["status"] == "found"
    assert exact["records"][0]["supportStatus"] == "supported"
    missing = tools.lookup_hardware("board", "UNKNOWN_BOARD_123")
    assert missing["status"] == "unknown"
    assert tools.reason_and_solve("Explain something outside these tools.")["answer"] == ""
    assert tools.generation_source == "deterministic-tools"


def test_changed_timer_values_are_parsed_without_default_substitution():
    tools = DeterministicElectronicsTools()
    first = tools.reason_and_solve("Calculate 555 frequency R1=10k R2=47k C=10uF")
    second = tools.reason_and_solve("Calculate 555 frequency R1=10k R2=47k C=20uF")
    assert first["calculation"]["frequency_Hz"] == pytest.approx(2 * second["calculation"]["frequency_Hz"], abs=0.0001)
    assert "calculation" not in tools.reason_and_solve("Calculate 555 frequency")


def test_board_answer_preserves_canonical_claims_and_selected_variant():
    tools = DeterministicElectronicsTools()
    record = tools.lookup_hardware("board", "ARDUINO_UNO")["records"][0]
    result = tools.reason_and_solve("What board am I using?", board_type="ARDUINO_UNO")
    details = json.loads(result["answer"].split("\n", 1)[1])
    assert details == {"variant": record["subject"]["variant"], "claims": record["claims"]}
    assert result["evidence"] == [record["recordId"]]
    assert result["neuralAttempted"] is False
    unknown = tools.reason_and_solve("What board am I using?", board_type="UNKNOWN_BOARD_123")
    assert "unavailable or ambiguous" in unknown["answer"]
    assert unknown["evidence"] == []


@pytest.mark.parametrize("prompt", [
    "Calculate capacitive reactance for -100nF at 1kHz",
    "Calculate capacitive reactance for 100nF at -1kHz",
    "Calculate 555 frequency R1=-10k R2=47k C=10uF",
])
def test_negative_prompt_operands_are_not_silently_converted_to_positive(prompt):
    result = DeterministicElectronicsTools().reason_and_solve(prompt)
    assert "calculation" not in result
    assert "positive" in result["answer"]


def test_decimal_and_scientific_notation_do_not_lose_digits():
    result = DeterministicElectronicsTools().reason_and_solve(
        "Calculate capacitive reactance for .1e-6F at 1e3Hz"
    )
    assert result["calculation"]["reactanceOhm"] == pytest.approx(1591.5494309189535)
