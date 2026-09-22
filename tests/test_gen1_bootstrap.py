from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json

import pytest

try:
    installed_torch = version("torch")
except PackageNotFoundError:
    pytest.skip("Gen1 pinned PyTorch runtime is not installed", allow_module_level=True)
if installed_torch.partition("+")[0] != "2.8.0":
    pytest.skip(
        f"Gen1 bootstrap tests require pinned PyTorch 2.8.0, found {installed_torch}",
        allow_module_level=True,
    )

from gen1_bootstrap import (  # noqa: E402
    BootstrapContractError,
    check_bootstrap_scorecard,
    load_bootstrap_plan,
    pareto_checkpoint_steps,
    select_best_checkpoint,
)
from gen1_bootstrap.bootstrap import (  # noqa: E402
    DEFAULT_BEST_CONFIG_PATH,
    DEFAULT_GLOBAL_REPORT_PATH,
    DEFAULT_REPORT_PATH,
    prepare_output_heldout,
)
from gen1_training import load_approved_corpus  # noqa: E402
from model.gen1 import Gen1Config  # noqa: E402


def checkpoint_score(
    step: int,
    *,
    full_loss: float,
    full_accuracy: float,
    output_loss: float,
    output_accuracy: float,
    macro_loss: float,
    worst_loss: float,
    training_loss: float = 100.0,
) -> dict[str, object]:
    return {
        "step": step,
        "status": "completed",
        "trainingSnapshot": {"trainingLoss": training_loss},
        "fullValidation": {
            "nextTokenLoss": full_loss,
            "top1TokenAccuracy": full_accuracy,
        },
        "outputHeldout": {
            "nextTokenLoss": output_loss,
            "top1TokenAccuracy": output_accuracy,
            "taskMacroLoss": macro_loss,
            "worstTaskLoss": worst_loss,
        },
    }


def test_plan_binds_selected_architecture_and_frozen_multi_metric_policy() -> None:
    plan = load_bootstrap_plan()
    config = Gen1Config.from_dict(plan["selectedModelConfig"])
    assert plan["selectedConfig"]["candidateId"] == "edge-wide-gqa"
    assert config.parameter_count() == 14_198_464
    assert plan["controls"]["maxSteps"] == 256
    assert plan["controls"]["evaluationInterval"] == 32
    assert len(plan["heldoutPolicy"]["objectives"]) == 6
    assert plan["heldoutPolicy"]["trainingLossUsedForSelection"] is False
    assert plan["globalReleasePolicy"]["expectedMetricCount"] == 13
    assert plan["releaseBoundary"]["releaseApproved"] is False


def test_historical_training_token_ceiling_is_rejected_for_current_corpus() -> None:
    plan = load_bootstrap_plan()
    config = Gen1Config.from_dict(plan["selectedModelConfig"])
    corpus = load_approved_corpus(config, packing_block_size=128)
    assert corpus.manifest["tokenizer"]["version"] == "1.1.0"
    assert corpus.train.predicted_token_count == 155_690
    assert plan["controls"]["approvedUniqueTrainingPredictedTokens"] == 71_377
    assert (
        corpus.train.predicted_token_count
        != plan["controls"]["approvedUniqueTrainingPredictedTokens"]
    )
    assert plan["releaseBoundary"]["releaseApproved"] is False


def test_historical_heldout_policy_is_rejected_for_current_split() -> None:
    plan = load_bootstrap_plan()
    config = Gen1Config.from_dict(plan["selectedModelConfig"])
    corpus = load_approved_corpus(config, packing_block_size=128)
    assert corpus.manifest["validationRecordCount"] == 23
    assert plan["heldoutPolicy"]["recordCount"] == 16
    with pytest.raises(BootstrapContractError, match="held-out validation IDs changed"):
        prepare_output_heldout(corpus, config, plan["heldoutPolicy"])


def test_pareto_selection_ignores_training_loss_and_removes_dominated_steps() -> None:
    policy = load_bootstrap_plan()["heldoutPolicy"]
    quality = checkpoint_score(
        32,
        full_loss=2.0,
        full_accuracy=0.4,
        output_loss=2.1,
        output_accuracy=0.3,
        macro_loss=2.2,
        worst_loss=2.4,
        training_loss=50.0,
    )
    dominated = checkpoint_score(
        64,
        full_loss=2.2,
        full_accuracy=0.3,
        output_loss=2.3,
        output_accuracy=0.2,
        macro_loss=2.4,
        worst_loss=2.6,
        training_loss=0.01,
    )
    assert pareto_checkpoint_steps([quality, dominated], policy["objectives"]) == [32]
    decision = select_best_checkpoint([quality, dominated], policy)
    assert decision["selectedStep"] == 32
    assert decision["trainingLossUsedForSelection"] is False


def test_exact_multi_metric_tie_selects_earlier_checkpoint() -> None:
    policy = load_bootstrap_plan()["heldoutPolicy"]
    first = checkpoint_score(
        32,
        full_loss=2.0,
        full_accuracy=0.4,
        output_loss=2.1,
        output_accuracy=0.3,
        macro_loss=2.2,
        worst_loss=2.4,
    )
    second = {**first, "step": 64}
    decision = select_best_checkpoint([second, first], policy)
    assert decision["paretoFrontierSteps"] == [32, 64]
    assert decision["selectedStep"] == 32


def test_committed_bootstrap_evidence_is_self_verifying_and_tamper_evident(
    tmp_path,
) -> None:
    result = check_bootstrap_scorecard()
    assert result["decision"] == "pass"
    assert result["globalMetricCount"] == 13
    assert result["releaseApproved"] is False

    tampered = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    tampered["selection"]["selectedStep"] = -1
    path = tmp_path / "tampered-bootstrap-report.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(BootstrapContractError, match="checksum mismatch"):
        check_bootstrap_scorecard(
            report_path=path,
            global_report_path=DEFAULT_GLOBAL_REPORT_PATH,
            best_config_path=DEFAULT_BEST_CONFIG_PATH,
        )
