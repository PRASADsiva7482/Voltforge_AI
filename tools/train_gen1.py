"""CLI for new or resumed project-owned VFDLM Gen1 optimization runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from gen1_training import Gen1Trainer, Gen1TrainingConfig, load_approved_corpus
from model.gen1 import Gen1Config


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--peak-learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.resume:
        run_manifest = _read_json(args.run_directory / "run-manifest.json")
        config = Gen1TrainingConfig.from_dict(run_manifest["trainingConfig"])
        corpus = load_approved_corpus(config.model)
        trainer = Gen1Trainer.resume_latest(args.run_directory, corpus)
    else:
        if not args.run_id:
            raise ValueError("--run-id is required for a new training run")
        model_config = (
            Gen1Config.from_dict(_read_json(args.model_config))
            if args.model_config
            else Gen1Config()
        )
        config = Gen1TrainingConfig(
            model=model_config,
            seed=args.seed,
            max_steps=args.max_steps,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            peak_learning_rate=args.peak_learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            max_gradient_norm=args.max_gradient_norm,
            validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,
            precision=args.precision,
            device=args.device,
        )
        corpus = load_approved_corpus(config.model)
        trainer = Gen1Trainer.create(
            config,
            corpus,
            args.run_directory,
            run_id=args.run_id,
        )
    result = trainer.train(until_step=args.until_step)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
