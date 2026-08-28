"""Build or no-write verify the VFAI-010 tokenizer release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from tokenizer_training.pipeline import check_tokenizer_release, write_tokenizer_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Train and write the immutable release")
    mode.add_argument("--check", action="store_true", help="Retrain and compare every artifact without writes")
    args = parser.parse_args()
    report = write_tokenizer_release() if args.write else check_tokenizer_release()
    print(json.dumps({
        "decision": report["decision"],
        "vocabSize": report["tokenizer"]["vocabSize"],
        "mergeCount": report["tokenizer"]["mergeCount"],
        "tokenReductionPercent": report["compression"]["tokenReductionPercent"],
        "vocabularySizeReductionPercent": report["vocabularyCost"]["sizeReductionPercent"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
