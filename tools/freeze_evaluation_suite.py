"""CLI entrypoint for freezing or verifying release evaluation fixtures."""

from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from evaluation.freeze import main


if __name__ == "__main__":
    raise SystemExit(main())
