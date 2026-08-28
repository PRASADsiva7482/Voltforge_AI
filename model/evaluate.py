"""Compatibility entrypoint for the frozen VoltForge AI release evaluation.

This replaces the legacy print-only evaluator, which could claim every check
passed while no approved model was available.
"""

from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from evaluation.release_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
