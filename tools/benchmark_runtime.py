"""CLI entrypoint for the offline VoltForge AI runtime benchmark."""

from __future__ import annotations

from pathlib import Path
import sys


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from benchmarking.runtime_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
