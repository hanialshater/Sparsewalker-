#!/usr/bin/env python
"""Colab-safe bootstrap for the streaming-native training speed audit."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXP = ROOT / "experiments"

for p in (str(SRC), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Fail here with a useful path diagnostic rather than later inside the runner.
try:
    import sparsewalker  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        f"Could not import sparsewalker after adding SRC={SRC} to sys.path; "
        f"sys.path[:5]={sys.path[:5]}"
    ) from exc

TARGET = EXP / "run_ml1m_walker_streaming_speed.py"
runpy.run_path(str(TARGET), run_name="__main__")
