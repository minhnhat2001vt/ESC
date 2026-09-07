"""Portable filesystem configuration for ESC experiments.

The defaults keep data and outputs inside the repository. Override any root
with an environment variable when datasets or results live elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_ROOT = Path(
    os.environ.get("ESC_DATA_ROOT", REPO_ROOT / "original_data")
).expanduser()

PROCESSED_DATA_ROOT = Path(
    os.environ.get("ESC_PROCESSED_ROOT", REPO_ROOT / "processed_data")
).expanduser()

RESULTS_ROOT = Path(
    os.environ.get("ESC_RESULTS_ROOT", REPO_ROOT / "results")
).expanduser()

LOGS_ROOT = Path(
    os.environ.get("ESC_LOGS_ROOT", REPO_ROOT / "logs")
).expanduser()
