"""Logging utilities for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs/evaluation_logs.json")


def log_evaluation_records(records: list[dict[str, Any]], log_path: str | Path = LOG_PATH) -> Path:
    """Persist evaluation records to JSON for research traceability."""

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return target
