"""Dataset statistics helpers for dashboard and research reporting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from dataset.build_dataset import load_dataset

DATASET_PATH = Path("data/misinformation_dataset.json")


def dataset_statistics(path: str | Path = DATASET_PATH) -> dict[str, Any]:
    records = load_dataset(path)

    by_language = Counter(str(item.get("language", "unknown")) for item in records)
    by_label = Counter(str(item.get("label", "uncertain")) for item in records)
    by_topic = Counter(str(item.get("topic", "unknown")) for item in records)
    by_source_type = Counter(str(item.get("source_type", "unknown")) for item in records)

    return {
        "total_samples": len(records),
        "by_language": dict(by_language),
        "by_label": dict(by_label),
        "by_topic": dict(by_topic),
        "by_source_type": dict(by_source_type),
    }


if __name__ == "__main__":
    print(dataset_statistics())
