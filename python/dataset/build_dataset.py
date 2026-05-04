"""Dataset builder for multilingual misinformation research."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

DATASET_PATH = Path("data/misinformation_dataset.json")
VALID_LABELS = {"true", "false", "uncertain"}
VALID_LANGUAGES = {"en", "am"}
VALID_TOPICS = {"health", "politics", "science", "social"}
VALID_SOURCE_TYPES = {"news", "social_media", "rumor", "official"}
REQUIRED_FIELDS = {"id", "claim", "label", "language", "topic", "source_type", "evidence"}


def load_dataset(path: str | Path = DATASET_PATH) -> list[dict[str, Any]]:
    dataset_file = Path(path)
    if not dataset_file.exists():
        return []
    return json.loads(dataset_file.read_text(encoding="utf-8"))


def save_dataset(records: list[dict[str, Any]], path: str | Path = DATASET_PATH) -> Path:
    dataset_file = Path(path)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_file


def _normalize_claim(claim: str) -> str:
    return " ".join(claim.strip().lower().split())


def _validate_record(record: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    if not record["claim"].strip():
        raise ValueError("Claim cannot be empty.")

    if record["label"] not in VALID_LABELS:
        raise ValueError(f"Invalid label '{record['label']}'.")
    if record["language"] not in VALID_LANGUAGES:
        raise ValueError(f"Invalid language '{record['language']}'.")
    if record["topic"] not in VALID_TOPICS:
        raise ValueError(f"Invalid topic '{record['topic']}'.")
    if record["source_type"] not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source type '{record['source_type']}'.")


def add_sample(
    claim: str,
    label: str,
    language: str,
    topic: str,
    source_type: str,
    path: str | Path = DATASET_PATH,
) -> tuple[bool, str]:
    """Add a validated sample unless an equivalent claim already exists."""

    sample = {
        "id": str(uuid.uuid4()),
        "claim": claim.strip(),
        "label": label.strip().lower(),
        "language": language.strip().lower(),
        "topic": topic.strip().lower(),
        "source_type": source_type.strip().lower(),
        "evidence": [],
        "retrieved_evidence": [],
        "human_verified_evidence": [],
    }
    _validate_record(sample)

    records = load_dataset(path)
    normalized = _normalize_claim(sample["claim"])
    if any(_normalize_claim(str(item.get("claim", ""))) == normalized for item in records):
        return False, "Duplicate claim skipped."

    records.append(sample)
    save_dataset(records, path)
    return True, f"Added sample: {sample['id']}"


def interactive_builder(path: str | Path = DATASET_PATH) -> None:
    """Simple manual entry CLI for dataset creation."""

    print("Manual dataset builder. Leave claim empty to stop.")
    while True:
        claim = input("Claim: ").strip()
        if not claim:
            break
        label = input("Label (true/false/uncertain): ").strip().lower()
        language = input("Language (en/am): ").strip().lower()
        topic = input("Topic (health/politics/science/social): ").strip().lower()
        source_type = input("Source type (news/social_media/rumor/official): ").strip().lower()

        try:
            added, message = add_sample(claim, label, language, topic, source_type, path=path)
            status = "ADDED" if added else "SKIPPED"
            print(f"[{status}] {message}")
        except ValueError as exc:
            print(f"[ERROR] {exc}")

    print("Done.")


if __name__ == "__main__":
    interactive_builder()
