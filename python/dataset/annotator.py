"""Simple CLI annotation tool for misinformation labels."""

from __future__ import annotations

from pathlib import Path

from dataset.build_dataset import load_dataset, save_dataset

DATASET_PATH = Path("data/misinformation_dataset.json")
LABEL_MAP = {"1": "true", "2": "false", "3": "uncertain"}


def annotate(path: str | Path = DATASET_PATH) -> None:
    records = load_dataset(path)
    if not records:
        print("Dataset is empty.")
        return

    for idx, sample in enumerate(records, start=1):
        claim = sample.get("claim", "")
        current = sample.get("label", "uncertain")

        print("\n" + "=" * 72)
        print(f"Sample {idx}/{len(records)}")
        print(f"Claim: {claim}")
        print(f"Current label: {current}")
        print("Choose label: 1=true, 2=false, 3=uncertain, Enter=keep")

        choice = input("Your choice: ").strip()
        if choice in LABEL_MAP:
            sample["label"] = LABEL_MAP[choice]
            print(f"Updated label -> {sample['label']}")
        else:
            print("No change")

    save_dataset(records, path)
    print("\nAnnotation complete. Dataset saved.")


if __name__ == "__main__":
    annotate()
