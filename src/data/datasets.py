"""
Dataset loading + teacher splits
Supported: sst2, ag_news, yelp_polarity
"""

import numpy as np
from datasets import load_dataset
from typing import List, Dict, Tuple


# ---- Label maps for each dataset ----------------------------------------

DATASET_CONFIG = {
    "sst2": {
        "hf_name": "stanfordnlp/sst2",
        "text_col": "sentence",
        "label_col": "label",
        "labels": ["negative", "positive"],
        "train_split": "train",
        "test_split": "validation",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "text_col": "text",
        "label_col": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "train_split": "train",
        "test_split": "test",
    },
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "text_col": "text",
        "label_col": "label",
        "labels": ["negative", "positive"],
        "train_split": "train",
        "test_split": "test",
    },
}


def load_splits(
    dataset_name: str,
    num_test: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (train_examples, test_examples).
    Each example: {"text": str, "label": int, "label_name": str}
    """
    cfg = DATASET_CONFIG[dataset_name]
    ds = load_dataset(cfg["hf_name"])

    rng = np.random.default_rng(seed)

    def _convert(split_data):
        examples = []
        for item in split_data:
            examples.append({
                "text": item[cfg["text_col"]],
                "label": item[cfg["label_col"]],
                "label_name": cfg["labels"][item[cfg["label_col"]]],
            })
        return examples

    train = _convert(ds[cfg["train_split"]])
    test  = _convert(ds[cfg["test_split"]])

    # Randomly sample the test set
    idx = rng.choice(len(test), size=min(num_test, len(test)), replace=False)
    test = [test[i] for i in idx]

    return train, test


def partition_for_teachers(
    train_examples: List[Dict],
    num_teachers: int,
    shots_per_teacher: int,
    seed: int = 42,
) -> List[List[Dict]]:
    """
    Split the training set into num_teachers disjoint subsets.
    Each teacher gets shots_per_teacher examples (class-balanced when possible).

    Returns: List[teacher_examples] of length num_teachers
    """
    rng = np.random.default_rng(seed)

    # Group by label
    by_label: Dict[int, List[Dict]] = {}
    for ex in train_examples:
        by_label.setdefault(ex["label"], []).append(ex)

    # Shuffle each label
    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    num_classes = len(by_label)
    shots_per_class = max(1, shots_per_teacher // num_classes)

    partitions = []
    label_ptr = {lbl: 0 for lbl in by_label}

    for _ in range(num_teachers):
        teacher_examples = []
        for lbl, examples in by_label.items():
            ptr = label_ptr[lbl]
            end = ptr + shots_per_class
            # Wrap around if there is not enough data
            chunk = [examples[i % len(examples)] for i in range(ptr, end)]
            teacher_examples.extend(chunk)
            label_ptr[lbl] = end % len(examples)
        rng.shuffle(teacher_examples)
        partitions.append(teacher_examples)

    return partitions


def get_label_names(dataset_name: str) -> List[str]:
    return DATASET_CONFIG[dataset_name]["labels"]


if __name__ == "__main__":
    train, test = load_splits("sst2", num_test=100)
    print(f"Train: {len(train)}  Test: {len(test)}")
    parts = partition_for_teachers(train, num_teachers=5, shots_per_teacher=4)
    for i, p in enumerate(parts):
        print(f"Teacher {i}: {[ex['label_name'] for ex in p]}")
