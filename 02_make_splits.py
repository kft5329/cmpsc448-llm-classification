"""
Step 2: Build a balanced family-level dataset and split it by question_id.

Input : data/arena_flat.jsonl  (from 01_prepare_data.py)
Output: data/train.jsonl, data/val.jsonl, data/test.jsonl, data/label_map.json

Run:
    python 02_make_splits.py
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 42
IN_PATH = "data/arena_flat.jsonl"
OUT_DIR = "data"

# samples per family (divisible by 2 and 3 so every family is exactly this size)
TARGET_PER_FAMILY = 2298

FAMILIES = {
    "GPT": ["gpt-4", "gpt-3.5-turbo"],
    "Claude": ["claude-v1", "claude-instant-v1"],
    "PaLM": ["palm-2"],
    "Llama-derived": ["vicuna-13b", "koala-13b", "alpaca-13b"],
    "Pythia-based": ["oasst-pythia-12b", "dolly-v2-12b"],
}


def sample_family(df: pd.DataFrame, models: list, target: int) -> pd.DataFrame:
    """Take (almost) equal numbers from each member model of a family."""
    parts = []
    remaining = target
    # smaller pools first so that any shortfall is covered by larger pools
    models_sorted = sorted(models, key=lambda m: (df["model_name"] == m).sum())
    for i, m in enumerate(models_sorted):
        pool = df[df["model_name"] == m]
        quota = remaining // (len(models_sorted) - i)
        n = min(quota, len(pool))
        parts.append(pool.sample(n=n, random_state=SEED))
        remaining -= n
    out = pd.concat(parts)
    assert len(out) == target, f"family sampling short: {len(out)} < {target}"
    return out


def main():
    df = pd.read_json(IN_PATH, lines=True)
    print("loaded:", len(df))

    pieces = []
    for family, models in FAMILIES.items():
        sub = df[df["model_name"].isin(models)]
        s = sample_family(sub, models, TARGET_PER_FAMILY)
        s = s.assign(family=family)
        pieces.append(s)
    data = pd.concat(pieces).reset_index(drop=True)

    label_map = {fam: i for i, fam in enumerate(FAMILIES)}
    data["label"] = data["family"].map(label_map)

    # ---- group split by question_id (no prompt shared across splits) ----
    qids = np.array(sorted(data["question_id"].unique()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(qids)
    n = len(qids)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    split_of = {}
    for q in qids[:n_train]:
        split_of[q] = "train"
    for q in qids[n_train:n_train + n_val]:
        split_of[q] = "val"
    for q in qids[n_train + n_val:]:
        split_of[q] = "test"
    data["split"] = data["question_id"].map(split_of)

    os.makedirs(OUT_DIR, exist_ok=True)
    for sp in ("train", "val", "test"):
        part = data[data["split"] == sp].sample(frac=1.0, random_state=SEED)
        part.to_json(os.path.join(OUT_DIR, f"{sp}.jsonl"),
                     orient="records", lines=True, force_ascii=False)

    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("\n=== family x split counts ===")
    print(pd.crosstab(data["family"], data["split"]).to_string())
    print("\n=== model x split counts ===")
    print(pd.crosstab(data["model_name"], data["split"]).to_string())

    # sanity check: no prompt in more than one split
    chk = data.groupby("question_id")["split"].nunique()
    assert (chk == 1).all(), "leak: a question_id appears in multiple splits"
    print("\nno question_id leakage across splits: OK")
    print("saved train/val/test.jsonl and label_map.json in", OUT_DIR)


if __name__ == "__main__":
    main()