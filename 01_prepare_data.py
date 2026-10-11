"""
Step 1: Load lmsys/chatbot_arena_conversations and flatten it to
(question_id, model_name, input, output) rows.

Setup (once):
    pip install datasets pandas
    huggingface-cli login      # or: hf auth login   (account must have dataset access)

Run:
    python 01_prepare_data.py
"""
import os
import pandas as pd
from datasets import load_dataset

OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "arena_flat.jsonl")


def is_flagged(moderation) -> bool:
    """Robustly check the openai_moderation field (dict or list of dicts)."""
    if moderation is None:
        return False
    items = moderation if isinstance(moderation, list) else [moderation]
    for m in items:
        if not isinstance(m, dict):
            continue
        if m.get("flagged"):
            return True
        cats = m.get("categories") or {}
        if any(bool(v) for v in cats.values()):
            return True
    return False


def extract_pair(conv):
    """Return (user_text, assistant_text) for a single-turn conversation, else None."""
    if not conv or len(conv) < 2:
        return None
    user, assistant = conv[0], conv[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    u, a = (user.get("content") or "").strip(), (assistant.get("content") or "").strip()
    if not u or not a:
        return None
    return u, a


def main():
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    print("raw rows:", len(ds))
    print("columns:", ds.column_names)

    rows = []
    for ex in ds:
        # English, single-turn only
        if ex.get("language") != "English" or ex.get("turn") != 1:
            continue
        # drop conversations flagged by moderation (avoid sensitive content)
        if is_flagged(ex.get("openai_moderation")):
            continue

        for side in ("a", "b"):
            pair = extract_pair(ex[f"conversation_{side}"])
            if pair is None:
                continue
            rows.append({
                "question_id": ex["question_id"],
                "model_name": ex[f"model_{side}"],
                "input": pair[0],
                "output": pair[1],
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["question_id", "model_name"])
    print("\nflattened samples:", len(df))
    print("unique prompts:", df["question_id"].nunique())

    counts = df["model_name"].value_counts()
    print("\n=== samples per model ===")
    print(counts.to_string())

    # quick length stats per model (useful for RQ4 later)
    df["out_len_words"] = df["output"].str.split().str.len()
    print("\n=== mean output length (words) per model ===")
    print(df.groupby("model_name")["out_len_words"].mean().round(1).sort_values().to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    df.drop(columns=["out_len_words"]).to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()