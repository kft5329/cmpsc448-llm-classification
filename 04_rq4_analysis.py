"""
Step 4: RQ4 - what characteristics distinguish different LLMs?
 
Part A: descriptive statistics of handcrafted features (per family / per model)
Part B: classifier on handcrafted features only (RandomForest)
        -> feature importance + feature-GROUP ablation (drop group / only group)
Part C: text ablation with TF-IDF + LogisticRegression
        -> remove formatting, punctuation, length cues, opening sentence ... and
           see how much accuracy drops
 
Setup (once):
    pip install scikit-learn pandas numpy matplotlib
 
Run:
    python 04_rq4_analysis.py
Outputs go to results/rq4/
"""
import os
import re
import time
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
 
SEED = 0
DATA_DIR = "data"
OUT_DIR = os.path.join("results", "rq4")
os.makedirs(OUT_DIR, exist_ok=True)
 
# ----------------------------------------------------------------------------
# Part A: handcrafted features
# ----------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
 
 
def extract_features(text: str) -> dict:
    n_chars = len(text)
    words = WORD_RE.findall(text)
    n_words = len(words)
    lines = text.split("\n")
    nonempty = [l for l in lines if l.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    first100 = [w.lower() for w in words[:100]]
    letters = [c for c in text if c.isalpha()]
    denom_c = max(n_chars, 1)
 
    return {
        # length group
        "n_chars": n_chars,
        "n_words": n_words,
        "avg_sent_len": n_words / max(len(sentences), 1),
        # lexical group
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr_first100": len(set(first100)) / max(len(first100), 1),
        "upper_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "digit_ratio": sum(c.isdigit() for c in text) / denom_c,
        # structure / formatting group
        "n_lines": len(nonempty),
        "n_paragraphs": len(paragraphs),
        "n_bullets": sum(bool(re.match(r"\s*[-*\u2022]\s+", l)) for l in lines),
        "n_numbered": sum(bool(re.match(r"\s*\d+[.)]\s+", l)) for l in lines),
        "n_bold": text.count("**") // 2,
        "n_headers": sum(bool(re.match(r"\s{0,3}#{1,6}\s", l)) for l in lines),
        "n_code_fence": text.count("```"),
        # punctuation group (rate per char)
        "comma_rate": text.count(",") / denom_c,
        "period_rate": text.count(".") / denom_c,
        "excl_rate": text.count("!") / denom_c,
        "quest_rate": text.count("?") / denom_c,
        "colon_rate": text.count(":") / denom_c,
        "semicolon_rate": text.count(";") / denom_c,
        "quote_rate": (text.count('"') + text.count("'")) / denom_c,
        "paren_rate": (text.count("(") + text.count(")")) / denom_c,
        # phrase group
        "has_apology": int(bool(re.search(r"\bsorry\b|\bapologi[sz]e", text, re.I))),
        "has_as_an_ai": int(bool(re.search(r"\bas an ai\b|language model|\bai assistant\b", text, re.I))),
        "ends_with_punct": int(text.rstrip()[-1:] in ".!?") if text.strip() else 0,
    }
 
 
FEATURE_GROUPS = {
    "length": ["n_chars", "n_words", "avg_sent_len"],
    "lexical": ["avg_word_len", "ttr_first100", "upper_ratio", "digit_ratio"],
    "structure": ["n_lines", "n_paragraphs", "n_bullets", "n_numbered",
                  "n_bold", "n_headers", "n_code_fence"],
    "punctuation": ["comma_rate", "period_rate", "excl_rate", "quest_rate",
                    "colon_rate", "semicolon_rate", "quote_rate", "paren_rate"],
    "phrases": ["has_apology", "has_as_an_ai", "ends_with_punct"],
}
ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
 
 
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([extract_features(t) for t in df["output"]], index=df.index)
 
 
def run_rf(Xtr, ytr, Xte, yte, cols):
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr[cols], ytr)
    pred = rf.predict(Xte[cols])
    return rf, accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
 
 
# ----------------------------------------------------------------------------
# Part C: text ablations
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")
 
 
def tokenize(text):
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(text)]
 
 
def strip_markdown(t):
    t = re.sub(r"```+", " ", t)
    t = t.replace("`", "")
    t = re.sub(r"\*\*|__", "", t)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+", "", t)
    return t
 
 
def flatten_ws(t):
    return re.sub(r"\s+", " ", t).strip()
 
 
def lower_no_punct(t):
    return flatten_ws(re.sub(r"[^\w\s]", " ", t.lower()))
 
 
def truncate_50_words(t):
    m = list(re.finditer(r"\S+", t))[:50]
    return t[:m[-1].end()] if m else t
 
 
def drop_first_sentence(t):
    return re.sub(r"^.*?(?:[.!?](?:\s|$)|\n)", "", t, count=1, flags=re.S) or t
 
 
ABLATIONS = {
    "original": lambda t: t,
    "strip_markdown": strip_markdown,
    "flatten_whitespace(no newlines)": flatten_ws,
    "no_structure(markdown+newlines)": lambda t: flatten_ws(strip_markdown(t)),
    "lowercase_no_punct": lower_no_punct,
    "truncate_50_words(length cue)": truncate_50_words,
    "drop_first_sentence(openers)": drop_first_sentence,
}
 
 
def run_tfidf(train_texts, ytr, test_texts, yte):
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(vec.transform(test_texts))
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
 
 
# ----------------------------------------------------------------------------
def main():
    tr = pd.read_json(os.path.join(DATA_DIR, "train.jsonl"), lines=True)
    va = pd.read_json(os.path.join(DATA_DIR, "val.jsonl"), lines=True)
    te = pd.read_json(os.path.join(DATA_DIR, "test.jsonl"), lines=True)
    full = pd.concat([tr, va, te]).reset_index(drop=True)
 
    # ---------------- Part A ----------------
    print("[A] extracting features ...")
    t0 = time.time()
    F_full = featurize(full)
    print(f"    done in {time.time() - t0:.1f}s")
    desc = pd.concat([full[["family", "model_name"]], F_full], axis=1)
    fam_means = desc.groupby("family")[ALL_FEATURES].mean().round(4)
    model_means = desc.groupby("model_name")[ALL_FEATURES].mean().round(4)
    fam_means.to_csv(os.path.join(OUT_DIR, "feature_means_by_family.csv"))
    model_means.to_csv(os.path.join(OUT_DIR, "feature_means_by_model.csv"))
    print("\n=== key feature means by family ===")
    key = ["n_words", "avg_sent_len", "ttr_first100", "n_lines", "n_bullets",
           "n_bold", "excl_rate", "has_apology", "has_as_an_ai"]
    print(fam_means[key].to_string())
 
    # boxplot: length by family (clip extreme tail for readability)
    fams = list(fam_means.index)
    cap = desc["n_words"].quantile(0.99)
    plt.figure(figsize=(7, 4))
    plt.boxplot([desc.loc[desc.family == f, "n_words"].clip(upper=cap) for f in fams],
                showfliers=False)
    plt.xticks(range(1, len(fams) + 1), fams)
    plt.ylabel("response length (words)")
    plt.title("Response length by LLM family")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_by_family.png"), dpi=200)
    plt.close()
 
    # ---------------- Part B ----------------
    print("\n[B] handcrafted-feature classifier (RandomForest) ...")
    n_tr, n_va = len(tr), len(va)
    Ftr = F_full.iloc[:n_tr]
    Fte = F_full.iloc[n_tr + n_va:]
    ytr, yte = tr["label"].values, te["label"].values
 
    rf, acc, f1 = run_rf(Ftr, ytr, Fte, yte, ALL_FEATURES)
    print(f"    all features: test acc={acc:.4f} macro-F1={f1:.4f}")
    imp = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), header=["importance"])
    print("    top-10 features:\n" + imp.head(10).round(4).to_string())
 
    top = imp.head(15)[::-1]
    plt.figure(figsize=(6, 5))
    plt.barh(top.index, top.values)
    plt.xlabel("RandomForest importance")
    plt.title("Top handcrafted features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rf_feature_importance.png"), dpi=200)
    plt.close()
 
    rows = [{"setting": "all features", "acc": acc, "macro_f1": f1}]
    for g, cols in FEATURE_GROUPS.items():
        rest = [c for c in ALL_FEATURES if c not in cols]
        _, a1, f1_ = run_rf(Ftr, ytr, Fte, yte, rest)
        _, a2, f2_ = run_rf(Ftr, ytr, Fte, yte, cols)
        rows.append({"setting": f"drop {g}", "acc": a1, "macro_f1": f1_})
        rows.append({"setting": f"only {g}", "acc": a2, "macro_f1": f2_})
    grp = pd.DataFrame(rows)
    grp["delta_f1_vs_all"] = grp["macro_f1"] - f1
    grp.round(4).to_csv(os.path.join(OUT_DIR, "feature_group_ablation.csv"), index=False)
    print("\n=== feature-group ablation (test) ===")
    print(grp.round(4).to_string(index=False))
 
    # ---------------- Part C ----------------
    print("\n[C] text ablation with TF-IDF + LR (this takes a few minutes) ...")
    rows = []
    base_f1 = None
    for name, fn in ABLATIONS.items():
        t0 = time.time()
        a, f = run_tfidf([fn(t) for t in tr["output"]], ytr,
                         [fn(t) for t in te["output"]], yte)
        if base_f1 is None:
            base_f1 = f
        rows.append({"ablation": name, "acc": a, "macro_f1": f, "delta_f1_vs_original": f - base_f1})
        print(f"    {name:36s} acc={a:.4f} F1={f:.4f}  ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT_DIR, "text_ablation.csv"), index=False)
 
    print("\nsaved outputs in", OUT_DIR)
 
 
if __name__ == "__main__":
    main()
 