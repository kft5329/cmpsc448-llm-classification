"""
Step 5: Summarize results/*.json into tables and figures for the report.

Run (after 03_train.py runs finished):
    python 05_summarize_results.py

Outputs (in results/summary/):
    summary_long.csv         one row per run
    summary_acc.csv / summary_f1.csv   model x mode tables (mean +- std over seeds)
    per_class_f1_output.csv  per-family F1 for the output-only setting
    rq2_bar.png              macro-F1 for input / output / both
    confusion_<model>_<mode>.png   row-normalized confusion matrices (lowest seed)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES_DIR = "results"
OUT_DIR = os.path.join(RES_DIR, "summary")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER = ["tfidf", "cnn", "lstm"]
MODE_ORDER = ["output", "input", "both"]


def fmt(values):
    v = np.array(values, dtype=float) * 100
    if len(v) == 1:
        return f"{v[0]:.1f} (n=1)"
    return f"{v.mean():.1f} \u00b1 {v.std(ddof=1):.1f} (n={len(v)})"


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(RES_DIR, "*_seed*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["_path"] = path
        runs.append(r)
    if not runs:
        raise SystemExit("no result files found in results/")

    df = pd.DataFrame([{
        "model": r["model"], "mode": r["mode"], "seed": r["seed"],
        "test_acc": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
        "val_macro_f1": r["val"]["macro_f1"],
        "epochs_run": r.get("epochs_run", np.nan),
        "train_time_sec": r.get("train_time_sec", np.nan),
    } for r in runs]).sort_values(["model", "mode", "seed"])
    df.round(4).to_csv(os.path.join(OUT_DIR, "summary_long.csv"), index=False)

    # ---- model x mode tables ----
    for key, name in (("test_acc", "acc"), ("test_macro_f1", "f1")):
        table = pd.DataFrame(index=[m for m in MODEL_ORDER if m in set(df.model)],
                             columns=MODE_ORDER, dtype=object)
        for (m, mode), g in df.groupby(["model", "mode"]):
            table.loc[m, mode] = fmt(g[key].values)
        table = table.fillna("-")
        table.to_csv(os.path.join(OUT_DIR, f"summary_{name}.csv"))
        print(f"\n=== test {'accuracy' if name == 'acc' else 'macro-F1'} (%) ===")
        print(table.to_string())

    # ---- RQ2 bar chart (macro-F1, mean over seeds, error bar = std) ----
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    width = 0.8 / max(len(models), 1)
    plt.figure(figsize=(7, 4))
    for i, m in enumerate(models):
        means, stds = [], []
        for mode in MODE_ORDER:
            v = df[(df.model == m) & (df["mode"] == mode)]["test_macro_f1"].values
            means.append(v.mean() if len(v) else 0.0)
            stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        xs = np.arange(len(MODE_ORDER)) + i * width
        plt.bar(xs, means, width=width, yerr=stds, capsize=3, label=m)
    plt.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="chance (5 classes)")
    plt.xticks(np.arange(len(MODE_ORDER)) + width * (len(models) - 1) / 2,
               ["output only", "input only", "input + output"])
    plt.ylabel("test macro-F1")
    plt.title("RQ2: effect of the input")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_bar.png"), dpi=200)
    plt.close()

    # ---- per-class F1 (output only), lowest seed per model ----
    per_class = {}
    for m in models:
        cand = [r for r in runs if r["model"] == m and r["mode"] == "output"]
        if not cand:
            continue
        r = min(cand, key=lambda x: x["seed"])
        per_class[m] = {c: r["test"]["per_class"][c]["f1-score"] for c in r["class_names"]}
    if per_class:
        pc = pd.DataFrame(per_class).round(3)
        pc.to_csv(os.path.join(OUT_DIR, "per_class_f1_output.csv"))
        print("\n=== per-family F1, output only (lowest seed) ===")
        print(pc.to_string())

    # ---- confusion matrices (output and both), lowest seed ----
    for m in models:
        for mode in ("output", "both"):
            cand = [r for r in runs if r["model"] == m and r["mode"] == mode]
            if not cand:
                continue
            r = min(cand, key=lambda x: x["seed"])
            cm = np.array(r["test"]["confusion_matrix"], dtype=float)
            cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            names = r["class_names"]
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticklabels(names)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{m} / {mode} (row-normalized)")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"confusion_{m}_{mode}.png"), dpi=200)
            plt.close(fig)

    print("\nsaved outputs in", OUT_DIR)


if __name__ == "__main__":
    main()