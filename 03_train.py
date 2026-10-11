"""
Step 3: Train / evaluate classifiers for RQ1 and RQ2.

Models : cnn (TextCNN), lstm (BiLSTM), tfidf (TF-IDF + LogisticRegression baseline)
Modes  : output  -> LLM_output only          (RQ1)
         input   -> LLM_input only           (RQ2)
         both    -> LLM_input + LLM_output   (RQ2)

Setup (once):
    pip install torch scikit-learn pandas numpy

Examples:
    python 03_train.py --model cnn   --mode output
    python 03_train.py --model lstm  --mode both --seed 1
    python 03_train.py --model tfidf --mode input

Results are saved to results/<model>_<mode>_seed<seed>.json
"""
import argparse
import copy
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA_DIR = "data"
RES_DIR = "results"

# ----------------------------------------------------------------------------
# Tokenization (keeps punctuation and newlines: they are style signals)
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def tokenize(text: str):
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(text)]


def build_tokens(row, mode, max_in, max_out):
    if mode == "output":
        return tokenize(row["output"])[:max_out]
    if mode == "input":
        return tokenize(row["input"])[:max_in]
    return tokenize(row["input"])[:max_in] + ["<sep>"] + tokenize(row["output"])[:max_out]


def build_vocab(token_lists, max_size=30000, min_freq=2):
    specials = ["<pad>", "<unk>", "<nl>", "<sep>"]
    counter = Counter(t for toks in token_lists for t in toks)
    itos = list(specials)
    for w, f in counter.most_common(max_size):
        if f >= min_freq and w not in specials:
            itos.append(w)
    return {w: i for i, w in enumerate(itos)}


def encode(token_lists, vocab, max_len):
    n = len(token_lists)
    X = np.zeros((n, max_len), dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    unk = vocab["<unk>"]
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(t, unk) for t in toks][:max_len]
        if not ids:
            ids = [unk]
        X[i, :len(ids)] = ids
        lens[i] = len(ids)
    return X, lens


# ----------------------------------------------------------------------------
# Metrics / saving
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_results(args, result):
    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, f"{args.model}_{args.mode}_seed{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("saved ->", path)


def load_data():
    tr = pd.read_json(os.path.join(DATA_DIR, "train.jsonl"), lines=True)
    va = pd.read_json(os.path.join(DATA_DIR, "val.jsonl"), lines=True)
    te = pd.read_json(os.path.join(DATA_DIR, "test.jsonl"), lines=True)
    with open(os.path.join(DATA_DIR, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return tr, va, te, class_names


# ----------------------------------------------------------------------------
# TF-IDF baseline
# ----------------------------------------------------------------------------
def run_tfidf(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr, va, te, class_names = load_data()

    def text(df):
        if args.mode == "output":
            return df["output"].tolist()
        if args.mode == "input":
            return df["input"].tolist()
        return (df["input"] + " [SEP] " + df["output"]).tolist()

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    Xtr = vec.fit_transform(text(tr))
    clf = LogisticRegression(max_iter=2000, random_state=args.seed)
    clf.fit(Xtr, tr["label"].values)

    val_pred = clf.predict(vec.transform(text(va)))
    test_pred = clf.predict(vec.transform(text(te)))
    val_m = compute_metrics(va["label"].values, val_pred, class_names)
    test_m = compute_metrics(te["label"].values, test_pred, class_names)
    print(f"[tfidf/{args.mode}] val acc={val_m['accuracy']:.4f} f1={val_m['macro_f1']:.4f} | "
          f"test acc={test_m['accuracy']:.4f} f1={test_m['macro_f1']:.4f}")
    save_results(args, {
        "model": "tfidf", "mode": args.mode, "seed": args.seed,
        "class_names": class_names, "train_time_sec": time.time() - t0,
        "val": val_m, "test": test_m,
    })


# ----------------------------------------------------------------------------
# Neural models
# ----------------------------------------------------------------------------
def run_neural(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, n_classes, emb=128, n_filters=128,
                     kernel_sizes=(3, 4, 5), dropout=0.5):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(emb, n_filters, k, padding=k // 2) for k in kernel_sizes])
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

        def forward(self, x, lengths=None):
            e = self.emb(x).transpose(1, 2)                      # (B, E, L)
            feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
            return self.fc(self.drop(torch.cat(feats, dim=1)))

    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, n_classes, emb=128, hidden=128, dropout=0.5):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * hidden, n_classes)

        def forward(self, x, lengths):
            e = self.drop(self.emb(x))
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)                     # h_n: (2, B, H)
            h = torch.cat([h_n[0], h_n[1]], dim=1)
            return self.fc(self.drop(h))

    tr, va, te, class_names = load_data()
    n_classes = len(class_names)

    tok = {name: [build_tokens(r, args.mode, args.max_in, args.max_out)
                  for r in df.to_dict("records")]
           for name, df in (("train", tr), ("val", va), ("test", te))}
    vocab = build_vocab(tok["train"], args.vocab_size, args.min_freq)
    max_len = max(5, max(len(t) for t in tok["train"]))
    print(f"vocab size: {len(vocab)} | max_len: {max_len}")

    def make_loader(name, df, shuffle):
        X, lens = encode(tok[name], vocab, max_len)
        y = df["label"].values.astype(np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(lens), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    loaders = {"train": make_loader("train", tr, True),
               "val": make_loader("val", va, False),
               "test": make_loader("test", te, False)}

    if args.model == "cnn":
        model = TextCNN(len(vocab), n_classes, args.emb_dim, args.n_filters, dropout=args.dropout)
    else:
        model = BiLSTM(len(vocab), n_classes, args.emb_dim, args.hidden, dropout=args.dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} | params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def predict(loader):
        model.eval()
        preds, ys = [], []
        for xb, lb, yb in loader:
            logits = model(xb.to(device), lb)
            preds.append(logits.argmax(1).cpu().numpy())
            ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(preds)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, lb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        yv, pv = predict(loaders["val"])
        v_acc, v_f1 = accuracy_score(yv, pv), f1_score(yv, pv, average="macro")
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": v_acc, "val_macro_f1": v_f1})
        print(f"epoch {epoch:02d} | train loss {total / n:.4f} | val acc {v_acc:.4f} | val F1 {v_f1:.4f}")
        if v_f1 > best_f1:
            best_f1, best_state, bad = v_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    yv, pv = predict(loaders["val"])
    yt, pt = predict(loaders["test"])
    val_m = compute_metrics(yv, pv, class_names)
    test_m = compute_metrics(yt, pt, class_names)
    print(f"[{args.model}/{args.mode}] best val F1={best_f1:.4f} | "
          f"test acc={test_m['accuracy']:.4f} macro-F1={test_m['macro_f1']:.4f}")

    save_results(args, {
        "model": args.model, "mode": args.mode, "seed": args.seed,
        "class_names": class_names, "n_params": n_params,
        "vocab_size": len(vocab), "max_len": max_len,
        "hyperparams": {k: getattr(args, k) for k in (
            "max_in", "max_out", "vocab_size", "min_freq", "emb_dim", "n_filters",
            "hidden", "dropout", "lr", "batch_size", "epochs", "patience")},
        "epochs_run": len(history), "train_time_sec": time.time() - t0,
        "history": history, "val": val_m, "test": test_m,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "lstm", "tfidf"], required=True)
    p.add_argument("--mode", choices=["output", "input", "both"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_in", type=int, default=128, help="max tokens kept from the input")
    p.add_argument("--max_out", type=int, default=256, help="max tokens kept from the output")
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--n_filters", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    args = p.parse_args()

    if args.model == "tfidf":
        run_tfidf(args)
    else:
        run_neural(args)


if __name__ == "__main__":
    main()