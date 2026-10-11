# cmpsc448-llm-classification
CMPSC 448 project: Identifying LLMs from their responses using CNN and RNN
# Who Wrote It? Identifying LLMs from Their Responses

- **Author:** Kanon Tsuda

## Overview

This project trains CNN and RNN (BiLSTM) text classifiers to predict which LLM family wrote a response, and study:

- **RQ1:** Can we identify which LLM generated a response? (output only)
- **RQ2:** Does the user's prompt help? (input only vs. output only vs. input + output)
- **RQ4 (extra credit):** What characteristics distinguish different LLMs? (feature analysis and ablations)

A TF-IDF + Logistic Regression model is included as a non-neural baseline.

## Data

- **Source:** [`lmsys/chatbot_arena_conversations`](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) (Hugging Face, 33k pairwise conversations, gated dataset).
- **The dataset is NOT included in this repository** because access requires accepting the dataset's terms on Hugging Face. Follow the steps below to rebuild it.
- **Filtering:** English only, single-turn conversations only, conversations flagged by the dataset's moderation field removed, duplicates removed. Each conversation row yields two samples (model A and model B), giving 51,822 samples from 25,928 unique prompts.
- **Classes (5 LLM families, balanced to 2,298 samples each, 11,490 total):**

| Class | Member models |
|---|---|
| GPT | gpt-4, gpt-3.5-turbo |
| Claude | claude-v1, claude-instant-v1 |
| PaLM | palm-2 |
| Llama-derived | vicuna-13b, koala-13b, alpaca-13b |
| Pythia-based | oasst-pythia-12b, dolly-v2-12b |

- **Split:** 80 / 10 / 10 (train 9,170 / val 1,158 / test 1,162), split **by `question_id`** so that the same prompt never appears in more than one split. The same prompt is answered by two different models in this dataset, so grouping avoids leakage.

## Repository structure

```
01_prepare_data.py       load + filter + flatten the dataset to (model, input, output)
02_make_splits.py        balanced 5-family sampling + grouped train/val/test split
03_train.py              CNN / BiLSTM / TF-IDF training and evaluation (RQ1, RQ2)
04_rq4_analysis.py       handcrafted features, feature-group ablation, text ablation (RQ4)
05_summarize_results.py  aggregates results/*.json into tables and figures
results/                 per-run JSON results, RQ4 outputs, summary tables/figures
```


## How to reproduce

```bash
pip install torch scikit-learn pandas numpy matplotlib datasets
```

1. Request access to the dataset on Hugging Face, then authenticate:
   ```bash
   python -c "from huggingface_hub import login; login()"
   ```
   (Use your own access token. Never commit tokens to the repository.)
2. Build the data:
   ```bash
   python 01_prepare_data.py
   python 02_make_splits.py
   ```
3. Train and evaluate (models: `cnn`, `lstm`, `tfidf`; modes: `output`, `input`, `both`):
   ```bash
   python 03_train.py --model cnn --mode output --seed 0
   ```
   Run all settings (PowerShell):
   ```powershell
   foreach ($m in "tfidf","cnn","lstm") { foreach ($mode in "output","input","both") { python 03_train.py --model $m --mode $mode } }
   ```
   Extra seeds (used for the CNN results in the report):
   ```powershell
   foreach ($s in 1,2) { foreach ($mode in "output","input","both") { python 03_train.py --model cnn --mode $mode --seed $s } }
   ```
4. RQ4 analysis and summary:
   ```bash
   python 04_rq4_analysis.py
   python 05_summarize_results.py
   ```

## Models and training

- **Tokenization:** words, punctuation and newlines (`<nl>`) as tokens; case is preserved. Vocabulary built from the training set only (max 30k tokens, min frequency 2).
- **CNN (TextCNN):** Embedding (128) → Conv1D with kernel sizes 3/4/5 (128 filters each) → max-over-time pooling → Dropout 0.5 → linear.
- **RNN (BiLSTM):** Embedding (128) → 1-layer bidirectional LSTM (hidden 128) → concatenated final states → Dropout 0.5 → linear.
- **Training:** Adam (lr 1e-3), batch size 64, max 20 epochs, early stopping on validation macro-F1 (patience 3), best-validation checkpoint evaluated on the test set, gradient clipping 5.0.
- **Input length:** at most 256 tokens of output and 128 tokens of input (`both` = input + `<sep>` + output).
- **Modes:** `output` (RQ1), `input`, `both` (RQ2).

## Main results (test set, macro-F1 %, chance = 20%)

| Model | Output only (RQ1) | Input only | Input + Output |
|---|---|---|---|
| TF-IDF + LR (1 run) | 56.9 | 20.3 | 56.3 |
| CNN (mean ± std, 3 seeds) | 57.6 ± 1.0 | 21.4 ± 0.6 | 57.6 ± 2.6 |
| BiLSTM (1 seed) | 48.0 | 21.0 | 51.0 |

Test set: 1,162 samples. See `results/summary/` for all tables and figures (accuracy, per-family F1, confusion matrices) and the report for discussion.

Main findings:

- **RQ1:** The LLM family can be identified from the response (about 57% macro-F1 vs. 20% chance), but the CNN is not better than the TF-IDF baseline.
- **RQ2:** The prompt alone is at chance level, and adding it to the response gives no consistent improvement (each prompt is answered by two different models in this dataset).
- **RQ4:** Both structural signals (length, lists, formatting, punctuation) and vocabulary carry information; structure/formatting features were the most important group in the handcrafted-feature ablation.

## RQ4 outputs

`results/rq4/` contains feature means per family and per model, RandomForest feature importances, feature-group ablation, and text-ablation results. See the report for interpretation.

## Notes and limitations

- Models in this dataset are from 2023 and the data is English, single-turn only.
- Neural models were trained on CPU with limited hyperparameter tuning; the CNN was run with 3 seeds, while the BiLSTM and the TF-IDF baseline were run once (the BiLSTM because of compute time).
- Class labels are LLM families (some classes group several related models).

## Use of AI assistants

Claude (Anthropic) was used to help plan experiments, write and debug code, and interpret results. I reviewed, ran and verified all code and results submitted here.
