# Waveform Technical Challenge — Submission

**Author:** Tien Nguyen
**Data:** MIMIC-III Blood Pressure dataset (Harvard Dataverse, doi:10.7910/DVN/DBM1NF)

## How to run
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Open `notebook/waveform_challenge.ipynb` and run it top to bottom.

The raw waveform files are **not** included; `data/README.md` explains how to download them.

## What's in each folder
```
waveform-challenge/
├── README.md                  <- this file:  how to run + checklist
├── requirements.txt           <- exact package versions
├── notebook/
│   └── waveform_challenge.ipynb   <- the main analysis (Parts 1–4)
├── src/
│   ├── qc.py                  <- quality checks on each 30-s clip
│   └── features.py            <- turns clean clips into numbers, plus a safe ranking helper
├── tests/
│   └── test_features.py       <- 15 quick tests on made-up signals
├── outputs/
│   ├── features.csv                 <- the finished feature table (3,000 clips × 49 features)
│   ├── feature_bp_correlations.csv  <- how each feature relates to blood pressure
│   ├── feature_ranking_bp.csv       <- ranked "best blood-pressure predictors" (leak-safe)
│   └── figures/               <- saved plots used in the memos
├── docs/
│   ├── qc_memo.md             <- Part 1: short quality-control memo
│   ├── feature_dictionary.md  <- Part 2: what every feature/column means
│   ├── methodology_audit.md   <- Part 3: audit of a colleague's model + the fix
│   ├── reflection.md          <- Part 4: what I'd do next
│   └── ai_usage_memo.md       <- how I used AI assistants
├── ai_conversation/
│   └── (1-6).md      <- the full AI conversations
└── data/                      <- raw waveforms live here locally; NOT submitted
    └── README.md
```

## What each part delivers (maps to the challenge)
- **Part 1  — Look at the data, check its quality.** Quality control in the notebook
  (8 figures) plus a short memo, `docs/qc_memo.md`.
- **Part 2 — Turn signals into features.** Feature code in `src/features.py`, the output
  table `outputs/features.csv` (3,000 clips × 49 features), and `docs/feature_dictionary.md`.
- **Part 3 — Audit a colleague's model.** `docs/methodology_audit.md` (what's wrong and
  how to fix it, with a runnable corrected version) plus `tests/test_features.py` (15 passing tests).
- **Part 4 — Reflect.** `docs/reflection.md`.
- **AI usage memo** — `docs/ai_usage_memo.md`; the full transcript is in `ai_conversation/`.

