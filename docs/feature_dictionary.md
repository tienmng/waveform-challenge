# Feature Dictionary — `outputs/features.csv`

This file lists every column in the feature table. There is **one row per 30-second clip**,
and the columns are the numbers ("features") we squeezed out of the raw signals. Features are
computed for **every** clip; quality flags are carried along so any later task can filter as it
likes. No prediction target is baked in — the features are extracted now to be used later.

## First, the one rule that prevents cheating

Whether a column is safe to use as an input **depends on what you're trying to predict.** The
arterial-line (ABP) columns are the danger:

| What you're predicting | Safe inputs | Do NOT use |
|---|---|---|
| **Cuffless blood pressure** (predict `sbp`/`dbp`) | ECG/HRV, PPG, and PAT columns | **the `ABP_*` columns and `shock_index`** — the blood-pressure labels *are* the arterial trace (systolic ≈ its 95th percentile, diastolic ≈ its 5th, correlation ≈ 0.99), so these columns are literally the answer. |
| **An ICU outcome** (death / becoming unstable) | ECG/HRV, PPG, PAT, **and `ABP_*`** | features taken from a time window that overlaps or follows the outcome (that's a different kind of leakage); also watch for treatment effects. No outcome label ships with this dataset. |

This rule is **enforced in code** by `features.predictor_columns(df, target)` (see the API
section). Use it instead of picking columns by hand, so a blood-pressure run can never
accidentally see the arterial columns.

## Housekeeping columns (id, labels, quality flags)
| Column | Meaning |
|---|---|
| `patient`, `segment`, `split` | patient id, clip number (0–29), and which data split it's in |
| `sbp`, `dbp` | the systolic / diastolic blood-pressure label for this clip (mmHg), taken from the arterial line |
| `ecg_ok`, `ppg_ok`, `abp_ok` | did each signal pass its quality check? (see `src/qc.py`) |
| `keep_for_model` | ECG, PPG, and label all clean — the recommended subset for a cuffless-BP model |

## ECG / heart-rate-variability (safe for any target)
Derived from the heartbeats detected in the ECG.
| Column | Plain meaning |
|---|---|
| `ecg_n_rpeaks` | how many heartbeats were found in the clip |
| `hr_bpm` | heart rate (beats per minute) |
| `rr_mean_ms` | average time between beats (milliseconds) |
| `sdnn_ms` | how much the beat-to-beat timing varies overall (a standard HRV measure) |
| `rmssd_ms` | short-term beat-to-beat variability (reflects the "rest and digest" nervous system) |
| `rr_cv` | beat-timing variability expressed relative to the average |

> **Why there's no frequency-based HRV (LF/HF).** Those measures need low-frequency rhythms
> that only show up over **1–2 minutes** of data. On a **30-second** clip you simply can't see
> them reliably, so they'd be mostly noise. They're left out on purpose. `sdnn_ms`, `rmssd_ms`,
> and `rr_cv` are the defensible short-window HRV measures. Frequency-based HRV becomes
> appropriate only if you use longer clips or stitch neighboring clips together.

## PPG (safe for any target) — most `_mean` columns have a matching `_std`
The PPG is the finger-clip pulse wave; these describe the *shape* of each pulse. For most
features we report the average (`_mean`) and the spread (`_std`) across the beats in a clip.
| Column (mean/std) | Plain meaning |
|---|---|
| `ppg_n_beats` | beats detected (a count; no `_std`) |
| `ppg_sys_amp` | pulse height (in rescaled units) |
| `ppg_upstroke_ms` | how long the pulse takes to rise from bottom to peak |
| `ppg_crest_frac` | where the peak sits within the pulse, as a fraction |
| `ppg_pulse_dur_ms` | how long one pulse lasts |
| `ppg_w25_ms`, `ppg_w50_ms`, `ppg_w75_ms` | pulse width measured at 25% / 50% / 75% of its height |
| `ppg_sysdia_area` | area under the rising part vs. the falling part of the pulse |
| `ppg_aug_index` | size of the reflected wave relative to the main pulse (a stiffness cue) |
| `ppg_vpg_max` | steepest part of the upstroke (first derivative) |
| `ppg_apg_ba` | a shape ratio from the second derivative, used as an arterial-stiffness index |
| `ppg_skew` | a pulse-shape quality score — **one value per clip, no `_std`** |
| `ppg_spec_purity` | how much of the signal's energy sits in the normal pulse frequency band — **one value per clip, no `_std`** |

> `ppg_skew` and `ppg_spec_purity` describe the whole clip (they're reused from the quality
> checks), so unlike the per-beat features they don't have a mean/std pair.

## PAT — cross-signal timing (safe; the classic cuffless-BP feature)
PAT = pulse arrival time: how long the pulse takes to travel from the heartbeat (ECG) to the
finger (PPG). This timing tracks blood pressure.
| Column (mean/std) | Plain meaning |
|---|---|
| `pat_foot_ms` | time from heartbeat to the *start* of the finger pulse |
| `pat_peak_ms` | time from heartbeat to the *peak* of the finger pulse |

> Both the start and the peak are measured on the same pulse trace, so they're consistent. In
> this dataset the absolute travel time is large but steady; the blood-pressure information is
> in the small **beat-to-beat changes**, not the absolute value.

## Arterial (ABP) hemodynamics  ⚠ (leaks blood pressure — for outcome targets only)
Measured directly from the arterial line. **Do not use these to predict blood pressure** (they
*are* the blood pressure). They're here as ready-made inputs for a different target, like ICU
outcomes.
| Column | Plain meaning |
|---|---|
| `abp_n_beats_hemo` | arterial beats detected |
| `abp_sbp`, `abp_dbp`, `abp_map`, `abp_pp` | average systolic / diastolic / mean / pulse pressure (mmHg) |
| `abp_map_std`, `abp_pp_std`, `abp_map_cv`, `abp_pp_cv` | how much mean and pulse pressure vary beat to beat |
| `abp_ppv_pct`, `abp_spv_pct` | a robust stand-in for respiratory pressure variation (a true one needs the breathing channel) |
| `abp_map_lt65_frac`, `abp_map_lt60_frac` | fraction of beats with dangerously low mean pressure (below 65 / 60 mmHg) |
| `shock_index` | heart rate ÷ systolic pressure (above ~0.9 suggests instability) |

> `abp_map` is the **true average of the pressure waveform**, not the textbook
> `(SBP + 2·DBP)/3` estimate (that estimate is only for when you don't have the waveform).
> `shock_index` reuses the same heart rate as `hr_bpm`.

---

# Feature selection / signal-ranking helper

The extraction step commits to **no** target, but `features.py` includes a small,
leakage-aware helper to flag the **most promising signals** for a chosen target *before* you
train a full model. It imports scikit-learn only when needed, so plain feature extraction has
no machine-learning dependency. All of these work on a loaded feature table (e.g.
`outputs/features.csv`) plus a target you supply.

## `predictor_columns(df, target)`
Returns the safe input columns and **enforces the rule above**:
- `target="bp"` → ECG/HRV + PPG + PAT only (the `abp_*` columns and `shock_index` are removed).
- `target="outcome"` → all feature columns, including `abp_*`.
- Housekeeping / label columns (`patient`, `segment`, `split`, `sbp`, `dbp`, `*_ok`,
  `keep_for_model`, …) are never returned as inputs.

## `rank_features(df, y, target, task=...)`
Ranks the safe inputs using **several methods together**, so no single method's blind spot
dominates:

| Method | Output column | What it catches |
|---|---|---|
| Mutual information | `mutual_info` | any feature↔target relationship, even non-straight-line |
| Random-forest importance | `rf_importance` | importance that accounts for feature interactions |
| Lasso \|coef\| (regression only) | `lasso_abs_coef` | straight-line signal after removing overlap between features |
| Variance | `variance`, `near_zero_var` | flags flat features worth pruning |

The result is a table, one row per feature, sorted by `consensus_rank` (the average of the
method rankings; **1 = strongest**), tagged with its `modality` (`ECG/HRV`, `PPG`, `PAT`,
`ABP`). Set `task` to `"regression"` (blood pressure / continuous) or `"classification"`
(an outcome label). Missing values are filled in **for ranking only** — the saved table is
left untouched.

> **Easy upgrade:** swap the random-forest step for LightGBM/XGBoost importances for large
> automated feature sets. The spot to change is marked in `features.py`.

## `stability_selection(df, y, target, groups=..., ...)`
Re-runs the ranking on many bootstrap resamples and reports how often each feature lands near
the top — a robustness check for messy ICU data. **Resampling is done over patients** (pass
`groups=df['patient']`), never over rows: resampling rows would mix a patient's clips across
samples and make features look more stable than they are. The output has a
`selection_frequency`, a `stable` flag, and the `modality` tag.

## `suggest_signals(df, target="bp", y=None, groups=None, top=15, stability=True)`
One call that runs the ranking and (optionally) the stability check and joins them:
- `target="bp"` → `y` defaults to `df['sbp']`; arterial columns are auto-excluded; treated as regression.
- `target="outcome"` → **you must pass `y`** (no outcome label ships); arterial columns allowed; treated as classification.
- `groups` defaults to `df['patient']` so resampling respects patient structure.

Returns `(ranking, stability_df | None)`.

```python
import pandas as pd, features as F
df = pd.read_csv("outputs/features.csv")

# Which signals look most predictive of systolic BP? (arterial columns auto-excluded)
ranking, stability = F.suggest_signals(df, target="bp", top=15)
print(ranking[["modality", "mutual_info", "rf_importance",
               "consensus_rank", "selection_frequency"]])

# For an outcome target you supply the label yourself; arterial columns are then allowed:
# ranking, stability = F.suggest_signals(df, target="outcome", y=my_outcome_vector)
```

## Reminders for when you actually model
- **Split by `patient`, not by clip** (`GroupKFold(groups=df['patient'])`). With ~30 clips per
  patient, splitting by clip lets the model peek at the same patient in train and test and
  inflates the score — the same reason stability resamples patients.
- For an **outcome** target, take features from a window that **ends before** the moment you're
  predicting, and account for treatments (drugs and fluids move arterial pressure on their own).
- The `sbp`/`dbp` label columns sit in the same table as the features; `predictor_columns`
  drops them for you, but if you build the input matrix by hand, remember to drop them (and the
  arterial block for a blood-pressure target).
