# Methodology Audit — Part 3

A colleague built a model that predicts whether an ICU patient dies, and reported it as very
accurate (ROC-AUC ≈ 0.94). Their recipe: take SAPS-II-style severity features → rescale them
→ split the data randomly into train and test → train a gradient-boosted tree
(`GradientBoostingClassifier`).


## What really explains the 0.94 (and what doesn't)

Ordered by how much it actually matters here:

| Concern | Does it matter for *this* model? | The fix |
|---|---|---|
| **Patient leakage (the main suspect).** The patient id (`stay_id`) is correctly dropped from the features, but the train/test split is **random** instead of keeping each patient entirely on one side. | **Possibly large — but only if a patient appears in more than one row.** If there's one row per patient, the leak mostly disappears. This is the most likely reason 0.94 is inflated, but it depends on the data's shape, which the snippet doesn't reveal. | Split by patient (`StratifiedGroupKFold` / `GroupShuffleSplit` on `stay_id`). First, check how many rows each patient has. |
| **Only one split, one number, no error bars.** | **Matters a lot as a criticism of the reported score.** 0.94 has no confidence interval and could swing several points on a different split, especially with few deaths. | Grouped k-fold cross-validation; report the mean ± a 95% interval. |
| **ROC-AUC as the only score for a rare event.** | **Matters a lot.** AUC hides how well the model finds the rare deaths and whether its probabilities are trustworthy — the things that count clinically. | Also report PR-AUC, calibration (Brier score + reliability curve), and sensitivity at a fixed specificity. |
| **Rescaling done before the split.** | **Essentially zero effect here.** A tree splits on thresholds, so rescaling does nothing — leaking the scaler's mean/variance changes the score by thousandths at most. It is **not** what's inflating 0.94. | Just remove the scaler (a tree doesn't need it). It would only need to go inside the cross-validation loop *if* you switched to a model that cares about scale. |
| **A `saps_mortality_risk` feature said to "leak the target."** | **Zero effect — and mislabeled.** It's a fixed formula applied to `saps_ii` (constants, not learned from the labels), so it's redundancy, not train/test leakage. And `saps_ii` itself stays in the features; a tree is unaffected by re-expressing one feature, so removing it changes the score by exactly zero. | Drop it for tidiness, but know it won't move the number. The real issue is *intent* (below). |
| **Class imbalance.** | The "0.5 threshold hurts accuracy" critique **doesn't apply** — the code scores probabilities with AUC, so no threshold is used. The fair point is that imbalance isn't handled during training. | This model has no `class_weight`, so pass `sample_weight` to `.fit()` (or resample), and evaluate at a realistic operating point. |
| **Imputation leakage.** | **Not applicable — there's no imputation in the snippet.** (Reminder only: if you add one, fit it inside the cross-validation loop.) | — |
| **Timing / treatment confounding.** | **Can't tell from the snippet** — it depends on how the feature file was built. | Make sure the observation window ends before the outcome; treat drugs/fluids as covariates. |

## The concern the checklist misses: what is the model *for*?

The whole SAPS-II family of features was hand-designed to predict mortality. Feeding them (plus
a rearranged copy of one) into a classifier means the 0.94 **may just be re-deriving a score we
already have.** That's completely fine if the goal is "predict death," and uninformative if the
goal is "learn something *beyond* SAPS-II." This isn't a bug — it's a question about the goal —
and it's the most important thing to settle before celebrating 0.94.

## Corrected approach (end to end)

Verified to run — see `_audit_corrected_pipeline.py`. I **kept the same model family** so the
numbers reflect the *method* fixes, not a switch to a different model.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# X: features with the redundant transform dropped (saps_mortality_risk); NO scaler (tree model).
# y: 0/1 mortality;  groups: stay_id.  Imbalance via sample_weight (GBC has no class_weight).
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
auc, pr, brier = [], [], []
for tr, te in cv.split(X, y, groups):
    w = np.where(y[tr] == 1, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0)
    clf = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr], sample_weight=w)
    p = clf.predict_proba(X[te])[:, 1]
    auc.append(roc_auc_score(y[te], p)); pr.append(average_precision_score(y[te], p))
    brier.append(brier_score_loss(y[te], p))
# report mean ± std for each metric; a grouped AUC well below 0.94 would reveal patient leakage.
```

The changes that **actually matter here**: (i) split by patient; (ii) report cross-validated
results with error bars, plus PR-AUC and calibration alongside AUC. Removing the scaler and the
redundant feature are cleanliness fixes with ~zero effect on this tree. If you want to know how
much of the 0.94 was patient leakage, compare the grouped-split AUC to the random-split AUC —
the gap is your answer.

## Metrics for clinical deployment

Report ROC-AUC **and** PR-AUC (the positive class is rare); calibration (Brier score +
reliability curve) so the probabilities can be trusted at the bedside; sensitivity at a fixed,
clinically acceptable specificity, plus the alert rate at that threshold (how many alarms staff
would get); optionally net benefit (decision-curve analysis); all as mean ± interval from
patient-grouped cross-validation, and broken out by key subgroups to surface bias.

## Unit test (bonus)

`../tests/test_features.py` — **15 tests, all passing** (`pytest -q`).
`test_predictor_columns_enforce_leakage_rules` turns the anti-leakage rule into an automatic
check: the feature selector can never hand back outcome-derived or label columns.
