"""Runnable demo of the corrected ICU-mortality method (Part 3, section 2).

Matched to the colleague model: keep the gradient-boosted tree, use NO scaler (a tree does not
care about scale), drop the redundant saps_mortality_risk feature, handle the rare positive
class with sample_weight (this model has no class_weight), and evaluate with patient-grouped,
stratified cross-validation that reports error bars + PR-AUC + calibration.

Made-up data with patient groups stands in for the real cohort so this runs anywhere.
"""
import numpy as np
# Searched for "scikit-learn grouped stratified k-fold for patient-level splits";
# StratifiedGroupKFold + sample_weight usage per the scikit-learn model_selection docs.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def make_synth(n_patients=200, seed=0):
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_patients), rng.integers(1, 6, n_patients))
    patient_effect = rng.normal(size=(n_patients, 6))
    X = rng.normal(size=(len(gid), 6)) + patient_effect[gid]     # gives each patient its own tendency
    risk = rng.random(n_patients)
    y = (rng.random(len(gid)) < 0.15 + 0.2 * risk[gid]).astype(int)   # rare outcome, tied to the patient
    return X, y, gid

def grouped_cv(X, y, gid, n_splits=5, seed=0):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc, pr, brier, leaks = [], [], [], []
    for tr, te in cv.split(X, y, gid):
        w = np.where(y[tr] == 1, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0)
        clf = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr], sample_weight=w)
        p = clf.predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y[te], p)); pr.append(average_precision_score(y[te], p))
        brier.append(brier_score_loss(y[te], p))
        leaks.append(bool(set(gid[tr]) & set(gid[te])))
    return auc, pr, brier, leaks

def main():
    X, y, gid = make_synth()
    auc, pr, brier, leaks = grouped_cv(X, y, gid)
    f = lambda a: f"{np.mean(a):.3f} +/- {np.std(a):.3f}"
    print("roc_auc :", f(auc)); print("pr_auc  :", f(pr)); print("brier   :", f(brier))
    print("event rate: %.1f%%" % (100 * y.mean()))
    print("any patient in both train & test (should be False):", any(leaks))

if __name__ == "__main__":
    main()
