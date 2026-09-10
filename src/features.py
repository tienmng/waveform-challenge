"""
Feature engineering for cuffless blood-pressure estimation (Part 2).

This turns each clean 30-second clip into a row of numbers ("features") a model could later
learn from. It pulls features from every signal -- the ECG, the PPG (finger pulse), the timing
between them (PAT), and the arterial line (ABP) -- and does NOT commit to a prediction target.
Every clip gets features, and quality flags travel with them so a later step can filter.

One rule matters most: whether a feature is safe to use depends on what you're predicting
(details in docs/feature_dictionary.md):
* To predict blood pressure (SBP/DBP), the labels ARE the arterial trace (SBP ~ its 95th
  percentile, DBP ~ its 5th; correlation ~ 0.99), so the ABP_* columns must NOT be used as
  inputs -- use the ECG/PPG/PAT features only.
* To predict an ICU outcome (death / instability), the ABP_* columns are fair, useful inputs
  (as long as they come from a time window before the outcome).

Each clip becomes ONE row: the per-beat measurements are summarized as their average and spread.

A few implementation notes:
* ABP mean pressure (MAP) is the true average of the waveform, not the (SBP+2*DBP)/3 estimate.
* Heartbeats and the pulse landmarks are found once per clip and reused, for speed and to keep
  the numbers consistent.
* shock_index reuses the same heart rate as hr_bpm, so the two values agree.

At the bottom is a small, leakage-aware ranking helper (predictor_columns / rank_features /
stability_selection / suggest_signals) that flags the most promising signals for a target
BEFORE training a full model, and enforces the safety rule above.

Pulse heights are measured on the rescaled PPG (raw PPG units aren't comparable between
patients); timings are in ms and ratios are unitless, so both compare across patients.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
import qc
from qc import FS, HR_RANGE_BPM

HR_MIN, HR_MAX = HR_RANGE_BPM


def _agg(vals, name):
    """mean + std of a per-beat list, NaN-safe."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
    return {f"{name}_mean": float(np.mean(v)), f"{name}_std": float(np.std(v))}


# ----------------------------- ECG: HR / HRV -----------------------------
def hrv_features(ecg, fs=FS, r=None):
    """Heart rate and heart-rate variability from the ECG beats.
    Pass precomputed beats `r` to skip re-detecting them."""
    if r is None:
        r = qc.find_rpeaks(ecg, fs)
    out = dict(ecg_n_rpeaks=int(len(r)))
    nan_keys = ("hr_bpm", "rr_mean_ms", "sdnn_ms", "rmssd_ms", "rr_cv")
    if len(r) < 3:
        out.update({k: np.nan for k in nan_keys})
        return out
    rr = np.diff(r) / fs                                    # seconds
    rr = rr[(rr >= 60.0 / HR_MAX) & (rr <= 60.0 / HR_MIN)]  # keep only plausible beat gaps
    if rr.size < 2:
        out.update({k: np.nan for k in nan_keys})
        return out
    rr_ms = rr * 1000.0
    # SDNN, RMSSD, and RR-CV are standard heart-rate-variability measures (ESC/NASPE 1996).
    # ddof=1 uses the sample standard deviation, the HRV convention. Note: because implausible
    # beat gaps were filtered out, a difference taken across a dropped beat can slightly inflate
    # RMSSD/SDNN; over a clean 30-s clip this effect is small.
    out.update(hr_bpm=float(60.0 / rr.mean()), rr_mean_ms=float(rr_ms.mean()),
               sdnn_ms=float(rr_ms.std(ddof=1)),
               rmssd_ms=float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))),
               rr_cv=float(rr_ms.std(ddof=1) / (rr_ms.mean() + 1e-12)))
    return out


# ----------------------------- PPG fiducials -----------------------------
def _ppg_norm_and_feet(ppg, fs=FS):
    try:
        xf = qc._bandpass(ppg, 0.5, 8.0)
    except Exception:
        xf = ppg - np.mean(ppg)
    xn = (xf - xf.mean()) / (xf.std() + 1e-12)
    feet, _ = find_peaks(-xn, distance=int(fs * 60 / HR_MAX), prominence=0.3)
    return xn, feet


def _ppg_peaks_from_feet(xn, feet):
    """The peak of each pulse (from one pulse start to the next). It's read off the SAME
    rescaled trace as the pulse starts, so the two PAT landmarks stay consistent."""
    pk = []
    for i in range(len(feet) - 1):
        seg = xn[feet[i]:feet[i + 1]]
        if seg.size:
            pk.append(feet[i] + int(np.argmax(seg)))
    return np.asarray(pk, dtype=int)


# ----------------------------- PPG morphology -----------------------------
def ppg_morphology_features(ppg, fs=FS, xn=None, feet=None):
    if xn is None or feet is None:
        xn, feet = _ppg_norm_and_feet(ppg, fs)
    amp, up_t, crest, dur, w25, w50, w75, sys_dia, aug = ([] for _ in range(9))
    for i in range(len(feet) - 1):
        beat = xn[feet[i]:feet[i + 1]]
        if len(beat) < 8:
            continue
        foot = beat[0]
        pidx = int(np.argmax(beat))
        if pidx == 0:
            continue
        a = beat[pidx] - foot                       # pulse height (rescaled units)
        if a <= 1e-6:
            continue
        amp.append(a)
        up_t.append(pidx / fs * 1000.0)             # rise time, bottom to peak (ms)
        crest.append(pidx / len(beat))              # where the peak sits, as a fraction
        dur.append(len(beat) / fs * 1000.0)         # how long the pulse lasts (ms)
        # pulse width measured at 25/50/75% of its height (ms)
        for frac, store in ((0.25, w25), (0.50, w50), (0.75, w75)):
            lvl = foot + frac * a
            above = np.where(beat >= lvl)[0]
            store.append((above[-1] - above[0]) / fs * 1000.0 if above.size >= 2 else np.nan)
        # area of the rising part vs the falling part of the pulse
        b0 = beat - foot
        s_area, d_area = b0[:pidx].sum(), b0[pidx:].sum()
        sys_dia.append(s_area / (d_area + 1e-9))
        # augmentation index: size of the reflected wave vs the main pulse height
        dia = beat[pidx:]
        sp, _ = find_peaks(dia)
        aug.append((dia[sp[0]] - foot) / a if sp.size else np.nan)
    out = dict(ppg_n_beats=int(max(0, len(feet) - 1)))
    out.update(_agg(amp, "ppg_sys_amp")); out.update(_agg(up_t, "ppg_upstroke_ms"))
    out.update(_agg(crest, "ppg_crest_frac")); out.update(_agg(dur, "ppg_pulse_dur_ms"))
    out.update(_agg(w25, "ppg_w25_ms")); out.update(_agg(w50, "ppg_w50_ms"))
    out.update(_agg(w75, "ppg_w75_ms")); out.update(_agg(sys_dia, "ppg_sysdia_area"))
    out.update(_agg(aug, "ppg_aug_index"))
    return out


# ----------------------------- PPG derivatives (VPG/APG) -----------------------------
def ppg_derivative_features(ppg, fs=FS, xn=None, feet=None):
    if xn is None or feet is None:
        xn, feet = _ppg_norm_and_feet(ppg, fs)
    vpg = np.gradient(xn) * fs
    apg = np.gradient(vpg) * fs
    vmax, ba = [], []
    for i in range(len(feet) - 1):
        vseg, aseg = vpg[feet[i]:feet[i + 1]], apg[feet[i]:feet[i + 1]]
        if len(aseg) < 8:
            continue
        vmax.append(float(np.max(vseg)))                     # max systolic upslope
        # Concept/reference: second-derivative PPG (APG) a- and b-waves as an
        # arterial-stiffness index (Takazawa et al., 1998).
        # APG a-wave (first, largest positive) then b-wave (following negative) -> b/a
        ap, _ = find_peaks(aseg)
        if ap.size:
            a_i = ap[int(np.argmax(aseg[ap]))]
            after = aseg[a_i:]
            bn, _ = find_peaks(-after)
            if bn.size and aseg[a_i] > 1e-6:
                ba.append(float(after[bn[0]] / aseg[a_i]))
    out = {}
    out.update(_agg(vmax, "ppg_vpg_max")); out.update(_agg(ba, "ppg_apg_ba"))
    return out


# ----------------------------- cross-signal PAT / PTT -----------------------------
def pat_features(ecg, ppg, fs=FS, r=None, xn=None, feet=None, ppg_peaks=None):
    """Pulse arrival time: R-peak -> next PPG foot and -> next PPG peak (ms).
    In this dataset the beat-to-pulse delay is large (~400-490 ms) but very steady; only the
    small BEAT-TO-BEAT changes carry blood-pressure information, so we just measure it
    consistently. The per-beat search is capped BELOW the time between beats so we never
    accidentally grab a landmark from the next heartbeat. The pulse start and peak come from
    the same rescaled PPG trace, so the two landmarks stay consistent."""
    # Concept: pulse arrival/transit time (ECG R-peak -> PPG foot/peak) as a cuffless-BP
    # surrogate; see the PAT/PTT cuffless-BP literature (e.g. Mukkamala et al., 2015).
    if r is None:
        r = qc.find_rpeaks(ecg, fs)
    if xn is None or feet is None:
        xn, feet = _ppg_norm_and_feet(ppg, fs)
    if ppg_peaks is None:
        ppg_peaks = _ppg_peaks_from_feet(xn, feet)
    feet = np.asarray(feet, dtype=int)
    pk = np.asarray(ppg_peaks, dtype=int)
    rr = float(np.median(np.diff(r)) / fs) if len(r) >= 2 else 0.8   # seconds
    lo = 0.05 * fs
    hi_f = min(0.65 * fs, 0.90 * rr * fs)     # foot: below RR
    hi_p = min(0.90 * fs, 0.97 * rr * fs)     # peak: below RR
    pat_foot, pat_peak = [], []
    for rp in r:
        ff = feet[(feet - rp >= lo) & (feet - rp <= hi_f)]
        if ff.size:
            pat_foot.append((ff[0] - rp) / fs * 1000.0)
        pp = pk[(pk - rp >= lo) & (pk - rp <= hi_p)]
        if pp.size:
            pat_peak.append((pp[0] - rp) / fs * 1000.0)
    out = {}
    out.update(_agg(pat_foot, "pat_foot_ms")); out.update(_agg(pat_peak, "pat_peak_ms"))
    return out


# ===================== ABP HEMODYNAMIC FEATURES (separate group) =====================
# IMPORTANT: these come straight from the ARTERIAL LINE (ABP). Do NOT use them to predict
# blood pressure -- the SBP/DBP labels ARE the arterial trace (SBP ~ its 95th percentile,
# DBP ~ its 5th; correlation ~ 0.99), so using them would be handing the model the answer.
# They're here as ready-made inputs for a DIFFERENT target -- an ICU outcome like death or
# instability -- where the arterial signal is fair to use (and ICU patients usually already
# have an arterial line). If you use them, take them from a time window that ENDS BEFORE the
# moment you're predicting, and remember that drugs and fluids move arterial pressure on their own.
_ABP_NAN_KEYS = ("abp_sbp", "abp_dbp", "abp_map", "abp_pp", "abp_map_std", "abp_pp_std",
                 "abp_map_cv", "abp_pp_cv", "abp_ppv_pct", "abp_spv_pct",
                 "abp_map_lt65_frac", "abp_map_lt60_frac", "shock_index")


def abp_hemodynamic_features(abp, ecg=None, fs=FS, hr_bpm=None):
    """Arterial-pressure summaries for one clip (mmHg).
    Returns mean/systolic/diastolic and pulse pressure, how much they vary beat to beat, a
    robust stand-in for breathing-related pressure swings (PPV/SPV), the fraction of beats
    with dangerously low mean pressure, and a shock index (heart rate / systolic).

    Mean pressure (MAP) is the true average of the pressure waveform, not the textbook
    (SBP+2*DBP)/3 estimate, because we have the actual waveform. shock_index reuses the heart
    rate from hrv_features() when you pass hr_bpm; otherwise it computes it from `ecg` the
    same way (not just a raw beat count)."""
    abp = np.asarray(abp, dtype=float)
    out = dict(abp_n_beats_hemo=0)
    if not np.isfinite(abp).any():
        out.update({k: np.nan for k in _ABP_NAN_KEYS})
        return out
    # NaN-guard: fill gaps for peak detection, but skip any beat overlapping missing samples.
    abp_f = abp.copy()
    if np.isnan(abp_f).any():
        abp_f[np.isnan(abp_f)] = np.nanmedian(abp_f)
    prom = float(np.nanstd(abp)) * 0.3 + 1e-6
    feet, _ = find_peaks(-abp_f, distance=int(fs * 60 / HR_MAX), prominence=prom)
    sbp_b, dbp_b, pp_b, map_b = [], [], [], []
    for i in range(len(feet) - 1):
        sl = slice(feet[i], feet[i + 1])
        if not np.isfinite(abp[sl]).all():   # original had missing samples here -> skip
            continue
        beat = abp_f[sl]
        if len(beat) < 5:
            continue
        dbp = float(beat.min()); sbp = float(beat.max())
        if sbp - dbp < 5:            # implausibly small pulse -> skip
            continue
        sbp_b.append(sbp); dbp_b.append(dbp); pp_b.append(sbp - dbp)
        map_b.append(float(beat.mean()))     # true waveform MAP (area-under-curve)
    out["abp_n_beats_hemo"] = len(pp_b)
    if len(pp_b) < 3:
        out.update({k: np.nan for k in _ABP_NAN_KEYS})
        return out
    sbp_b, dbp_b, pp_b, map_b = map(np.asarray, (sbp_b, dbp_b, pp_b, map_b))

    def robust_var(v):  # robust % spread: (90th - 10th percentile) / median, outlier-safe
        return float((np.percentile(v, 90) - np.percentile(v, 10)) / (np.median(v) + 1e-9) * 100.0)

    out.update(
        abp_sbp=float(sbp_b.mean()), abp_dbp=float(dbp_b.mean()),
        abp_map=float(map_b.mean()), abp_pp=float(pp_b.mean()),
        abp_map_std=float(map_b.std()), abp_pp_std=float(pp_b.std()),
        abp_map_cv=float(map_b.std() / (map_b.mean() + 1e-9)),
        abp_pp_cv=float(pp_b.std() / (pp_b.mean() + 1e-9)),
        # stand-in for breathing-related pressure variation (robust % spread).
        # A true respiratory PPV needs the breathing channel (excluded here), so this is a
        # robust spread proxy, not a calibrated PPV -- treat it as report-only.
        abp_ppv_pct=robust_var(pp_b),
        abp_spv_pct=robust_var(sbp_b),
        # low-blood-pressure burden: fraction of beats with mean pressure below the threshold
        abp_map_lt65_frac=float((map_b < 65).mean()),
        abp_map_lt60_frac=float((map_b < 60).mean()),
    )
    # shock index = heart rate / systolic; heart rate matches hr_bpm.
    if hr_bpm is None and ecg is not None:
        hr_bpm = hrv_features(ecg, fs).get("hr_bpm", np.nan)
    ok_hr = hr_bpm is not None and np.isfinite(hr_bpm) and out["abp_sbp"] > 0
    out["shock_index"] = float(hr_bpm / out["abp_sbp"]) if ok_hr else np.nan
    return out


def build_hemodynamic_matrix(dbpath, qc_df, abp_clean_only=True):
    """Build the SEPARATE arterial-pressure feature table (one row per clip).
    Uses clips where the arterial signal passed quality (abp_ok), not ECG/PPG. Not for
    blood-pressure prediction."""
    import os, pandas as pd
    sub = qc_df[qc_df.abp_ok] if (abp_clean_only and "abp_ok" in qc_df) else qc_df
    rows, cache = [], {}
    for _, q in sub.iterrows():
        pid, seg = q["patient"], int(q["segment"])
        if pid not in cache:
            cache.clear()
            cache[pid] = dict(abp=np.load(os.path.join(dbpath, "abp", f"{pid}_abp.npy")),
                              ecg=np.load(os.path.join(dbpath, "ecg", f"{pid}_ecg.npy")))
        d = cache[pid]
        row = dict(patient=pid, segment=seg)
        if "split" in q: row["split"] = q["split"]
        # labels kept only for reference / linking to an outcome later, never as inputs
        row["label_sbp"] = float(q.get("label_sbp", np.nan)); row["label_dbp"] = float(q.get("label_dbp", np.nan))
        row.update(abp_hemodynamic_features(d["abp"][seg], d["ecg"][seg]))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------- orchestration -----------------------------
def extract_features(ecg, ppg, fs=FS):
    """All ECG/PPG/PAT features for one clip, as a flat dictionary.
    The heartbeats and the rescaled PPG + its pulse landmarks are computed once here and
    reused by the helper functions."""
    r = qc.find_rpeaks(ecg, fs)                     # once
    xn, feet = _ppg_norm_and_feet(ppg, fs)          # once
    peaks = _ppg_peaks_from_feet(xn, feet)          # once (consistent with feet)
    f = {}
    f.update(hrv_features(ecg, fs, r=r))
    f.update(ppg_morphology_features(ppg, fs, xn=xn, feet=feet))
    f.update(ppg_derivative_features(ppg, fs, xn=xn, feet=feet))
    f.update(pat_features(ecg, ppg, fs, r=r, xn=xn, feet=feet, ppg_peaks=peaks))
    # lightweight PPG shape/quality descriptors (reuse qc)
    f["ppg_skew"] = qc.ppg_skew_sqi(ppg)
    f["ppg_spec_purity"] = qc.ppg_spectral_purity(ppg)
    return f


FEATURE_COLUMNS = None  # set after first build for reference


def build_feature_matrix(dbpath, qc_df, only_keep=True, verbose=False):
    """Build the per-clip feature table over a quality-checked sample.
    qc_df: table from qc.patient_qc (needs patient, segment, keep_for_model, label_sbp,
    label_dbp, and optionally split). Returns a table with one row per clip."""
    import os
    sub = qc_df[qc_df.keep_for_model] if only_keep else qc_df
    rows = []
    cache = {}
    for _, q in sub.iterrows():
        pid, seg = q["patient"], int(q["segment"])
        if pid not in cache:
            cache.clear()
            cache[pid] = dict(
                ecg=np.load(os.path.join(dbpath, "ecg", f"{pid}_ecg.npy")),
                ppg=np.load(os.path.join(dbpath, "ppg", f"{pid}_ppg.npy")),
            )
        d = cache[pid]
        row = dict(patient=pid, segment=seg,
                   sbp=float(q.get("label_sbp", np.nan)), dbp=float(q.get("label_dbp", np.nan)))
        if "split" in q:
            row["split"] = q["split"]
        row.update(extract_features(d["ecg"][seg], d["ppg"][seg]))
        rows.append(row)
    return __import__("pandas").DataFrame(rows)


# ----------------------------- comprehensive feature bank -----------------------------
def build_all_features(dbpath, qc_df, only_keep=False):
    """One row per clip with ALL the features (ECG/HRV, PPG shape and derivatives, the
    cross-signal PAT timing, and arterial-pressure summaries) plus id columns and quality
    flags. By default it covers EVERY clip (only_keep=False) and carries
    ecg_ok/ppg_ok/abp_ok/keep_for_model so a later step can filter. Nothing is pre-excluded;
    the safety rules are in docs/feature_dictionary.md."""
    import os, pandas as pd
    sub = qc_df[qc_df.keep_for_model] if only_keep else qc_df
    meta = ["patient", "segment", "split", "sbp", "dbp", "ecg_ok", "ppg_ok", "abp_ok", "keep_for_model"]
    rows, cache = [], {}
    for _, q in sub.iterrows():
        pid, seg = q["patient"], int(q["segment"])
        if pid not in cache:
            cache.clear()
            cache[pid] = {m: np.load(os.path.join(dbpath, m, f"{pid}_{m}.npy")) for m in ("ecg", "ppg", "abp")}
        d = cache[pid]
        row = dict(patient=pid, segment=seg, split=q.get("split", "train"),
                   sbp=float(q.get("label_sbp", np.nan)), dbp=float(q.get("label_dbp", np.nan)),
                   ecg_ok=bool(q.get("ecg_ok", False)), ppg_ok=bool(q.get("ppg_ok", False)),
                   abp_ok=bool(q.get("abp_ok", False)), keep_for_model=bool(q.get("keep_for_model", False)))
        feat = extract_features(d["ecg"][seg], d["ppg"][seg])           # ECG/HRV, PPG, PAT, shape
        row.update(feat)
        # reuse the filtered-RR HR from extract_features so shock_index and hr_bpm agree,
        # and so R-peaks are not detected a third time for this segment.
        row.update(abp_hemodynamic_features(d["abp"][seg], d["ecg"][seg], fs=FS,
                                            hr_bpm=feat.get("hr_bpm")))  # ABP hemodynamics
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = meta + [c for c in df.columns if c not in meta]
    return df[ordered]


# ============================================================================
#  Leakage-aware feature ranking: "which signals look most predictive?"
# ----------------------------------------------------------------------------
#  These work on a built feature table + a chosen target. They rank features by
#  several methods and, importantly, ENFORCE the safety rules from
#  feature_dictionary.md so a blood-pressure run can never see the ABP_* columns.
#  scikit-learn is imported only when needed, so plain extraction needs no ML libs.
# ============================================================================
_META_COLS = {"patient", "segment", "split", "sbp", "dbp", "label_sbp", "label_dbp",
              "ecg_ok", "ppg_ok", "abp_ok", "keep_for_model"}
_ECG_HRV = {"ecg_n_rpeaks", "hr_bpm", "rr_mean_ms", "sdnn_ms", "rmssd_ms", "rr_cv"}


def _modality_of(col):
    if col.startswith("pat_"):
        return "PAT"
    if col.startswith("ppg_"):
        return "PPG"
    if col.startswith("abp_") or col == "shock_index":
        return "ABP"
    if col in _ECG_HRV:
        return "ECG/HRV"
    return "other"


def predictor_columns(df, target="bp"):
    """The safe input columns for a target.
    target='bp'      -> ECG/HRV, PPG, PAT only (the ABP_* columns and shock_index are the answer).
    target='outcome' -> all feature columns including ABP_* (fair for ICU outcomes).
    Id, quality-flag, and label columns are never returned as inputs."""
    cols = [c for c in df.columns if c not in _META_COLS]
    if target == "bp":
        cols = [c for c in cols if not (c.startswith("abp_") or c == "shock_index")]
    elif target != "outcome":
        raise ValueError("target must be 'bp' or 'outcome'")
    return cols


def _prep_Xy(df, cols, y):
    """Numeric X (median-imputed, all-NaN columns dropped) and finite-y rows. Ranking-time
    imputation only -- it never touches the persisted feature bank."""
    import pandas as pd
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    X, y = X.loc[ok], y[ok]
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep].fillna(X[keep].median(numeric_only=True))
    return X, y, keep, ok


def rank_features(df, y, target="bp", task="regression", n_estimators=300, random_state=0):
    """Rank the safe inputs by AGREEMENT across several methods:
      * mutual information (catches any relationship, even non-straight-line),
      * random-forest importance (accounts for interactions),
      * Lasso |coef| on standardized inputs (straight-line signal; regression only).
    Also reports each feature's spread and a near-flat flag. Returns a table indexed by
    feature, sorted by consensus_rank (1 = strongest), with a `modality` column.

    task: 'regression' (blood pressure / continuous) or 'classification' (an outcome label).
    To use a stronger method, swap the random-forest block for LightGBM/XGBoost importances;
    nothing else changes."""
    import pandas as pd
    # Searched for "scikit-learn feature selection: mutual_info, RandomForest importance,
    # LassoCV"; estimator usage follows the scikit-learn User Guide (Feature selection).
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV

    cols = predictor_columns(df, target)
    X, yv, cols, _ = _prep_Xy(df, cols, y)
    res = pd.DataFrame(index=cols)
    res["modality"] = [_modality_of(c) for c in cols]

    var = X.var(axis=0, ddof=1)
    res["variance"] = var.values
    res["near_zero_var"] = (var.values <= 1e-8)

    mi_fn = mutual_info_regression if task == "regression" else mutual_info_classif
    res["mutual_info"] = mi_fn(X.values, yv, random_state=random_state)

    RF = RandomForestRegressor if task == "regression" else RandomForestClassifier
    rf = RF(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X.values, yv if task == "regression" else yv.astype(int))
    res["rf_importance"] = rf.feature_importances_

    if task == "regression":
        Xs = StandardScaler().fit_transform(X.values)
        lasso = LassoCV(cv=5, random_state=random_state, n_jobs=-1, max_iter=5000).fit(Xs, yv)
        res["lasso_abs_coef"] = np.abs(lasso.coef_)

    score_cols = [c for c in ("mutual_info", "rf_importance", "lasso_abs_coef") if c in res]
    res["consensus_rank"] = res[score_cols].rank(ascending=False).mean(axis=1)
    return res.sort_values("consensus_rank")


def stability_selection(df, y, target="bp", task="regression", groups=None,
                        n_boot=50, top_frac=0.25, sel_threshold=0.60,
                        n_estimators=200, random_state=0):
    """How often each feature lands near the top across many bootstrap resamples. ICU data is
    noisy, so we resample over PATIENTS (pass groups=df['patient']) -- resampling rows would
    mix a patient's clips across samples and make features look steadier than they are.
    Features with selection_frequency >= sel_threshold are the stable ones. Returns a table
    sorted by selection_frequency."""
    # Concept: stability selection (Meinshausen & Buhlmann, 2010) -- bootstrap the ranking
    # and keep features chosen often; resampling over patients avoids group leakage.
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    cols = predictor_columns(df, target)
    X, yv, cols, ok = _prep_Xy(df, cols, y)
    g = np.asarray(groups)[ok] if groups is not None else np.arange(len(yv))
    uniq = np.unique(g)
    k_top = max(1, int(round(top_frac * len(cols))))
    rng = np.random.default_rng(random_state)
    counts = pd.Series(0.0, index=cols)
    RF = RandomForestRegressor if task == "regression" else RandomForestClassifier
    Xv = X.values
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)         # bootstrap patients
        idx = np.concatenate([np.where(g == p)[0] for p in pick])
        rf = RF(n_estimators=n_estimators, random_state=int(rng.integers(1_000_000_000)), n_jobs=-1)
        yi = yv[idx]
        rf.fit(Xv[idx], yi if task == "regression" else yi.astype(int))
        top = np.argsort(rf.feature_importances_)[::-1][:k_top]
        counts.iloc[top] += 1.0
    out = (counts / n_boot).to_frame("selection_frequency")
    out["stable"] = out["selection_frequency"] >= sel_threshold
    out["modality"] = [_modality_of(c) for c in out.index]
    return out.sort_values("selection_frequency", ascending=False)


def suggest_signals(df, target="bp", y=None, task=None, groups=None, top=15,
                    stability=True, n_boot=50, random_state=0):
    """One call to flag the likely-most-predictive signals for a target, leakage-safe.

    target='bp'      : y defaults to df['sbp']; ABP_* and shock_index are auto-excluded;
                       task defaults to 'regression'.
    target='outcome' : you MUST pass y (no outcome label ships with the dataset); ABP_* is
                       allowed; task defaults to 'classification'.

    Returns (ranking, stability_df|None). `ranking` is the consensus table (top `top` rows,
    or all if top is None) with the bootstrap selection_frequency joined in when stability=True.
    Grouping defaults to df['patient'] so bootstraps respect patient structure."""
    if task is None:
        task = "regression" if target == "bp" else "classification"
    if y is None:
        if target == "bp" and "sbp" in df:
            y = df["sbp"].values
        else:
            raise ValueError("Provide y: no default target column for this target.")
    if groups is None and "patient" in df:
        groups = df["patient"].values
    ranking = rank_features(df, y, target=target, task=task, random_state=random_state)
    stab = None
    if stability:
        stab = stability_selection(df, y, target=target, task=task, groups=groups,
                                   n_boot=n_boot, random_state=random_state)
        ranking = ranking.join(stab["selection_frequency"])
    return (ranking.head(top) if top else ranking), stab
