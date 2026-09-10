"""
Signal quality control for the MIMIC-BP dataset (Part 1).

How the data is laid out (per patient p######):
    <signal>/p######_<signal>.npy  shape (30, 3750)  -> 30 clips of 30 s at 125 Hz
    labels/p######_labels.npy      shape (30, 2)     -> [systolic, diastolic] per clip (mmHg)
Signals: abp is in mmHg; ecg, ppg, resp are in rescaled/arbitrary units.

Only numpy + scipy are used and the functions are pure, so they're easy to unit-test (see
tests/) and reuse from src/features.py.

How it's organized: each check returns a plain, interpretable NUMBER; QC_THRESHOLDS turns those
numbers into pass/fail FLAGS; segment_qc() combines the flags into a keep-or-drop decision.

Notes on the methods (tuned while exploring the data -- see notebook Part 1):
* A "spike" test (first-difference / Hampel) was tried and dropped: ECG is nearly flat between
  beats, so normal sharp beats get flagged as spikes (~99% false alarms). It is NOT used.
* qc_ecg/qc_abp/qc_ppg also record REPORT-ONLY diagnostics that do NOT affect keep/drop until
  they're calibrated on known-good real signals:
    ECG -- frequency-band powers, kurtosis/skew, lead steps (level/gain).
    ABP -- high-frequency ringing (>=10 Hz), upstroke, fast-flush, pulse pressure, mean, skew/kurt.
    PPG -- skewness quality score (Elgendi), spectral purity, beat-template match, kurtosis.
         (perfusion index isn't computed: the rescaled PPG has no DC level.)
* Instead, signal quality is judged by a periodicity score: how strongly the clip repeats at a
  plausible heart rate (the peak of the normalized autocorrelation). Clean cardiac clips score
  ~0.85-0.98; genuinely noisy ones fall below ~0.6. This is a standard autocorrelation-based
  quality index.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch, medfilt
from scipy.stats import kurtosis as _kurtosis, skew as _skew
from numpy.lib.stride_tricks import sliding_window_view


def has_nan(x) -> bool:
    """True if the clip has any missing value (NaN). Checked FIRST in every qc_* so bad data is
    rejected outright: NaNs break the filters, and comparisons against a NaN quietly return
    False, so no other check would catch it."""
    return bool(np.isnan(np.asarray(x, dtype=float)).any())

FS = 125
SEG_SAMPLES = 3750
N_SEG = 30

ABP_MIN_MMHG, ABP_MAX_MMHG = 20.0, 300.0
SBP_RANGE = (50.0, 250.0)
DBP_RANGE = (20.0, 150.0)
HR_RANGE_BPM = (30.0, 220.0)

QC_THRESHOLDS = dict(
    flat_std_eps=1e-6,        # spread below this -> a flat line
    flat_run_frac=0.20,       # >20% of the clip stuck at one value -> frozen sensor
    clip_frac=0.02,           # >2% of samples stuck at a sensor limit -> clipping
    ppg_rail_hi=4.0,          # nominal PPG ceiling; rarely reached (<0.3% of segments)
    ppg_rail_lo=0.0,          # PPG floor; dropout-to-0 is the real PPG artifact (see EDA)
    dropout_frac=0.02,        # >2% of samples stuck at 0 -> sensor lost contact
    sqi_min=0.50,             # periodicity score below this -> no steady rhythm / noisy
    min_beats=int(30 * HR_RANGE_BPM[0] / 60),
    max_beats=int(30 * HR_RANGE_BPM[1] / 60),
)


# ----------------------------- generic signal checks -----------------------------
def flat_std(x: np.ndarray) -> float:
    return float(np.std(x))


def max_constant_run_frac(x: np.ndarray) -> float:
    """Longest stretch of (nearly) identical samples in a row, as a fraction of the clip.
    Catches a frozen sensor that overall spread alone would miss when a flat patch is followed
    by noise. Uses a fast run-length method (runs on every clip)."""
    # Searched for "numpy find runs of consecutive equal values / longest run length";
    # vectorized run-length-encoding idiom adapted from Stack Overflow.
    if x.size == 0:
        return 1.0
    tol = 1e-9 + 1e-6 * (np.max(np.abs(x)) + 1e-12)
    is_same = np.abs(np.diff(x)) <= tol
    if not is_same.any():
        return 1.0 / x.size
    padded = np.concatenate(([0], is_same.view(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    runs = edges[1::2] - edges[::2]
    best = int(runs.max()) if runs.size else 0
    return (best + 1) / x.size            # k equal diffs span k+1 samples


def clip_fraction(x: np.ndarray, lo: float | None = None, hi: float | None = None,
                  tol: float = 1e-6) -> float:
    """Fraction of samples pinned at the low or high rail (saturation/clipping)."""
    lo = np.min(x) if lo is None else lo
    hi = np.max(x) if hi is None else hi
    at_hi = np.abs(x - hi) <= tol * (abs(hi) + 1.0)
    at_lo = np.abs(x - lo) <= tol * (abs(lo) + 1.0)
    return float((at_hi | at_lo).mean())


def _bandpass(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    # Zero-phase Butterworth band-pass (butter + filtfilt) per the scipy.signal docs;
    # filtfilt avoids the phase distortion that would shift fiducial timings.
    nyq = FS / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def periodicity_sqi(x: np.ndarray, lo_hz: float = 0.5, hi_hz: float = 8.0,
                    hr_min: float = HR_RANGE_BPM[0], hr_max: float = HR_RANGE_BPM[1]) -> float:
    """How strongly the clip repeats at a plausible heart rate (peak of the normalized
    autocorrelation). Close to 1 => a clean, steady beat; near 0 => noise."""
    # Concept: autocorrelation-based signal-quality index, a standard periodicity SQI for
    # cardiac waveforms (searched "autocorrelation signal quality index ECG/PPG").
    try:
        xf = _bandpass(x, lo_hz, hi_hz)
    except Exception:
        xf = x - np.mean(x)
    xf = xf - xf.mean()
    if np.std(xf) < 1e-12:
        return 0.0
    ac = np.correlate(xf, xf, "full")[len(xf) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(FS * 60 / hr_max)
    hi = min(int(FS * 60 / hr_min), len(ac) - 1)
    if hi <= lo:
        return 0.0
    return float(np.max(ac[lo:hi]))


def count_beats(x: np.ndarray) -> int:
    """Count the pulses in a clip (band-pass filter, then peak detection)."""
    try:
        xf = _bandpass(x, 0.5, 8.0)
    except Exception:
        xf = x - np.mean(x)
    xf = xf / (np.std(xf) + 1e-12)
    min_dist = int(FS * 60 / HR_RANGE_BPM[1])
    peaks, _ = find_peaks(xf, distance=min_dist, prominence=0.3)
    return int(len(peaks))


def estimate_hr(x: np.ndarray) -> float:
    """Heart rate (bpm) from the beat count over the clip."""
    return count_beats(x) * 60.0 / (len(x) / FS)


# ----------------------------- ECG diagnostics (REPORT-ONLY) -----------------------------
# These numbers are recorded but do NOT decide keep/drop. Their cutoffs aren't calibrated to
# this dataset yet; the notebook plots them on known-good ECG so a cutoff can be set above the
# clean range before any of them is ever used as a real check.
def find_rpeaks(x, fs=FS):
    """Positions of the heartbeats (R-peaks). It squares the beat-band signal so an
    upside-down lead still shows up as a strong peak instead of being missed."""
    # QRS-band filter + squaring is a Pan-Tompkins-style energy step; peak picking uses
    # scipy.signal.find_peaks (distance/prominence) per the SciPy docs.
    try:
        xf = _bandpass(x, 5.0, 20.0)
    except Exception:
        xf = x - np.mean(x)
    e = (xf - xf.mean()) ** 2
    e = e / (np.max(e) + 1e-12)
    peaks, _ = find_peaks(e, distance=int(fs * 60 / HR_RANGE_BPM[1]), prominence=0.05)
    return peaks


def find_ppg_peaks(x, fs=FS):
    """Positions of the PPG pulse peaks. The prominence setting skips the smaller secondary
    bump so a single pulse isn't counted twice."""
    try:
        xf = _bandpass(x, 0.5, 8.0)
    except Exception:
        xf = x - np.mean(x)
    xf = (xf - xf.mean()) / (np.std(xf) + 1e-12)
    peaks, _ = find_peaks(xf, distance=int(fs * 60 / HR_RANGE_BPM[1]), prominence=0.4)
    return peaks


def _gain_step(x, fs=FS):
    """Largest lasting change in beat height, as a fraction of the bigger side. It looks for a
    step in the (median-smoothed) beat heights, so a real gain change (the whole run shifts)
    is caught while a single tall beat is ignored."""
    rpk = find_rpeaks(x, fs)
    if len(rpk) < 6:
        return 0.0
    a = medfilt(np.abs(x[rpk] - np.median(x)), kernel_size=3)
    best = 0.0
    for s in range(2, len(a) - 2):
        b, f = np.median(a[:s]), np.median(a[s:])
        best = max(best, abs(f - b) / (max(b, f) + 1e-12))
    return float(best)


def ecg_step(x, fs=FS, win_ms=200.0):
    """Returns (level_shift, gain_shift): a sudden baseline jump (e.g. the monitor switching
    leads) and a change in beat height (a gain change). level_shift is the biggest jump in the
    rolling median across a ~win_ms gap, as a fraction of the signal height."""
    w = max(3, int(fs * win_ms / 1000.0))
    lvl = 0.0
    if x.size >= 2 * w:
        med = np.median(sliding_window_view(x, w), axis=1)
        ptp = np.percentile(x, 99) - np.percentile(x, 1)
        ptp = ptp if ptp > 1e-9 else (np.std(x) + 1e-12)
        d = np.abs(med[w:] - med[:-w]) / ptp
        lvl = float(d.max()) if d.size else 0.0
    return lvl, _gain_step(x, fs)


def ecg_spectral(x, fs=FS):
    """How the signal energy splits across frequency bands. It keeps genuine slow baseline
    drift, and the high-frequency band skips the 50/60 Hz notches so broadband muscle noise
    and narrow electrical-line interference stay separate."""
    # Searched for "scipy welch power spectral density band power"; PSD estimate and
    # band-power summation follow the scipy.signal.welch documentation.
    f, pxx = welch(x, fs=fs, nperseg=min(1024, len(x)), detrend="constant")
    total = pxx.sum() + 1e-12
    nyq = fs / 2.0
    band = lambda lo, hi: pxx[(f >= lo) & (f < hi)].sum()
    pl50 = band(49.0, 51.0) / total
    pl60 = band(59.0, min(61.0, nyq)) / total
    hf = (band(40.0, nyq) - band(49.0, 51.0) - band(59.0, min(61.0, nyq))) / total
    return dict(baseline_frac=float(band(0.0, 0.5) / total), hf_frac=float(max(0.0, hf)),
                powerline_frac=float(pl50 + pl60), pl50_frac=float(pl50), pl60_frac=float(pl60))


def ecg_stats(x):
    """"Peakedness" (sharp, rare beats make the signal spiky -> high value; noise flattens it)
    and skew (which way the lead points; diagnostic only)."""
    return dict(kurtosis=float(_kurtosis(x, fisher=True, bias=False)),
                skewness=float(_skew(x, bias=False)))


# ----------------------------- ABP diagnostics (REPORT-ONLY) -----------------------------
# Recorded, NOT gated. Damping/flush thresholds need calibration on known-good ABP first.
def abp_damping(x, fs=FS):
    """Catheter "ringing" (under-damping) shows up as high-frequency energy a real arterial
    wave should not have: hf10_frac = share of energy at >=10 Hz. upslope = the steepest rise,
    a damping cue. Over-damping cannot be told apart from a simply scaled wave without a flush
    reference, so it is not reported on its own."""
    f, pxx = welch(x, fs=fs, nperseg=min(1024, len(x)), detrend="constant")
    hf10 = float(pxx[f >= 10.0].sum() / (pxx.sum() + 1e-12))
    try:
        xf = _bandpass(x, 0.5, 8.0)
    except Exception:
        xf = x - np.mean(x)
    pp = np.percentile(x, 95) - np.percentile(x, 5) + 1e-12
    return dict(hf10_frac=hf10, upslope=float(np.max(np.diff(xf)) / pp))


def abp_flush(x, fs=FS):
    """Detects a fast-flush test: a long near-flat plateau away from the normal beat range.
    Returns how long the plateau lasts (s) and how far off it sits (as a fraction of pulse
    pressure)."""
    tol = 1e-6 * (np.max(np.abs(x)) + 1e-12) + 1e-9
    same = np.abs(np.diff(x)) <= tol
    if not same.any():
        return dict(flush_plateau_s=1.0 / fs, flush_offset=0.0)
    padded = np.concatenate(([0], same.view(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    runs = edges[1::2] - edges[::2]
    k = int(runs.argmax()); best = int(runs[k]); start = int(edges[2 * k])
    med = np.median(x); pp = np.percentile(x, 95) - np.percentile(x, 5) + 1e-12
    offset = float(abs(np.median(x[start:start + best + 1]) - med) / pp)
    return dict(flush_plateau_s=float((best + 1) / fs), flush_offset=offset)


def abp_pressures(x):
    """Pulse pressure (95th minus 5th percentile) and mean arterial pressure (clip average)."""
    return dict(pp=float(np.percentile(x, 95) - np.percentile(x, 5)),
                map=float(np.mean(x)))


# ----------------------------- PPG diagnostics (REPORT-ONLY) -----------------------------
def _extract_beats(x, fs=FS, L=50):
    """Individual pulses (start to start), each stretched to L samples so their shapes line up."""
    try:
        xf = _bandpass(x, 0.5, 8.0)
    except Exception:
        xf = x - np.mean(x)
    xn = (xf - xf.mean()) / (np.std(xf) + 1e-12)
    tr, _ = find_peaks(-xn, distance=int(fs * 60 / HR_RANGE_BPM[1]), prominence=0.3)
    beats = []
    for i in range(len(tr) - 1):
        s = xn[tr[i]:tr[i + 1]]
        if len(s) >= 5:
            beats.append(np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s))
    return np.array(beats)


def ppg_skew_sqi(x):
    """Skew of the PPG pulse wave -- the standard Elgendi quality score (cleaner pulses lean
    more to one side). Works on the rescaled PPG (unlike perfusion index, which needs the raw
    DC level and so is not computed here)."""
    # Reference: Elgendi (2016), "Optimal Signal Quality Index for Photoplethysmogram
    # Signals" -- skewness as a PPG signal-quality index.
    try:
        xf = _bandpass(x, 0.5, 8.0)
    except Exception:
        xf = x - np.mean(x)
    return float(_skew(xf, bias=False))


def ppg_spectral_purity(x, fs=FS):
    """Share of the signal energy in the normal pulse band (0.5-8 Hz). Low => mostly noise."""
    f, pxx = welch(x, fs=fs, nperseg=min(1024, len(x)), detrend="constant")
    return float(pxx[(f >= 0.5) & (f < 8.0)].sum() / (pxx.sum() + 1e-12))


def ppg_beat_consistency(x, fs=FS, L=50):
    """How closely each pulse matches the clip typical pulse shape (median correlation to the
    average pulse). Close to 1 => consistent pulses; low => misshapen or noisy ones. Needs at
    least 4 pulses, otherwise NaN (report-only)."""
    beats = _extract_beats(x, fs, L)
    if len(beats) < 4:
        return dict(beat_corr=float("nan"), n_beats_morph=int(len(beats)))
    beats = (beats - beats.mean(1, keepdims=True)) / (beats.std(1, keepdims=True) + 1e-12)
    templ = np.median(beats, 0)
    corr = np.array([np.corrcoef(b, templ)[0, 1] for b in beats])
    return dict(beat_corr=float(np.median(corr)), n_beats_morph=int(len(beats)))


# ----------------------------- per-modality QC -----------------------------
def qc_abp(seg, thr=QC_THRESHOLDS):
    if has_nan(seg):
        return dict(flag_nan=True, ok=False)
    m = dict(flag_nan=False, std=flat_std(seg), const_run_frac=max_constant_run_frac(seg),
             clip_frac=clip_fraction(seg), sqi=periodicity_sqi(seg),
             frac_out_of_range=float(((seg < ABP_MIN_MMHG) | (seg > ABP_MAX_MMHG)).mean()),
             n_beats=count_beats(seg), seg_min=float(seg.min()), seg_max=float(seg.max()))
    # report-only diagnostics (recorded, NOT gated)
    m.update(abp_damping(seg)); m.update(abp_flush(seg)); m.update(abp_pressures(seg))
    m.update(skewness=float(_skew(seg, bias=False)), kurtosis=float(_kurtosis(seg, fisher=True, bias=False)))
    m["flag_flat"] = m["std"] < thr["flat_std_eps"] or m["const_run_frac"] > thr["flat_run_frac"]
    m["flag_clip"] = m["clip_frac"] > thr["clip_frac"]
    m["flag_range"] = m["frac_out_of_range"] > 0.0
    m["flag_beats"] = not (thr["min_beats"] <= m["n_beats"] <= thr["max_beats"])
    m["flag_sqi"] = m["sqi"] < thr["sqi_min"]
    m["ok"] = not (m["flag_nan"] or m["flag_flat"] or m["flag_clip"] or m["flag_range"] or m["flag_beats"] or m["flag_sqi"])
    return m


def qc_ppg(seg, thr=QC_THRESHOLDS):
    if has_nan(seg):
        return dict(flag_nan=True, ok=False)
    # PPG cannot go negative; the main problem is dropping to 0 (sensor loses contact), not
    # hitting the 4.0 ceiling (which rarely happens). Track both.
    dropout = clip_fraction(seg, lo=thr["ppg_rail_lo"], hi=thr["ppg_rail_lo"])   # frac at 0
    ceil = clip_fraction(seg, lo=thr["ppg_rail_hi"], hi=thr["ppg_rail_hi"])      # frac at 4.0
    m = dict(flag_nan=False, std=flat_std(seg), const_run_frac=max_constant_run_frac(seg),
             dropout_frac=dropout, clip_frac=ceil,
             sqi=periodicity_sqi(seg), n_beats=count_beats(seg),
             seg_min=float(seg.min()), seg_max=float(seg.max()))
    # report-only diagnostics (recorded, NOT gated). Perfusion index omitted: needs raw DC.
    m.update(skew_sqi=ppg_skew_sqi(seg), spec_purity=ppg_spectral_purity(seg),
             kurtosis=float(_kurtosis(seg, fisher=True, bias=False)))
    m.update(ppg_beat_consistency(seg))
    m["flag_flat"] = m["std"] < thr["flat_std_eps"] or m["const_run_frac"] > thr["flat_run_frac"]
    m["flag_dropout"] = m["dropout_frac"] > thr["dropout_frac"]
    m["flag_clip"] = m["clip_frac"] > thr["clip_frac"]
    m["flag_beats"] = not (thr["min_beats"] <= m["n_beats"] <= thr["max_beats"])
    m["flag_sqi"] = m["sqi"] < thr["sqi_min"]
    m["ok"] = not (m["flag_nan"] or m["flag_flat"] or m["flag_dropout"] or m["flag_clip"] or m["flag_beats"] or m["flag_sqi"])
    return m


def qc_ecg(seg, thr=QC_THRESHOLDS):
    if has_nan(seg):
        return dict(flag_nan=True, ok=False)
    m = dict(flag_nan=False, std=flat_std(seg), const_run_frac=max_constant_run_frac(seg),
             clip_frac=clip_fraction(seg), sqi=periodicity_sqi(seg),
             seg_min=float(seg.min()), seg_max=float(seg.max()), seg_mean=float(seg.mean()))
    # report-only diagnostics (recorded, NOT gated -- thresholds uncalibrated for this data)
    lvl, gain = ecg_step(seg)
    m.update(level_shift=lvl, gain_shift=gain)
    m.update(ecg_spectral(seg)); m.update(ecg_stats(seg))
    m["flag_flat"] = m["std"] < thr["flat_std_eps"] or m["const_run_frac"] > thr["flat_run_frac"]
    m["flag_clip"] = m["clip_frac"] > thr["clip_frac"]
    m["flag_sqi"] = m["sqi"] < thr["sqi_min"]
    m["ok"] = not (m["flag_nan"] or m["flag_flat"] or m["flag_clip"] or m["flag_sqi"])
    return m


def label_sanity(sbp, dbp, abp_seg=None):
    if np.isnan(sbp) or np.isnan(dbp) or (abp_seg is not None and has_nan(abp_seg)):
        return dict(flag_nan=True, ok=False)
    m = dict(sbp=float(sbp), dbp=float(dbp), flag_nan=False)
    m["flag_sbp_range"] = not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1])
    m["flag_dbp_range"] = not (DBP_RANGE[0] <= dbp <= DBP_RANGE[1])
    m["flag_order"] = not (sbp > dbp)
    m["flag_pp"] = not (10.0 <= (sbp - dbp) <= 120.0)
    if abp_seg is not None:
        m["abp_p95"] = float(np.percentile(abp_seg, 95))
        m["abp_p05"] = float(np.percentile(abp_seg, 5))
        m["flag_sbp_mismatch"] = abs(sbp - m["abp_p95"]) > 25.0
        m["flag_dbp_mismatch"] = abs(dbp - m["abp_p05"]) > 20.0
    m["ok"] = not any(v for k, v in m.items() if k.startswith("flag_"))
    return m


# ----------------------------- orchestration -----------------------------
def segment_qc(abp, ecg, ppg, sbp, dbp, thr=QC_THRESHOLDS):
    """Run every check on one clip and decide keep or drop.
    For cuffless BP the inputs are ECG + PPG and the target is the label (from the arterial
    line), so keep_for_model needs the ECG, the PPG, and the label to all be clean. Arterial
    quality is recorded for the report. Breathing is left out of QC (it is not a model input)."""
    a = qc_abp(abp, thr); e = qc_ecg(ecg, thr); p = qc_ppg(ppg, thr)
    lab = label_sanity(sbp, dbp, abp)
    out = {}
    for name, d in dict(abp=a, ecg=e, ppg=p, label=lab).items():
        for k, v in d.items():
            out[f"{name}_{k}"] = v
    out["keep_for_model"] = bool(e["ok"] and p["ok"] and lab["ok"])
    out["keep_all_signals"] = bool(a["ok"] and e["ok"] and p["ok"] and lab["ok"])
    return out


def load_patient(dbpath, pid):
    import os
    out = {m: np.load(os.path.join(dbpath, m, f"{pid}_{m}.npy")) for m in ("abp", "ecg", "ppg")}
    out["labels"] = np.load(os.path.join(dbpath, "labels", f"{pid}_labels.npy"))
    return out


def patient_qc(dbpath, pid, thr=QC_THRESHOLDS):
    d = load_patient(dbpath, pid)
    rows = []
    for i in range(d["abp"].shape[0]):
        sbp, dbp = d["labels"][i]
        row = dict(patient=pid, segment=i)
        row.update(segment_qc(d["abp"][i], d["ecg"][i], d["ppg"][i], sbp, dbp, thr))
        rows.append(row)
    return rows
