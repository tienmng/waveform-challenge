"""
Unit tests for the QC and feature-extraction code (Part 3 bonus).

The tests run on FIXED, MADE-UP signals rather than the real MIMIC-BP files, so they are fast,
repeatable, and can run automatically without the 5 GB dataset. Run them with:  pytest -q
"""
import os, sys
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import qc
import features as F

FS = qc.FS            # 125 Hz
N = qc.SEG_SAMPLES    # 3750 (30 s)


# ----------------------------- synthetic signal builders -----------------------------
def _t():
    return np.arange(N) / FS

def sine(hz, lo=-1.0, hi=1.0, phase=0.0):
    x = np.sin(2 * np.pi * hz * _t() + phase)
    return lo + (hi - lo) * (x + 1) / 2

def ecg_spikes(hr_bpm=75.0):
    """A train of sharp spikes (fake heartbeats) at a fixed heart rate on a quiet baseline."""
    x = 0.01 * np.random.default_rng(0).standard_normal(N)
    step = int(round(FS * 60.0 / hr_bpm))
    x[np.arange(step, N, step)] += 3.0
    return x


# ----------------------------- generic QC helpers -----------------------------
def test_max_constant_run_frac_extremes():
    assert F.qc.max_constant_run_frac(np.zeros(100)) == 1.0            # fully flat
    x = np.arange(100.0)                                               # strictly increasing
    assert F.qc.max_constant_run_frac(x) == pytest.approx(1.0 / 100)   # no repeats

def test_max_constant_run_frac_matches_naive():
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(500), 1)      # forces some repeats
    # naive longest-run reference
    d = np.abs(np.diff(x)) <= (1e-9 + 1e-6 * (np.max(np.abs(x)) + 1e-12))
    best = run = 0
    for s in d:
        run = run + 1 if s else 0
        best = max(best, run)
    assert F.qc.max_constant_run_frac(x) == pytest.approx((best + 1) / x.size)

def test_clip_fraction_known():
    x = np.concatenate([np.full(20, 5.0), np.linspace(0.5, 4.0, 80)])  # only 20 of 100 stuck at hi=5
    assert qc.clip_fraction(x, lo=0.0, hi=5.0) == pytest.approx(0.20, abs=1e-6)


# ----------------------------- NaN guard (correctness fix) -----------------------------
def test_has_nan():
    assert qc.has_nan(np.array([1.0, np.nan, 3.0])) is True
    assert qc.has_nan(np.zeros(10)) is False

@pytest.mark.parametrize("fn", ["qc_ecg", "qc_ppg", "qc_abp"])
def test_nan_is_hard_rejected(fn):
    seg = sine(1.25); seg[123] = np.nan
    out = getattr(qc, fn)(seg)
    assert out["flag_nan"] is True and out["ok"] is False


# ----------------------------- QC decisions -----------------------------
def test_flatline_flagged():
    out = qc.qc_ecg(np.full(N, 0.7))
    assert out["flag_flat"] is True and out["ok"] is False

def test_periodicity_sqi_sine_high_noise_low():
    clean = qc.periodicity_sqi(sine(1.25))                       # 75 bpm
    noise = qc.periodicity_sqi(np.random.default_rng(2).standard_normal(N))
    assert clean > 0.8
    assert noise < 0.5
    assert clean > noise

def test_count_beats_matches_rate():
    # 75 bpm over 30 s -> ~37-38 beats
    assert qc.count_beats(sine(1.25)) == pytest.approx(37, abs=3)


# ----------------------------- feature extraction -----------------------------
def test_hrv_hr_matches_construction():
    out = F.hrv_features(ecg_spikes(hr_bpm=75.0))
    assert out["hr_bpm"] == pytest.approx(75.0, abs=3.0)
    assert out["rr_mean_ms"] == pytest.approx(800.0, abs=30.0)

def test_extract_features_schema():
    feats = F.extract_features(ecg_spikes(75.0), sine(1.25, lo=0.5, hi=3.5))
    for k in ("hr_bpm", "sdnn_ms", "ppg_sys_amp_mean", "ppg_upstroke_ms_mean",
              "ppg_apg_ba_mean", "pat_foot_ms_mean", "ppg_skew", "ppg_spec_purity"):
        assert k in feats
    assert np.isfinite(feats["hr_bpm"])

def test_abp_hemodynamics_on_known_sine():
    # a fake arterial wave from 60 to 120 -> pulse pressure 60, true waveform mean 90
    abp = sine(1.25, lo=60.0, hi=120.0)
    h = F.abp_hemodynamic_features(abp, ecg=ecg_spikes(75.0))
    assert h["abp_sbp"] == pytest.approx(120.0, abs=2.0)
    assert h["abp_dbp"] == pytest.approx(60.0, abs=2.0)
    assert h["abp_pp"] == pytest.approx(60.0, abs=3.0)
    assert h["abp_map"] == pytest.approx(90.0, abs=3.0)          # true waveform mean, not the (SBP+2DBP)/3=80 estimate
    assert np.isfinite(h["shock_index"]) and 0.3 < h["shock_index"] < 1.5


# ----------------------------- leakage-safety guarantee -----------------------------
def test_predictor_columns_enforce_leakage_rules():
    cols = ["patient","segment","split","sbp","dbp","ecg_ok","ppg_ok","abp_ok","keep_for_model",
            "hr_bpm","ppg_sys_amp_mean","pat_foot_ms_mean",
            "abp_map","abp_sbp","shock_index"]
    df = pd.DataFrame({c: [0.0] for c in cols})
    bp = F.predictor_columns(df, "bp")
    assert not any(c.startswith("abp_") or c == "shock_index" for c in bp)   # arterial columns never reach a BP run
    assert "hr_bpm" in bp and "pat_foot_ms_mean" in bp
    assert not any(c in bp for c in ("sbp", "dbp", "patient", "keep_for_model"))  # no id/label columns
    outcome = F.predictor_columns(df, "outcome")
    assert "abp_map" in outcome and "shock_index" in outcome                # arterial columns allowed for outcomes

def test_predictor_columns_rejects_bad_target():
    df = pd.DataFrame({"hr_bpm": [1.0], "sbp": [120.0]})
    with pytest.raises(ValueError):
        F.predictor_columns(df, "mortality")
