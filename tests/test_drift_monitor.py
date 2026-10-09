"""
test_drift_monitor.py
======================
Unit tests for concept drift monitoring (PSI, KS test, retrain recommendation, drift simulation)
and 4-window disjoint drift detection.
"""

import numpy as np
import pandas as pd
import pytest
from src.drift_monitor import DriftMonitor, compute_psi, simulate_drift


def test_compute_psi_identical_distributions():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(0, 1, 1000)
    psi = compute_psi(ref, curr)
    assert psi < 0.10, f"Expected stable PSI < 0.10 for identical distributions, got {psi:.4f}"


def test_drift_monitor_clean_versus_shifted():
    train_df = pd.read_parquet("data/features_train.parquet")
    ignore_cols = {"isFraud", "step", "mechanism"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]

    clean_sample = train_df[feature_cols].sample(frac=0.5, random_state=42)
    monitor = DriftMonitor(train_df)

    clean_res = monitor.check(clean_sample)
    assert clean_res["retrain_recommended"] is False, "Clean training distribution slice should not trigger retrain recommendation"

    shifted_df = simulate_drift(clean_sample, amount_scale=5.0)
    shifted_res = monitor.check(shifted_df)

    assert shifted_res["retrain_recommended"] is True, "Shifted distribution must trigger retrain recommendation"
    assert shifted_res["overall_status"] == "DRIFT_DETECTED"


def test_disjoint_windows_drift_behavior():
    train_df = pd.read_parquet("data/features_train.parquet")
    test_df = pd.read_parquet("data/features_test.parquet")
    shift_df = pd.read_parquet("data/features_shift.parquet")

    monitor = DriftMonitor(train_df)

    # 1. Clean Test (steps 541..629) should NOT recommend retrain
    clean_res = monitor.check(test_df)
    assert clean_res["retrain_recommended"] is False, "Clean test split should not trigger retrain recommendation"

    # 2. Shift Window (steps 630..743) MUST trigger DRIFT_DETECTED and retrain recommendation
    shift_res = monitor.check(shift_df)
    assert shift_res["overall_status"] == "DRIFT_DETECTED", f"Expected DRIFT_DETECTED on shift split, got {shift_res['overall_status']}"
    assert shift_res["retrain_recommended"] is True, "Shift window must trigger retrain recommendation"

    # 3. Amount x5 Simulation MUST trigger DRIFT_DETECTED and retrain recommendation
    amt_x5_df = simulate_drift(test_df, amount_scale=5.0)
    x5_res = monitor.check(amt_x5_df)

    assert x5_res["overall_status"] == "DRIFT_DETECTED", f"Expected DRIFT_DETECTED on amount x5, got {x5_res['overall_status']}"
    assert x5_res["retrain_recommended"] is True, "Amount x5 simulation must trigger retrain recommendation"
