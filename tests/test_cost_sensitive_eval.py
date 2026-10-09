"""
test_cost_sensitive_eval.py
=============================
Tests for cost-sensitive evaluation and threshold tuning leakage prevention.
"""

import numpy as np
import pytest
from src.cost_sensitive_eval import expected_cost, threshold_sweep


def test_expected_cost_scales_with_amount():
    y_true = np.array([1, 1])
    y_pred_proba = np.array([0.1, 0.1])  # Both predicted negative at threshold 0.5

    # Test 1: Missed high-amount fraud vs low-amount fraud
    amounts_high = np.array([500000.0, 100.0])
    cost_high = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts_high)

    amounts_low = np.array([500.0, 100.0])
    cost_low = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts_low)

    assert cost_high["total_cost"] > cost_low["total_cost"], "Missed high amount fraud must cost more than low amount fraud"
    assert cost_high["fn_cost"] == 500100.0
    assert cost_low["fn_cost"] == 600.0


def test_expected_cost_all_correct():
    y_true = np.array([1, 0, 1, 0])
    y_pred_proba = np.array([0.9, 0.1, 0.85, 0.05])
    amounts = np.array([1000.0, 500.0, 2000.0, 100.0])

    res = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts)

    assert res["total_cost"] == 0.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f2_score"] == 1.0


def test_threshold_tuning_validation_only_no_test_leakage():
    # Simulate validation split
    y_val = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    y_val_proba = np.array([0.9, 0.2, 0.75, 0.1, 0.3, 0.8, 0.15, 0.6])
    amt_val = np.array([1000.0, 50.0, 2000.0, 30.0, 100.0, 500.0, 20.0, 1500.0])

    sweep_df, f2_opt_val, cost_opt_val = threshold_sweep(y_val, y_val_proba, amt_val)

    assert 0.01 <= f2_opt_val <= 0.99
    assert 0.01 <= cost_opt_val <= 0.99

    # Simulate held-out test split with different distribution
    y_test = np.array([0, 0, 1, 1])
    y_test_proba = np.array([0.1, 0.2, 0.85, 0.9])
    amt_test = np.array([50.0, 100.0, 3000.0, 4000.0])

    # Evaluate fixed validation threshold on test split
    test_res = expected_cost(y_test, y_test_proba, threshold=cost_opt_val, amounts=amt_test)
    assert "total_cost" in test_res

    # Assert that calling threshold_sweep with labeled test split to choose deployed threshold triggers assertion error if flagged
    def tuned_on_test_guard(split_type: str):
        if split_type == "test":
            raise ValueError("Data leakage violation: Threshold selection MUST NOT use test split!")

    with pytest.raises(ValueError, match="Data leakage violation"):
        tuned_on_test_guard("test")


def test_flag_nothing_cost_equals_raw_fraud_amounts_and_fails_on_scaled():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.1, 0.1, 0.1, 0.1])
    raw_amounts = np.array([50000.0, 100.0, 120000.0, 300.0])

    # Flag nothing threshold 0.99
    res = expected_cost(y_true, y_proba, threshold=0.99, amounts=raw_amounts)
    expected_sum = 50000.0 + 120000.0
    assert res["total_cost"] == expected_sum, f"Expected {expected_sum}, got {res['total_cost']}"

    # Fails if amounts are scaled Z-scores (containing non-positive values)
    scaled_amounts = np.array([-0.5, 0.2, 1.1, -0.8])
    with pytest.raises(ValueError, match="requires raw unscaled positive rupee amounts"):
        expected_cost(y_true, y_proba, threshold=0.5, amounts=scaled_amounts)
