"""
drift_monitor.py — Concept Drift Monitoring Module for FinSecure AI v2.
========================================================================
Computes Population Stability Index (PSI), Type-Mix distribution shifts,
and score distribution shifts across model features and prediction probabilities.
"""

import os
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

DB_PATH = Path("data/finsecure.db")
DRIFT_PSI_STABLE = 0.10
DRIFT_PSI_WARNING = 0.25

# Cumulative graph counts & calendar features excluded from retraining triggers to avoid false positives
EXCLUDED_PSI_COLS = {
    "step", "hour_of_day", "day_of_week", "high_risk_hour",
    "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_component_size", "orig_fraud_exposure",
    "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_component_size", "dest_fraud_exposure"
}


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-4,
    is_probability: bool = False,
) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current distributions.
    Handles zero-inflated features by placing exact zeros in a dedicated bin and quantile binning non-zero values.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    if is_probability:
        bin_edges = np.linspace(0.0, 1.0, 11)
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        ref_pct = ref_counts / len(reference)
        curr_pct = curr_counts / len(current)
    else:
        unique_ref = np.unique(reference)
        min_val = np.min(reference)
        ref_min_mask = np.isclose(reference, min_val, atol=1e-6) | (reference == 0.0)
        ref_min_pct = np.mean(ref_min_mask)

        if len(unique_ref) <= 10:
            all_vals = np.unique(np.concatenate([reference, current]))
            ref_pcts = [np.mean(np.isclose(reference, val, atol=1e-6)) for val in all_vals]
            curr_pcts = [np.mean(np.isclose(current, val, atol=1e-6)) for val in all_vals]

            ref_pct = np.array(ref_pcts)
            curr_pct = np.array(curr_pcts)
        elif ref_min_pct >= 0.05:
            curr_min_mask = np.isclose(current, min_val, atol=1e-6) | (current == 0.0)
            curr_min_pct = np.mean(curr_min_mask)

            ref_nz = reference[~ref_min_mask]
            curr_nz = current[~curr_min_mask]

            if len(ref_nz) > 0 and len(curr_nz) > 0:
                q_edges = np.unique(np.percentile(ref_nz, np.linspace(0, 100, bins + 1)))
                if len(q_edges) >= 2:
                    ref_nz_counts, _ = np.histogram(ref_nz, bins=q_edges)
                    curr_nz_counts, _ = np.histogram(curr_nz, bins=q_edges)

                    ref_nz_pcts = ref_nz_counts / len(reference)
                    curr_nz_pcts = curr_nz_counts / len(current)

                    ref_pct = np.array([ref_min_pct] + list(ref_nz_pcts))
                    curr_pct = np.array([curr_min_pct] + list(curr_nz_pcts))
                else:
                    ref_pct = np.array([ref_min_pct, len(ref_nz) / len(reference)])
                    curr_pct = np.array([curr_min_pct, len(curr_nz) / len(current)])
            else:
                ref_pct = np.array([ref_min_pct, len(ref_nz) / len(reference)])
                curr_pct = np.array([curr_min_pct, len(curr_nz) / len(current)])
        else:
            percentiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(reference, percentiles)
            bin_edges = np.unique(bin_edges)

            if len(bin_edges) < 2:
                return 0.0

            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            ref_pct = ref_counts / len(reference)
            curr_pct = curr_counts / len(current)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    curr_pct = np.where(curr_pct == 0, eps, curr_pct)

    psi_vec = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    psi_total = float(np.sum(psi_vec))
    return max(0.0, psi_total)


def init_drift_db(db_path: Path = DB_PATH):
    """Initialize SQLite table for persisting drift snapshots."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                retrain_recommended INTEGER NOT NULL,
                score_psi REAL NOT NULL,
                drifted_feature_count INTEGER NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )
    conn.close()


class DriftMonitor:
    def __init__(self, reference_df: pd.DataFrame, reference_probas: Optional[np.ndarray] = None):
        """Initialize reference distributions and feature names from training split."""
        warmup_frac = float(os.getenv("WARMUP_FRACTION", "0.2"))
        if "step" in reference_df.columns:
            min_s = reference_df["step"].min()
            max_s = reference_df["step"].max()
            step_range = max_s - min_s
            warmup_cutoff = min_s + step_range * warmup_frac
            stable_ref_df = reference_df[reference_df["step"] >= warmup_cutoff]
            if len(stable_ref_df) < 500:
                stable_ref_df = reference_df
        ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
        self.feature_cols = [c for c in reference_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(reference_df[c])]

        self.reference_data = {col: stable_ref_df[col].values for col in self.feature_cols}

        if reference_probas is not None:
            self.reference_probas = reference_probas
        else:
            model_path = Path("models/xgboost_model.joblib")
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    X_ref = stable_ref_df[self.feature_cols]
                    self.reference_probas = model.predict_proba(X_ref)[:, 1]
                except Exception:
                    self.reference_probas = np.array([0.01, 0.03, 0.05, 0.02, 0.04])
            else:
                self.reference_probas = np.array([0.01, 0.03, 0.05, 0.02, 0.04])

        init_drift_db()

    def check(self, recent_df: pd.DataFrame, recent_probas: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate drift metrics across non-excluded features, type mix, amount quantiles, and prediction probabilities.
        """
        feature_psi_map = {}
        feature_status_map = {}
        inclusion_map = {}

        drifted_features = []
        warning_features = []

        if recent_probas is None:
            model_path = Path("models/xgboost_model.joblib")
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    X_rec = recent_df[self.feature_cols]
                    recent_probas = model.predict_proba(X_rec)[:, 1]
                except Exception:
                    recent_probas = None

        for col in self.feature_cols:
            if col in recent_df.columns:
                ref_vec = self.reference_data[col]
                curr_vec = recent_df[col].values

                psi_val = compute_psi(ref_vec, curr_vec)
                feature_psi_map[col] = float(psi_val)

                is_included = col not in EXCLUDED_PSI_COLS
                inclusion_map[col] = "INCLUDED" if is_included else "EXCLUDED"

                if is_included:
                    if psi_val > DRIFT_PSI_WARNING:
                        feature_status_map[col] = "DRIFT"
                        drifted_features.append(col)
                    elif psi_val >= DRIFT_PSI_STABLE:
                        feature_status_map[col] = "WARNING"
                        warning_features.append(col)
                    else:
                        feature_status_map[col] = "STABLE"
                else:
                    if col in {"hour_of_day", "high_risk_hour", "day_of_week"}:
                        feature_status_map[col] = "EXCLUDED_CALENDAR"
                    else:
                        feature_status_map[col] = "EXCLUDED_CUMULATIVE"

        # Check prediction score distribution PSI using fixed probability bins [0.0..1.0]
        score_psi = 0.0
        if recent_probas is not None and len(recent_probas) > 0 and self.reference_probas is not None and len(self.reference_probas) > 0:
            score_psi = float(compute_psi(self.reference_probas, recent_probas, is_probability=True))

        # Check type mix and amount shift
        type_debit_psi = feature_psi_map.get("type_DEBIT", 0.0)
        type_cashout_psi = feature_psi_map.get("type_CASH_OUT", 0.0)
        amt_psi = feature_psi_map.get("amount", 0.0)
        log_amt_psi = feature_psi_map.get("log_amount", 0.0)

        type_or_amount_drift = (type_debit_psi > DRIFT_PSI_WARNING) or (amt_psi > DRIFT_PSI_WARNING) or (log_amt_psi > DRIFT_PSI_WARNING)
        retrain_recommended = bool(len(drifted_features) >= 2 or score_psi > DRIFT_PSI_WARNING or type_or_amount_drift)

        if retrain_recommended:
            overall_status = "DRIFT_DETECTED"
        elif len(warning_features) >= 2 or score_psi >= DRIFT_PSI_STABLE:
            overall_status = "WARNING"
        else:
            overall_status = "STABLE"

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "retrain_recommended": retrain_recommended,
            "score_psi": score_psi,
            "drifted_feature_count": len(drifted_features),
            "warning_feature_count": len(warning_features),
            "drifted_features": drifted_features,
            "feature_psi": feature_psi_map,
            "feature_status": feature_status_map,
            "feature_inclusion": inclusion_map,
        }

        try:
            conn = sqlite3.connect(DB_PATH)
            with conn:
                conn.execute(
                    """
                    INSERT INTO drift_snapshots (timestamp, overall_status, retrain_recommended, score_psi, drifted_feature_count, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot["timestamp"],
                        snapshot["overall_status"],
                        1 if retrain_recommended else 0,
                        snapshot["score_psi"],
                        snapshot["drifted_feature_count"],
                        json.dumps(snapshot),
                    ),
                )
            conn.close()
        except Exception:
            pass

        return snapshot


def simulate_drift(df: pd.DataFrame, amount_scale: float = 5.0) -> pd.DataFrame:
    """
    Standardized drift simulator matching verify_drift_psi.py:
    Multiplies raw amount by amount_scale, recomputes unscaled dependent features,
    and scales features with scaler.joblib.
    """
    scaler_path = Path("models/scaler.joblib")
    df_shifted = df.copy()

    if "amount_raw" in df_shifted.columns:
        raw_amt = df_shifted["amount_raw"].values * amount_scale
        df_shifted["amount_raw"] = raw_amt
    else:
        raw_amt = df_shifted["amount"].values * amount_scale

    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            feature_names = list(getattr(scaler, "feature_names_in_", []))
            if feature_names and "amount" in feature_names:
                def get_raw(col_name):
                    idx = feature_names.index(col_name)
                    return df_shifted[col_name].values * scaler.scale_[idx] + scaler.mean_[idx]

                raw_old_orig = get_raw("oldbalanceOrg")
                raw_new_orig = get_raw("newbalanceOrig")
                raw_old_dest = get_raw("oldbalanceDest")
                raw_new_dest = get_raw("newbalanceDest")

                raw_err_orig = raw_new_orig + raw_amt - raw_old_orig
                raw_err_dest = raw_old_dest + raw_amt - raw_new_dest
                raw_log_amt = np.log1p(np.maximum(0.0, raw_amt))
                raw_ratio_orig = raw_amt / (raw_old_orig + 1.0)
                raw_ratio_dest = raw_amt / (raw_old_dest + 1.0)

                def set_scaled(col_name, raw_vec):
                    idx = feature_names.index(col_name)
                    df_shifted[col_name] = (raw_vec - scaler.mean_[idx]) / scaler.scale_[idx]

                set_scaled("amount", raw_amt)
                set_scaled("log_amount", raw_log_amt)
                set_scaled("errorBalanceOrig", raw_err_orig)
                set_scaled("errorBalanceDest", raw_err_dest)
                set_scaled("amount_to_oldbalanceOrg_ratio", raw_ratio_orig)
                set_scaled("amount_to_oldbalanceDest_ratio", raw_ratio_dest)
                return df_shifted
        except Exception:
            pass

    df_shifted["amount"] = df_shifted["amount"] * amount_scale
    df_shifted["log_amount"] = np.log1p(np.maximum(0.0, df_shifted["amount"]))
    return df_shifted
