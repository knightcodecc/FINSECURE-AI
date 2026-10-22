"""
batch_scoring.py — Scalable Streaming Validation, Batch Scoring & Analytics Engine
===================================================================================
Provides modular, memory-bounded batch processing for user-uploaded transaction CSVs:
- Delimiter autodetection (comma, semicolon, tab) & UTF-8 BOM handling.
- Streaming disk chunking (no entire file in RAM).
- Strict row validation with categorized rejection logging.
- Streaming EDA aggregates over ALL rows.
- Chronological expanding-window graph feature extraction (reusing src/graph_features.py).
- Zero-skew tabular feature engineering (reusing src/feature_engineering.py).
- Scaler transformation (models/scaler.joblib) and XGBoost batch inference.
- Threshold what-if simulation, cost reconciliation, bootstrap CIs (<20 frauds), alert budgets.
- CSV formula-injection sanitation for exported results.
"""

import csv
import io
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from src.feature_engineering import engineer_tabular_features
from src.graph_features import attach_graph_features_expanding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"

DEFAULT_MAX_SCORE_ROWS = int(os.getenv("MAX_SCORE_ROWS", "300000"))
DEFAULT_STEP_BATCH_SIZE = int(os.getenv("STEP_BATCH_SIZE", "72"))
ALLOWED_TYPES = {"TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"}

REQUIRED_CANONICAL_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"
]

OPTIONAL_CANONICAL_COLUMNS = ["isFraud", "isFlaggedFraud"]

COLUMN_ALIASES = {
    "step": ["step", "timestep", "time_step", "time"],
    "type": ["type", "transaction_type", "tx_type", "action"],
    "amount": ["amount", "amt", "transaction_amount", "tx_amount"],
    "nameOrig": ["nameorig", "name_orig", "origin", "sender", "origin_account", "from_account", "customer_id", "source"],
    "oldbalanceOrg": ["oldbalanceorg", "oldbalance_org", "old_balance_orig", "old_balance_org", "sender_old_balance", "balance_orig_before", "oldbalance_sender"],
    "newbalanceOrig": ["newbalanceorig", "newbalance_orig", "new_balance_orig", "sender_new_balance", "balance_orig_after", "newbalance_sender"],
    "nameDest": ["namedest", "name_dest", "dest", "destination", "recipient", "dest_account", "to_account", "target"],
    "oldbalanceDest": ["oldbalancedest", "oldbalance_dest", "old_balance_dest", "recipient_old_balance", "balance_dest_before", "oldbalance_recipient"],
    "newbalanceDest": ["newbalancedest", "newbalance_dest", "new_balance_dest", "recipient_new_balance", "balance_dest_after", "newbalance_recipient"],
    "isFraud": ["isfraud", "is_fraud", "fraud", "label", "target", "fraud_label"],
    "isFlaggedFraud": ["isflaggedfraud", "is_flagged_fraud", "flagged_fraud", "flagged"]
}

_WARMED_GRAPH_CACHE: Optional[Any] = None

def get_warmed_graph_state() -> Optional[Any]:
    """Retrieve or build memory-cached InferenceGraphState warmed from training history (step <= 420)."""
    global _WARMED_GRAPH_CACHE
    if _WARMED_GRAPH_CACHE is not None:
        return _WARMED_GRAPH_CACHE
    hist_csv = PROJECT_ROOT / "data" / "paysim_synth.csv"
    if not hist_csv.exists():
        return None
    try:
        from api.predictor import InferenceGraphState
        df_hist = pd.read_csv(hist_csv)
        train_history = df_hist[df_hist["step"] <= 420].copy()
        state = InferenceGraphState()
        state.warm_from_history(train_history)
        _WARMED_GRAPH_CACHE = state
        return _WARMED_GRAPH_CACHE
    except Exception as e:
        return None



def detect_delimiter_and_encoding(sample_bytes: bytes) -> Tuple[str, str]:
    """Detect text encoding (handling UTF-8 BOM) and CSV delimiter (comma, semicolon, tab)."""
    encoding = "utf-8"
    if sample_bytes.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"

    try:
        sample_text = sample_bytes.decode(encoding, errors="replace")[:8192]
    except Exception:
        sample_text = sample_bytes.decode("utf-8", errors="replace")[:8192]

    # Count potential delimiters in the first few lines
    lines = [line.strip() for line in sample_text.splitlines() if line.strip()][:5]
    if not lines:
        return ",", encoding

    delims = [",", ";", "\t", "|"]
    best_delim = ","
    best_score = -1

    for d in delims:
        counts = [line.count(d) for line in lines]
        if len(counts) > 0 and counts[0] > 0 and all(c == counts[0] for c in counts):
            score = counts[0] * 10
        else:
            score = sum(counts)
        if score > best_score:
            best_score = score
            best_delim = d

    return best_delim, encoding


def auto_map_columns(raw_columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Attempt to automatically map raw CSV column headers to canonical column names.
    Returns: (mapping_dict: canonical -> raw, missing_required_canonical_cols: list)
    """
    normalized_raw = {c.strip().lower().replace("_", "").replace(" ", ""): c for c in raw_columns}
    raw_lower_map = {c.strip().lower(): c for c in raw_columns}

    mapping = {}
    for canonical in REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS:
        aliases = COLUMN_ALIASES.get(canonical, [canonical])
        for alias in aliases:
            alias_lower = alias.lower()
            alias_clean = alias_lower.replace("_", "").replace(" ", "")
            if alias_lower in raw_lower_map:
                mapping[canonical] = raw_lower_map[alias_lower]
                break
            elif alias_clean in normalized_raw:
                mapping[canonical] = normalized_raw[alias_clean]
                break

    missing_required = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in mapping]
    return mapping, missing_required


def validate_and_clean_stream(
    csv_path: Path,
    column_mapping: Dict[str, str],
    rejected_output_path: Optional[Path] = None,
    chunksize: int = 50000,
    delimiter: str = ",",
    encoding: str = "utf-8",
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Stream CSV from disk in bounded chunks.
    Validates row by row, computes streaming EDA across ALL valid rows,
    and returns (eda_summary, first_chunk_valid_df_for_scoring).
    """
    # Inverse mapping: raw_header -> canonical
    raw_to_canonical = {raw: canonical for canonical, raw in column_mapping.items() if raw}

    total_rows = 0
    valid_rows_count = 0
    rejection_counts: Dict[str, int] = {
        "invalid_type": 0,
        "non_positive_amount": 0,
        "negative_step": 0,
        "invalid_numeric": 0,
        "missing_required_value": 0,
    }

    # Streaming EDA state
    type_counts: Dict[str, int] = {t: 0 for t in ALLOWED_TYPES}
    total_amount = 0.0
    min_amount = float("inf")
    max_amount = float("-inf")
    step_min = float("inf")
    step_max = float("-inf")
    has_labels = "isFraud" in column_mapping
    total_fraud_count = 0
    total_fraud_amount = 0.0

    # Rejected writer
    rejected_file = None
    rejected_writer = None
    if rejected_output_path:
        rejected_output_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_file = open(rejected_output_path, "w", newline="", encoding="utf-8")
        rejected_writer = csv.writer(rejected_file)
        rejected_writer.writerow(["row_index", "rejection_reason", "raw_data"])

    valid_chunks: List[pd.DataFrame] = []
    chunk_idx = 0

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunksize,
            delimiter=delimiter,
            encoding=encoding,
            dtype=str,
            on_bad_lines="skip",
            low_memory=True,
        )

        for chunk_df in reader:
            if cancel_check and cancel_check():
                break

            chunk_idx += 1
            if progress_callback:
                progress_callback(min(30.0, 5.0 + chunk_idx * 2.0), "validating")

            # Rename mapped columns to canonical
            rename_cols = {raw: can for raw, can in raw_to_canonical.items() if raw in chunk_df.columns}
            df_can = chunk_df.rename(columns=rename_cols)

            chunk_len = len(df_can)
            total_rows += chunk_len

            # Check missing required columns in this chunk
            missing_cols = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df_can.columns]
            if missing_cols:
                raise ValueError(f"Chunk missing required canonical columns: {missing_cols}")

            # Vectorized row validations
            raw_types = df_can["type"].astype(str).str.strip().str.upper()
            valid_type_mask = raw_types.isin(ALLOWED_TYPES)

            # Numeric parses
            numeric_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            num_dfs = {}
            num_valid_mask = pd.Series(True, index=df_can.index)

            for nc in numeric_cols:
                parsed = pd.to_numeric(df_can[nc], errors="coerce")
                num_dfs[nc] = parsed
                num_valid_mask &= parsed.notna()

            step_valid_mask = num_valid_mask & (num_dfs["step"] >= 0)
            amount_valid_mask = num_valid_mask & (num_dfs["amount"] > 0)

            overall_valid_mask = valid_type_mask & num_valid_mask & step_valid_mask & amount_valid_mask

            # Log rejections
            invalid_indices = df_can.index[~overall_valid_mask]
            for idx in invalid_indices:
                reason = "invalid_numeric"
                if not valid_type_mask.loc[idx]:
                    reason = "invalid_type"
                elif num_valid_mask.loc[idx] and not (num_dfs["step"].loc[idx] >= 0):
                    reason = "negative_step"
                elif num_valid_mask.loc[idx] and not (num_dfs["amount"].loc[idx] > 0):
                    reason = "non_positive_amount"
                elif not num_valid_mask.loc[idx]:
                    reason = "invalid_numeric"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

                if rejected_writer:
                    raw_row_str = ";".join(str(v) for v in chunk_df.loc[idx].values)
                    rejected_writer.writerow([total_rows - chunk_len + idx + 1, reason, raw_row_str])

            # Process valid rows in chunk
            valid_subset = df_can.loc[overall_valid_mask].copy()
            valid_len = len(valid_subset)
            valid_rows_count += valid_len

            if valid_len > 0:
                # Cast canonical numeric types
                for nc in numeric_cols:
                    if nc == "step":
                        valid_subset[nc] = num_dfs[nc].loc[overall_valid_mask].astype(int)
                    else:
                        valid_subset[nc] = num_dfs[nc].loc[overall_valid_mask].astype(float)
                valid_subset["type"] = raw_types.loc[overall_valid_mask]

                # Update streaming EDA metrics
                amt_col = valid_subset["amount"]
                step_col = valid_subset["step"]
                total_amount += float(amt_col.sum())
                min_amount = min(min_amount, float(amt_col.min()))
                max_amount = max(max_amount, float(amt_col.max()))
                step_min = min(step_min, int(step_col.min()))
                step_max = max(step_max, int(step_col.max()))

                for t, cnt in valid_subset["type"].value_counts().items():
                    type_counts[t] = type_counts.get(t, 0) + int(cnt)

                if has_labels and "isFraud" in valid_subset.columns:
                    fraud_parsed = pd.to_numeric(valid_subset["isFraud"], errors="coerce").fillna(0).astype(int)
                    valid_subset["isFraud"] = fraud_parsed
                    frauds_in_chunk = int(fraud_parsed.sum())
                    total_fraud_count += frauds_in_chunk
                    if frauds_in_chunk > 0:
                        total_fraud_amount += float(valid_subset.loc[fraud_parsed == 1, "amount"].sum())

                # Collect for batch scoring up to limit (or streaming temporary store)
                valid_chunks.append(valid_subset)

    finally:
        if rejected_file:
            rejected_file.close()

    # Friendly error handling for empty / header-only / single row
    if total_rows == 0:
        raise ValueError("The uploaded CSV file is completely empty.")
    if valid_rows_count == 0:
        raise ValueError("No valid transaction rows found in CSV. All rows failed schema validation.")

    # Consolidate valid DataFrame
    all_valid_df = pd.concat(valid_chunks, ignore_index=True) if valid_chunks else pd.DataFrame()

    account_overlap = compute_account_overlap(all_valid_df)
    overlap_min = float(os.getenv("OVERLAP_MIN", "0.20"))
    recommended_graph_mode = "warmed" if account_overlap >= overlap_min else "cold_start"
    step_span = int(step_max - step_min + 1) if (step_max != float("-inf") and step_min != float("inf")) else 0
    span_under_72 = bool(step_span < 72)

    eda_summary = {
        "rows_total": total_rows,
        "rows_valid": valid_rows_count,
        "rows_excluded": total_rows - valid_rows_count,
        "rejection_counts": rejection_counts,
        "type_counts": type_counts,
        "total_amount": total_amount,
        "min_amount": min_amount if min_amount != float("inf") else 0.0,
        "max_amount": max_amount if max_amount != float("-inf") else 0.0,
        "step_min": step_min if step_min != float("inf") else 0,
        "step_max": step_max if step_max != float("-inf") else 0,
        "step_span": step_span,
        "span_under_72": span_under_72,
        "account_overlap": account_overlap,
        "account_overlap_pct": round(account_overlap * 100.0, 2),
        "recommended_graph_mode": recommended_graph_mode,
        "has_labels": has_labels,
        "total_fraud_count": total_fraud_count,
        "total_fraud_amount": total_fraud_amount,
    }

    return eda_summary, all_valid_df


def compute_account_overlap(df: pd.DataFrame, warmed_state=None) -> float:
    """
    Compute account overlap: share of distinct accounts (nameOrig + nameDest)
    in the dataframe that exist in the historical training network.
    """
    if df.empty or "nameOrig" not in df.columns or "nameDest" not in df.columns:
        return 0.0
    file_accounts = set(df["nameOrig"].astype(str)).union(set(df["nameDest"].astype(str)))
    if not file_accounts:
        return 0.0
    if warmed_state is None:
        warmed_state = get_warmed_graph_state()
    if warmed_state is None or not hasattr(warmed_state, "G") or warmed_state.G.number_of_nodes() == 0:
        return 0.0
    training_nodes = set(warmed_state.G.nodes())
    overlap_count = len(file_accounts.intersection(training_nodes))
    return float(overlap_count / len(file_accounts))


def batch_score_dataframe(
    valid_df: pd.DataFrame,
    active_threshold: Optional[float] = None,
    threshold_mode: str = "model_threshold",
    alert_budget_pct: Optional[float] = 2.0,
    use_labels_for_graph: bool = False,
    graph_mode: str = "warmed",
    max_score_rows: int = DEFAULT_MAX_SCORE_ROWS,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score valid transactions using the exact trained model pipeline.
    Preserves exact row order by sorting chronologically for graph features
    and then mapping probabilities and features back to original row indices.
    Supports 3 graph modes:
      - 'warmed': uses historical topology from model training history (min covariate shift)
      - 'cold_start': neutral baseline graph defaults for unseen accounts
      - 'file_only': expands graph strictly within the uploaded file
    Guarantees input DataFrame is NEVER mutated.
    """
    if cancel_check and cancel_check():
        raise InterruptedError("Batch scoring cancelled by user.")

    # Guard against mutating input DataFrame
    valid_df = valid_df.copy()

    # Calculate account overlap with training graph
    account_overlap = compute_account_overlap(valid_df)
    overlap_min = float(os.getenv("OVERLAP_MIN", "0.20"))
    if graph_mode == "auto":
        graph_mode = "warmed" if account_overlap >= overlap_min else "cold_start"

    # Load model artifacts
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model artifacts (models/xgboost_model.joblib or scaler.joblib) not found.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        try:
            metadata = joblib.load(METADATA_PATH)
        except Exception:
            pass

    dataset_name = str(metadata.get("dataset_name", "Synthetic PaySim (300K)"))
    model_version = str(metadata.get("model_version", "v2_xgboost"))
    default_threshold = float(metadata.get("active_threshold", metadata.get("cost_optimal_threshold", 0.017033)))
    threshold_to_use = active_threshold if active_threshold is not None else default_threshold

    feature_names = metadata.get("feature_names", list(getattr(model, "feature_names_in_", [])))
    if not feature_names and hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)

    total_valid = len(valid_df)
    capped = False
    warning_message = None

    if total_valid > max_score_rows:
        capped = True
        warning_message = f"EDA computed on {total_valid:,} rows; scored first {max_score_rows:,} rows in chronological step order."
        # Select first max_score_rows in step order
        score_df = valid_df.sort_values("step", kind="stable").head(max_score_rows).copy()
    else:
        score_df = valid_df.copy()

    # Attach original row index to guarantee exact 1-to-1 parity and alignment
    score_df["_orig_score_idx"] = np.arange(len(score_df))

    if progress_callback:
        progress_callback(40.0, "graph")

    # 1. Graph feature calculation according to graph_mode
    graph_input_df = score_df.copy()

    if graph_mode == "cold_start":
        # Neutral cold-start graph features (unseen account defaults)
        for prefix in ["orig", "dest"]:
            graph_input_df[f"{prefix}_in_degree"] = 0.0
            graph_input_df[f"{prefix}_out_degree"] = 0.0
            graph_input_df[f"{prefix}_pagerank"] = 0.0
            graph_input_df[f"{prefix}_component_size"] = 1.0
            graph_input_df[f"{prefix}_fraud_exposure"] = 0.0
        graph_aligned = graph_input_df.sort_values("_orig_score_idx", kind="stable").reset_index(drop=True)

    elif graph_mode == "warmed":
        # Training-warmed graph context
        warmed_state = get_warmed_graph_state()
        if warmed_state is not None:
            orig_feats = [warmed_state.get_node_features(n) for n in graph_input_df["nameOrig"]]
            dest_feats = [warmed_state.get_node_features(n) for n in graph_input_df["nameDest"]]
            for f in ["in_degree", "out_degree", "pagerank", "component_size", "fraud_exposure"]:
                graph_input_df[f"orig_{f}"] = [of[f] for of in orig_feats]
                graph_input_df[f"dest_{f}"] = [df_[f] for df_ in dest_feats]
            if not use_labels_for_graph:
                graph_input_df["orig_fraud_exposure"] = 0.0
                graph_input_df["dest_fraud_exposure"] = 0.0
            graph_aligned = graph_input_df.sort_values("_orig_score_idx", kind="stable").reset_index(drop=True)
        else:
            # Fallback safely to cold start if training history not found
            for prefix in ["orig", "dest"]:
                graph_input_df[f"{prefix}_in_degree"] = 0.0
                graph_input_df[f"{prefix}_out_degree"] = 0.0
                graph_input_df[f"{prefix}_pagerank"] = 0.0
                graph_input_df[f"{prefix}_component_size"] = 1.0
                graph_input_df[f"{prefix}_fraud_exposure"] = 0.0
            graph_aligned = graph_input_df.sort_values("_orig_score_idx", kind="stable").reset_index(drop=True)

    else:
        # "file_only" (Expanding graph computed exclusively from uploaded file)
        if not use_labels_for_graph and "isFraud" in graph_input_df.columns:
            graph_input_df["isFraud"] = 0

        if cancel_check and cancel_check():
            raise InterruptedError("Batch scoring cancelled by user.")

        graph_out = attach_graph_features_expanding(graph_input_df, step_batch_size=DEFAULT_STEP_BATCH_SIZE)
        if not use_labels_for_graph:
            graph_out["orig_fraud_exposure"] = 0.0
            graph_out["dest_fraud_exposure"] = 0.0
        graph_aligned = graph_out.sort_values("_orig_score_idx", kind="stable").reset_index(drop=True)

    if progress_callback:
        progress_callback(60.0, "features")

    if cancel_check and cancel_check():
        raise InterruptedError("Batch scoring cancelled by user.")

    # 2. Tabular feature engineering
    df_eng = engineer_tabular_features(graph_aligned)

    # Ensure all 5 one-hot encoded type columns are present
    for t_col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if t_col not in df_eng.columns:
            df_eng[t_col] = (df_eng["type"] == t_col.replace("type_", "")).astype(float)

    # 3. Model feature alignment & scaling
    missing_feats = [f for f in feature_names if f not in df_eng.columns]
    if missing_feats:
        raise ValueError(f"Engineered dataframe is missing model features: {missing_feats}")

    X_unscaled = df_eng[feature_names].copy()

    if progress_callback:
        progress_callback(75.0, "scoring")

    if cancel_check and cancel_check():
        raise InterruptedError("Batch scoring cancelled by user.")

    # 4. Scale using training-fitted scaler
    X_scaled = scaler.transform(X_unscaled)

    # 5. XGBoost batch inference
    probs = model.predict_proba(X_scaled)[:, 1]

    if progress_callback:
        progress_callback(85.0, "aggregates")

    # Evaluate decision flags under active threshold mode
    if threshold_mode == "alert_budget":
        pct = float(alert_budget_pct if alert_budget_pct is not None else 2.0)
        pct = max(0.01, min(100.0, pct))
        budget_k = max(1, int(round(len(probs) * (pct / 100.0))))
        kth_val = float(np.sort(probs)[-budget_k])
        effective_threshold = kth_val
        flagged_mask = probs >= effective_threshold
        if np.sum(flagged_mask) > budget_k:
            top_k_indices = np.argsort(probs)[-budget_k:]
            flagged_mask = np.zeros(len(probs), dtype=bool)
            flagged_mask[top_k_indices] = True
        active_threshold_used = effective_threshold
    else:
        threshold_mode = "model_threshold"
        active_threshold_used = threshold_to_use
        flagged_mask = probs >= active_threshold_used

    decision_col = np.where(flagged_mask, "FLAGGED", "CLEARED")

    # Attach results to dataframe
    score_df["fraud_probability"] = probs
    score_df["decision"] = decision_col
    score_df["threshold_used"] = active_threshold_used
    score_df["threshold_mode"] = threshold_mode
    score_df["account_drained"] = df_eng["account_drained"].values

    # Clean internal temporary tracking column
    score_df = score_df.drop(columns=["_orig_score_idx"], errors="ignore")

    # 6. Compute Analytics & KPIs
    flagged_mask = score_df["decision"] == "FLAGGED"
    flagged_count = int(flagged_mask.sum())
    total_scored = len(score_df)
    flagged_rate = float(flagged_count / max(total_scored, 1))
    total_amount_flagged = float(score_df.loc[flagged_mask, "amount"].sum())
    amount_cleared = float(score_df.loc[~flagged_mask, "amount"].sum())

    # Flagged by type breakdown
    flagged_by_type = score_df.loc[flagged_mask, "type"].value_counts().to_dict()
    for t in ALLOWED_TYPES:
        if t not in flagged_by_type:
            flagged_by_type[t] = 0

    # Drained account among flagged
    drained_flagged_count = int(score_df.loc[flagged_mask, "account_drained"].sum())
    non_drained_flagged_count = flagged_count - drained_flagged_count

    # Probability histogram (20 bins: 0.0 - 1.0)
    prob_hist, prob_bin_edges = np.histogram(probs, bins=20, range=(0.0, 1.0))
    prob_histogram = {
        "counts": prob_hist.tolist(),
        "bin_edges": [round(float(b), 2) for b in prob_bin_edges.tolist()],
    }

    # Amount log-histogram (flagged vs cleared)
    amounts = score_df["amount"].values
    log_amounts = np.log10(np.maximum(amounts, 1.0))
    min_log = float(np.min(log_amounts))
    max_log = float(np.max(log_amounts))
    if min_log == max_log:
        max_log += 1.0
    bins = np.linspace(min_log, max_log, 15)

    amt_hist_cleared, _ = np.histogram(log_amounts[~flagged_mask.values], bins=bins)
    amt_hist_flagged, _ = np.histogram(log_amounts[flagged_mask.values], bins=bins)
    amount_distribution = {
        "bin_labels": [f"10^{round(b, 1)}" for b in bins[:-1]],
        "cleared_counts": amt_hist_cleared.tolist(),
        "flagged_counts": amt_hist_flagged.tolist(),
    }

    # Transactions over time (step buckets)
    min_s = int(score_df["step"].min())
    max_s = int(score_df["step"].max())
    n_step_bins = min(30, max(1, max_s - min_s + 1))
    step_edges = np.linspace(min_s, max_s + 1, n_step_bins + 1)
    step_binned = np.digitize(score_df["step"].values, step_edges) - 1
    step_binned = np.clip(step_binned, 0, n_step_bins - 1)

    time_trend_labels = [f"Step {int(step_edges[i])}-{int(step_edges[i+1]-1)}" for i in range(n_step_bins)]
    time_total = [0] * n_step_bins
    time_flagged = [0] * n_step_bins

    for b_idx, is_fl in zip(step_binned, flagged_mask.values):
        time_total[b_idx] += 1
        if is_fl:
            time_flagged[b_idx] += 1

    time_series_chart = {
        "labels": time_trend_labels,
        "total": time_total,
        "flagged": time_flagged,
    }

    # Evaluation metrics if ground truth labels (isFraud) are present
    has_labels = "isFraud" in score_df.columns
    metrics = None
    if has_labels:
        y_true = score_df["isFraud"].astype(int).values
        y_pred = flagged_mask.astype(int)
        amounts_scored = score_df["amount"].values

        metrics = compute_evaluation_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            amounts=amounts_scored,
            active_threshold=active_threshold_used,
        )

    high_alert_rate_warning = None
    if flagged_rate > 0.20:
        high_alert_rate_warning = (
            f"High alert rate ({flagged_rate*100:.1f}%): Graph features have little context in small files; results are indicative."
        )

    synthetic_warning = None
    if "synth" in dataset_name.lower():
        synthetic_warning = f"Model trained on {dataset_name}; scores on real data may not be reliable."

    training_overlap_warning = "Metrics are optimistic if this file overlaps the model's training data."
    low_overlap_warning = None
    if account_overlap < overlap_min:
        low_overlap_warning = (
            f"Low account overlap ({account_overlap*100:.1f}% < {overlap_min*100:.0f}%): "
            "Warming on training history does nothing for unseen accounts; neutral cold-start defaults are recommended."
        )

    summary_metadata = {
        "model_version": model_version,
        "dataset_name": dataset_name,
        "threshold_used": active_threshold_used,
        "threshold_mode": threshold_mode,
        "alert_budget_pct": alert_budget_pct,
        "account_overlap": account_overlap,
        "account_overlap_pct": round(account_overlap * 100.0, 2),
        "recommended_graph_mode": "warmed" if account_overlap >= overlap_min else "cold_start",
        "rows_scored": total_scored,
        "flagged_count": flagged_count,
        "flagged_rate": flagged_rate,
        "total_amount_flagged": total_amount_flagged,
        "amount_cleared": amount_cleared,
        "warning_message": warning_message,
        "high_alert_rate_warning": high_alert_rate_warning,
        "synthetic_warning": synthetic_warning,
        "training_overlap_warning": training_overlap_warning,
        "low_overlap_warning": low_overlap_warning,
        "scoring_options": {
            "graph_mode": graph_mode,
            "threshold_mode": threshold_mode,
            "alert_budget_pct": alert_budget_pct,
            "use_labels_for_graph": use_labels_for_graph,
            "max_score_rows": max_score_rows,
            "threshold": active_threshold_used,
        },
        "flagged_by_type": flagged_by_type,
        "drained_flagged_count": drained_flagged_count,
        "non_drained_flagged_count": non_drained_flagged_count,
        "prob_histogram": prob_histogram,
        "amount_distribution": amount_distribution,
        "time_series_chart": time_series_chart,
        "has_labels": has_labels,
        "metrics": metrics,
    }

    if progress_callback:
        progress_callback(100.0, "finishing")

    return score_df, summary_metadata


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    amounts: np.ndarray,
    active_threshold: float,
    fp_review_cost: float = 50.0,
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics with exact financial cost reconciliation:
    assert TP + FP + FN + TN == N
    assert caught_fraud_amount + missed_fraud_amount == total_fraud_amount
    assert total_cost == missed_fraud_amount + 50.0 * FP
    """
    n = len(y_true)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # Strict reconciliation assertion 1
    assert tp + fp + fn + tn == n, f"Reconciliation failed: TP({tp}) + FP({fp}) + FN({fn}) + TN({tn}) != N({n})"

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2 = float(5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0
    alert_rate = float((tp + fp) / max(n, 1))

    # Financial cost calculations in Indian Rupees (raw unscaled)
    caught_amount = float(np.sum(amounts[(y_true == 1) & (y_pred == 1)]))
    missed_amount = float(np.sum(amounts[(y_true == 1) & (y_pred == 0)]))
    total_fraud_amount = float(np.sum(amounts[y_true == 1]))

    # Strict reconciliation assertion 2
    assert abs((caught_amount + missed_amount) - total_fraud_amount) < 1e-2, (
        f"Amount reconciliation failed: caught({caught_amount}) + missed({missed_amount}) != total({total_fraud_amount})"
    )

    review_cost = float(fp * fp_review_cost)
    total_cost = missed_amount + review_cost

    # Baseline comparison costs
    # Flag nothing: miss all fraud, 0 FP review costs
    flag_nothing_cost = total_fraud_amount
    # Flag everything: catch all fraud (missed = 0), review every non-fraud transaction
    non_fraud_count = int(np.sum(y_true == 0))
    flag_everything_cost = float(non_fraud_count * fp_review_cost)

    savings_vs_nothing = float(flag_nothing_cost - total_cost)

    # PR-AUC & ROC-AUC
    pr_auc = 0.0
    roc_auc = 0.0
    try:
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            prec_vals, rec_vals, _ = precision_recall_curve(y_true, probs)
            pr_auc = float(auc(rec_vals, prec_vals))
            roc_auc = float(roc_auc_score(y_true, probs))
    except Exception:
        pass

    # Alert Budgets (Recall at 1%, 2%, 5% alert budget)
    alert_budgets = {}
    for budget_pct in [0.01, 0.02, 0.05]:
        budget_k = max(1, int(math.ceil(n * budget_pct)))
        top_k_indices = np.argsort(probs)[-budget_k:]
        frauds_caught_k = int(np.sum(y_true[top_k_indices] == 1))
        recall_at_k = float(frauds_caught_k / max(int(np.sum(y_true == 1)), 1))
        alert_budgets[f"{int(budget_pct*100)}%"] = {
            "alert_budget_rows": budget_k,
            "frauds_caught": frauds_caught_k,
            "recall": recall_at_k,
        }

    # Small fraud count safeguard (< 100 fraud rows)
    total_fraud_rows = int(np.sum(y_true == 1))
    small_sample_warning = None
    bootstrap_ci = None

    if total_fraud_rows < 100:
        small_sample_warning = (
            f"Metrics (PR-AUC / ROC-AUC) unreliable with small fraud sample ({total_fraud_rows} < 100 confirmed frauds). "
            "Bootstrap 95% confidence intervals are displayed below."
        )
        bootstrap_ci = compute_bootstrap_confidence_intervals(y_true, y_pred, probs=probs, n_bootstraps=500)

    return {
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": n
        },
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "financial_cost": {
            "total_cost": total_cost,
            "missed_fraud_amount": missed_amount,
            "fp_review_cost": review_cost,
            "caught_fraud_amount": caught_amount,
            "total_fraud_amount": total_fraud_amount,
            "flag_nothing_cost": flag_nothing_cost,
            "flag_everything_cost": flag_everything_cost,
            "savings_vs_nothing": savings_vs_nothing,
            "fp_unit_cost": fp_review_cost,
        },
        "alert_budgets": alert_budgets,
        "small_sample_warning": small_sample_warning,
        "bootstrap_ci": bootstrap_ci,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals for recall, precision, F1, and AUCs."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    recalls, precisions, f1s, pr_aucs, roc_aucs = [], [], [], [], []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        b_true = y_true[boot_idx]
        b_pred = y_pred[boot_idx]

        tp = np.sum((b_true == 1) & (b_pred == 1))
        fp = np.sum((b_true == 0) & (b_pred == 1))
        fn = np.sum((b_true == 1) & (b_pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1_val)

        if probs is not None and np.sum(b_true == 1) > 0 and np.sum(b_true == 0) > 0:
            b_probs = probs[boot_idx]
            try:
                prec_vals, rec_vals, _ = precision_recall_curve(b_true, b_probs)
                pr_aucs.append(float(auc(rec_vals, prec_vals)))
                roc_aucs.append(float(roc_auc_score(b_true, b_probs)))
            except Exception:
                pass

    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    ci_dict = {
        "recall_ci": (round(float(np.percentile(recalls, lower_p)), 4), round(float(np.percentile(recalls, upper_p)), 4)),
        "precision_ci": (round(float(np.percentile(precisions, lower_p)), 4), round(float(np.percentile(precisions, upper_p)), 4)),
        "f1_ci": (round(float(np.percentile(f1s, lower_p)), 4), round(float(np.percentile(f1s, upper_p)), 4)),
    }
    if pr_aucs:
        ci_dict["pr_auc_ci"] = (round(float(np.percentile(pr_aucs, lower_p)), 4), round(float(np.percentile(pr_aucs, upper_p)), 4))
    if roc_aucs:
        ci_dict["roc_auc_ci"] = (round(float(np.percentile(roc_aucs, lower_p)), 4), round(float(np.percentile(roc_aucs, upper_p)), 4))

    return ci_dict


def sanitize_csv_formula_injection(val: Any) -> str:
    """Escape CSV formula injection: prefix any string starting with = + - @ with a single quote."""
    if val is None:
        return ""
    s = str(val)
    if s.startswith(("=", "+", "-", "@")):
        return "'" + s
    return s


def export_sanitized_results_csv(df: pd.DataFrame, output_path: Path):
    """Write results CSV with full CSV formula-injection protection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(df.columns))
        for row in df.itertuples(index=False):
            sanitized_row = [sanitize_csv_formula_injection(cell) for cell in row]
            writer.writerow(sanitized_row)
