"""
preprocess.py
=============
Data preprocessing and graph feature merging for FinSecure AI v2.

Why this exists
----------------
Raw PaySim transaction records contain literal account identifier columns
(`nameOrig` and `nameDest`). In tabular models, raw IDs cause severe overfitting
and leakage because arbitrary alphanumeric IDs do not generalize to unseen
accounts. However, dropping them immediately destroys the graph structure of the
financial network.

This module enforces the critical pipeline ordering:
1. Load raw transaction log (`paysim.csv`).
2. Merge graph features extracted by `graph_features.py` using `(step, nameOrig, nameDest)`
   BEFORE dropping ID columns.
3. Drop literal ID leakage columns (`nameOrig`, `nameDest`, `isFlaggedFraud`).
4. Clean missing values, handle categorical encoding (`type`), and output clean data.

Pipeline position:
    paysim.csv -> graph_features.py -> preprocess.py -> feature_engineering.py -> train_model.py
"""

import os
from pathlib import Path
import pandas as pd

_data_path_env = os.getenv("DATA_PATH") or os.getenv("PAYSIM_DATA_PATH")
if _data_path_env:
    RAW_DATA_PATH = Path(_data_path_env)
elif Path("data/paysim_synth.csv").exists():
    RAW_DATA_PATH = Path("data/paysim_synth.csv")
else:
    RAW_DATA_PATH = Path("data/paysim.csv")

GRAPH_DATA_PATH = Path("data/transactions_with_graph_features.parquet")
OUTPUT_DATA_PATH = Path("data/cleaned_transactions.parquet")

_sample_env = os.getenv("SAMPLE_ROWS")
if _sample_env is not None and _sample_env.strip() != "":
    SAMPLE_ROWS = int(_sample_env)
else:
    SAMPLE_ROWS = None


def load_raw(path: Path = RAW_DATA_PATH, sample_rows: int | None = SAMPLE_ROWS) -> pd.DataFrame:
    """
    Load raw PaySim data prior to any ID column removal or transformation.

    Why this exists:
    Maintains raw dataframe state including `nameOrig` and `nameDest` so graph
    relational features can be successfully joined.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw PaySim dataset not found at {path}")
    df = pd.read_csv(path, nrows=sample_rows)
    required = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PaySim columns: {missing}")
    return df


def merge_graph_features(
    df: pd.DataFrame,
    graph_path: Path = GRAPH_DATA_PATH
) -> pd.DataFrame:
    """
    Merge graph features from parquet output of graph_features.py on (step, nameOrig, nameDest).

    Why this exists:
    Graph features are precomputed using temporal expanding-window logic in
    `graph_features.py`. Joining on transaction key identifiers (step, nameOrig, nameDest)
    attaches relational features (degree, PageRank, component size, fraud exposure)
    to each transaction record before raw account IDs are discarded.
    """
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph features file not found at {graph_path}. "
            "Please run `python src/graph_features.py` first."
        )

    graph_df = pd.read_parquet(graph_path)

    graph_cols = [
        c for c in graph_df.columns
        if c.startswith(("orig_", "dest_")) and c not in ("nameOrig", "nameDest")
    ]

    if not graph_cols:
        raise ValueError(f"No graph feature columns (orig_*, dest_*) found in {graph_path}")

    # Key columns for joining graph features back onto raw transaction records
    key_cols = ["step", "nameOrig", "nameDest"]

    # Check that key columns exist in both dataframes
    for col in key_cols:
        if col not in df.columns:
            raise KeyError(f"Key column '{col}' missing from input DataFrame")
        if col not in graph_df.columns:
            raise KeyError(f"Key column '{col}' missing from graph DataFrame ({graph_path})")

    # Diagnostic 1: Print merge key dtypes on both sides
    print("--- DIAGNOSTIC: MERGE KEY DTYPES ---")
    for col in key_cols:
        print(f"Key '{col}': raw df dtype = {df[col].dtype}, graph df dtype = {graph_df[col].dtype}")

    # Drop any graph feature columns if already in df to prevent suffix duplication
    df_clean = df.drop(columns=[c for c in graph_cols if c in df.columns])

    # Extract lean subset of keys and graph columns, deduplicating keys if necessary
    graph_subset = graph_df[key_cols + graph_cols].drop_duplicates(subset=key_cols)

    # Perform left join with indicator=True to track matching
    merged = pd.merge(df_clean, graph_subset, on=key_cols, how="left", indicator=True)

    # Diagnostic 2: Print pandas _merge indicator value counts
    print("\n--- DIAGNOSTIC: MERGE INDICATOR VALUE COUNTS ---")
    print(merged["_merge"].value_counts())

    # Diagnostic 3: Null count per graph feature column immediately after merge (before fillna)
    print("\n--- DIAGNOSTIC: NULL COUNTS PER GRAPH FEATURE (BEFORE FILLNA) ---")
    print(merged[graph_cols].isna().sum())

    # Diagnostic 4: Cold-start detection analysis
    is_cold = (
        (merged["orig_in_degree"] == 0) &
        (merged["orig_out_degree"] == 0) &
        (merged["orig_pagerank"] == 0) &
        (merged["dest_in_degree"] == 0) &
        (merged["dest_out_degree"] == 0) &
        (merged["dest_pagerank"] == 0)
    )
    merged["is_cold_start"] = is_cold
    total_rows = len(merged)
    cold_count = is_cold.sum()
    cold_pct = (cold_count / total_rows) * 100

    print("\n--- DIAGNOSTIC: COLD-START ROW ANALYSIS ---")
    print(f"Overall Cold-Start Rows: {cold_count}/{total_rows} ({cold_pct:.2f}%)")

    batch_num = merged["step"] // 24
    batch_breakdown = merged.groupby(batch_num)["is_cold_start"].agg(
        total_rows="count",
        cold_start_count="sum",
        cold_start_rate=lambda x: f"{x.mean() * 100:.2f}%"
    )
    print("\nCold-Start Rate Breakdown by Step Batch (step // 24):")
    print(batch_breakdown)
    print("---------------------------------------------------\n")

    # Remove the temporary indicator and diagnostic columns
    merged = merged.drop(columns=["_merge", "is_cold_start"])


    # Impute neutral default values for any missing graph feature entries
    for col in graph_cols:
        default_val = 1.0 if "component_size" in col else 0.0
        merged[col] = merged[col].fillna(default_val)

    return merged



def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove literal ID columns and non-generalizable rule-based flags.

    Why this exists:
    `nameOrig` and `nameDest` are high-cardinality string IDs that cause tabular tree models
    to overfit to specific entity strings rather than learning generalizable fraud patterns.
    `isFlaggedFraud` is a static rule flag (>200k amount in PaySim) that induces target leakage.
    After graph features have been safely merged, these literal columns must be dropped.
    """
    df = df.copy()

    leakage_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    existing_leakage = [c for c in leakage_cols if c in df.columns]
    if existing_leakage:
        df = df.drop(columns=existing_leakage)

    # Preserve unscaled raw amount in rupees for financial cost evaluations
    if "amount" in df.columns:
        df["amount_raw"] = df["amount"].astype(float)

    # Clean missing values & remove duplicate rows
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # One-hot encode categorical transaction 'type' column
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="type", dtype=float)

    return df


def main():
    print(f"Loading raw transaction data from {RAW_DATA_PATH}...")
    raw_df = load_raw()
    print(f"Loaded {len(raw_df):,} raw transactions.")

    print(f"Merging graph features from {GRAPH_DATA_PATH}...")
    df_merged = merge_graph_features(raw_df)
    print(f"Graph features merged. Total columns: {len(df_merged.columns)}")

    print("Dropping leakage columns (nameOrig, nameDest, isFlaggedFraud) and encoding categories...")
    df_clean = drop_leakage_columns(df_merged)
    print(f"Preprocessing complete. Clean shape: {df_clean.shape}")

    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUTPUT_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()

