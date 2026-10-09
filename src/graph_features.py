"""
graph_features.py
==================
Fully vectorized, high-performance graph feature extraction for FinSecure AI v2.

Pipeline position:
    paysim.csv -> graph_features.py -> preprocess.py -> feature_engineering.py -> train_model.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

_data_path_env = os.getenv("DATA_PATH") or os.getenv("PAYSIM_DATA_PATH")
if _data_path_env:
    RAW_DATA_PATH = Path(_data_path_env)
elif Path("data/paysim_synth.csv").exists():
    RAW_DATA_PATH = Path("data/paysim_synth.csv")
else:
    RAW_DATA_PATH = Path("data/paysim.csv")

OUTPUT_PATH = Path("data/transactions_with_graph_features.parquet")

USE_EXPANDING_WINDOW = True
STEP_BATCH_SIZE = int(os.getenv("STEP_BATCH_SIZE", "72"))

_sample_env = os.getenv("SAMPLE_ROWS")
if _sample_env is not None and _sample_env.strip() != "":
    SAMPLE_ROWS = int(_sample_env)
else:
    SAMPLE_ROWS = None


def load_raw(path: Path = RAW_DATA_PATH, sample_rows: int | None = SAMPLE_ROWS) -> pd.DataFrame:
    """Load PaySim1 BEFORE nameOrig/nameDest are dropped."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    df = pd.read_csv(path, nrows=sample_rows)
    required = {"step", "nameOrig", "nameDest", "amount", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected PaySim columns: {missing}")
    return df


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph collapsed to weighted DiGraph."""
    G = nx.DiGraph()
    if edges_df.empty:
        return G

    grouped = (
        edges_df.groupby(["nameOrig", "nameDest"])
        .agg(
            weight=("amount", "count"),
            total_amount=("amount", "sum"),
            fraud_count=("isFraud", "sum"),
        )
        .reset_index()
    )

    edges = [
        (orig, dest, {"weight": w, "total_amount": a, "fraud_count": f})
        for orig, dest, w, a, f in grouped.itertuples(index=False)
    ]
    G.add_edges_from(edges)
    return G


def compute_node_features(history_df: pd.DataFrame | nx.DiGraph = None, G: nx.DiGraph = None) -> pd.DataFrame:
    """
    Compute per-node structural + fraud-exposure features. Supports flexible argument passing.
    """
    if isinstance(history_df, nx.DiGraph):
        G = history_df
        history_df = None

    if G is None:
        G = nx.DiGraph()

    if G.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "node", "in_degree", "out_degree", "pagerank",
                "component_size", "fraud_exposure",
            ]
        )

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=25, tol=1e-2)
    except Exception:
        pagerank = {n: 1.0 / len(G) for n in G.nodes()}

    component_size = {}
    for component in nx.weakly_connected_components(G):
        sz = len(component)
        for node in component:
            component_size[node] = sz

    fraud_exp_dict = {}
    if history_df is not None and not history_df.empty and isinstance(history_df, pd.DataFrame):
        orig_edges = history_df[["nameOrig", "isFraud"]].rename(columns={"nameOrig": "node"})
        dest_edges = history_df[["nameDest", "isFraud"]].rename(columns={"nameDest": "node"})
        all_edges = pd.concat([orig_edges, dest_edges], ignore_index=True)
        fraud_exp_dict = all_edges.groupby("node")["isFraud"].mean().to_dict()

    nodes = list(G.nodes())
    return pd.DataFrame(
        {
            "node": nodes,
            "in_degree": [in_degree.get(n, 0) for n in nodes],
            "out_degree": [out_degree.get(n, 0) for n in nodes],
            "pagerank": [pagerank.get(n, 0.0) for n in nodes],
            "component_size": [component_size.get(n, 1) for n in nodes],
            "fraud_exposure": [fraud_exp_dict.get(n, 0.0) for n in nodes],
        }
    )


def attach_graph_features_expanding(df: pd.DataFrame, step_batch_size: int | None = None) -> pd.DataFrame:
    """
    Leakage-safe expanding window graph feature attachment with vectorized map lookups.
    """
    if step_batch_size is None:
        step_batch_size = int(os.getenv("STEP_BATCH_SIZE", "72"))

    df = df.sort_values("step").reset_index(drop=True)
    steps = df["step"]
    max_step = int(steps.max())

    out_chunks = []

    for batch_start in range(0, max_step + 1, step_batch_size):
        batch_end = batch_start + step_batch_size
        batch_mask = (steps >= batch_start) & (steps < batch_end)
        history_mask = steps < batch_start

        batch_df = df.loc[batch_mask].copy()
        if batch_df.empty:
            continue

        history_df = df.loc[history_mask]
        if history_df.empty:
            node_feats = compute_node_features(history_df, nx.DiGraph())
        else:
            G = build_graph(history_df)
            node_feats = compute_node_features(history_df, G)

        if not node_feats.empty:
            feats_indexed = node_feats.set_index("node")
        else:
            feats_indexed = pd.DataFrame()

        for prefix, id_col in [("orig", "nameOrig"), ("dest", "nameDest")]:
            for feat in ["in_degree", "out_degree", "pagerank", "component_size", "fraud_exposure"]:
                default_val = 1.0 if feat == "component_size" else 0.0
                if not feats_indexed.empty and feat in feats_indexed.columns:
                    mapping = feats_indexed[feat]
                    batch_df[f"{prefix}_{feat}"] = batch_df[id_col].map(mapping).fillna(default_val)
                else:
                    batch_df[f"{prefix}_{feat}"] = default_val

        out_chunks.append(batch_df)

    return pd.concat(out_chunks, ignore_index=True)


def attach_graph_features_full(df: pd.DataFrame) -> pd.DataFrame:
    """Fast non-leakage-safe version for EDA."""
    G = build_graph(df)
    node_feats = compute_node_features(df, G)
    if not node_feats.empty:
        feats_indexed = node_feats.set_index("node")
    else:
        feats_indexed = pd.DataFrame()

    df = df.copy()
    for prefix, id_col in [("orig", "nameOrig"), ("dest", "nameDest")]:
        for feat in ["in_degree", "out_degree", "pagerank", "component_size", "fraud_exposure"]:
            default_val = 1.0 if feat == "component_size" else 0.0
            if not feats_indexed.empty and feat in feats_indexed.columns:
                mapping = feats_indexed[feat]
                df[f"{prefix}_{feat}"] = df[id_col].map(mapping).fillna(default_val)
            else:
                df[f"{prefix}_{feat}"] = default_val
    return df


def main():
    print(f"Loading raw PaySim data (path={RAW_DATA_PATH}, sample_rows={SAMPLE_ROWS})...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows.")

    if USE_EXPANDING_WINDOW:
        print(f"Building leakage-safe expanding-window graph features (batch size = {STEP_BATCH_SIZE} steps)...")
        df_out = attach_graph_features_expanding(df)
    else:
        print("WARNING: using full-graph mode. EDA only.")
        df_out = attach_graph_features_full(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_out):,} rows with graph features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
