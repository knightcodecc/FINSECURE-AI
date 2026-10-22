"""
analytics.py — PaySim Precomputed Aggregates API Endpoints for FinSecure AI v2.
================================================================================
Serves memory-cached precomputed aggregates for large-scale PaySim datasets.
Ensures zero browser raw-data loading.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends

import os
from api.auth import get_current_user, get_data_dir

router = APIRouter(prefix="/analytics", tags=["PaySim Analytics"])

PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# In-memory cache: dataset_key -> loaded dict
_AGGREGATE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_aggregate_data(dataset: str = "synthetic") -> Dict[str, Any]:
    """Retrieve precomputed aggregates from memory cache or disk."""
    dataset_clean = dataset.lower().strip()
    if dataset_clean in _AGGREGATE_CACHE:
        return _AGGREGATE_CACHE[dataset_clean]

    current_data_dir = get_data_dir()
    target_file = current_data_dir / f"aggregates_{dataset_clean}.json"
    if not target_file.exists():
        # Check alias in current data dir
        if dataset_clean in ("full", "paysim_full", "6.36m", "paysim"):
            target_file = current_data_dir / "aggregates_paysim.json"
        elif dataset_clean in ("synth", "synthetic_300k", "synthetic"):
            target_file = current_data_dir / "aggregates_synthetic.json"

    # Fallback to project root data/ if running under custom temp DATA_DIR
    if not target_file.exists() and PROJECT_DATA_DIR.exists():
        target_file = PROJECT_DATA_DIR / f"aggregates_{dataset_clean}.json"
        if not target_file.exists():
            if dataset_clean in ("full", "paysim_full", "6.36m", "paysim"):
                target_file = PROJECT_DATA_DIR / "aggregates_paysim.json"
            elif dataset_clean in ("synth", "synthetic_300k", "synthetic"):
                target_file = PROJECT_DATA_DIR / "aggregates_synthetic.json"

    if not target_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Aggregate file for dataset '{dataset}' was not found at {target_file.name}. "
                   "Run 'python scripts/build_aggregates.py' to precompute dataset aggregates."
        )

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            _AGGREGATE_CACHE[dataset_clean] = data
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read aggregate data: {e}")


@router.get("/datasets")
def list_available_datasets(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all available precomputed aggregate datasets and their summary metadata."""
    datasets = []
    candidates = [
        ("paysim", "PaySim Canonical (6.36M Rows)", "aggregates_paysim.json"),
        ("synthetic", "PaySim Synthetic (300K Rows)", "aggregates_synthetic.json"),
        ("sample", "PaySim Quick Sample (1K Rows)", "aggregates_sample.json"),
    ]

    for key, label, fname in candidates:
        p = DATA_DIR / fname
        if p.exists():
            try:
                data = get_aggregate_data(key)
                meta = data.get("metadata", {})
                summ = data.get("summary", {})
                datasets.append({
                    "id": key,
                    "name": label,
                    "filename": fname,
                    "row_count": meta.get("row_count", summ.get("total_transactions", 0)),
                    "fraud_count": meta.get("fraud_count", summ.get("total_fraud", 0)),
                    "build_time_utc": meta.get("build_time_utc", "N/A"),
                })
            except Exception:
                pass

    return {
        "datasets": datasets,
        "default": "paysim" if any(d["id"] == "paysim" for d in datasets) else "synthetic"
    }


@router.get("/summary")
def get_analytics_summary(
    dataset: str = Query("synthetic", description="Dataset key: 'paysim' or 'synthetic'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Overall dataset summary, fraud rates, volumes, and monetary ranges."""
    data = get_aggregate_data(dataset)
    return {
        "metadata": data.get("metadata", {}),
        "summary": data.get("summary", {})
    }


@router.get("/by-type")
def get_analytics_by_type(
    dataset: str = Query("synthetic", description="Dataset key: 'paysim' or 'synthetic'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Transaction counts, fraud counts, fraud rates, and financial volumes broken down by type."""
    data = get_aggregate_data(dataset)
    return {
        "dataset": dataset,
        "by_type": data.get("by_type", {})
    }


@router.get("/timeseries")
def get_analytics_timeseries(
    granularity: str = Query("hour", description="Granularity: 'hour', 'day', or 'step24'"),
    dataset: str = Query("synthetic", description="Dataset key: 'paysim' or 'synthetic'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Temporal transaction and fraud counts bucketed by hour-of-day, day, or 24-step windows."""
    data = get_aggregate_data(dataset)
    ts = data.get("timeseries", {})

    gran_clean = granularity.lower().strip()
    if gran_clean not in ts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity '{granularity}'. Must be one of: 'hour', 'day', 'step24'."
        )

    return {
        "dataset": dataset,
        "granularity": gran_clean,
        "series": ts[gran_clean]
    }


@router.get("/amount-distribution")
def get_analytics_amount_distribution(
    dataset: str = Query("synthetic", description="Dataset key: 'paysim' or 'synthetic'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Amount histogram frequency counts on log-spaced bins for legitimate vs fraudulent transactions."""
    data = get_aggregate_data(dataset)
    return {
        "dataset": dataset,
        "amount_distribution": data.get("amount_distribution", {})
    }


@router.get("/fraud-patterns")
def get_analytics_fraud_patterns(
    dataset: str = Query("synthetic", description="Dataset key: 'paysim' or 'synthetic'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fraud mechanism pattern shares: account draining and unchanged destination balance."""
    data = get_aggregate_data(dataset)
    return {
        "dataset": dataset,
        "fraud_patterns": data.get("fraud_patterns", {})
    }
