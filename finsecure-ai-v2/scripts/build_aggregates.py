"""
build_aggregates.py — Stream PaySim CSV & Precompute Memory-Light Small Aggregates.
===================================================================================
Streams PaySim CSV in 500k-row chunks with minimal memory footprint (usecols only).
Produces data/aggregates_<dataset>.json where <dataset> is "paysim", "synthetic", or "sample".

Includes:
- Total rows, fraud count, fraud rate, total fraud amount
- Transaction counts, fraud counts, fraud rates, and fraud volumes by type
- Temporal aggregates bucketed to:
  - hour_of_day (0..23)
  - day (1..31)
  - step24 (24-step windows)
- Amount distribution histogram on log-spaced bins for fraud vs legitimate
- Fraud pattern shares (account drained: newbalanceOrig=0 and oldbalanceOrg>0; dest unchanged)
- Summary statistics (mean, p95 approx, max) for fraud vs legit
- Metadata block (source file name, row count, build time, dataset name)

Fails loudly if any assertion check fails.
"""

import os
import sys
import time
import json
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# 16 log-spaced bins spanning from 10^0 (Rs. 1) to 10^8 (Rs. 100M)
LOG_BINS = np.logspace(0, 8, 17)


def build_aggregates_for_file(csv_path: Path, dataset_name: str, out_path: Path) -> Dict[str, Any]:
    print(f"\n[*] Processing dataset '{dataset_name}' from: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    tracemalloc.start()
    start_time = time.time()

    usecols = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ]

    total_rows = 0
    total_fraud = 0
    total_amount = 0.0
    total_fraud_amount = 0.0

    # By-type accumulators: type -> {count, fraud_count, amount, fraud_amount}
    by_type = {}
    valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    for t in valid_types:
        by_type[t] = {"count": 0, "fraud_count": 0, "amount": 0.0, "fraud_amount": 0.0}

    # Time series accumulators
    hour_counts = [0] * 24
    hour_frauds = [0] * 24
    day_counts = {}
    day_frauds = {}
    step24_counts = {}
    step24_frauds = {}

    # Amount histograms on log-spaced bins
    # Bins: [0, B0], [B0, B1], ..., [Bn-1, inf]
    num_bins = len(LOG_BINS) + 1
    hist_legit = [0] * num_bins
    hist_fraud = [0] * num_bins

    # Pattern counters
    legit_drained_count = 0
    fraud_drained_count = 0
    legit_dest_unchanged_count = 0
    fraud_dest_unchanged_count = 0

    # Numeric summaries
    max_amount_legit = 0.0
    max_amount_fraud = 0.0
    legit_amount_sum = 0.0
    fraud_amount_sum = 0.0
    legit_count = 0

    chunksize = 500000
    chunk_idx = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk_idx += 1
        n = len(chunk)
        total_rows += n

        frauds_mask = (chunk["isFraud"] == 1).values
        n_frauds = int(np.sum(frauds_mask))
        total_fraud += n_frauds
        legit_count += (n - n_frauds)

        amounts = chunk["amount"].values
        total_amount += float(np.sum(amounts))
        total_fraud_amount += float(np.sum(amounts[frauds_mask]))
        legit_amount_sum += float(np.sum(amounts[~frauds_mask]))
        fraud_amount_sum += float(np.sum(amounts[frauds_mask]))

        if n_frauds > 0:
            max_amount_fraud = max(max_amount_fraud, float(np.max(amounts[frauds_mask])))
        if (n - n_frauds) > 0:
            max_amount_legit = max(max_amount_legit, float(np.max(amounts[~frauds_mask])))

        # By-type accumulation
        for t, group in chunk.groupby("type"):
            if t not in by_type:
                by_type[t] = {"count": 0, "fraud_count": 0, "amount": 0.0, "fraud_amount": 0.0}
            t_amounts = group["amount"].values
            t_frauds = (group["isFraud"] == 1).values
            by_type[t]["count"] += len(group)
            by_type[t]["fraud_count"] += int(np.sum(t_frauds))
            by_type[t]["amount"] += float(np.sum(t_amounts))
            by_type[t]["fraud_amount"] += float(np.sum(t_amounts[t_frauds]))

        # Temporal accumulation
        steps = chunk["step"].values
        hours = steps % 24
        days = (steps - 1) // 24 + 1
        step24_keys = (steps - 1) // 24

        # Hour of day (0..23)
        for h in range(24):
            h_mask = (hours == h)
            hour_counts[h] += int(np.sum(h_mask))
            hour_frauds[h] += int(np.sum(h_mask & frauds_mask))

        # Day
        unique_days = np.unique(days)
        for d in unique_days:
            d_int = int(d)
            d_mask = (days == d)
            day_counts[d_int] = day_counts.get(d_int, 0) + int(np.sum(d_mask))
            day_frauds[d_int] = day_frauds.get(d_int, 0) + int(np.sum(d_mask & frauds_mask))

        # Step 24 window
        unique_step24 = np.unique(step24_keys)
        for s24 in unique_step24:
            s_int = int(s24)
            s_mask = (step24_keys == s24)
            step24_counts[s_int] = step24_counts.get(s_int, 0) + int(np.sum(s_mask))
            step24_frauds[s_int] = step24_frauds.get(s_int, 0) + int(np.sum(s_mask & frauds_mask))

        # Histogram accumulation
        legit_amounts = amounts[~frauds_mask]
        fraud_amounts = amounts[frauds_mask]

        bin_idx_legit = np.digitize(legit_amounts, LOG_BINS)
        bin_idx_fraud = np.digitize(fraud_amounts, LOG_BINS)

        counts_l = np.bincount(bin_idx_legit, minlength=num_bins)
        counts_f = np.bincount(bin_idx_fraud, minlength=num_bins)

        for b in range(num_bins):
            hist_legit[b] += int(counts_l[b])
            hist_fraud[b] += int(counts_f[b])

        # Fraud patterns
        old_orig = chunk["oldbalanceOrg"].values
        new_orig = chunk["newbalanceOrig"].values
        drained_mask = (new_orig == 0.0) & (old_orig > 0.0)
        legit_drained_count += int(np.sum(drained_mask & (~frauds_mask)))
        fraud_drained_count += int(np.sum(drained_mask & frauds_mask))

        old_dest = chunk["oldbalanceDest"].values
        new_dest = chunk["newbalanceDest"].values
        dest_unchanged_mask = (new_dest == old_dest)
        legit_dest_unchanged_count += int(np.sum(dest_unchanged_mask & (~frauds_mask)))
        fraud_dest_unchanged_count += int(np.sum(dest_unchanged_mask & frauds_mask))

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.time() - start_time

    # Calculate rates and format by-type dict
    for t, data in by_type.items():
        data["fraud_rate"] = (data["fraud_count"] / data["count"]) if data["count"] > 0 else 0.0

    # Build histogram bin labels
    bin_labels = []
    for i in range(num_bins):
        if i == 0:
            bin_labels.append(f"< Rs. {int(LOG_BINS[0]):,}")
        elif i == num_bins - 1:
            bin_labels.append(f">= Rs. {int(LOG_BINS[-1]):,}")
        else:
            bin_labels.append(f"Rs. {int(LOG_BINS[i-1]):,} - {int(LOG_BINS[i]):,}")

    # Build timeseries sorted structures
    ts_hour = [{"hour": h, "count": hour_counts[h], "fraud_count": hour_frauds[h]} for h in range(24)]
    ts_day = [{"day": d, "count": day_counts[d], "fraud_count": day_frauds[d]} for d in sorted(day_counts.keys())]
    ts_step24 = [{"window": w, "count": step24_counts[w], "fraud_count": step24_frauds[w]} for w in sorted(step24_counts.keys())]

    # Pattern percentages
    legit_drained_pct = (legit_drained_count / legit_count) if legit_count > 0 else 0.0
    fraud_drained_pct = (fraud_drained_count / total_fraud) if total_fraud > 0 else 0.0
    legit_dest_unchanged_pct = (legit_dest_unchanged_count / legit_count) if legit_count > 0 else 0.0
    fraud_dest_unchanged_pct = (fraud_dest_unchanged_count / total_fraud) if total_fraud > 0 else 0.0

    # Summary stats
    mean_amount_legit = (legit_amount_sum / legit_count) if legit_count > 0 else 0.0
    mean_amount_fraud = (fraud_amount_sum / total_fraud) if total_fraud > 0 else 0.0

    result = {
        "metadata": {
            "source_file": csv_path.name,
            "dataset_name": dataset_name,
            "row_count": total_rows,
            "fraud_count": total_fraud,
            "build_time_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(duration, 2),
            "peak_ram_mb": round(peak_mem / (1024 * 1024), 2)
        },
        "summary": {
            "total_transactions": total_rows,
            "total_fraud": total_fraud,
            "total_legitimate": legit_count,
            "overall_fraud_rate": (total_fraud / total_rows) if total_rows > 0 else 0.0,
            "total_volume_rs": total_amount,
            "total_fraud_volume_rs": total_fraud_amount,
            "mean_amount_legit": mean_amount_legit,
            "mean_amount_fraud": mean_amount_fraud,
            "max_amount_legit": max_amount_legit,
            "max_amount_fraud": max_amount_fraud
        },
        "by_type": by_type,
        "timeseries": {
            "hour": ts_hour,
            "day": ts_day,
            "step24": ts_step24
        },
        "amount_distribution": {
            "bin_labels": bin_labels,
            "hist_legit": hist_legit,
            "hist_fraud": hist_fraud
        },
        "fraud_patterns": {
            "drained_origin": {
                "fraud_count": fraud_drained_count,
                "fraud_pct": fraud_drained_pct,
                "legit_count": legit_drained_count,
                "legit_pct": legit_drained_pct
            },
            "dest_unchanged": {
                "fraud_count": fraud_dest_unchanged_count,
                "fraud_pct": fraud_dest_unchanged_pct,
                "legit_count": legit_dest_unchanged_count,
                "legit_pct": legit_dest_unchanged_pct
            }
        }
    }

    # =========================================================================
    # BUILT-IN LOUD CHECKS (FAIL IMMEDIATELY IF INCONSISTENT)
    # =========================================================================
    sum_type_counts = sum(v["count"] for v in by_type.values())
    assert sum_type_counts == total_rows, f"Check Failed: sum of type counts ({sum_type_counts}) != total rows ({total_rows})"

    sum_type_frauds = sum(v["fraud_count"] for v in by_type.values())
    assert sum_type_frauds == total_fraud, f"Check Failed: sum of fraud by type ({sum_type_frauds}) != total frauds ({total_fraud})"

    sum_hist_total = sum(hist_legit) + sum(hist_fraud)
    assert sum_hist_total == total_rows, f"Check Failed: histogram total ({sum_hist_total}) != total rows ({total_rows})"

    # PaySim canonical audit check:
    if "paysim.csv" in csv_path.name.lower() and "sample" not in csv_path.name.lower() and "synth" not in csv_path.name.lower():
        assert total_rows == 6362620, f"Check Failed: PaySim total rows ({total_rows}) != 6,362,620"
        assert total_fraud == 8213, f"Check Failed: PaySim total fraud ({total_fraud}) != 8,213"
        print(f"[CANONICAL PAYSIM AUDIT CONFIRMED]: Rows = 6,362,620 | Frauds = 8,213")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[+] Aggregates successfully saved to: {out_path}")
    print(f"    Total Rows:       {total_rows:,}")
    print(f"    Fraud Count:      {total_fraud:,} ({result['summary']['overall_fraud_rate']:.4%})")
    print(f"    Runtime:          {duration:.2f} seconds")
    print(f"    Peak RAM:         {peak_mem / (1024 * 1024):.2f} MB")
    print(f"    All Assertion Checks: PASSED.")

    return result


def main():
    print("=" * 70)
    print(" FinSecure AI v2 — PaySim Chunked Aggregate Builder")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data"

    # 1. Sample (1,000 rows)
    sample_csv = data_dir / "paysim_sample.csv"
    if sample_csv.exists():
        build_aggregates_for_file(sample_csv, "sample", data_dir / "aggregates_sample.json")

    # 2. Synthetic (300,000 rows)
    synth_csv = data_dir / "paysim_synth.csv"
    if synth_csv.exists():
        build_aggregates_for_file(synth_csv, "synthetic", data_dir / "aggregates_synthetic.json")

    # 3. Canonical PaySim Full (6,362,620 rows) if present on machine
    paysim_csv = data_dir / "paysim.csv"
    if paysim_csv.exists():
        build_aggregates_for_file(paysim_csv, "paysim", data_dir / "aggregates_paysim.json")
    else:
        print("\n[!] Canonical paysim.csv not present locally. Aggregates for synthetic & sample built.")


if __name__ == "__main__":
    main()
