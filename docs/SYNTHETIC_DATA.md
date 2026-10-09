# Synthetic PaySim Dataset Specification & Generator Guide

## Overview & Purpose

To enable full end-to-end local validation of FinSecure AI v2 without memory bottlenecks on laptops, `scripts/make_synthetic_paysim.py` generates a **300,000-row synthetic PaySim dataset** (`data/paysim_synth.csv`).

> **Important Distinction**:
> Results generated on `data/paysim_synth.csv` serve strictly for local pipeline verification, unit test validation, and UI/serving testing. Headline performance metrics reported in final thesis deliverables are produced by running the exact same pipeline on the full 6.36M-row PaySim dataset via Google Colab (`scripts/colab_run.md`).

---

## 🛠️ Dataset Schema & Properties

- **Total Rows**: `300,000`
- **Time Window**: Steps `1` to `743` (30 days of hourly transactions)
- **Total Fraud Instances**: `~5,400` (`~1.8%` fraud rate)
- **Column Schema**: Matched 100% to PaySim1 (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`).

---

## 🎯 Planted Fraud Mechanisms

The generator explicitly injects three distinct fraud mechanisms to test model capabilities:

### 1. Mechanism A: Account-Drain Fraud (~2,200 instances)
- **Pattern**: Origin account is drained to zero (`oldbalanceOrg > 0`, `newbalanceOrig = 0`), but destination balance remains unchanged.
- **Detectable via**: Tabular balance accounting equations (`errorBalanceOrig`, `errorBalanceDest`).

### 2. Mechanism B: Mule / Ring Network Fraud (~2,000 instances)
- **Pattern**: Transactions exhibit completely normal amounts and valid accounting balance updates (`oldbalanceDest + amount == newbalanceDest`). However, transactions are routed through shared mule accounts (`C_MULE_RING_*`) sending to central cash-out hub accounts (`M_HUB_CASHOUT_*`).
- **Detectable via**: Relational graph features (`orig_fraud_exposure`, `dest_fraud_exposure`, `orig_pagerank`, `component_size`). Single-transaction tabular features miss this pattern entirely.

### 3. Mechanism C: Late-Period Tactic Shift (~1,200 instances in steps 630..743)
- **Pattern**: Shift in transaction types (DEBIT & CASH_OUT), new amount distribution, and late-night execution hours during the final 15% of steps.
- **Detectable via**: Concept drift monitoring (`src/drift_monitor.py`).

---

## 🚀 Execution & Environment Configuration

Generate the dataset locally:
```bash
python scripts/make_synthetic_paysim.py
```

Set environment variable `DATA_PATH` to target the synthetic dataset:
```bash
export DATA_PATH=data/paysim_synth.csv   # Linux/macOS
$env:DATA_PATH="data/paysim_synth.csv"  # PowerShell
```

---

## 📌 Note on Graph Features (`orig_in_degree` and `dest_out_degree`)

In `data/paysim_synth.csv`, origin accounts (`C*`) act strictly as transaction senders and destination accounts (`M*`) act strictly as transaction receivers. Consequently, `orig_in_degree` and `dest_out_degree` evaluate to constant `0` across all windows in synthetic runs.

On real PaySim data (6.36M rows) and production banking networks, customer accounts participate in bidirectional transfers (peer-to-peer transfers, refunds, merchant payouts), so these degree columns will vary. They are retained as 33-column model features for full compatibility with production datasets.

