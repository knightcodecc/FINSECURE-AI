# FinSecure AI v2 — Sample Datasets Documentation (Demo Samples)

## Overview
This document records the exact provenance, sampling methodology, random seed, and statistical properties of demo samples included under `frontend/samples/`.

---

## 1. Demo Sample: `sample_heldout.csv`
- **Purpose**: Demonstration test dataset for the "Analyze CSV" batch upload feature, simulating held-out operational traffic.
- **Label**: **Demo Sample** (Clean Test Window: steps 541..629).
- **Source Dataset**: `data/paysim_synth.csv` (temporal split where `step > 420` represents the post-training operational window).
- **Sampling Methodology**:
  - Filter: all synthetic transactions in the held-out window `541 <= step <= 629` (35,954 total rows).
  - Seeded random sampling (`seed=42`) without replacement:
    - 25 confirmed frauds from Mechanism A (`MECH_A_DRAIN`).
    - 25 confirmed frauds from Mechanism B (`MECH_B_RING`).
    - 1,950 legit (non-fraud) transactions (`isFraud == 0`), sampled across all 5 transaction types (not first N per type).
  - Total rows: **2,000**.
  - Total confirmed frauds: **50**.
  - Fraud Prevalence: **2.50%** (50 / 2,000).
  - Step range: Steps **541 to 629** (spanning 89 chronological hours).
  - Transaction Type Mix:
    - `CASH_OUT`: 701 (35.05%)
    - `PAYMENT`: 666 (33.30%)
    - `CASH_IN`: 396 (19.80%)
    - `TRANSFER`: 197 (9.85%)
    - `DEBIT`: 40 (2.00%)
- **Small Sample Safeguard (< 100 Frauds)**:
  - Because `sample_heldout.csv` contains only 50 confirmed frauds (< 100 frauds), raw point estimates of PR-AUC and ROC-AUC are statistically noisy.
  - The evaluation pipeline emits a **Small Sample Warning**:
    > *"Metrics (PR-AUC / ROC-AUC) unreliable with small fraud sample (50 < 100 confirmed frauds). Bootstrap 95% confidence intervals are displayed below."*
  - Empirical 95% bootstrap confidence intervals (500 iterations, seed 42) are computed and reported alongside point metrics for precision, recall, F1, PR-AUC, and ROC-AUC.

---

## 2. Demo Sample: `sample_training_rows.csv`
- **Purpose**: Demonstration dataset drawn directly from the historical training window to verify behavior on accounts with high graph connectivity.
- **Label**: **Demo Sample** (Training Window: steps 1..420).
- **Source Dataset**: `data/paysim_synth.csv` (`step <= 420`).
- **Total Rows**: 2,000.
- **Confirmed Frauds**: 50.
- **Account Overlap with Training Graph**: **94.60%**.

---

## 3. Demo Sample: `sample_unlabeled.csv`
- **Purpose**: Real-world operational file without ground truth `isFraud` or `isFlaggedFraud` columns.
- **Label**: **Demo Sample** (Unlabeled).
- **Total Rows**: 2,000.
- **Columns**: 9 canonical attributes (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`).
