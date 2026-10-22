"""
make_synthetic_paysim.py — Synthetic PaySim Data Generator (v3)
================================================================
Generates 300,000-row synthetic PaySim dataset (data/paysim_synth.csv) with ~1.8% fraud rate (5,400 fraud rows).

Mechanisms:
1. Mechanism A (Account-Drain Fraud, steps 1..629):
   65% full drain, 15% partial drain, 20% destination accounting discrepancy.
   Normal rows have at most 0.5% accidental drain.
2. Mechanism B (Ring/Mule Network Fraud, steps 1..629):
   Identical tabular amount/balance distribution to normal transactions.
   Uses persistent mule ring accounts (C_MULE_RING_*) and cashout hubs (M_HUB_CASHOUT_*) starting at step 1
   so expanding window calculates non-zero fraud_exposure > 0 for later steps in val and clean test.
3. Mechanism C (Late Tactic Shift, steps 630..743):
   Shifted type mix (DEBIT/CASH_OUT emphasis) and uniform amount distribution (200k..900k).
4. Asymmetric Label Flips:
   Flip 1.5% of fraud labels to 0, and 0.04% of legitimate labels to 1.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_PATH = Path("data/paysim_synth.csv")


def generate_synthetic_paysim(
    n_rows: int = 300_000,
    seed: int = 42,
    output_path: Path = OUTPUT_PATH
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    print(f"Generating {n_rows:,} synthetic PaySim records across steps 1..743 (seed={seed})...")

    # 1. Step generation (1 to 743 hours)
    steps = rng.randint(1, 744, size=n_rows)
    steps.sort()

    # 2. Base Transaction Types
    types = rng.choice(
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_rows,
        p=[0.35, 0.35, 0.20, 0.08, 0.02]
    )

    # 3. Base Amounts & Balances
    amounts = np.round(np.exp(rng.normal(loc=9.8, scale=1.4, size=n_rows)), 2)
    amounts = np.clip(amounts, 5.0, 1_500_000.0)

    old_balance_org = np.round(np.exp(rng.normal(loc=10.2, scale=1.6, size=n_rows)), 2)
    old_balance_dest = np.round(np.exp(rng.normal(loc=9.5, scale=1.8, size=n_rows)), 2)

    dest_zero_mask = rng.rand(n_rows) < 0.35
    old_balance_dest[dest_zero_mask] = 0.0

    new_balance_orig = np.maximum(0.0, old_balance_org - amounts)
    new_balance_dest = old_balance_dest + amounts

    # 4. Normal Account Pool & Legitimate High-Degree Merchant Hubs
    n_orig_users = 70_000
    n_dest_users = 35_000
    orig_users = [f"C{rng.randint(1000000, 9999999)}" for _ in range(n_orig_users)]
    dest_users = [f"M{rng.randint(1000000, 9999999)}" for _ in range(n_dest_users)]
    legit_hubs = [f"M_LEGIT_MERCHANT_{i}" for i in range(1, 41)]

    name_orig = rng.choice(orig_users, size=n_rows)
    name_dest = rng.choice(dest_users, size=n_rows)

    # Route 12% of normal destination transactions to legitimate high-degree merchant hubs
    legit_hub_mask = rng.rand(n_rows) < 0.12
    name_dest[legit_hub_mask] = rng.choice(legit_hubs, size=legit_hub_mask.sum())

    # Normal account drain noise (0.5% of normal transactions)
    normal_drain_mask = (rng.rand(n_rows) < 0.005) & (types != "CASH_IN")
    new_balance_orig[normal_drain_mask] = 0.0

    is_fraud = np.zeros(n_rows, dtype=int)
    is_flagged_fraud = np.zeros(n_rows, dtype=int)
    mech_labels = np.array(["NONE"] * n_rows, dtype=object)

    # -------------------------------------------------------------------------
    # PLANT FRAUD MECHANISMS (TARGET ~1.8% OVERALL FRAUD RATE)
    # -------------------------------------------------------------------------

    # Mechanism A: Account-Drain Fraud (~2,400 rows across steps < 630)
    drain_candidates = np.where(steps < 630)[0]
    drain_indices = rng.choice(drain_candidates, size=2400, replace=False)

    for idx in drain_indices:
        is_fraud[idx] = 1
        mech_labels[idx] = "MECH_A_DRAIN"
        types[idx] = "TRANSFER" if rng.rand() > 0.45 else "CASH_OUT"
        amt = round(float(np.exp(rng.normal(loc=10.5, scale=1.2))), 2)
        amt = min(amt, 1_200_000.0)
        amounts[idx] = amt
        old_balance_org[idx] = round(amt + rng.uniform(0.0, 5000.0), 2)

        r = rng.rand()
        if r < 0.65:
            new_balance_orig[idx] = 0.0  # Full drain
        elif r < 0.80:
            new_balance_orig[idx] = round(old_balance_org[idx] * rng.uniform(0.10, 0.30), 2)  # Partial drain
        else:
            new_balance_orig[idx] = max(0.0, round(old_balance_org[idx] - amt, 2))
            new_balance_dest[idx] = old_balance_dest[idx]  # Accounting error

    # Mechanism B: Ring/Mule Network Fraud (~2,100 rows across steps 1..629)
    # Uses persistent mule ring accounts so earlier steps populate fraud exposure for later steps
    ring_mule_accounts = [f"C_MULE_RING_{i}" for i in range(1, 31)]
    ring_hub_accounts = [f"M_HUB_CASHOUT_{j}" for j in range(1, 11)]

    ring_candidates = np.setdiff1d(np.where(steps < 630)[0], drain_indices)
    ring_indices = rng.choice(ring_candidates, size=2100, replace=False)
    # Sort ring_indices so earlier steps get assigned mule/hub accounts first
    ring_indices.sort()

    for idx in ring_indices:
        is_fraud[idx] = 1
        mech_labels[idx] = "MECH_B_RING"
        types[idx] = "TRANSFER" if rng.rand() > 0.35 else "CASH_OUT"

        # Tabular distribution identical to normal transactions
        amt = round(float(np.exp(rng.normal(loc=9.8, scale=1.4))), 2)
        amt = np.clip(amt, 10.0, 500_000.0)
        amounts[idx] = amt
        old_bal_o = round(float(np.exp(rng.normal(loc=10.2, scale=1.6))), 2)
        old_balance_org[idx] = old_bal_o
        new_balance_orig[idx] = max(0.0, round(old_bal_o - amt, 2))

        old_bal_d = round(float(np.exp(rng.normal(loc=9.5, scale=1.8))), 2)
        if rng.rand() < 0.35:
            old_bal_d = 0.0
        old_balance_dest[idx] = old_bal_d
        new_balance_dest[idx] = round(old_bal_d + amt, 2)  # Clean accounting

        # Connect to persistent mule & hub accounts
        name_orig[idx] = rng.choice(ring_mule_accounts)
        name_dest[idx] = rng.choice(ring_hub_accounts)

    # Mechanism C: Late Tactic Shift (~900 rows in steps 630..743) + Type Mix Shift in steps >= 630
    late_candidates = np.where(steps >= 630)[0]
    late_indices = rng.choice(late_candidates, size=900, replace=False)

    for idx in late_indices:
        is_fraud[idx] = 1
        mech_labels[idx] = "MECH_C_SHIFT"
        types[idx] = "DEBIT" if rng.rand() < 0.60 else "CASH_OUT"
        amt = round(float(rng.uniform(200_000.0, 900_000.0)), 2)
        amounts[idx] = amt
        old_balance_org[idx] = round(amt + rng.uniform(500.0, 10_000.0), 2)
        new_balance_orig[idx] = round(old_balance_org[idx] - amt, 2)
        old_balance_dest[idx] = 0.0
        new_balance_dest[idx] = 0.0

    # Type-mix shift in steps >= 630 for normal transactions (convert 25% PAYMENT to DEBIT)
    shift_normal_candidates = np.setdiff1d(late_candidates, late_indices)
    payment_mask = (types[shift_normal_candidates] == "PAYMENT")
    debit_flip = shift_normal_candidates[payment_mask][rng.rand(payment_mask.sum()) < 0.40]
    types[debit_flip] = "DEBIT"

    # 5. Asymmetric Label Noise
    # Flip 1.5% of true fraud labels to 0
    fraud_indices_all = np.where(is_fraud == 1)[0]
    fn_flip_indices = rng.choice(fraud_indices_all, size=int(len(fraud_indices_all) * 0.015), replace=False)
    is_fraud[fn_flip_indices] = 0

    # Flip 0.04% of true normal labels to 1
    normal_indices_all = np.where(is_fraud == 0)[0]
    fp_flip_indices = rng.choice(normal_indices_all, size=int(len(normal_indices_all) * 0.0004), replace=False)
    is_fraud[fp_flip_indices] = 1

    # Flagged fraud rule (>200k in TRANSFER)
    flagged_mask = (is_fraud == 1) & (types == "TRANSFER") & (amounts > 200_000.0)
    is_flagged_fraud[flagged_mask] = 1

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged_fraud,
        "mechanism": mech_labels,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} synthetic rows to {output_path}")
    print(f"Total fraud rows: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
    return df


if __name__ == "__main__":
    generate_synthetic_paysim()
