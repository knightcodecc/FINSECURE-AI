"""
prepare_samples.py
Creates frontend/samples/sample_heldout.csv from held-out window (steps 541..629),
preserves the old sample as frontend/samples/sample_training_rows.csv,
and builds frontend/samples/sample_unlabeled.csv.
"""
import shutil
from pathlib import Path
import pandas as pd

def main():
    old_sample = Path("frontend/samples/sample_transactions.csv")
    renamed_sample = Path("frontend/samples/sample_training_rows.csv")
    if old_sample.exists():
        shutil.copy(old_sample, renamed_sample)
        print(f"Preserved existing training-window sample as: {renamed_sample}")

    df_all = pd.read_csv("data/paysim_synth.csv")
    heldout = df_all[(df_all["step"] >= 541) & (df_all["step"] <= 629)].copy()

    # Sample 50 frauds (25 MECH_A_DRAIN, 25 MECH_B_RING) and 1,950 non-frauds
    frauds_a = heldout[heldout["mechanism"] == "MECH_A_DRAIN"].sample(n=25, random_state=42)
    frauds_b = heldout[heldout["mechanism"] == "MECH_B_RING"].sample(n=25, random_state=42)
    non_frauds = heldout[heldout["isFraud"] == 0].sample(n=1950, random_state=42)

    sample_heldout = pd.concat([frauds_a, frauds_b, non_frauds]).sort_values("step", kind="stable").reset_index(drop=True)

    cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    sample_heldout = sample_heldout[cols]

    heldout_path = Path("frontend/samples/sample_heldout.csv")
    sample_heldout.to_csv(heldout_path, index=False)
    print(f"Created {heldout_path}: shape {sample_heldout.shape}, frauds {sample_heldout['isFraud'].sum()}")
    print(f"Heldout steps: {sample_heldout['step'].min()}..{sample_heldout['step'].max()}")
    print(f"Heldout type mix: {sample_heldout['type'].value_counts().to_dict()}")

    # Also keep sample_transactions.csv as copy of heldout
    shutil.copy(heldout_path, old_sample)

    # Create unlabeled sample from sample_heldout
    unlabeled_cols = [c for c in cols if c not in ["isFraud", "isFlaggedFraud"]]
    sample_unlabeled = sample_heldout[unlabeled_cols].copy()
    unlabeled_path = Path("frontend/samples/sample_unlabeled.csv")
    sample_unlabeled.to_csv(unlabeled_path, index=False)
    print(f"Created {unlabeled_path}: shape {sample_unlabeled.shape}")

if __name__ == "__main__":
    main()
