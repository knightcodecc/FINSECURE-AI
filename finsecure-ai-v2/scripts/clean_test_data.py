"""
clean_test_data.py — Interactive Cleanup Utility for Test Users & Prediction Records
====================================================================================
Lists test users (e.g., 'Auditor Jane Doe') and test transaction predictions created
during automated test runs, displays them to the operator, and requests explicit
confirmation before deletion. Never deletes data without explicit 'yes' confirmation.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.auth import get_users_db_path, get_excel_export_path, export_users_to_excel
from api.db import get_db_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_test_data():
    users_db = get_users_db_path()
    predictions_db = get_db_path()

    print("\n=======================================================")
    print("🧹 FINSECURE AI v2 — TEST DATA CLEANUP AUDIT")
    print("=======================================================\n")
    print(f"Users Database:       {users_db}")
    print(f"Predictions Database: {predictions_db}")

    # 1. Inspect test users
    matching_users = []
    if users_db.exists():
        conn = sqlite3.connect(users_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, role, created_at FROM users
            WHERE full_name LIKE '%Jane Doe%'
               OR email LIKE '%test%'
               OR email LIKE '%auditor%'
               OR email LIKE '%e2e%'
               OR email LIKE '%lead_auditor%'
            ORDER BY id ASC
            """
        )
        matching_users = cursor.fetchall()
        conn.close()

    print(f"\n[1] Found {len(matching_users)} test user account(s) to remove:")
    for u in matching_users:
        print(f"    - ID #{u[0]}: '{u[1]}' <{u[2]}> (Role: {u[3]})")

    # 2. Inspect test predictions
    matching_preds = []
    if predictions_db.exists():
        conn = sqlite3.connect(predictions_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, name_orig, name_dest, amount, is_fraud FROM predictions
            WHERE name_orig LIKE '%TEST%'
               OR name_orig LIKE '%BATCH%'
               OR name_orig LIKE '%AUDIT%'
               OR name_orig LIKE '%CLEAN%'
               OR name_orig LIKE '%LEGIT%'
               OR name_orig LIKE '%MULE%'
            ORDER BY id ASC
            """
        )
        matching_preds = cursor.fetchall()
        conn.close()

    print(f"\n[2] Found {len(matching_preds)} test prediction row(s) to remove:")
    if matching_preds:
        for p in matching_preds[:5]:
            print(f"    - ID #{p[0]}: {p[1]} | {p[2]} -> {p[3]} | Rs. {p[4]:,.2f} (Fraud: {bool(p[5])})")
        if len(matching_preds) > 5:
            print(f"      ... and {len(matching_preds) - 5} more test rows")

    if not matching_users and not matching_preds:
        print("\nNo test records found. Databases are already clean.")
        return

    # 3. Explicit confirmation prompt
    print("\n-------------------------------------------------------")
    print("⚠️ WARNING: Deletion is permanent.")
    try:
        confirmation = input("Type 'yes' to confirm deletion of the listed test records: ")
    except EOFError:
        print("\n[ABORTED] Non-interactive environment; cannot prompt for confirmation. No changes made.")
        return

    if confirmation.strip().lower() != "yes":
        print("\n[CANCELLED] Operation cancelled by user. No records were deleted.\n")
        return

    # 4. Perform deletion
    if matching_users and users_db.exists():
        user_ids = [u[0] for u in matching_users]
        conn = sqlite3.connect(users_db)
        with conn:
            conn.execute(f"DELETE FROM login_events WHERE user_id IN ({','.join(['?']*len(user_ids))})", user_ids)
            conn.execute(f"DELETE FROM users WHERE id IN ({','.join(['?']*len(user_ids))})", user_ids)
        conn.close()
        print(f"[SUCCESS] Deleted {len(user_ids)} test user(s) and associated login event(s).")
        try:
            export_users_to_excel()
            print(f"[SUCCESS] Re-synchronized Excel export to {get_excel_export_path()}.")
        except Exception as e:
            print(f"[WARNING] Could not update Excel export: {e}")

    if matching_preds and predictions_db.exists():
        pred_ids = [p[0] for p in matching_preds]
        conn = sqlite3.connect(predictions_db)
        with conn:
            conn.execute(f"DELETE FROM predictions WHERE id IN ({','.join(['?']*len(pred_ids))})", pred_ids)
        conn.close()
        print(f"[SUCCESS] Deleted {len(pred_ids)} test prediction row(s) from SQLite.")

    print("\n=======================================================")
    print("✅ TEST DATA CLEANUP COMPLETE")
    print("=======================================================\n")


if __name__ == "__main__":
    clean_test_data()
