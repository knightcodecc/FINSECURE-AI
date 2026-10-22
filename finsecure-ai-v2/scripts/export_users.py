"""
export_users.py — Standalone CLI utility to export SQLite users to Excel (.xlsx).
================================================================================
Reads data/finsecure_users.db and outputs data/users_export.xlsx (password_hash excluded).
Verifies the generated Excel workbook using pandas and displays a formatted summary.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from api.auth import init_users_db, export_users_to_excel, USERS_DB_PATH, EXCEL_EXPORT_PATH


def main():
    print("=" * 70)
    print(" FinSecure AI v2 — User Database Excel Exporter")
    print("=" * 70)

    init_users_db()
    print(f"[*] Reading users and login events from: {USERS_DB_PATH}")
    export_users_to_excel(USERS_DB_PATH, EXCEL_EXPORT_PATH)
    print(f"[+] Successfully exported to: {EXCEL_EXPORT_PATH}")
    print("-" * 70)

    # Verification: Read back sheets using pandas
    try:
        users_df = pd.read_excel(EXCEL_EXPORT_PATH, sheet_name="Users")
        events_df = pd.read_excel(EXCEL_EXPORT_PATH, sheet_name="Login Events")

        print("[AUDIT VERIFICATION]: Successfully verified sheets with pandas:")
        print(f"  - Sheet 'Users': {len(users_df)} rows")
        print(f"    Columns: {list(users_df.columns)}")
        assert "password_hash" not in users_df.columns, "CRITICAL ERROR: password_hash found in export!"
        print("    [PASS] password_hash strictly excluded from export.")

        print(f"\n  - Users Table Preview:")
        print(users_df.to_string(index=False))

        print(f"\n  - Sheet 'Login Events': {len(events_df)} rows")
        if len(events_df) > 0:
            print(events_df.head(5).to_string(index=False))
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR]: Failed to verify exported Excel file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
