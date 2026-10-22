"""
diagnose_old_router.py
Simulates the old substring-matching router logic on the 30 diagnostic queries.
"""
import re

queries = [
    "How many transactions are there?",
    "total transactions",
    "number of predictions",
    "count of rows",
    "how many were flagged",
    "how many frauds",
    "fraud rate",
    "what's the threshold",
    "is the model drifting",
    "show me the biggest transaction",
    "how much money saved",
    "what is PSI",
    "explain transaction 5",
    "why was #2 flagged",
    "which graph mode was used",
    "what is the overlap",
    "what threshold mode is active",
    "precision and recall of my upload",
    "how many rows in my file",
    "how many trasactions scord",
    "HOW MANY TRANSACTIONS ARE THERE?",
    "Count of Scored Transactions",
    "whats the fraud rate",
    "is model drifting?",
    "highest transaction amount",
    "why was transaction #1 flagged",
    "tell me about graph mode",
    "account overlap with training",
    "what is the active threshold",
    "how many rows in upload"
]

def match_old_router(query: str) -> str:
    q = query.lower().strip()
    
    # 1. Scored count
    if "how many transactions were scored" in q or "how many scored" in q:
        return "count_scored"
    # 2. Flagged fraud count
    if "how many were flagged as fraud" in q or "how many flagged" in q:
        return "count_flagged"
    # 3. Last decision
    if "explain the last decision" in q or "explain last decision" in q:
        return "explain_last"
    # 4. Active threshold
    if "what is the active threshold" in q or "active threshold" in q:
        return "active_threshold"
    # 5. Threshold comparison
    if "why is the cost-optimal threshold different" in q or "why thresholds differ" in q:
        return "threshold_comparison"
    # 6. Drift
    if "what is the drift status" in q or "drift status" in q:
        return "drift_status"
    # 7. PaySim frauds
    if "how many frauds are in the dataset" in q or "paysim frauds" in q:
        return "dataset_frauds"
    # 8. SHAP
    if "what is shap" in q:
        return "shap_explanation"
    # 9. PSI
    if "what is psi" in q:
        return "psi_definition"
    # 10. Cost savings
    if "how much money does the model save" in q or "cost savings" in q:
        return "cost_savings"
    # 11. Highest transaction
    if "highest transaction amount" in q:
        return "highest_transaction"
    # 12. Specific tx
    m_tx = re.search(r"transaction\s+#?(\d+)", q)
    if m_tx:
        return f"explain_tx_{m_tx.group(1)}"
    # 13. Upload specific
    if "what graph mode was used" in q or "graph mode of upload" in q:
        return "upload_graph_mode"
    if "account overlap with training" in q:
        return "upload_account_overlap"
    if "what threshold mode was applied" in q:
        return "upload_threshold_mode"
    if "precision of my upload" in q:
        return "upload_precision"
    if "recall of my upload" in q:
        return "upload_recall"
    if "f1 score of my upload" in q or "f1 of my upload" in q:
        return "upload_f1"
    if "how many rows in my file" in q:
        return "upload_row_count"

    return "NONE (FALLBACK)"

print("=== OLD ROUTER: 30 DIAGNOSTIC QUERIES ===")
print(f"{'#':<3} | {'QUERY':<38} | {'MATCHED INTENT':<26}")
print("-" * 72)
matched = 0
for idx, q in enumerate(queries, 1):
    intent = match_old_router(q)
    if intent != "NONE (FALLBACK)":
        matched += 1
    print(f"{idx:<3} | {q:<38} | {intent:<26}")
print("-" * 72)
print(f"Total Matched: {matched}/30 ({matched/30*100:.1f}%), Fallback: {len(queries)-matched}/30")
