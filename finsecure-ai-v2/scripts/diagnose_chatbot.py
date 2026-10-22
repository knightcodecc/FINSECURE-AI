"""
diagnose_chatbot.py
===================
Run 30 queries through the current router and print query -> matched intent or NONE / FALLBACK.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.rag_chatbot import answer_analyst_query

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

print("=== DIAGNOSTIC: 30 QUERIES ON CURRENT ROUTER ===")
print(f"{'#':<3} | {'QUERY':<38} | {'STATUS':<8} | {'FIRST 55 CHARS OF RESPONSE':<55}")
print("-" * 110)

matched_cnt = 0
fallback_cnt = 0

for idx, q in enumerate(queries, 1):
    res = answer_analyst_query(q)
    ans = str(res)
    is_fallback = "I couldn't find a direct match" in ans or "No concept drift snapshots" in ans
    status = "FALLBACK" if "I couldn't find a direct match" in ans else "MATCHED"
    if status == "MATCHED":
        matched_cnt += 1
    else:
        fallback_cnt += 1
    snippet = ans.replace("\n", " ")[:53]
    print(f"{idx:<3} | {q:<38} | {status:<8} | {snippet:<55}")

print("-" * 110)
print(f"Summary: {matched_cnt} MATCHED, {fallback_cnt} FALLBACK out of {len(queries)} queries.")
