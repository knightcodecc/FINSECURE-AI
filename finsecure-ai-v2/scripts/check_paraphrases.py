"""
check_paraphrases.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.rag_chatbot import classify_intent_scored

catalog = {
    "count_all_scopes": [
        "how many transactions are there",
        "total number of transactions",
        "count of all rows",
        "how many total records",
        "what is the total transaction volume",
        "how many entries in total",
        "total transactions in system",
        "count of transactions across all scopes",
        "number of rows in the platform",
        "transactions total count",
    ],
    "flagged_count": [
        "how many transactions were flagged",
        "number of flagged frauds",
        "how many frauds caught",
        "how many alerts were raised",
        "total flagged count",
        "count of fraudulent transactions flagged",
        "how many were flagged as fraud",
        "how many suspicious transactions flagged",
        "flagged transactions count",
        "how many frauds detected",
    ],
    "fraud_rate": [
        "what is the fraud rate",
        "whats the fraud rate",
        "overall fraud rate",
        "fraud percentage in dataset",
        "rate of fraudulent transactions",
        "what percentage of transactions are fraud",
        "fraud prevalence",
        "proportion of fraud",
        "fraud rate benchmark",
        "dataset fraud rate",
    ],
    "active_threshold": [
        "what is the active threshold",
        "what's the threshold",
        "current decision threshold",
        "what threshold is currently used",
        "active model threshold",
        "model cutoff threshold",
        "what is the operating threshold",
        "current active threshold",
        "threshold used for decisions",
        "active probability threshold",
    ],
    "drift_status": [
        "is the model drifting",
        "is model drifting",
        "what is the drift status",
        "check concept drift",
        "current drift status",
        "has the model drifted",
        "is retraining recommended",
        "drift monitor status",
        "model stability status",
        "check drift status",
    ],
    "explain_transaction": [
        "explain transaction 2",
        "why was #2 flagged",
        "why was transaction #2 flagged as fraud",
        "explain transaction #1",
        "audit transaction 2",
        "breakdown transaction #5",
        "why was #1 flagged",
        "explain decision for transaction 2",
        "explain transaction 3",
        "why was transaction 2 caught",
    ],
    "cost_savings": [
        "how much money saved",
        "how much money does the model save",
        "cost savings vs nothing",
        "financial savings from the model",
        "how many rupees saved",
        "total money saved by the system",
        "savings compared to flag nothing",
        "prevented fraud losses savings",
        "financial cost savings",
        "how much rupees saved",
    ],
    "shap_explanation": [
        "what is shap",
        "explain shap",
        "what are shapley values",
        "how does shap work",
        "explain shapley additive explanations",
        "what is the shap framework",
        "explain the shap explainer",
        "what does shap measure",
        "shapley explanations",
        "what is shap value",
    ],
    "graph_features": [
        "what are the graph features",
        "explain graph features",
        "what are the 10 graph features",
        "explain pagerank and degree features",
        "tell me about the graph topological features",
        "what graph structural metrics are used",
        "expanding window graph features",
        "list graph features",
        "graph topological features",
        "what are graph features",
    ],
    "precision_recall_file": [
        "precision and recall of my upload",
        "what was the recall on my file",
        "what was the precision on my file",
        "precision and recall for my analysis",
        "what is the recall on my upload",
        "what is the precision on my upload",
        "precision and recall metrics for my file",
        "recall and precision of upload",
        "show precision and recall for my file",
        "file precision and recall",
    ]
}

mismatches = []
for expected_intent, phrasings in catalog.items():
    for phrase in phrasings:
        matched_id, score, _, _ = classify_intent_scored(phrase)
        if matched_id != expected_intent:
            mismatches.append((phrase, expected_intent, matched_id, score))

print(f"Total mismatches: {len(mismatches)}")
for m in mismatches:
    print(f"  - '{m[0]}': expected {m[1]}, got {m[2]} (score: {m[3]:.2f})")
