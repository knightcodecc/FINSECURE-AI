"""
test_intent_matcher.py
======================
Prototype and verify the normalized, keyword/synonym scored intent router.
"""
import re
import difflib
from typing import Dict, List, Any, Tuple, Optional

DOMAIN_VOCAB = [
    "transactions", "transaction", "predictions", "prediction", "scored",
    "drifting", "drift", "threshold", "thresholds", "frauds", "fraud",
    "flagged", "overlap", "precision", "recall", "explain", "features",
    "dataset", "savings", "saved", "biggest", "highest", "amount"
]

SYNONYM_MAP = {
    "txns": "transactions",
    "txn": "transaction",
    "records": "transactions",
    "record": "transaction",
    "rows": "transactions",
    "row": "transaction",
    "predictions": "transactions",
    "prediction": "transaction",
    "items": "transactions",
    "entries": "transactions",
    
    "how many": "count",
    "how much": "count",
    "number of": "count",
    "total": "count",
    "there are": "count",
    "volume of": "count",
    "count of": "count",
    
    "frauds": "fraud",
    "fraudulent": "fraud",
    "suspicious": "flagged",
    "alerts": "flagged",
    "alerted": "flagged",
    "caught": "flagged",
    
    "drifting": "drift",
    "covariate shift": "drift",
    "stability": "drift",
    
    "thresholds": "threshold",
    "cutoff": "threshold",
    "cut-off": "threshold",
    
    "savings": "saved",
    "save": "saved",
    "money saved": "saved",
    
    "biggest": "highest",
    "largest": "highest",
    "maximum": "highest",
    "max": "highest",
    
    "my file": "upload",
    "my upload": "upload",
    "in my file": "upload",
    "on my file": "upload",
    "uploaded file": "upload",
    "last upload": "upload",
    "my analysis": "upload",
}

def normalize_text(text: str) -> str:
    t = text.lower().strip()
    # Replace punctuation with spaces
    t = re.sub(r"[?!,;:#*\"\'\(\)\[\]\{\}\<\>/\\_`~]", " ", t)
    words = t.split()
    corrected = []
    for w in words:
        if w in DOMAIN_VOCAB:
            corrected.append(w)
        else:
            matches = difflib.get_close_matches(w, DOMAIN_VOCAB, n=1, cutoff=0.75)
            if matches:
                corrected.append(matches[0])
            else:
                corrected.append(w)
    normalized = " ".join(corrected)
    # Apply synonym map
    for k, v in SYNONYM_MAP.items():
        normalized = re.sub(rf"\b{k}\b", v, normalized)
    return normalized


INTENT_DEFS = [
    {
        "id": "count_all_scopes",
        "canonical": "How many transactions are there?",
        "keywords": ["count", "transaction"],
        "regex": r"\b(how many transactions|total transactions|count of (rows|transactions)|number of (predictions|transactions)|transactions (there are|in total))\b",
        "scope_priority": 1.0,
    },
    {
        "id": "count_scored",
        "canonical": "How many transactions were scored?",
        "keywords": ["count", "transaction", "scored"],
        "regex": r"\b(scored|logged)\b.*(transaction|prediction)|(transaction|prediction).*\b(scored|logged)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "count_dataset",
        "canonical": "How many transactions are in the dataset?",
        "keywords": ["count", "transaction", "dataset", "paysim"],
        "regex": r"\b(in the dataset|dataset count|paysim count)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "count_file",
        "canonical": "How many rows were in my uploaded file?",
        "keywords": ["count", "transaction", "upload"],
        "regex": r"\b(rows in (my )?(file|upload)|upload(ed)? (rows|count))\b",
        "scope_priority": 1.3,
    },
    {
        "id": "flagged_count",
        "canonical": "How many transactions were flagged as fraud?",
        "keywords": ["count", "flagged"],
        "alt_keywords": ["count", "fraud"],
        "regex": r"\b(how many (were )?(flagged|fraud|frauds)|how many frauds|flagged count|frauds flagged)\b",
        "scope_priority": 1.1,
    },
    {
        "id": "fraud_rate",
        "canonical": "What is the fraud rate?",
        "keywords": ["fraud", "rate"],
        "regex": r"\b(fraud rate|fraud percentage|rate of fraud|whats the fraud rate)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "active_threshold",
        "canonical": "What is the active threshold?",
        "keywords": ["threshold", "active"],
        "regex": r"\b(active threshold|what('s| is) the threshold|current threshold)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "cost_vs_f2_threshold",
        "canonical": "Why is the cost-optimal threshold different from the F2-optimal one?",
        "keywords": ["threshold", "cost", "f2"],
        "regex": r"\b(differ|difference|why).*(cost|f2)|(cost).*(f2)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "threshold_mode_info",
        "canonical": "What threshold mode is active?",
        "keywords": ["threshold", "mode"],
        "regex": r"\b(threshold mode|alert budget|budget mode)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "drift_status",
        "canonical": "What is the concept drift status?",
        "keywords": ["drift"],
        "regex": r"\b(drift status|is (the )?model drifting|model drift|retrain recommended)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "psi_explanation",
        "canonical": "What is PSI (Population Stability Index)?",
        "keywords": ["psi"],
        "regex": r"\b(what is psi|explain psi|population stability index)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "explain_transaction",
        "canonical": "Why was transaction #1 flagged?",
        "keywords": ["explain", "flagged"],
        "regex": r"\b(explain|why|audit|breakdown)\b.*(#?\d+|transaction|txn)|(#\d+)\b",
        "scope_priority": 1.4,
    },
    {
        "id": "biggest_transaction",
        "canonical": "Show me the biggest transaction flagged",
        "keywords": ["highest", "transaction"],
        "regex": r"\b(biggest|highest|largest|max)\b.*(transaction|amount)",
        "scope_priority": 1.2,
    },
    {
        "id": "cost_savings",
        "canonical": "How much money does the model save?",
        "keywords": ["saved", "money"],
        "regex": r"\b(how much money saved|cost savings|money save|rupees save)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "expected_cost",
        "canonical": "What is the expected financial cost?",
        "keywords": ["expected", "cost"],
        "regex": r"\b(expected cost|financial cost)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "shap_explanation",
        "canonical": "What is SHAP?",
        "keywords": ["shap"],
        "regex": r"\b(what is shap|explain shap|shapley)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "graph_features",
        "canonical": "What are the 10 graph features?",
        "keywords": ["graph", "features"],
        "regex": r"\b(graph features|pagerank|in degree|out degree|component size)\b",
        "scope_priority": 1.2,
    },
    {
        "id": "graph_mode_info",
        "canonical": "Which graph mode was used?",
        "keywords": ["graph", "mode"],
        "regex": r"\b(which graph mode|graph mode|warmed vs cold|tell me about graph mode)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "account_overlap_info",
        "canonical": "What is the account overlap with training?",
        "keywords": ["overlap"],
        "regex": r"\b(overlap|seen in training|account overlap)\b",
        "scope_priority": 1.3,
    },
    {
        "id": "precision_recall_file",
        "canonical": "What is the precision and recall of my upload?",
        "keywords": ["precision", "recall"],
        "regex": r"\b(precision and recall|recall on my|precision on my|upload metrics)\b",
        "scope_priority": 1.3,
    },
]

def match_query(raw_q: str) -> Tuple[Optional[str], float, List[str]]:
    norm_q = normalize_text(raw_q)
    raw_lower = raw_q.lower()
    
    scores = []
    
    # Check for transaction ID entity
    tx_match = re.search(r"#\s*(\d+)|\b(?:transaction|txn|id)\s*#?\s*(\d+)\b", raw_lower)
    has_tx_id = tx_match is not None
    
    for idef in INTENT_DEFS:
        score = 0.0
        
        # 1. Regex check on raw or normalized
        if "regex" in idef and (re.search(idef["regex"], raw_lower) or re.search(idef["regex"], norm_q)):
            score += 0.55
            
        # 2. Keywords check on normalized
        kw_hits = sum(1 for kw in idef.get("keywords", []) if kw in norm_q)
        max_kws = len(idef.get("keywords", []))
        if "alt_keywords" in idef:
            alt_hits = sum(1 for kw in idef["alt_keywords"] if kw in norm_q)
            if alt_hits > kw_hits:
                kw_hits = alt_hits
                max_kws = len(idef["alt_keywords"])
        if max_kws > 0:
            score += 0.35 * (kw_hits / max_kws)
            
        # 3. Fuzzy similarity to canonical
        seq_sim = difflib.SequenceMatcher(None, raw_lower, idef["canonical"].lower()).ratio()
        score += 0.25 * seq_sim
        
        # Entity boost
        if idef["id"] == "explain_transaction" and has_tx_id:
            score += 0.5
            
        # Apply scope priority
        score *= idef.get("scope_priority", 1.0)
        scores.append((idef["id"], score, idef["canonical"]))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score, _ = scores[0]
    
    top_suggestions = [s[2] for s in scores[:3]]
    if best_score >= 0.45:
        return best_id, best_score, top_suggestions
    return None, best_score, top_suggestions


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

print("=== VERIFYING INTENT MATCHER ON 30 TEST QUERIES ===")
for i, q in enumerate(queries, 1):
    matched_id, score, suggs = match_query(q)
    print(f"{i:2d}. '{q}' -> {matched_id} (score: {score:.2f})")
    assert matched_id is not None, f"Query '{q}' was not matched!"

print("\n[SUCCESS] All 30 queries matched successfully!")
