"""
xai_engine.py — Explainability layer.
=======================================
Objective (from literature gap): regulatory pressure is driving adoption of
genuinely interpretable models in finance — a paraphrase-only LLM explanation
doesn't satisfy that bar.

v2 approach:
    1. Compute SHAP values for the scored transaction (TreeExplainer, fast for XGBoost).
    2. Feed the TOP-5 SHAP-ranked features (name + value + contribution direction)
       into the LLM prompt as grounding — the LLM's job is to narrate the SHAP output
       in plain English, not to invent a reason.
    3. Keep Gemini primary -> OpenAI fallback -> deterministic rule-based fallback for offline/failures.
"""

from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain.shap_explainer import (
    ShapExplainer,
    build_grounded_prompt,
    validate_faithfulness,
    deterministic_template_explanation,
    generate_llm_explanation,
    FEATURE_HUMAN_NAMES,
)

# Singleton explainer instance for FastAPI
_EXPLAINER: ShapExplainer | None = None


def get_explainer() -> ShapExplainer:
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = ShapExplainer()
    return _EXPLAINER


def shap_explain(
    transaction_df,
    top_k: int = 5,
    raw_tx_dict: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Compute SHAP values for a given transaction DataFrame row.
    Returns (top_features, base_value, probability).
    """
    explainer = get_explainer()
    return explainer.explain_transaction(transaction_df, top_k=top_k, raw_tx_dict=raw_tx_dict)


def generate_explanation(
    top_features: List[Dict[str, Any]],
    probability: float,
    threshold: float
) -> str:
    """
    Generate grounded natural language explanation using Gemini -> OpenAI -> Deterministic template.
    """
    return generate_llm_explanation(top_features, probability, threshold)
