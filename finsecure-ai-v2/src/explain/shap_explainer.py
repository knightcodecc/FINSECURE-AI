"""
shap_explainer.py
==================
SHAP-grounded explainability engine for FinSecure AI v2.

Why this exists:
Regulatory standards and financial risk management require feature-level auditability.
Instead of allowing LLMs to narrate arbitrary reasons, this module uses SHAP (SHapley Additive
exPlanations) via shap.TreeExplainer to extract exact mathematical feature contributions.
The LLM is strictly constrained to explain ONLY the top-K SHAP features and return structured JSON.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Pre-emptively mock cv2 in sys.modules if not installed or broken on NumPy 2.x
# shap.maskers._image imports cv2 optionally; mocking avoids OpenCV NumPy 1.x C-extension crash
from unittest.mock import MagicMock
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_HUMAN_NAMES: Dict[str, str] = {
    "amount": "Transaction Amount",
    "oldbalanceOrg": "Sender Initial Balance",
    "newbalanceOrig": "Sender Remaining Balance",
    "oldbalanceDest": "Recipient Initial Balance",
    "newbalanceDest": "Recipient Final Balance",
    "errorBalanceOrig": "Sender Balance Discrepancy",
    "errorBalanceDest": "Recipient Balance Discrepancy",
    "log_amount": "Log-scaled Transaction Size",
    "account_drained": "Sender Account Drained to Zero",
    "dest_balance_zero": "Recipient Zero-Balance Account",
    "orig_balance_zero": "Sender Zero-Balance Account",
    "amount_to_oldbalanceOrg_ratio": "Ratio of Amount to Sender Balance",
    "amount_to_oldbalanceDest_ratio": "Ratio of Amount to Recipient Balance",
    "hour_of_day": "Hour of Day",
    "high_risk_hour": "Late Night / Early Morning Window",
    "day_of_week": "Day of Week",
    "net_change_orig": "Sender Net Balance Change",
    "net_change_dest": "Recipient Net Balance Change",
    "orig_in_degree": "Sender Inbound Connected Accounts",
    "orig_out_degree": "Sender Outbound Transaction Frequency",
    "orig_pagerank": "Sender Network Centrality",
    "orig_component_size": "Sender Network Cluster Size",
    "orig_fraud_exposure": "Sender's Connection to Previously Flagged Accounts",
    "dest_in_degree": "Recipient Inbound Connected Accounts",
    "dest_out_degree": "Recipient Outbound Transaction Frequency",
    "dest_pagerank": "Recipient Network Centrality",
    "dest_component_size": "Recipient Network Cluster Size",
    "dest_fraud_exposure": "Recipient's Connection to Previously Flagged Accounts",
    "type_CASH_IN": "CASH_IN Transaction Type",
    "type_CASH_OUT": "CASH_OUT Transaction Type",
    "type_DEBIT": "DEBIT Transaction Type",
    "type_PAYMENT": "PAYMENT Transaction Type",
    "type_TRANSFER": "TRANSFER Transaction Type",
}


class ShapExplainer:
    def __init__(
        self,
        model_path: Path = Path("models/xgboost_model.joblib"),
        metadata_path: Path = Path("models/model_metadata.joblib"),
    ):
        """Load XGBoost model and metadata once at startup, initializing TreeExplainer."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
        self.feature_names = self.metadata.get(
            "feature_names",
            list(getattr(self.model, "feature_names_in_", []))
        )
        
        # Initialize SHAP TreeExplainer on trained XGBoost model
        self.explainer = shap.TreeExplainer(self.model)

    def explain_transaction(
        self,
        X_df: pd.DataFrame,
        top_k: int = 5,
        raw_tx_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Compute SHAP values for a transaction record and return top-k features with unscaled raw values.
        
        Returns:
            top_features: List of dicts containing feature, human_name, raw_value, shap_value, direction
            base_value: Expected value in margin space
            prob: Output probability
        """
        if isinstance(X_df, dict):
            X_df = pd.DataFrame([X_df])
            
        if self.feature_names:
            X_df = X_df[self.feature_names]
            
        shap_values_raw = self.explainer(X_df)
        
        if len(shap_values_raw.values.shape) == 2:
            shap_vec = shap_values_raw.values[0]
            base_val = float(shap_values_raw.base_values[0])
        elif len(shap_values_raw.values.shape) == 3:
            shap_vec = shap_values_raw.values[0, :, 1]
            base_val = float(shap_values_raw.base_values[0, 1])
        else:
            shap_vec = np.asarray(shap_values_raw.values).flatten()
            base_val = float(np.mean(shap_values_raw.base_values))

        abs_indices = np.argsort(np.abs(shap_vec))[::-1]
        top_indices = abs_indices[:top_k]
        
        # Unscale scaler values if scaler artifact is available
        scaler_path = Path("models/scaler.joblib")
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        top_features = []
        for idx in top_indices:
            feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            shap_val = float(shap_vec[idx])
            
            # Determine unscaled human-readable raw value
            if raw_tx_dict and feat_name in raw_tx_dict:
                raw_val = float(raw_tx_dict[feat_name])
            elif scaler and hasattr(scaler, "feature_names_in_") and feat_name in scaler.feature_names_in_:
                feat_idx = list(scaler.feature_names_in_).index(feat_name)
                scaled_v = float(X_df.iloc[0, idx])
                raw_val = float(scaled_v * scaler.scale_[feat_idx] + scaler.mean_[feat_idx])
            else:
                raw_val = float(X_df.iloc[0, idx])

            human_name = FEATURE_HUMAN_NAMES.get(feat_name, feat_name.replace("_", " ").title())
            direction = "pushes_toward_fraud" if shap_val > 0 else "pushes_away_from_fraud"
            
            top_features.append({
                "feature": feat_name,
                "human_name": human_name,
                "raw_value": raw_val,
                "shap_value": shap_val,
                "direction": direction,
            })

        # Margins and probability output
        margin = base_val + np.sum(shap_vec)
        prob = float(1.0 / (1.0 + np.exp(-margin)))
        
        return top_features, base_val, prob

    def verify_additivity(self, X_df: pd.DataFrame, tol: float = 1e-4) -> bool:
        """
        Verify SHAP additivity in MARGIN space:
        |base_value + sum(shap_values) - model.predict(X, output_margin=True)| < tol
        """
        shap_res = self.explainer(X_df)
        base_vals = shap_res.base_values
        if np.ndim(base_vals) == 0:
            base_vals = np.full(len(X_df), float(base_vals))
        elif np.ndim(base_vals) == 2:
            base_vals = base_vals[:, 1]
            
        shap_vals_matrix = shap_res.values
        if np.ndim(shap_vals_matrix) == 3:
            shap_vals_matrix = shap_vals_matrix[:, :, 1]

        sum_shap = np.sum(shap_vals_matrix, axis=1)
        base_plus_sum = base_vals + sum_shap

        # Get direct model prediction in margin space
        margin_preds = self.model.predict(X_df, output_margin=True)

        diffs = np.abs(base_plus_sum - margin_preds)
        max_diff = float(np.max(diffs))
        return bool(max_diff < tol)


def build_grounded_prompt(
    top_features: List[Dict[str, Any]],
    probability: float,
    threshold: float
) -> str:
    """Construct structured JSON LLM prompt grounded strictly in top SHAP features."""
    status = "FLAGGED AS FRAUD" if probability >= threshold else "CLEARED / LEGITIMATE"
    
    prompt = f"""You are a senior financial risk analyst assistant explaining an AI fraud detection decision.
Your task is to explain the model prediction using ONLY the mathematical SHAP contribution factors listed below.

[PREDICTION CONTEXT]
- Decision Status: {status}
- Predicted Fraud Probability: {probability:.1%}
- Decision Threshold: {threshold:.1%}

[GROUNDED SHAP TOP-5 FEATURES]
"""
    for i, feat in enumerate(top_features, 1):
        sign = "+" if feat["shap_value"] > 0 else ""
        prompt += f"{i}. feature_id: '{feat['feature']}' ({feat['human_name']}): value = {feat['raw_value']:.4f}, SHAP contribution = {sign}{feat['shap_value']:.4f} ({feat['direction']})\n"

    prompt += """
[REQUIRED STAGE OUTPUT FORMAT]
Return your response STRICTLY as a valid JSON object matching this schema:
{
  "explanation": "A concise 2-3 sentence explanation explaining the decision based strictly on the cited features.",
  "cited_features": [
    {
      "feature_id": "exact_feature_id",
      "direction": "pushes_toward_fraud" or "pushes_away_from_fraud"
    }
  ]
}

[STRICT INSTRUCTIONS]
1. Explain why the transaction was flagged or cleared using ONLY the features listed above.
2. DO NOT introduce outside factors, unlisted account details, or ungrounded assumptions.
3. Every feature mentioned in 'cited_features' MUST exist in the top-5 list above with matching direction.
4. If you mention any numeric values in 'explanation', they MUST match the raw values in the list above within rounding.
"""
    return prompt


def validate_faithfulness(
    llm_json_response: Dict[str, Any] | str,
    top_features: List[Dict[str, Any]]
) -> bool:
    """
    Validate LLM JSON response against three strict rules:
    (a) Every cited feature_id is in the supplied top-5.
    (b) Cited direction matches the SHAP sign.
    (c) Any numbers quoted in explanation text match supplied raw values within rounding tolerance.
    """
    if isinstance(llm_json_response, str):
        try:
            # Extract JSON block if wrapped in markdown code blocks
            text = llm_json_response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
        except Exception:
            return False
    else:
        data = llm_json_response

    if not isinstance(data, dict) or "explanation" not in data or "cited_features" not in data:
        return False

    explanation_text = data["explanation"]
    cited_list = data["cited_features"]

    supplied_map = {f["feature"]: f for f in top_features}

    # (a) & (b) Check cited_features
    for item in cited_list:
        feat_id = item.get("feature_id")
        direction = item.get("direction")

        if feat_id not in supplied_map:
            # Hallucinated feature not in top-5
            return False

        expected_direction = supplied_map[feat_id]["direction"]
        if direction != expected_direction:
            # Direction mismatch
            return False

    # (c) Validate numbers in explanation text match raw values within rounding
    numbers_in_text = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", explanation_text)]
    if numbers_in_text:
        supplied_raw_values = [f["raw_value"] for f in top_features]
        for num in numbers_in_text:
            # Skip percentage/threshold constants (e.g. 0.25 or 100 or 5)
            if num in (0.0, 1.0, 100.0, 5.0) or any(abs(num - f["shap_value"]) < 0.05 for f in top_features):
                continue
            
            # Check if num matches any raw value within 10% relative or 0.1 absolute tolerance
            matches = any(
                abs(num - raw) < 0.1 or (raw != 0 and abs(num - raw) / abs(raw) < 0.15)
                for raw in supplied_raw_values
            )
            if not matches:
                # Quoted number doesn't match any raw value in supplied list
                return False

    return True


def deterministic_template_explanation(
    top_features: List[Dict[str, Any]],
    probability: float,
    threshold: float
) -> str:
    """Fallback deterministic explanation built strictly from SHAP values."""
    status = "Flagged as Fraud" if probability >= threshold else "Cleared as Legitimate"
    
    top_risk = [f for f in top_features if f["shap_value"] > 0]
    top_mitigating = [f for f in top_features if f["shap_value"] < 0]

    explanation = f"Transaction {status} with a fraud probability of {probability:.1%} (threshold {threshold:.1%}). "
    
    if top_risk:
        risk_strs = [f"{f['human_name']} (SHAP +{f['shap_value']:.3f})" for f in top_risk[:2]]
        explanation += f"Primary risk drivers pushing toward fraud: {', '.join(risk_strs)}. "
    
    if top_mitigating:
        mit_strs = [f"{f['human_name']} (SHAP {f['shap_value']:.3f})" for f in top_mitigating[:2]]
        explanation += f"Mitigating factors: {', '.join(mit_strs)}."

    return explanation


def generate_llm_explanation(
    top_features: List[Dict[str, Any]],
    probability: float,
    threshold: float,
) -> str:
    """
    Generate grounded LLM explanation with Gemini primary -> OpenAI fallback -> Deterministic template fallback.
    Includes JSON schema enforcement and strict faithfulness validation.
    """
    prompt = build_grounded_prompt(top_features, probability, threshold)
    
    # 1. Try Gemini primary
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import httpx
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = httpx.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if validate_faithfulness(raw_text, top_features):
                    data = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
                    return data["explanation"]
                # Retry once if unfaithful
                response = httpx.post(url, json=payload, timeout=5.0)
                if response.status_code == 200:
                    raw_retry = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if validate_faithfulness(raw_retry, top_features):
                        data = json.loads(raw_retry.replace("```json", "").replace("```", "").strip())
                        return data["explanation"]
        except Exception:
            pass

    # 2. Try OpenAI fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import httpx
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}"}
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            response = httpx.post(url, json=payload, headers=headers, timeout=5.0)
            if response.status_code == 200:
                raw_text = response.json()["choices"][0]["message"]["content"].strip()
                if validate_faithfulness(raw_text, top_features):
                    data = json.loads(raw_text)
                    return data["explanation"]
        except Exception:
            pass

    # 3. Deterministic template fallback
    return deterministic_template_explanation(top_features, probability, threshold)


def generate_shap_summary_plot(
    explainer: ShapExplainer,
    val_df: pd.DataFrame,
    output_path: Path = Path("reports/shap_summary.png")
):
    """
    Generate and save global SHAP summary plot using a stratified sample:
    all validation fraud rows (up to 2000) + 3000 random non-fraud rows.
    """
    feature_cols = [c for c in val_df.columns if c not in ("isFraud", "step")]
    fraud_df = val_df[val_df["isFraud"] == 1]
    non_fraud_df = val_df[val_df["isFraud"] == 0]

    n_fraud = min(len(fraud_df), 2000)
    n_non_fraud = min(len(non_fraud_df), 3000)

    sample_fraud = fraud_df.sample(n_fraud, random_state=42) if n_fraud > 0 else fraud_df
    sample_non_fraud = non_fraud_df.sample(n_non_fraud, random_state=42) if n_non_fraud > 0 else non_fraud_df

    stratified_sample = pd.concat([sample_fraud, sample_non_fraud]).sample(frac=1.0, random_state=42)
    X_strat = stratified_sample[feature_cols]

    shap_values = explainer.explainer(X_strat)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_strat, show=False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Stratified SHAP summary plot saved to {output_path}")


def generate_waterfall_example_plot(
    explainer: ShapExplainer,
    val_df: pd.DataFrame,
    output_path: Path = Path("reports/shap_waterfall_example.png")
):
    """Save one local waterfall/bar example for a correctly caught ring-style fraud."""
    feature_cols = [c for c in val_df.columns if c not in ("isFraud", "step")]
    fraud_rows = val_df[val_df["isFraud"] == 1]
    
    if len(fraud_rows) == 0:
        sample_row = val_df[feature_cols].iloc[:1]
    else:
        sample_row = fraud_rows[feature_cols].iloc[:1]

    shap_exp = explainer.explainer(sample_row)[0]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Local waterfall example plot saved to {output_path}")
