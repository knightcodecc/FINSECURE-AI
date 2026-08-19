"""
Explainable AI Engine Module for FinSecure AI Fraud Detection System.

This module provides the XAIEngine class that generates natural-language
explanations for fraud predictions using an LLM with graceful fallback
to rule-based explanations.

Author: FinSecure AI Team
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import os
from dotenv import load_dotenv

# Try to import OpenAI, fallback if not available
try:
    from openai import AsyncOpenAI, RateLimitError, Timeout
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import schemas for type hints
from api.schemas import RiskFactor, RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Prompt template for LLM
XAI_PROMPT_TEMPLATE = """You are a senior financial fraud analyst AI assistant.

A machine learning model has flagged a financial transaction as potentially fraudulent.

Transaction Details:
- Transaction Type: {type}
- Amount: ₹{amount:,.2f}
- Sender Balance Before: ₹{oldbalanceOrg:,.2f}
- Sender Balance After: ₹{newbalanceOrig:,.2f}
- Receiver Balance Before: ₹{oldbalanceDest:,.2f}
- Receiver Balance After: ₹{newbalanceDest:,.2f}
- Fraud Probability Score: {fraud_probability:.1%}
- Risk Level: {risk_level}

Top Risk Factors Identified by the Model:
{top_risk_factors_formatted}

Task: In 2-3 concise sentences, explain to a bank analyst WHY this transaction is suspicious.
Focus on the behavioral anomalies. Do NOT include PII. Be specific and actionable.
Output ONLY the explanation, no preamble.
"""


def format_risk_factors(risk_factors: List[RiskFactor]) -> str:
    """
    Format risk factors for the prompt.

    Args:
        risk_factors: List of RiskFactor objects.

    Returns:
        Formatted string of risk factors.
    """
    if not risk_factors:
        return "No specific risk factors identified."

    lines = []
    for i, factor in enumerate(risk_factors, 1):
        # Format based on feature type
        if factor.feature == 'account_drained' and factor.value == 1:
            desc = "Account completely drained"
        elif factor.feature == 'amount_gt_orig_balance' and factor.value == 1:
            desc = "Amount exceeds sender's balance"
        elif factor.feature == 'orig_balance_zero' and factor.value == 1:
            desc = "Sender had zero balance before transaction"
        elif factor.feature == 'dest_balance_unchanged' and factor.value == 1:
            desc = "Recipient balance unchanged (suspicious)"
        elif factor.feature == 'is_high_risk_hour' and factor.value == 1:
            desc = "Transaction during high-risk hours (midnight-5AM)"
        elif factor.feature == 'dest_balance_zero' and factor.value == 1:
            desc = "Recipient had zero balance before"
        else:
            desc = f"Feature value: {factor.value:.4f}"

        lines.append(f"{i}. {factor.feature}: {desc}")

    return "\n".join(lines)


def generate_rule_based_explanation(risk_factors: List[RiskFactor],
                                     probability: float,
                                     risk_level: RiskLevel) -> str:
    """
    Generate rule-based fallback explanation.

    Args:
        risk_factors: List of identified risk factors.
        probability: Fraud probability.
        risk_level: Risk level classification.

    Returns:
        Rule-based explanation string.
    """
    reasons = []

    # Analyze risk factors
    for factor in risk_factors:
        if factor.feature == 'account_drained' and factor.value == 1:
            reasons.append("complete account drainage detected")
        elif factor.feature == 'amount_gt_orig_balance' and factor.value == 1:
            reasons.append("transaction amount exceeds available balance")
        elif factor.feature == 'orig_balance_zero' and factor.value == 1:
            reasons.append("transfer from zero-balance account")
        elif factor.feature == 'dest_balance_unchanged' and factor.value == 1:
            reasons.append("recipient balance unchanged (invalid transfer)")
        elif factor.feature == 'is_high_risk_hour' and factor.value == 1:
            reasons.append("transaction during high-risk nighttime hours")
        elif factor.feature == 'dest_balance_zero' and factor.value == 1:
            reasons.append("transfer to previously zero-balance account")

    # Build explanation
    if not reasons:
        if probability >= 0.8:
            return f"Transaction flagged as critical risk (probability: {probability:.1%}) due to multiple anomalies in transaction patterns."
        elif probability >= 0.6:
            return f"Transaction flagged as high risk (probability: {probability:.1%}) with unusual behavioral patterns detected."
        else:
            return f"Transaction shows moderate fraud indicators (probability: {probability:.1%})."

    # Combine reasons
    explanation = "Transaction flagged due to: "
    explanation += ", ".join(reasons[:3])  # Take top 3 reasons

    if len(reasons) > 3:
        explanation += f", and {len(reasons) - 3} other factors"

    explanation += "."

    return explanation


class XAIEngine:
    """
    Explainable AI engine for generating fraud explanations.

    This class uses an LLM to generate natural-language explanations for
    fraud predictions, with graceful degradation to rule-based explanations
    if the LLM fails.
    """

    def __init__(self, timeout_seconds: int = 3):
        """
        Initialize XAI engine.

        Args:
            timeout_seconds: Timeout for LLM API calls.
        """
        self.timeout_seconds = timeout_seconds
        self.client: Optional[Any] = None
        self.gemini_client: Optional[str] = None
        self.model: str = "gpt-3.5-turbo"
        self.provider: Optional[str] = None

        # Load keys
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Initialize Gemini if available
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            self.gemini_client = gemini_key
            env_model = os.getenv("LLM_MODEL")
            if env_model and "gemini" in env_model.lower():
                self.model = env_model
            else:
                self.model = "gemini-2.5-flash"
            self.provider = "gemini"
            logger.info(f"XAI Engine initialized with Gemini using model {self.model}")
        # Otherwise initialize OpenAI
        elif OPENAI_AVAILABLE and openai_key and openai_key != "your_openai_api_key_here":
            try:
                self.client = AsyncOpenAI(api_key=openai_key)
                env_model = os.getenv("LLM_MODEL")
                self.model = env_model if env_model else "gpt-3.5-turbo"
                self.provider = "openai"
                logger.info(f"XAI Engine initialized with OpenAI using model {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.provider = None
        else:
            logger.info("XAI Engine using rule-based explanations (no LLM API key)")

    async def generate_explanation(self, context: Dict[str, Any]) -> str:
        """
        Generate explanation for a fraud prediction.

        Args:
            context: Dictionary containing:
                - transaction: TransactionInput object
                - fraud_probability: float
                - risk_level: RiskLevel
                - top_risk_factors: List[RiskFactor]

        Returns:
            Explanation string.
        """
        transaction = context.get('transaction')
        fraud_probability = context.get('fraud_probability', 0.0)
        risk_level = context.get('risk_level', RiskLevel.LOW)
        risk_factors = context.get('top_risk_factors', [])

        # Format risk factors
        risk_factors_formatted = format_risk_factors(risk_factors)

        # Build prompt
        prompt = XAI_PROMPT_TEMPLATE.format(
            type=transaction.type.value,
            amount=transaction.amount,
            oldbalanceOrg=transaction.oldbalanceOrg,
            newbalanceOrig=transaction.newbalanceOrig,
            oldbalanceDest=transaction.oldbalanceDest,
            newbalanceDest=transaction.newbalanceDest,
            fraud_probability=fraud_probability,
            risk_level=risk_level.value,
            top_risk_factors_formatted=risk_factors_formatted
        )

        # Try LLM if available
        if self.provider == "gemini" and self.gemini_client:
            try:
                explanation = await asyncio.wait_for(
                    self._call_gemini(prompt),
                    timeout=self.timeout_seconds
                )
                logger.info(f"Generated Gemini explanation for fraud probability: {fraud_probability:.2f}")
                return explanation
            except asyncio.TimeoutError:
                logger.warning("Gemini call timed out, using rule-based fallback")
            except Exception as e:
                logger.warning(f"Gemini call failed: {e}, using rule-based fallback")
        elif self.provider == "openai" and self.client:
            try:
                explanation = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout_seconds
                )
                logger.info(f"Generated OpenAI explanation for fraud probability: {fraud_probability:.2f}")
                return explanation
            except asyncio.TimeoutError:
                logger.warning("OpenAI call timed out, using rule-based fallback")
            except Exception as e:
                logger.warning(f"OpenAI call failed: {e}, using rule-based fallback")

        # Fallback to rule-based explanation
        return generate_rule_based_explanation(
            risk_factors,
            fraud_probability,
            risk_level
        )

    async def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API for explanation.

        Args:
            prompt: Formatted prompt string.

        Returns:
            LLM response string.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial fraud analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    async def _call_gemini(self, prompt: str) -> str:
        """
        Call Google Gemini API for explanation via raw HTTP requests.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Gemini response string.
        """
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.gemini_client}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": "You are a financial fraud analysis expert."}
                ]
            },
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 150
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            # Extract text response from Gemini response format
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()

    def generate_explanation_sync(self, context: Dict[str, Any]) -> str:
        """
        Synchronous wrapper for generate_explanation.

        For use in sync contexts, executing the underlying coroutine properly.

        Args:
            context: Explanation context dictionary.

        Returns:
            Explanation string.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            # Schedule the coroutine in the running loop, and wait for the result
            # inside a ThreadPoolExecutor to prevent blocking runtime errors/deadlocks
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: asyncio.run_coroutine_threadsafe(
                        self.generate_explanation(context), loop
                    ).result()
                )
                return future.result()
        else:
            # No running event loop in this thread, run with asyncio.run
            return asyncio.run(self.generate_explanation(context))


def create_xai_engine(timeout_seconds: int = 3) -> XAIEngine:
    """
    Factory function to create XAI engine.

    Args:
        timeout_seconds: Timeout for LLM calls.

    Returns:
        Initialized XAIEngine instance.
    """
    return XAIEngine(timeout_seconds=timeout_seconds)