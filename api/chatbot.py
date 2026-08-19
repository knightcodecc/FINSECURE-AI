"""
Process Q&A Chatbot Module for FinSecure AI Fraud Detection System.

Provides RAG-based conversational Q&A for explaining historical fraud predictions,
aggregate stats, recent alerts, and analyst feedback without hallucination.

Author: FinSecure AI Team
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from api.db import get_rag_context_for_chatbot
from api.xai_engine import XAIEngine

logger = logging.getLogger(__name__)

CHATBOT_PROMPT_TEMPLATE = """You are FinSecure AI's Senior Fraud Operations Assistant.
Your job is to answer the bank analyst's question accurately using ONLY the retrieved database context provided below.

Retrieved Database Context:
- Overall Stats: {total_predictions} total predictions scored, {fraud_detected} fraud cases caught ({fraud_rate}% fraud rate).
- Total Fraud Amount: ₹{total_fraud_amount:,.2f}
- Average Fraud Probability: {average_fraud_probability:.1%}
- Risk Level Counts: {risk_level_counts}

Recent Fraud Alerts (Last 10 Flagged):
{recent_fraud_alerts_formatted}

Queried Transaction Detail (if specific ID was mentioned):
{queried_transaction_formatted}

Recent Analyst Feedback:
{recent_feedback_formatted}

User Question: "{user_query}"

Instructions:
1. Provide a concise, clear, and professional answer based STRICTLY on the retrieved data above.
2. Do NOT invent or hallucinate transaction details or statistics not present in the context.
3. If a specific transaction ID was asked about and found, explain its status, probability score, and risk level.
4. Keep the tone authoritative, helpful, and concise (2-4 sentences max).
Output ONLY the final response.
"""

def format_alerts_summary(alerts: list) -> str:
    if not alerts:
        return "No recent fraud alerts recorded."
    lines = []
    for a in alerts[:5]:
        tx_id = str(a.get('transaction_id', ''))[:8]
        lines.append(
            f"- TxID ..{tx_id}: Step {a.get('step')}, Type {a.get('type')}, Amount ₹{a.get('amount', 0):,.2f}, "
            f"Prob {a.get('fraud_probability', 0):.1%}, Risk {a.get('risk_level')}"
        )
    return "\n".join(lines)

def format_tx_detail(tx: Optional[dict]) -> str:
    if not tx:
        return "No specific transaction ID identified in prompt or found in records."
    explanation = tx.get('explanation_text', 'No custom explanation text stored.')
    return (
        f"- ID: {tx.get('transaction_id')}\n"
        f"- Step: {tx.get('step')}, Type: {tx.get('type')}, Amount: ₹{tx.get('amount', 0):,.2f}\n"
        f"- Probability: {tx.get('fraud_probability', 0):.1%}, Risk Level: {tx.get('risk_level')}, Flagged: {bool(tx.get('is_fraud'))}\n"
        f"- Explanation: {explanation}"
    )

def format_feedback_summary(feedback: list) -> str:
    if not feedback:
        return "No analyst feedback logged yet."
    lines = []
    for f in feedback[:3]:
        tx_id = str(f.get('transaction_id', ''))[:8]
        lines.append(f"- TxID ..{tx_id}: {f.get('feedback_label')} - Notes: {f.get('notes', 'None')}")
    return "\n".join(lines)

async def generate_chatbot_response(user_query: str, xai_engine: Optional[XAIEngine]) -> Dict[str, Any]:
    """
    Process Q&A Chatbot request using RAG retrieval + LLM synthesis with graceful fallback.

    Args:
        user_query: The analyst's question.
        xai_engine: Initialized XAIEngine instance.

    Returns:
        Dict containing answer string, context metadata, and timestamp.
    """
    # 1. Retrieve factual database context
    rag_context = get_rag_context_for_chatbot(user_query)
    stats = rag_context.get('stats_summary', {})
    
    # 2. Format context for prompt
    prompt = CHATBOT_PROMPT_TEMPLATE.format(
        total_predictions=stats.get('total_predictions', 0),
        fraud_detected=stats.get('fraud_detected', 0),
        fraud_rate=stats.get('fraud_rate', 0.0),
        total_fraud_amount=stats.get('total_fraud_amount', 0.0),
        average_fraud_probability=stats.get('average_fraud_probability', 0.0),
        risk_level_counts=stats.get('risk_level_distribution', {}),
        recent_fraud_alerts_formatted=format_alerts_summary(rag_context.get('recent_fraud_alerts', [])),
        queried_transaction_formatted=format_tx_detail(rag_context.get('queried_transaction')),
        recent_feedback_formatted=format_feedback_summary(rag_context.get('recent_feedback', [])),
        user_query=user_query
    )

    answer = None
    
    # 3. Try LLM generation via xai_engine if available
    if xai_engine and xai_engine.provider == "gemini" and xai_engine.gemini_client:
        try:
            answer = await asyncio.wait_for(
                xai_engine._call_gemini(prompt),
                timeout=xai_engine.timeout_seconds
            )
            logger.info("Chatbot answer generated via Gemini")
        except Exception as e:
            logger.warning(f"Gemini call failed for chatbot: {e}")
    elif xai_engine and xai_engine.provider == "openai" and xai_engine.client:
        try:
            res = await asyncio.wait_for(
                xai_engine.client.chat.completions.create(
                    model=xai_engine.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.3
                ),
                timeout=xai_engine.timeout_seconds
            )
            answer = res.choices[0].message.content.strip()
            logger.info("Chatbot answer generated via OpenAI")
        except Exception as e:
            logger.warning(f"OpenAI call failed for chatbot: {e}")

    # 4. Fallback to deterministic rule-based response if LLM failed or no LLM key
    if not answer:
        tx_detail = rag_context.get('queried_transaction')
        if tx_detail:
            answer = (
                f"Transaction {tx_detail.get('transaction_id')} ({tx_detail.get('type')} of ₹{tx_detail.get('amount', 0):,.2f}) "
                f"was evaluated at step {tx_detail.get('step')} with a fraud probability of {tx_detail.get('fraud_probability', 0):.1%} ({tx_detail.get('risk_level')} risk). "
                f"Verdict: {'FLAGGED FRAUD' if tx_detail.get('is_fraud') else 'CLEARED SAFE'}."
            )
        else:
            answer = (
                f"FinSecure AI has processed {stats.get('total_predictions', 0):,} total transactions to date, "
                f"detecting {stats.get('fraud_detected', 0):,} fraud cases ({stats.get('fraud_rate', 0.0)}% fraud rate) "
                f"totaling ₹{stats.get('total_fraud_amount', 0.0):,.2f}. "
                f"Risk breakdown: {stats.get('risk_level_distribution', {})}."
            )

    return {
        'query': user_query,
        'answer': answer,
        'has_queried_tx': rag_context.get('queried_transaction') is not None,
        'timestamp': datetime.now().isoformat()
    }
