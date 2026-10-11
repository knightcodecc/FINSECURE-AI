"""
predictor.py — ML Inference Wrapper for FinSecure AI v2.
=========================================================
Loads models/xgboost_model.joblib + models/scaler.joblib + models/model_metadata.joblib.
Ensures ZERO train/serve feature skew, maintains real-time expanding graph state (zero lookahead),
and computes SHAP explanations and cost-sensitive threshold decisions.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from api.db import log_prediction, log_predictions_batch
from api.alerts import send_alert_webhook
from api.xai_engine import shap_explain, generate_explanation
from src.feature_engineering import engineer_tabular_features

MODEL_PATH = Path("models/xgboost_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
METADATA_PATH = Path("models/model_metadata.joblib")


class InferenceGraphState:
    """
    Maintains real-time expanding-window graph state for online inference.
    Prevents future lookahead leakage by only incorporating past transactions and confirmed fraud labels.
    """
    def __init__(self, refresh_interval: int = 100):
        self.G = nx.DiGraph()
        self.confirmed_fraud_edges: Dict[Tuple[str, str], int] = {}
        self.tx_count = 0
        self.refresh_interval = refresh_interval
        self._cached_pagerank: Dict[str, float] = {}
        self._cached_component_size: Dict[str, int] = {}

    def update_graph(self, name_orig: str, name_dest: str, amount: float, is_confirmed_fraud: bool = False):
        """Update graph state with a newly processed transaction."""
        self.tx_count += 1
        
        if self.G.has_edge(name_orig, name_dest):
            self.G[name_orig][name_dest]["weight"] += 1
            self.G[name_orig][name_dest]["total_amount"] += amount
            if is_confirmed_fraud:
                self.G[name_orig][name_dest]["fraud_count"] += 1
        else:
            self.G.add_edge(
                name_orig,
                name_dest,
                weight=1,
                total_amount=amount,
                fraud_count=1 if is_confirmed_fraud else 0,
            )

        # Refresh PageRank and component size periodically to balance speed and centrality accuracy
        if self.tx_count == 1 or self.tx_count % self.refresh_interval == 0:
            self._recompute_structural_metrics()

    def _recompute_structural_metrics(self):
        if self.G.number_of_nodes() == 0:
            self._cached_pagerank = {}
            self._cached_component_size = {}
            return
        
        self._cached_pagerank = nx.pagerank(self.G, weight="weight")
        
        comp_size = {}
        for component in nx.weakly_connected_components(self.G):
            size = len(component)
            for node in component:
                comp_size[node] = size
        self._cached_component_size = comp_size

    def warm_from_history(self, history_df: pd.DataFrame):
        """
        Warm real-time inference graph state from historical transactions (e.g. training set).
        Computes initial degrees, PageRank, connected component sizes, and fraud exposures.
        """
        from src.graph_features import build_graph, compute_node_features
        self.G = build_graph(history_df)
        self.tx_count = len(history_df)
        node_feats = compute_node_features(history_df, self.G)
        if not node_feats.empty:
            nfi = node_feats.set_index("node")
            self._cached_in_degree = nfi["in_degree"].to_dict()
            self._cached_out_degree = nfi["out_degree"].to_dict()
            self._cached_pagerank = nfi["pagerank"].to_dict()
            self._cached_component_size = nfi["component_size"].to_dict()
            self._cached_fraud_exposure = nfi["fraud_exposure"].to_dict()
        else:
            self._cached_in_degree = {}
            self._cached_out_degree = {}
            self._cached_pagerank = {}
            self._cached_component_size = {}
            self._cached_fraud_exposure = {}

    def get_node_features(self, node: str) -> Dict[str, float]:
        """Extract graph features for a node with neutral cold-start defaults."""
        if not self.G.has_node(node):
            return {
                "in_degree": 0.0,
                "out_degree": 0.0,
                "pagerank": 0.0,
                "component_size": 1.0,
                "fraud_exposure": 0.0,
            }

        in_deg = float(getattr(self, "_cached_in_degree", {}).get(node, self.G.in_degree(node)))
        out_deg = float(getattr(self, "_cached_out_degree", {}).get(node, self.G.out_degree(node)))
        pr = float(self._cached_pagerank.get(node, 0.0))
        comp_s = float(self._cached_component_size.get(node, 1.0))

        if hasattr(self, "_cached_fraud_exposure") and node in self._cached_fraud_exposure:
            fraud_exp = float(self._cached_fraud_exposure[node])
        else:
            incident_edges = list(self.G.in_edges(node, data=True)) + list(self.G.out_edges(node, data=True))
            if not incident_edges:
                fraud_exp = 0.0
            else:
                fraud_edges = sum(1 for *_x, data in incident_edges if data.get("fraud_count", 0) > 0)
                fraud_exp = float(fraud_edges / len(incident_edges))

        return {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "pagerank": pr,
            "component_size": comp_s,
            "fraud_exposure": fraud_exp,
        }


class Predictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
            
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self._metadata_mtime = 0.0
        self._reload_metadata_if_needed()

        # Real-time inference graph state
        self.graph_state = InferenceGraphState()

    def warm_graph_on_train(self, history_df: Optional[pd.DataFrame] = None):
        """Warm the inference graph on training history transactions (step <= 420)."""
        if history_df is None:
            if Path("data/paysim_synth.csv").exists():
                df_raw = pd.read_csv("data/paysim_synth.csv")
            elif Path("data/transactions_with_graph_features.parquet").exists():
                df_raw = pd.read_parquet("data/transactions_with_graph_features.parquet")
            elif Path("data/paysim.csv").exists():
                df_raw = pd.read_csv("data/paysim.csv")
            else:
                return
            history_df = df_raw[df_raw["step"] <= 420].copy()
            
        self.graph_state.warm_from_history(history_df)

    def _reload_metadata_if_needed(self):
        if METADATA_PATH.exists():
            try:
                current_mtime = METADATA_PATH.stat().st_mtime
                if getattr(self, "_metadata_mtime", 0.0) != current_mtime:
                    self.metadata = joblib.load(METADATA_PATH)
                    self._metadata_mtime = current_mtime
                    self.active_threshold = float(self.metadata.get("active_threshold", self.metadata.get("cost_optimal_threshold", 0.017033)))
                    self.f2_optimal_threshold = float(self.metadata.get("f2_optimal_threshold", 0.201402))
                    self.cost_optimal_threshold = float(self.metadata.get("cost_optimal_threshold", 0.017033))
                    self.active_strategy = str(self.metadata.get("active_strategy", "cost_optimal"))
                    self.feature_names = self.metadata.get("feature_names", list(getattr(self.model, "feature_names_in_", [])))
            except Exception:
                pass

    def prepare_features(self, raw_tx: Dict[str, Any]) -> pd.DataFrame:
        """
        Shared feature pipeline matching training feature creation exactly:
        1. Extract graph features from graph_state for nameOrig and nameDest
        2. One-hot encode type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
        3. Create 13 engineered tabular features
        4. Scale numerical features using scaler.joblib
        """
        amount = float(raw_tx.get("amount", 0.0))
        assert amount > 0, (
            f"Serving assertion failed: raw rupee amount must be strictly positive (> 0), got {amount}. "
            "Never pass scaled/z-score values to predict_one!"
        )

        orig_id = str(raw_tx.get("nameOrig", ""))
        dest_id = str(raw_tx.get("nameDest", ""))

        orig_feats = self.graph_state.get_node_features(orig_id)
        dest_feats = self.graph_state.get_node_features(dest_id)

        tx_dict = {
            "step": int(raw_tx.get("step", 1)),
            "amount": amount,
            "oldbalanceOrg": float(raw_tx.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(raw_tx.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(raw_tx.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(raw_tx.get("newbalanceDest", 0.0)),
            
            # One-hot transaction types
            "type_CASH_IN": 1.0 if raw_tx.get("type") == "CASH_IN" else 0.0,
            "type_CASH_OUT": 1.0 if raw_tx.get("type") == "CASH_OUT" else 0.0,
            "type_DEBIT": 1.0 if raw_tx.get("type") == "DEBIT" else 0.0,
            "type_PAYMENT": 1.0 if raw_tx.get("type") == "PAYMENT" else 0.0,
            "type_TRANSFER": 1.0 if raw_tx.get("type") == "TRANSFER" else 0.0,

            # Graph features
            "orig_in_degree": orig_feats["in_degree"],
            "orig_out_degree": orig_feats["out_degree"],
            "orig_pagerank": orig_feats["pagerank"],
            "orig_component_size": orig_feats["component_size"],
            "orig_fraud_exposure": orig_feats["fraud_exposure"],
            "dest_in_degree": dest_feats["in_degree"],
            "dest_out_degree": dest_feats["out_degree"],
            "dest_pagerank": dest_feats["pagerank"],
            "dest_component_size": dest_feats["component_size"],
            "dest_fraud_exposure": dest_feats["fraud_exposure"],
        }

        df_raw = pd.DataFrame([tx_dict])
        
        # Engineer 13 tabular features
        df_eng = engineer_tabular_features(df_raw)

        # Scale using TRAIN-fitted scaler
        if self.feature_names:
            df_eng = df_eng[self.feature_names]

        df_scaled = df_eng.copy()
        df_scaled[self.feature_names] = self.scaler.transform(df_eng[self.feature_names])

        return df_scaled

    def predict_one(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full end-to-end inference flow for a single transaction:
        Input -> Prepare Features -> XGBoost Score -> SHAP Top 5 -> LLM Explanation -> Decision -> Log & Alert.
        """
        self._reload_metadata_if_needed()
        df_scaled = self.prepare_features(raw_tx)
        
        # XGBoost probability
        proba = float(self.model.predict_proba(df_scaled)[0, 1])
        is_fraud = bool(proba >= self.active_threshold)

        # SHAP Top 5 features & Grounded LLM Explanation
        top_features, base_val, _ = shap_explain(df_scaled, top_k=5, raw_tx_dict=raw_tx)
        explanation = generate_explanation(top_features, proba, self.active_threshold)

        timestamp = datetime.now().isoformat()

        response_payload = {
            "prediction_proba": proba,
            "is_fraud": is_fraud,
            "threshold_used": self.active_threshold,
            "active_strategy": self.active_strategy,
            "f2_optimal_threshold": self.f2_optimal_threshold,
            "cost_optimal_threshold": self.cost_optimal_threshold,
            "shap_top_features": top_features,
            "explanation": explanation,
            "timestamp": timestamp,
        }

        # Update real-time expanding graph state
        self.graph_state.update_graph(
            name_orig=raw_tx.get("nameOrig", ""),
            name_dest=raw_tx.get("nameDest", ""),
            amount=raw_tx.get("amount", 0.0),
            is_confirmed_fraud=is_fraud
        )

        # Log prediction to SQLite
        db_log_record = {
            "timestamp": timestamp,
            "step": raw_tx.get("step", 1),
            "nameOrig": raw_tx.get("nameOrig", ""),
            "nameDest": raw_tx.get("nameDest", ""),
            "amount": raw_tx.get("amount", 0.0),
            "prediction_proba": proba,
            "is_fraud": is_fraud,
            "threshold_used": self.active_threshold,
            "features": raw_tx,
            "shap_top_features": top_features,
            "explanation": explanation,
        }
        log_prediction(db_log_record)

        # Dispatch alert webhook if high risk fraud
        if is_fraud:
            send_alert_webhook("HIGH_FRAUD_ALERT", {
                "nameOrig": raw_tx.get("nameOrig"),
                "nameDest": raw_tx.get("nameDest"),
                "amount": raw_tx.get("amount"),
                "prediction_proba": proba,
                "explanation": explanation
            })

        return response_payload


# Global Predictor singleton
_PREDICTOR: Predictor | None = None


def get_predictor(warm: bool = True) -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
        if warm:
            _PREDICTOR.warm_graph_on_train()
    else:
        _PREDICTOR._reload_metadata_if_needed()
    return _PREDICTOR

