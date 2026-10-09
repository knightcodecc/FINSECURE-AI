# Future Enhancements — FinSecure AI v2

The scope of FinSecure AI v2 is explicitly anchored around 4 core architectural objectives for a VTU final-year major project (Graph Features, SHAP Grounded XAI, Cost-Sensitive Evaluation, and Concept Drift Monitoring). The following advanced capabilities are explicitly identified as **Future Enhancements** and reserved for post-thesis deployment:

---

## 1. Adaptive Per-Segment Decision Thresholds
- **Concept**: Instead of global thresholds ($T_{\text{F2}}$ and $T_{\text{cost}}$) applied uniformly across all transactions, operating dynamic threshold surfaces based on transaction metadata segments (e.g., high-net-worth accounts, cross-border transfers, new account age brackets, channel types).
- **Rationale & Benefit**: Allows risk teams to tighten control tolerances on ultra-high-value wire transfers while maintaining low friction for small micro-payments.
- **Architectural Prerequisite**: Requires offline segment-level precision-recall calibration and real-time segment classification within `api/predictor.py`.

---

## 2. Federated Learning & Privacy-Preserving Multi-Institution Training
- **Concept**: Decentralized model training across multiple banking nodes or financial institutions without centralizing raw customer ledger records or transaction PII.
- **Rationale & Benefit**: Enables cross-institutional fraud ring detection and shared threat intelligence while strictly satisfying regional data privacy regulations (GDPR, DPDP Act 2023).
- **Architectural Prerequisite**: Implementation of Secure Aggregation protocols (e.g., PySyft or Flower framework) and differential privacy bounds on XGBoost gradient updates.

---

## 3. Distributed Production Graph Engine (Neo4j / Memgraph / igraph)
- **Concept**: Transitioning from NetworkX dynamic in-memory graph objects to a distributed, persistent graph database (e.g., Neo4j, Memgraph, or Rust-backed `igraph`).
- **Rationale & Benefit**: Scales graph feature generation (PageRank, fraud exposure multi-hop propagation) to sub-millisecond latencies across billions of edge insertions.
- **Architectural Prerequisite**: Graph database cluster deployment with cypher query optimization and low-latency streaming sink (Kafka/Flink).
