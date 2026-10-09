# Feature Definitions (FinSecure AI v2)

Total Model Input Feature Count: **33 columns** (`step` is excluded from model input)

## 1. Raw Numeric Features (5 columns)
- `amount`: Raw transaction amount.
- `oldbalanceOrg`: Initial balance of origin account before transaction.
- `newbalanceOrig`: New balance of origin account after transaction.
- `oldbalanceDest`: Initial balance of destination account before transaction.
- `newbalanceDest`: New balance of destination account after transaction.

*(Note: `step` is used as a temporal index for chronological splitting and temporal feature creation, but is excluded from model input features).*

## 2. One-Hot Encoded Categorical Features (5 columns)
- `type_CASH_IN`: Binary flag for CASH_IN transaction.
- `type_CASH_OUT`: Binary flag for CASH_OUT transaction.
- `type_DEBIT`: Binary flag for DEBIT transaction.
- `type_PAYMENT`: Binary flag for PAYMENT transaction.
- `type_TRANSFER`: Binary flag for TRANSFER transaction.

## 3. Engineered Tabular Features (13 columns)
- `errorBalanceOrig`: Balance discrepancy in origin accounting (`newbalanceOrig + amount - oldbalanceOrg`).
- `errorBalanceDest`: Balance discrepancy in destination accounting (`oldbalanceDest + amount - newbalanceDest`).
- `log_amount`: Logarithm of transaction amount (`log1p(amount)`).
- `account_drained`: Flag indicating origin balance went from >0 to 0.
- `dest_balance_zero`: Flag indicating destination balance before and after are both 0.
- `orig_balance_zero`: Flag indicating origin balance before and after are both 0.
- `amount_to_oldbalanceOrg_ratio`: Ratio of transaction amount to initial origin balance (`amount / (oldbalanceOrg + 1)`).
- `amount_to_oldbalanceDest_ratio`: Ratio of transaction amount to initial destination balance (`amount / (oldbalanceDest + 1)`).
- `hour_of_day`: Hour of transaction step (`step % 24`).
- `high_risk_hour`: Flag for late night / early morning transactions (hours 0-5).
- `day_of_week`: Day index (`(step // 24) % 7`).
- `net_change_orig`: Net balance reduction in origin (`oldbalanceOrg - newbalanceOrig`).
- `net_change_dest`: Net balance increase in destination (`newbalanceDest - oldbalanceDest`).

## 4. Graph Relational Features (10 columns)
### Origin Account Graph Features (5 columns)
- `orig_in_degree`: In-degree of origin node in historical transaction graph.
- `orig_out_degree`: Out-degree of origin node in historical transaction graph.
- `orig_pagerank`: PageRank centrality of origin node in money-flow graph.
- `orig_component_size`: Weakly connected component size of origin node.
- `orig_fraud_exposure`: Fraction of incident edges with historical fraud exposure for origin node.

### Destination Account Graph Features (5 columns)
- `dest_in_degree`: In-degree of destination node in historical transaction graph.
- `dest_out_degree`: Out-degree of destination node in historical transaction graph.
- `dest_pagerank`: PageRank centrality of destination node in money-flow graph.
- `dest_component_size`: Weakly connected component size of destination node.
- `dest_fraud_exposure`: Fraction of incident edges with historical fraud exposure for destination node.
