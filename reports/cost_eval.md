# Cost-Sensitive Evaluation Report (FinSecure AI v2)

## Cost Model Parameters
- **False Negative (FN) Cost**: Transaction Amount ($	ext{amount} \times 1.0$)
- **False Positive (FP) Cost**: Fixed Analyst Review Cost (Default: Rs. 50.0 per alert)

## Model Performance on Held-Out Test Split
- **No Model Baseline (Flag Nothing)**: Total Cost = **Rs. 671,617,426.05**
- **Deployed Model (Tabular + Graph, T_F2 = 0.6633)**: Total Cost = **Rs. 70,250,281.89** | Savings = **Rs. 601,367,144.16**
- **Deployed Model (Tabular + Graph, T_cost = 0.0100)**: Total Cost = **Rs. 2,163,200.00** | Savings = **Rs. 669,454,226.05**
