# Full Feature Group Ablation & Impact Analysis (Clean Test Split)

Base Rate (Clean Test): **0.0173** (1.73%)
Baseline Costs:
- **No Model (Flag Nothing)**: Rs. 35,357,945.92
- **Flag Everything**: Rs. 1,766,600.00

| Model Variant | Feature Count | Base Rate | Validation T_F2 | PR-AUC (95% CI) | ROC-AUC | Overall Recall | Flag Rate | Ring-Fraud Recall (n=263) | Expected Cost (Rs.) | Beats Both Baselines? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Model (Flag Nothing)** | 0 | 0.0173 | N/A | N/A | N/A | 0.00% | 0.00% | N/A | Rs. 35,357,945.92 | N/A |
| **Flag Everything** | 0 | 0.0173 | 0.0000 | 0.0173 | 0.5000 | 100.00% | 100.00% | 100.00% | Rs. 1,766,600.00 | N/A |
| **Raw Numeric Only** | 5 | 0.0173 | 0.7425 | 0.5135 (0.4767..0.5501) | 0.7741 | 99.84% | 98.20% | 99.62% (99.0%..100.0%) | Rs. 1,744,819.50 | YES |
| **Raw + One-Hot Type** | 10 | 0.0173 | 0.8019 | 0.5518 (0.5138..0.5918) | 0.9351 | 98.55% | 43.31% | 99.62% (98.8%..100.0%) | Rs. 912,857.93 | YES |
| **No-Accounting-Error Tabular** | 19 | 0.0173 | 0.8613 | 0.6396 (0.6064..0.6718) | 0.9417 | 98.87% | 44.26% | 100.00% (100.0%..100.0%) | Rs. 922,323.79 | YES |
| **No-Accounting-Error + Graph** | 31 | 0.0173 | 0.7029 | 0.9140 (0.8918..0.9311) | 0.9870 | 90.35% | 2.46% | 83.27% (78.6%..86.6%) | Rs. 2,155,273.31 | NO |
| **Tabular Only (23 cols)** | 21 | 0.0173 | 0.8811 | 0.6516 (0.6163..0.6855) | 0.9415 | 98.39% | 43.42% | 98.86% (97.4%..100.0%) | Rs. 1,222,592.62 | YES |
| **Tabular + Graph (33 cols)** | 33 | 0.0173 | 0.1783 | 0.9607 (0.9459..0.9753) | 0.9878 | 97.59% | 1.77% | 100.00% (100.0%..100.0%) | Rs. 383,278.67 | YES |
