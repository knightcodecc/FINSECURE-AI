# Full Feature Group Ablation & Impact Analysis (Clean Test Split)

Base Rate (Clean Test): **0.0173** (1.73%)
Baseline Costs:
- **No Model (Flag Nothing)**: Rs. 35,357,945.92
- **Flag Everything**: Rs. 1,766,600.00

| Model Variant | Feature Count | Base Rate | Validation T_F2 | PR-AUC (95% CI) | ROC-AUC | Overall Recall | Flag Rate | Ring-Fraud Recall (n=263) | Expected Cost (Rs.) | Beats Both Baselines? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Model (Flag Nothing)** | 0 | 0.0173 | N/A | N/A | N/A | 0.00% | 0.00% | N/A | Rs. 35,357,945.92 | N/A |
| **Flag Everything** | 0 | 0.0173 | 0.0000 | 0.0173 | 0.5000 | 100.00% | 100.00% | 100.00% | Rs. 1,766,600.00 | N/A |
| **Raw Numeric Only** | 5 | 0.0173 | 0.7248 | 0.5135 (0.4767..0.5501) | 0.7741 | 99.84% | 98.18% | 99.62% (99.0%..100.0%) | Rs. 1,744,369.50 | YES |
| **Raw + One-Hot Type** | 10 | 0.0173 | 0.8053 | 0.5518 (0.5138..0.5918) | 0.9351 | 98.55% | 43.14% | 99.62% (98.8%..100.0%) | Rs. 909,857.93 | YES |
| **No-Accounting-Error Tabular** | 19 | 0.0173 | 0.8657 | 0.6396 (0.6064..0.6718) | 0.9417 | 98.87% | 44.85% | 100.00% (100.0%..100.0%) | Rs. 933,073.79 | YES |
| **No-Accounting-Error + Graph** | 31 | 0.0173 | 0.6979 | 0.9140 (0.8918..0.9311) | 0.9870 | 98.07% | 7.98% | 100.00% (100.0%..100.0%) | Rs. 343,845.98 | YES |
| **Tabular Only (23 cols)** | 21 | 0.0173 | 0.8858 | 0.6516 (0.6163..0.6855) | 0.9415 | 98.71% | 44.07% | 99.62% (98.8%..100.0%) | Rs. 920,724.63 | YES |
| **Tabular + Graph (33 cols)** | 33 | 0.0173 | 0.2014 | 0.9607 (0.9459..0.9753) | 0.9878 | 97.59% | 5.47% | 100.00% (100.0%..100.0%) | Rs. 449,778.67 | YES |
