# Full Feature Group Ablation & Impact Analysis (Clean Test Split)

Base Rate (Clean Test): **0.0173** (1.73%)
Baseline Costs:
- **No Model (Flag Nothing)**: Rs. 35,357,945.92
- **Flag Everything**: Rs. 1,766,600.00

| Model Variant | Feature Count | Base Rate | Validation T_F2 | PR-AUC (95% CI) | ROC-AUC | Overall Recall | Flag Rate | Ring-Fraud Recall (n=263) | Expected Cost (Rs.) | Beats Both Baselines? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Model (Flag Nothing)** | 0 | 0.0173 | N/A | N/A | N/A | 0.00% | 0.00% | N/A | Rs. 35,357,945.92 | N/A |
| **Flag Everything** | 0 | 0.0173 | 0.0000 | 0.0173 | 0.5000 | 100.00% | 100.00% | 100.00% | Rs. 1,766,600.00 | N/A |
| **Raw Numeric Only** | 5 | 0.0173 | 0.7114 | 0.5139 (0.4757..0.5519) | 0.7812 | 100.00% | 98.88% | 100.00% (100.0%..100.0%) | Rs. 1,746,550.00 | YES |
| **Raw + One-Hot Type** | 10 | 0.0173 | 0.7650 | 0.5570 (0.5219..0.5967) | 0.9358 | 98.71% | 43.63% | 99.62% (98.8%..100.0%) | Rs. 912,774.63 | YES |
| **No-Accounting-Error Tabular** | 19 | 0.0173 | 0.8456 | 0.6362 (0.6011..0.6700) | 0.9394 | 98.71% | 44.21% | 99.62% (98.8%..100.0%) | Rs. 923,324.63 | YES |
| **No-Accounting-Error + Graph** | 31 | 0.0173 | 0.6644 | 0.9080 (0.8820..0.9271) | 0.9874 | 92.93% | 2.99% | 88.97% (85.1%..92.6%) | Rs. 1,379,548.39 | YES |
| **Tabular Only (23 cols)** | 21 | 0.0173 | 0.8858 | 0.6474 (0.6127..0.6797) | 0.9405 | 98.87% | 44.66% | 100.00% (100.0%..100.0%) | Rs. 929,623.79 | YES |
| **Tabular + Graph (33 cols)** | 33 | 0.0173 | 0.4094 | 0.9577 (0.9405..0.9740) | 0.9876 | 97.59% | 2.32% | 100.00% (100.0%..100.0%) | Rs. 393,128.67 | YES |
