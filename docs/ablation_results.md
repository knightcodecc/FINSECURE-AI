# Ablation Study: Graph Feature Impact Analysis (Test Split)

Evaluated on the held-out test split using fixed decision thresholds tuned exclusively on the validation split (F2-Optimal Threshold = 0.0694).

| Model Variant | Feature Count | PR-AUC | ROC-AUC | Precision | Recall | F2 Score | Expected Cost (Rs.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Tabular Only)** | 23 | 0.0655 | 0.8270 | 0.0000 | 0.0000 | 0.0000 | Rs. 4,332.79 |
| **Model B (Tabular + Graph)** | **33** | **0.0686** | **0.8354** | **0.0000** | **0.0000** | **0.0000** | **Rs. 4,032.79** |

## Key Findings
- **PR-AUC Improvement**: Baseline PR-AUC 0.0655 vs. Tabular+Graph 0.0686.
- **Leakage Prevention**: Evaluated strictly on the held-out test split using fixed validation-tuned thresholds.
