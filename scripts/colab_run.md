# Google Colab High-RAM Staged Execution Plan for FinSecure AI v2

This document provides the exact staged execution plan to run FinSecure AI v2 on Google Colab High-RAM instances using environment variables `DATA_PATH` and `SAMPLE_ROWS`.

---

## 💻 Recommended Colab Setup

```python
# 1. Mount Google Drive to persist artifacts between stages
from google.colab import drive
drive.mount('/content/drive')

# 2. Set up working directory and install requirements
!git clone https://github.com/YOUR_ORGANIZATION/finsecure-ai-v2.git
%cd finsecure-ai-v2
!pip install -r requirements.txt

# 3. Create output directory in Drive
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_200k
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_1M
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_full
```

---

## 🚀 STAGED EXECUTION PIPELINE

The execution is divided into three incremental stages to validate correctness, performance, and resource usage before running on the full 6.36M dataset.

### ──────── Stage 1: 200k Rows Validation Run ────────

```bash
export DATA_PATH="/content/drive/MyDrive/paysim.csv"
export SAMPLE_ROWS="200000"

python -c "
import time, psutil, os
def run_stage(name, cmd):
    t0 = time.time()
    os.system(cmd)
    t1 = time.time()
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f'[{name}] Time: {t1-t0:.2f}s | Peak RSS RAM: {ram:.2f} MB')

run_stage('Preprocess', 'python src/preprocess.py')
run_stage('Graph Features', 'python src/graph_features.py')
run_stage('Feature Engineering', 'python src/feature_engineering.py')
run_stage('Model Training & Ablation', 'python src/train_model.py')
run_stage('Real Data Report', 'python scripts/real_data_report.py')
"

# Save Stage 1 artifacts to Google Drive
!cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_200k/
```

---

### ──────── Stage 2: 1,000,000 Rows Scale Run ────────

```bash
export DATA_PATH="/content/drive/MyDrive/paysim.csv"
export SAMPLE_ROWS="1000000"

python -c "
import time, psutil, os
def run_stage(name, cmd):
    t0 = time.time()
    os.system(cmd)
    t1 = time.time()
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f'[{name}] Time: {t1-t0:.2f}s | Peak RSS RAM: {ram:.2f} MB')

run_stage('Preprocess', 'python src/preprocess.py')
run_stage('Graph Features', 'python src/graph_features.py')
run_stage('Feature Engineering', 'python src/feature_engineering.py')
run_stage('Model Training & Ablation', 'python src/train_model.py')
run_stage('Real Data Report', 'python scripts/real_data_report.py')
"

# Save Stage 2 artifacts to Google Drive
!cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_1M/
```

---

### ──────── Stage 3: Full 6,362,620 Rows Final Run ────────

```bash
export DATA_PATH="/content/drive/MyDrive/paysim.csv"
unset SAMPLE_ROWS

python -c "
import time, psutil, os
def run_stage(name, cmd):
    t0 = time.time()
    os.system(cmd)
    t1 = time.time()
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f'[{name}] Time: {t1-t0:.2f}s | Peak RSS RAM: {ram:.2f} MB')

run_stage('Preprocess', 'python src/preprocess.py')
run_stage('Graph Features', 'python src/graph_features.py')
run_stage('Feature Engineering', 'python src/feature_engineering.py')
run_stage('Model Training & Ablation', 'python src/train_model.py')
run_stage('Real Data Report', 'python scripts/real_data_report.py')
"

# Save Stage 3 final artifacts to Google Drive
!cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_full/
```

---

## ⏱️ Memory and Runtime Resource Guidelines

| Stage | Input Rows | Est. Runtime | Peak RAM | Drive Save Location |
|---|:---:|:---:|:---:|---|
| **Stage 1** | 200,000 | ~30 seconds | ~0.5 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_200k/` |
| **Stage 2** | 1,000,000 | ~2.5 minutes | ~1.2 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_1M/` |
| **Stage 3** | 6,362,620 | ~12.0 minutes | ~2.0 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_full/` |
