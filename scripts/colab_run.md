# Google Colab High-RAM Staged Execution Plan for FinSecure AI v2

This document provides the complete, working staged execution plan to run FinSecure AI v2 on Google Colab High-RAM instances.

> **Important Note on Stages**:
> - **Stage 1 (200k rows)** and **Stage 2 (1M rows)** serve as **smoke tests** to verify correctness, timing, and memory allocation before full execution.
> - **Stage 3 (Full 6,362,620 rows)** is the primary production execution run that generates final deliverable thesis artifacts.

---

## 💻 CELL 1: Environment Initialization & Drive Mounting

```python
# Mount Google Drive to persist artifacts
from google.colab import drive
drive.mount('/content/drive')

# Create destination Drive directories FIRST to avoid file copy failures
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_200k
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_1M
!mkdir -p /content/drive/MyDrive/finsecure_artifacts/stage_full
!mkdir -p /content/drive/MyDrive/paysim_data

# Clone repository and install requirements
!git clone https://github.com/YOUR_ORGANIZATION/finsecure-ai-v2.git /content/finsecure-ai-v2
%cd /content/finsecure-ai-v2
!pip install -r requirements.txt
```

---

## ⚡ CELL 2: Stage 1 — 200,000 Rows Smoke Test

```bash
%%bash
set -e
cd /content/finsecure-ai-v2

# Copy dataset locally for fast I/O
cp /content/drive/MyDrive/paysim_data/paysim.csv /content/paysim_local.csv

export DATA_PATH="/content/paysim_local.csv"
export SAMPLE_ROWS="200000"
export MAX_GRAPH_RSS_GB="20.0"

mkdir -p /content/finsecure-ai-v2/reports/logs/colab_stage_200k

echo "=== STAGE 1: 200K ROWS SMOKE TEST ==="
/usr/bin/time -v python src/graph_features.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_200k/graph_features.log
/usr/bin/time -v python src/preprocess.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_200k/preprocess.log
/usr/bin/time -v python src/feature_engineering.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_200k/feature_engineering.log
/usr/bin/time -v python src/train_model.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_200k/train_model.log
/usr/bin/time -v python scripts/real_data_report.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_200k/real_data_report.log

# Copy stage 1 artifacts to Google Drive
cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_200k/
```

---

## ⚡ CELL 3: Stage 2 — 1,000,000 Rows Scale Smoke Test

```bash
%%bash
set -e
cd /content/finsecure-ai-v2

export DATA_PATH="/content/paysim_local.csv"
export SAMPLE_ROWS="1000000"
export MAX_GRAPH_RSS_GB="20.0"

mkdir -p /content/finsecure-ai-v2/reports/logs/colab_stage_1M

echo "=== STAGE 2: 1M ROWS SMOKE TEST ==="
/usr/bin/time -v python src/graph_features.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_1M/graph_features.log
/usr/bin/time -v python src/preprocess.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_1M/preprocess.log
/usr/bin/time -v python src/feature_engineering.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_1M/feature_engineering.log
/usr/bin/time -v python src/train_model.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_1M/train_model.log
/usr/bin/time -v python scripts/real_data_report.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_1M/real_data_report.log

# Copy stage 2 artifacts to Google Drive
cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_1M/
```

---

## ⚡ CELL 4: Stage 3 — Full 6,362,620 Rows Final Run

```bash
%%bash
set -e
cd /content/finsecure-ai-v2

export DATA_PATH="/content/paysim_local.csv"
unset SAMPLE_ROWS
export MAX_GRAPH_RSS_GB="20.0"

mkdir -p /content/finsecure-ai-v2/reports/logs/colab_stage_full

echo "=== STAGE 3: FULL 6.36M ROWS PRODUCTION RUN ==="
/usr/bin/time -v python src/graph_features.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_full/graph_features.log
/usr/bin/time -v python src/preprocess.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_full/preprocess.log
/usr/bin/time -v python src/feature_engineering.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_full/feature_engineering.log
/usr/bin/time -v python src/train_model.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_full/train_model.log
/usr/bin/time -v python scripts/real_data_report.py 2>&1 | tee /content/finsecure-ai-v2/reports/logs/colab_stage_full/real_data_report.log

# Copy stage 3 final artifacts to Google Drive
cp -r models/ reports/ data/*.parquet /content/drive/MyDrive/finsecure_artifacts/stage_full/
```

---

## 📊 Expected Runtime & Resource Benchmark

| Stage | Input Rows | Est. Runtime | Peak RSS RAM | Drive Destination |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 1 (Smoke)** | 200,000 | ~30 seconds | ~0.5 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_200k/` |
| **Stage 2 (Smoke)** | 1,000,000 | ~2.5 minutes | ~1.2 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_1M/` |
| **Stage 3 (Full)** | 6,362,620 | ~12.0 minutes | ~2.0 GB | `/content/drive/MyDrive/finsecure_artifacts/stage_full/` |
