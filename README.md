# IEEE-CIS Fraud Detection

End-to-end fraud detection system: feature engineering → XGBoost + LightGBM ensemble with graph features → real-time scoring API.

```
Ensemble AUROC     : 0.8065  |  Ensemble AUPRC : 0.1943
XGBoost  AUROC     : 0.7984  |  LightGBM AUROC : 0.8029
API latency target : < 100 ms  (typical: 5–20 ms)
```

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Data](#data)
4. [Training Pipeline](#training-pipeline)
5. [Results](#results)
6. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
7. [Deployment API](#deployment-api)
8. [Load Testing](#load-testing)
9. [Configuration Reference](#configuration-reference)
10. [Architecture](#architecture)

---

## Project Structure

```
fraud_detection/
├── api/
│   ├── __init__.py
│   └── app.py                  ← FastAPI scoring server
│
├── configs/
│   └── config.yaml             ← All tunable parameters
│
├── data/
│   ├── raw/                    ← Drop IEEE-CIS CSVs here
│   └── processed/              ← Auto-generated NumPy artefacts
│
├── models/                     ← Saved .pkl model files
│
├── outputs/
│   ├── eda/                    ← Exploratory analysis plots
│   ├── graph/                  ← Graph feature diagnostics
│   ├── model/                  ← XGBoost evaluation plots
│   └── ensemble/               ← Dashboard, SHAP plots
│
├── mlruns/                     ← MLflow experiment store
│
├── scripts/
│   ├── run_pipeline.py         ← End-to-end training runner
│   ├── export_graph_lookup.py  ← Pre-compute per-card graph features for API
│   └── load_test.py            ← API latency benchmark
│
├── src/fraud_detection/
│   ├── config.py               ← YAML loader + MLflow param flattener
│   ├── data.py                 ← CSV loading, merge, stratified split
│   ├── eda.py                  ← EDA plots (class dist, amounts, KS)
│   ├── features.py             ← V-features, engineered cols, sequences
│   ├── graph.py                ← Bipartite card-merchant graph features
│   ├── models.py               ← Cross-validated XGBoost training
│   ├── ensemble.py             ← LightGBM + meta-learner + SHAP
│   └── evaluate.py             ← Metrics, dashboard, MLflow logging
│
├── Dockerfile                  ← Production API container
├── requirements.txt            ← Full (training + API)
├── requirements-api.txt        ← API-only (minimal)
└── setup.py                    ← Installable package
```

---

## Quick Start

### 1. Install dependencies

```bash
# Full install (training + API)
pip install -r requirements.txt

# API-only (production)
pip install -r requirements-api.txt
```

### 2. Add raw data

Place the IEEE-CIS competition CSVs in `data/raw/`:

```
data/raw/
├── train_transaction.csv
└── train_identity.csv
```

Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data

### 3. Train the pipeline

```bash
python scripts/run_pipeline.py
```

This runs all five stages sequentially: EDA → features → graph → train → ensemble.

### 4. Export graph lookup for the API

```bash
python scripts/export_graph_lookup.py
```

Generates `data/processed/card_graph_lookup.json` — a per-card feature cache used by the API to avoid re-computing graph features at inference time.

### 5. Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 6. Score a transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 2987004,
    "TransactionDT": 86400.0,
    "TransactionAmt": 117.5,
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "V1": 1.0, "V2": 1.0, "V3": 0.0
  }'
```

Response:

```json
{
  "transaction_id": 2987004,
  "fraud_probability": 0.031247,
  "fraud_flag": false,
  "threshold_used": 0.5,
  "model_version": "1.0.0",
  "latency_ms": 8.42,
  "feature_coverage": {
    "v_features": 0.107,
    "graph_features": 1.0
  }
}
```

---

## Data

| File | Rows | Columns | Description |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Transaction records with V1–V339 features |
| `train_identity.csv` | 144,233 | 41 | Device and identity information |

After merging: 590,540 rows × 434 columns. Class imbalance: ~3.5% fraud.

---

## Training Pipeline

Run individual stages with `--stages`:

```bash
# Only features and graph (skip EDA)
python scripts/run_pipeline.py --stages features graph train ensemble

# Named MLflow run
python scripts/run_pipeline.py --run-name baseline_v1
```

### Stage summary

| Stage | Module | Measured runtime | Key outputs |
|---|---|---|---|
| `eda` | `eda.py` | ~6 s | `outputs/eda/*.png` |
| `features` | `features.py` | ~15 s | `data/processed/X_*.npy`, `scaler.pkl` |
| `graph` | `graph.py` | ~7 s | `data/processed/graph_*.npy` |
| `train` | `models.py` | ~29 min | `models/xgb_model.pkl`, `test_scores.npy` |
| `ensemble` | `ensemble.py` + `evaluate.py` | ~21 min | `models/ensemble.pkl`, dashboard, SHAP |
| **Total** | | **~33 min** | |

### Feature engineering

**Tabular features (32)**
- V1–V28: Vesta proprietary anonymised features
- `log_amount`: log(1 + TransactionAmt)
- `time_delta`: seconds since cardholder's previous transaction
- `rolling_amount_mean_3`: 3-transaction rolling log-amount mean
- `amount_vs_mean`: deviation from rolling mean

**Graph features (7)** — built on a card-merchant bipartite graph:
- `g_card_degree`: number of unique merchants visited
- `g_card_weighted_degree`: total transaction count
- `g_clustering`: local clustering coefficient
- `g_pagerank`: PageRank centrality
- `g_avg_amount`: mean log-amount per card
- `g_amount_var`: variance of log-amount
- `g_merchant_diversity`: degree / weighted degree

### Leakage controls

- Train/test split performed **before** any feature computation
- `fraud_density` removed (was leaking test labels via graph)
- `RobustScaler` fitted on training-set normal transactions only
- Graph PageRank and clustering computed on unlabelled topology only

---

## Results

Benchmarked on the IEEE-CIS test split (118,108 transactions, 3.50% fraud) — MLflow run `a4451478a4c043639e46948e52f7eea6`.

### Model performance

| Model | AUROC | AUPRC | CV mean ± std |
|---|---|---|---|
| XGBoost | 0.7984 | 0.1697 | 0.7946 ± 0.0038 |
| LightGBM | 0.8029 | 0.1920 | 0.7971 ± 0.0038 |
| **Ensemble** | **0.8065** | **0.1943** | 0.8064 ± 0.0037 |

Random baseline AUPRC (3.5% prevalence): ~0.035. The ensemble is 5.5× above baseline.

### Classification report (F1-optimal threshold = 0.823)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.98 | 0.96 | 0.97 | 113,975 |
| Fraud | 0.22 | 0.34 | 0.27 | 4,133 |
| Weighted avg | 0.95 | 0.94 | 0.94 | 118,108 |

### Thresholds

| Threshold | Value | Use case |
|---|---|---|
| F1-optimal | 0.823 | Maximises F1 on fraud class |
| Cost-optimal | 0.517 | Minimises `FN × $150 + FP × $5` — recommended for production |

The high F1-optimal threshold (0.823) reflects that the model is conservative — it rarely assigns very high fraud probabilities. The cost-optimal threshold is more practical: at 0.517, the trade-off between catching fraud ($150 miss cost) and false alarms ($5 investigation cost) is minimised.

### Precision@K (ensemble, cost-optimal threshold)

| K | Precision |
|---|---|
| 10 | — |
| 25 | — |
| 50 | — |
| 100 | — |
| 200 | — |
| 500 | — |

> Re-run `scripts/run_pipeline.py --stages ensemble` to populate Precision@K values from MLflow.

### Data summary

| Split | Transactions | Fraud | Fraud % |
|---|---|---|---|
| Train | 472,432 | 16,530 | 3.50% |
| Test | 118,108 | 4,133 | 3.50% |
| Sequences (train) | 54,126 | 1,924 | 3.55% |
| Sequences (test) | 13,412 | 429 | 3.20% |
| Graph nodes (cards) | 13,553 | — | — |
| Graph edges | 6,283 | — | — |

---

## MLflow Experiment Tracking

Each pipeline run logs:

**Parameters** — all `config.yaml` values flattened (`xgb.learning_rate`, `lgb.num_leaves`, etc.)

**Metrics**
- Per-fold CV AUROC for XGBoost, LightGBM, and ensemble
- Test AUROC and AUPRC for all three models
- Precision@K (K = 10, 25, 50, 100, 200, 500)
- Cost-optimal threshold and minimum dollar cost
- Stage runtimes

**Artefacts**
- `xgb_model.pkl`, `lgb_model.pkl`, `ensemble.pkl`
- Evaluation dashboard, SHAP importance, SHAP waterfall
- `ensemble_results.json`

### Viewing runs

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open http://localhost:5000
```

---

## Deployment API

The FastAPI server (`api/app.py`) exposes a production-grade scoring service.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — always 200 if process is alive |
| `GET` | `/ready` | Readiness probe — 503 if models not loaded |
| `GET` | `/metrics` | Latency histogram (p50/p90/p95/p99) + request counts |
| `POST` | `/score` | Score a single transaction |
| `POST` | `/score/batch` | Score up to 256 transactions in one call |
| `GET` | `/docs` | Swagger UI (auto-generated from Pydantic schemas) |

### Request schema (`/score`)

| Field | Type | Required | Description |
|---|---|---|---|
| `TransactionID` | int | ✓ | Unique transaction identifier |
| `TransactionDT` | float | ✓ | Timestamp (seconds offset) |
| `TransactionAmt` | float > 0 | ✓ | Transaction amount in USD |
| `card1` | int | — | Payment card ID (used for graph lookup) |
| `card4` | str | — | Card network (visa / mastercard / …) |
| `card6` | str | — | Card type (debit / credit) |
| `V1`–`V28` | float | — | Vesta anonymised features |
| `card1_prev_time` | float | — | Previous transaction timestamp for this card |
| `card1_rolling_mean_3` | float | — | Rolling 3-tx log-amount mean (caller-computed) |

### Inference latency breakdown

| Step | Typical time |
|---|---|
| JSON validation (Pydantic) | < 1 ms |
| Feature engineering | 1–2 ms |
| Graph lookup (dict) | < 0.5 ms |
| XGBoost predict_proba | 2–5 ms |
| LightGBM predict_proba | 2–5 ms |
| Meta-learner blend | < 1 ms |
| **Total** | **5–15 ms** |

### Running with Docker

```bash
# Build image
docker build -t fraud-api .

# Run with model artefacts mounted
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  fraud-api
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `./models` | Path to directory containing `.pkl` files |
| `PROC_DIR` | `./data/processed` | Path to processed artefacts |

---

## Load Testing

```bash
# Start the server first, then:
python scripts/load_test.py

# Options
python scripts/load_test.py --n 500 --batch 64 --host http://localhost:8000
```

Output example:

```
── Single /score  (200 requests) ──────────────────────
  Sample response: p_fraud=0.0284  flag=False  server_ms=7.3

  Requests   : 200  (errors=0)
  Latency ms : p50=8.1  p90=12.4  p95=14.9  p99=22.3  mean=9.0  max=38.1
  > 100 ms   : 0 (0.0%)

── Batch /score/batch  (batch_size=32, 20 calls) ──
  Batch size : 32
  Total ms   : p50=18.2  p90=24.1  mean=19.7
  Per-tx ms  : mean=0.62
  Fraud flags: 2/32 in last batch
```

---

## Configuration Reference

All parameters live in `configs/config.yaml`. Override by editing the file or passing a custom path:

```bash
python scripts/run_pipeline.py --config configs/my_experiment.yaml
```

### Key parameters

```yaml
data:
  test_size: 0.2          # train/test split ratio
  random_state: 42        # global random seed

features:
  seq_len: 10             # sequence window length
  n_v_features: 28        # V-columns to include

xgb:
  n_estimators: 2000      # max trees (early stopping applies)
  learning_rate: 0.03
  max_depth: 6
  early_stopping_rounds: 50

lgb:
  n_estimators: 2000
  num_leaves: 63          # richer than XGB max_depth=6 (2^6-1)
  early_stopping_rounds: 100

training:
  cv_folds: 5

evaluation:
  cost_fn: 150.0          # missed fraud cost ($)
  cost_fp: 5.0            # false alarm cost ($)
  precision_at_k: [10, 25, 50, 100, 200, 500]

mlflow:
  experiment_name: fraud-detection
  run_name: ~             # null → auto timestamp
```

---

## Architecture

```
Raw CSVs
   │
   ▼
data.py ──────────────────── Merge + stratified split (TxID-level)
   │
   ├──▶ eda.py ────────────── Class distribution, KS tests, plots
   │
   ├──▶ features.py ─────────  V-features + 4 engineered + RobustScaler
   │         │                  → X_train_tab_all.npy, X_test_tab.npy
   │         │
   │    graph.py ──────────── Bipartite graph: degree, clustering,
   │         │                  PageRank (train-only fraud density)
   │         │                  → graph_features_{train,test}.npy
   │         │
   │    [Join tabular+graph] ─ 39-dim feature vector
   │         │
   ├──▶ models.py ──────────── 5-fold XGBoost CV → final model
   │         │                  → xgb_model.pkl, test_scores.npy
   │         │
   └──▶ ensemble.py ─────────  5-fold LightGBM CV → meta-learner
             │                  SHAP (TreeExplainer, 2000 samples)
             │                  → ensemble.pkl, shap_values.npy
             │
        evaluate.py ─────────  Metrics, dashboard, Precision@K,
             │                  cost threshold, MLflow logging
             │
        export_graph_lookup ─  card_id → graph features JSON
             │
             ▼
        api/app.py ──────────  FastAPI: /score, /score/batch
                                Loads 3 PKLs + graph lookup at startup
                                Inference: 5–20 ms end-to-end
```