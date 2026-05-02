"""
Fraud Detection — Real-Time Scoring API
========================================
FastAPI service that scores a single transaction in < 100 ms.

Architecture
------------
At startup the server loads three artefacts into memory:
  1. scaler.pkl          — RobustScaler fitted on training normals
  2. xgb_model.pkl       — Final XGBoost classifier
  3. ensemble.pkl        — LightGBM + meta-learner bundle

At inference time:
  1. Raw transaction JSON is validated by Pydantic (< 1 ms)
  2. Tabular features are built deterministically (no DB calls)
  3. Graph features are approximated from lightweight lookup tables
     that are persisted at training time (< 2 ms)
  4. XGBoost + LightGBM score the 40-dim feature vector in parallel
  5. Meta-learner blends scores → final fraud probability (< 5 ms)

Total end-to-end latency target: < 100 ms (typically 5–20 ms).

Run
---
    # From repo root
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Or with auto-reload during development
    uvicorn api.app:app --reload --port 8000

Endpoints
---------
POST /score          — score a single transaction
POST /score/batch    — score up to 256 transactions
GET  /health         — liveness probe
GET  /ready          — readiness probe (fails if models not loaded)
GET  /metrics        — latency histogram + request counters
GET  /docs           — Swagger UI (auto-generated)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ── Paths (resolved from env or defaults relative to repo root) ───────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", _REPO_ROOT / "models"))
PROC_DIR   = Path(os.getenv("PROC_DIR",   _REPO_ROOT / "data" / "processed"))

# ── Feature constants (must match training) ────────────────────────────────────
N_V_FEATURES = 28
ENGINEERED   = ["log_amount", "time_delta", "rolling_amount_mean_3", "amount_vs_mean"]
V_FEATURES   = [f"V{i}" for i in range(1, N_V_FEATURES + 1)]
TAB_FEATURES = V_FEATURES + ENGINEERED          # 32 tabular features
GRAPH_FEATURES = [                               # 7 graph features
    "g_card_degree", "g_card_weighted_degree", "g_clustering",
    "g_pagerank", "g_avg_amount", "g_amount_var", "g_merchant_diversity",
]
ALL_FEATURES = TAB_FEATURES + GRAPH_FEATURES    # 39 total


# ── In-memory model store ──────────────────────────────────────────────────────
class ModelStore:
    """Holds all artefacts loaded at startup."""

    scaler: Any          = None   # sklearn RobustScaler
    xgb_model: Any       = None   # xgb.XGBClassifier
    lgb_model: Any       = None   # lgb.LGBMClassifier
    meta_learner: Any    = None   # sklearn LogisticRegression
    scaler_xgb: Any      = None   # MinMaxScaler for XGBoost scores
    scaler_lgb: Any      = None   # MinMaxScaler for LightGBM scores
    graph_lookup: dict   = {}     # card_id → {feature: value}
    sequence_meta: dict  = {}
    ready: bool          = False


store = ModelStore()


# ── Metrics store ─────────────────────────────────────────────────────────────
class Metrics:
    total_requests: int       = 0
    total_errors: int         = 0
    latencies_ms: list[float] = []   # last 1000 requests

    def record(self, latency_ms: float, error: bool = False) -> None:
        self.total_requests += 1
        if error:
            self.total_errors += 1
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 1000:
            self.latencies_ms.pop(0)

    def summary(self) -> dict:
        lats = self.latencies_ms
        if not lats:
            return {"total_requests": 0, "total_errors": 0}
        arr = np.array(lats)
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "latency_ms": {
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "mean": round(float(arr.mean()), 2),
                "max": round(float(arr.max()), 2),
            },
        }


metrics = Metrics()


# ── Startup / shutdown ─────────────────────────────────────────────────────────
def _load_pickle(path: Path, label: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}. Run the training pipeline first.")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    logger.info("  ✓ Loaded %s  (%s)", label, path.name)
    return obj


def _load_graph_lookup() -> dict:
    """Load per-card graph feature lookup from processed artefacts.

    Falls back to an empty dict if graph features haven't been computed yet
    (inference will fill those slots with zeros, i.e. 'unseen card' defaults).
    """
    lookup: dict[int, dict[str, float]] = {}
    train_path = PROC_DIR / "graph_features_train.npy"
    tx_path    = PROC_DIR / "graph_tx_ids_train.npy"
    meta_path  = PROC_DIR / "graph_meta.json"

    if not all(p.exists() for p in [train_path, tx_path, meta_path]):
        logger.warning("Graph lookup artefacts missing — graph features will default to 0.")
        return lookup

    with open(meta_path) as fh:
        gph_meta = json.load(fh)
    gph_feats = gph_meta["graph_features"]

    X_g  = np.load(train_path)
    tx_g = np.load(tx_path)

    # We need card_id → features, not txid → features.
    # The graph artefacts are indexed by transaction; we aggregate per card
    # by loading the raw mapping saved at feature-engineering time.
    card_map_path = PROC_DIR / "card_graph_lookup.json"
    if card_map_path.exists():
        with open(card_map_path) as fh:
            raw = json.load(fh)
        # Keys are string card_ids; convert back to int
        for card_str, feat_dict in raw.items():
            lookup[int(card_str)] = feat_dict
        logger.info("  ✓ Loaded card graph lookup (%s cards)", f"{len(lookup):,}")
    else:
        logger.warning(
            "card_graph_lookup.json not found — graph features will default to 0. "
            "Re-run the pipeline with graph stage to generate it."
        )
    return lookup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models into memory before accepting traffic."""
    logger.info("=" * 55)
    logger.info("Loading fraud detection models ...")
    t0 = time.perf_counter()

    try:
        store.scaler    = _load_pickle(PROC_DIR   / "scaler.pkl",    "RobustScaler")
        store.xgb_model = _load_pickle(MODELS_DIR / "xgb_model.pkl", "XGBoost")

        ens_bundle      = _load_pickle(MODELS_DIR / "ensemble.pkl",  "Ensemble bundle")
        store.lgb_model    = ens_bundle["lgb_model"]
        store.meta_learner = ens_bundle["meta_learner"]
        store.scaler_xgb   = ens_bundle["score_scaler_xgb"]
        store.scaler_lgb   = ens_bundle["score_scaler_lgb"]

        with open(PROC_DIR / "sequence_meta.json") as fh:
            store.sequence_meta = json.load(fh)

        store.graph_lookup = _load_graph_lookup()
        store.ready = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Models ready in %.0f ms", elapsed)
        logger.info("=" * 55)

    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        # Still yield so /health works; /ready will return 503
        store.ready = False

    yield  # ← server is live

    logger.info("Shutting down — releasing model memory.")
    store.ready = False


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time transaction scoring using an XGBoost + LightGBM ensemble "
        "with graph-based card features. Target latency < 100 ms."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Latency middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def track_latency(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        latency = (time.perf_counter() - t0) * 1000
        response.headers["X-Latency-Ms"] = f"{latency:.2f}"
        if request.url.path in ("/score", "/score/batch"):
            metrics.record(latency, error=response.status_code >= 400)
        return response
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        metrics.record(latency, error=True)
        raise


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """A single raw transaction, mirroring the IEEE-CIS schema."""

    # Required identifiers
    TransactionID: int = Field(..., description="Unique transaction ID", example=2987004)
    TransactionDT: float = Field(..., description="Transaction timestamp (seconds offset)", example=86400.0)
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD", example=117.5)

    # Card features
    card1: int | None = Field(None, description="Payment card ID", example=13926)
    card4: str | None = Field(None, description="Card network (visa/mastercard/…)", example="visa")
    card6: str | None = Field(None, description="Card type (debit/credit)", example="debit")
    ProductCD: str | None = Field(None, description="Product code", example="W")
    P_emaildomain: str | None = Field(None, description="Purchaser email domain", example="gmail.com")

    # V-features (Vesta proprietary, V1–V28 used)
    V1:  float | None = None; V2:  float | None = None; V3:  float | None = None
    V4:  float | None = None; V5:  float | None = None; V6:  float | None = None
    V7:  float | None = None; V8:  float | None = None; V9:  float | None = None
    V10: float | None = None; V11: float | None = None; V12: float | None = None
    V13: float | None = None; V14: float | None = None; V15: float | None = None
    V16: float | None = None; V17: float | None = None; V18: float | None = None
    V19: float | None = None; V20: float | None = None; V21: float | None = None
    V22: float | None = None; V23: float | None = None; V24: float | None = None
    V25: float | None = None; V26: float | None = None; V27: float | None = None
    V28: float | None = None

    # Context (optional — used for graph features if available)
    card1_prev_time: float | None = Field(None, description="Timestamp of card's previous transaction")
    card1_prev_log_amount: float | None = Field(None, description="log(1+amount) of card's previous transaction")
    card1_rolling_mean_3: float | None = Field(None, description="Rolling 3-tx log-amount mean for this card")

    @field_validator("TransactionAmt")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TransactionAmt must be > 0")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "TransactionID": 2987004,
            "TransactionDT": 86400.0,
            "TransactionAmt": 117.5,
            "card1": 13926,
            "card4": "visa",
            "card6": "debit",
            "ProductCD": "W",
            "V1": 1.0, "V2": 1.0, "V3": 1.0, "V4": 1.0,
            "V5": 0.0, "V6": 0.0, "V7": 0.0,
        }
    }}


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_flag: bool
    threshold_used: float
    model_version: str
    latency_ms: float
    feature_coverage: dict[str, float]  # fraction of features that were non-zero


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=256)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    batch_size: int
    total_latency_ms: float
    mean_latency_ms: float


# ── Feature engineering (inference-time, no leakage) ─────────────────────────

def _build_feature_vector(tx: TransactionRequest) -> tuple[np.ndarray, dict[str, float]]:
    """Convert a raw TransactionRequest into the 39-dim feature vector.

    Returns
    -------
    x      : (1, 39) float32 array, scaled and clipped
    coverage : dict reporting what fraction of each feature group was present
    """
    # ── 1. V-features ──────────────────────────────────────────────────
    v_vals = np.array(
        [getattr(tx, f"V{i}", None) or 0.0 for i in range(1, N_V_FEATURES + 1)],
        dtype=np.float32,
    )
    v_present = float((v_vals != 0).sum()) / N_V_FEATURES

    # ── 2. Engineered tabular features ────────────────────────────────
    log_amount = float(np.log1p(tx.TransactionAmt))

    # time_delta: use caller-provided previous timestamp if available
    if tx.card1_prev_time is not None:
        time_delta = max(0.0, float(tx.TransactionDT - tx.card1_prev_time))
    else:
        time_delta = 0.0  # unknown history → neutral

    # rolling mean: use caller-provided value or fall back to current
    if tx.card1_rolling_mean_3 is not None:
        rolling_mean = float(tx.card1_rolling_mean_3)
    elif tx.card1_prev_log_amount is not None:
        rolling_mean = (log_amount + float(tx.card1_prev_log_amount)) / 2.0
    else:
        rolling_mean = log_amount

    amount_vs_mean = log_amount - rolling_mean

    tab_vals = np.concatenate([v_vals, [log_amount, time_delta, rolling_mean, amount_vs_mean]])

    # ── 3. Apply scaler ────────────────────────────────────────────────
    tab_scaled = store.scaler.transform(tab_vals.reshape(1, -1)).astype(np.float32)
    tab_clipped = np.clip(tab_scaled, -5.0, 5.0)

    # ── 4. Graph features ──────────────────────────────────────────────
    card_id = int(tx.card1) if tx.card1 is not None else -1
    if card_id in store.graph_lookup:
        gph_dict = store.graph_lookup[card_id]
        gph_vals = np.array(
            [gph_dict.get(f, 0.0) for f in GRAPH_FEATURES], dtype=np.float32
        )
        gph_present = 1.0
    else:
        # Unseen card — zero-fill (model was trained to handle this)
        gph_vals = np.zeros(len(GRAPH_FEATURES), dtype=np.float32)
        gph_present = 0.0

    # ── 5. Concatenate ────────────────────────────────────────────────
    x = np.hstack([tab_clipped.ravel(), gph_vals]).reshape(1, -1).astype(np.float32)

    coverage = {
        "v_features": round(v_present, 3),
        "graph_features": round(gph_present, 3),
    }
    return x, coverage


def _ensemble_score(x: np.ndarray) -> tuple[float, float, float]:
    """Return (xgb_prob, lgb_prob, ensemble_prob)."""
    xgb_prob = float(store.xgb_model.predict_proba(x)[0, 1])
    lgb_prob = float(store.lgb_model.predict_proba(x)[0, 1])

    # Normalise using training-time MinMaxScalers
    xn = float(store.scaler_xgb.transform([[xgb_prob]])[0, 0])
    ln = float(store.scaler_lgb.transform([[lgb_prob]])[0, 0])

    # Meta-learner features (must match training construction)
    meta_x = np.array([[xn, ln, xn * ln, abs(xn - ln), max(xn, ln)]])
    ens_prob = float(store.meta_learner.predict_proba(meta_x)[0, 1])

    return xgb_prob, lgb_prob, ens_prob


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
def health() -> dict:
    """Liveness probe — always 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"])
def ready() -> dict:
    """Readiness probe — 503 if models are not loaded."""
    if not store.ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"status": "ready", "models_loaded": True}


@app.get("/metrics", tags=["Ops"])
def get_metrics() -> dict:
    """Latency histogram and request counters (last 1000 requests)."""
    return metrics.summary()


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_transaction(tx: TransactionRequest, threshold: float = 0.5) -> ScoreResponse:
    """Score a single transaction and return a fraud probability.

    - **fraud_probability**: ensemble model output in [0, 1]
    - **fraud_flag**: True if probability ≥ threshold
    - **latency_ms**: server-side inference time

    Typical latency: 5–20 ms.
    """
    if not store.ready:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    t0 = time.perf_counter()
    try:
        x, coverage = _build_feature_vector(tx)
        xgb_prob, lgb_prob, ens_prob = _ensemble_score(x)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "TxID=%s  amt=%.2f  p_fraud=%.4f  flag=%s  %.1f ms",
            tx.TransactionID, tx.TransactionAmt, ens_prob,
            ens_prob >= threshold, latency_ms,
        )

        return ScoreResponse(
            transaction_id=tx.TransactionID,
            fraud_probability=round(ens_prob, 6),
            fraud_flag=ens_prob >= threshold,
            threshold_used=threshold,
            model_version="1.0.0",
            latency_ms=round(latency_ms, 2),
            feature_coverage=coverage,
        )

    except Exception as exc:
        logger.error("Scoring error for TxID=%s: %s", tx.TransactionID, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
def score_batch(request: BatchRequest) -> BatchResponse:
    """Score up to 256 transactions in one call.

    Transactions are scored independently (no sequence context across batch items).
    """
    if not store.ready:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    t0 = time.perf_counter()
    results: list[ScoreResponse] = []

    # Vectorised path: build all feature vectors at once
    try:
        xs, coverages = zip(*[_build_feature_vector(tx) for tx in request.transactions])
        X_batch = np.vstack(xs).astype(np.float32)

        xgb_probs = store.xgb_model.predict_proba(X_batch)[:, 1]
        lgb_probs = store.lgb_model.predict_proba(X_batch)[:, 1]

        xn = store.scaler_xgb.transform(xgb_probs.reshape(-1, 1)).ravel()
        ln = store.scaler_lgb.transform(lgb_probs.reshape(-1, 1)).ravel()
        meta_X = np.column_stack([xn, ln, xn * ln, np.abs(xn - ln), np.maximum(xn, ln)])
        ens_probs = store.meta_learner.predict_proba(meta_X)[:, 1]

        total_ms = (time.perf_counter() - t0) * 1000
        per_ms = total_ms / len(request.transactions)

        for i, tx in enumerate(request.transactions):
            results.append(ScoreResponse(
                transaction_id=tx.TransactionID,
                fraud_probability=round(float(ens_probs[i]), 6),
                fraud_flag=float(ens_probs[i]) >= request.threshold,
                threshold_used=request.threshold,
                model_version="1.0.0",
                latency_ms=round(per_ms, 2),
                feature_coverage=coverages[i],
            ))

        logger.info(
            "Batch %s txns  total=%.1f ms  per_tx=%.1f ms  flagged=%s",
            len(request.transactions), total_ms, per_ms,
            sum(1 for r in results if r.fraud_flag),
        )

        return BatchResponse(
            results=results,
            batch_size=len(results),
            total_latency_ms=round(total_ms, 2),
            mean_latency_ms=round(per_ms, 2),
        )

    except Exception as exc:
        logger.error("Batch scoring error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {exc}")


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True, workers=1)
