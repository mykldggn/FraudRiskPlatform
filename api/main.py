import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import FEATURE_COLUMNS, MODEL_PATH, SAMPLE_DATA_PATH
from fraud_risk_platform.data import generate_synthetic_transactions, load_transactions
from fraud_risk_platform.model import load_model, save_model, train_model
from fraud_risk_platform.scoring import score_transactions

app = FastAPI(title="Fraud Risk Platform API", version="0.1.0")


class Transaction(BaseModel):
    transaction_id: str = Field(default="api_txn")
    amount: float
    hour: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1)
    customer_tenure_days: int = Field(ge=1)
    transactions_last_24h: int = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    is_foreign: int = Field(ge=0, le=1)
    is_card_not_present: int = Field(ge=0, le=1)
    is_fraud: Optional[int] = None


class ScoreRequest(BaseModel):
    transaction: Transaction
    threshold: float = Field(default=0.5, ge=0, le=1)


class BatchScoreRequest(BaseModel):
    transactions: List[Transaction]
    threshold: float = Field(default=0.5, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/score")
def score_single(request: ScoreRequest):
    model = _get_model()
    df = pd.DataFrame([_dump_model(request.transaction)])
    scored = score_transactions(model, _ensure_target(df), threshold=request.threshold)
    return _response_rows(scored)[0]


@app.post("/score-batch")
def score_batch(request: BatchScoreRequest):
    model = _get_model()
    df = pd.DataFrame([_dump_model(transaction) for transaction in request.transactions])
    scored = score_transactions(model, _ensure_target(df), threshold=request.threshold)
    return {"count": len(scored), "results": _response_rows(scored)}


def _get_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    df = load_transactions(SAMPLE_DATA_PATH) if SAMPLE_DATA_PATH.exists() else generate_synthetic_transactions()
    model, _ = train_model(df)
    save_model(model, MODEL_PATH)
    return model


def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    return df[["transaction_id", *FEATURE_COLUMNS, "is_fraud"]]


def _response_rows(scored: pd.DataFrame):
    columns = ["transaction_id", "risk_score", "fraud_probability", "alert", "review_reason"]
    return scored[columns].to_dict(orient="records")


def _dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
