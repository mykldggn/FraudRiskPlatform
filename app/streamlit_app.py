import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import MODEL_PATH, SAMPLE_DATA_PATH, SCORED_DATA_PATH
from fraud_risk_platform.data import generate_synthetic_transactions, load_transactions
from fraud_risk_platform.model import load_model, save_model, train_model
from fraud_risk_platform.monitoring import monitoring_summary
from fraud_risk_platform.scoring import precision_at_k, score_transactions


st.set_page_config(page_title="Fraud Risk Monitoring", layout="wide")
st.title("Fraud Risk Monitoring Platform")

threshold = st.sidebar.slider("Review threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded:
    transactions = pd.read_csv(uploaded)
elif SAMPLE_DATA_PATH.exists():
    transactions = load_transactions(SAMPLE_DATA_PATH)
else:
    transactions = generate_synthetic_transactions()

if MODEL_PATH.exists():
    model = load_model(MODEL_PATH)
else:
    model, _ = train_model(transactions)
    save_model(model, MODEL_PATH)

scored = score_transactions(model, transactions, threshold=threshold)
summary = monitoring_summary(scored, threshold=threshold)

metric_cols = st.columns(5)
metric_cols[0].metric("Transactions", f"{int(summary['transactions']):,}")
metric_cols[1].metric("Alert rate", f"{summary['alert_rate']:.1%}")
metric_cols[2].metric("Avg risk score", f"{summary['avg_risk_score']:.1f}")
metric_cols[3].metric("P95 risk score", f"{summary['p95_risk_score']:.1f}")
metric_cols[4].metric("Precision@100", f"{precision_at_k(scored, k=100):.1%}")

st.subheader("Risk Score Distribution")
st.bar_chart(scored["risk_score"].round(0).value_counts().sort_index())

st.subheader("Flagged Transactions")
flagged = scored[scored["alert"]].head(50)
st.dataframe(
    flagged[
        [
            "transaction_id",
            "amount",
            "risk_score",
            "rule_score",
            "review_reason",
            "is_fraud",
        ]
    ],
    use_container_width=True,
)

st.subheader("Highest-Risk Transactions")
st.dataframe(scored.head(25), use_container_width=True)

if st.sidebar.button("Save scored CSV"):
    SCORED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_DATA_PATH, index=False)
    st.sidebar.success(f"Saved to {SCORED_DATA_PATH}")

