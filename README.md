# Fraud Risk Monitoring Platform

An end-to-end fraud detection and analyst review workflow for highly imbalanced transaction data. The project turns a notebook-style credit-card fraud model into a production-minded portfolio project with data ingestion, baseline rules, ML scoring, threshold tuning, explainability, monitoring, and a lightweight dashboard.

## Why This Project Exists

The goal is to show business-focused ML engineering, not just a high offline accuracy score. Fraud teams care about catching risky transactions while managing false-positive review volume, so this repo emphasizes:

- PR-AUC, recall at fixed precision, and precision@top-k
- configurable risk thresholds and expected review volume
- rule-based and ML-based risk signals
- analyst-facing explanations for flagged transactions
- monitoring views for alert volume and score drift

## Project Structure

```text
FraudRiskPlatform/
├── app/                         # Streamlit analyst dashboard
├── data/
│   ├── raw/                     # source CSVs, not committed
│   └── processed/               # generated sample data, not committed
├── docs/                        # architecture and project notes
├── models/                      # trained model artifacts, not committed
├── notebooks/                   # exploration notebooks
├── scripts/                     # CLI workflows
├── src/fraud_risk_platform/     # reusable package code
└── tests/                       # focused unit tests
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_sample_data.py
python scripts/train_model.py
python scripts/score_transactions.py
streamlit run app/streamlit_app.py
```

The sample-data generator creates synthetic transactions so the repo runs without downloading a private or Kaggle dataset. To use a real dataset, place a CSV in `data/raw/` with a binary `is_fraud` column and update the path in the scripts.

## Core Workflow

1. Generate or ingest transactions.
2. Apply baseline rules to create operational risk flags.
3. Train a calibrated ML model on transaction features.
4. Evaluate with fraud-appropriate metrics.
5. Tune the alert threshold based on precision, recall, and review capacity.
6. Score new transactions and explain why each case was flagged.
7. Monitor score distribution, alert volume, and feature drift.

## Portfolio Framing

**Financial Fraud Monitoring Platform**  
Built an end-to-end fraud detection system on transaction data with baseline rules, ML risk scoring, threshold tuning, analyst-facing explanations, and monitoring views to support review workflows and reduce false positives.

## Next Enhancements

- Add XGBoost or LightGBM benchmark once dependencies are available.
- Add SHAP explanations for model-level feature attribution.
- Replace synthetic data with a public card-transaction fraud dataset.
- Add a FastAPI scoring endpoint for batch and single-transaction inference.
- Add a demo GIF and dashboard screenshot for the portfolio site.

