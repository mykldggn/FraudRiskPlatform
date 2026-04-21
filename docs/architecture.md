# Architecture

```text
Transaction CSV
      |
      v
Data validation and feature selection
      |
      +--> Baseline rule engine
      |
      +--> ML risk model
              |
              v
Fraud probability and risk score
      |
      +--> Threshold tuning
      +--> Analyst explanations
      +--> Monitoring summary
      |
      v
Streamlit review dashboard / scored CSV output
```

## MVP Components

- `data.py`: creates synthetic transaction data and validates incoming CSVs.
- `rules.py`: adds transparent review rules for high-risk transaction patterns.
- `model.py`: trains a gradient boosting classifier and reports fraud-focused metrics.
- `scoring.py`: creates fraud probabilities, risk scores, alerts, and review reasons.
- `monitoring.py`: summarizes alert rate, score distribution, and simple feature drift.
- `app/streamlit_app.py`: provides a local analyst dashboard.

## Business Metric Lens

Fraud detection is a review-capacity problem. A useful model is not just accurate; it catches enough true fraud at a manageable alert rate. This repo therefore tracks PR-AUC, recall at a target precision level, precision@k, and alert-rate changes as the review threshold moves.

