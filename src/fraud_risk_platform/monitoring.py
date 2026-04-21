import pandas as pd


def monitoring_summary(scored: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    alerts = scored["fraud_probability"] >= threshold
    return {
        "transactions": float(len(scored)),
        "alert_rate": float(alerts.mean()),
        "avg_risk_score": float(scored["risk_score"].mean()),
        "p95_risk_score": float(scored["risk_score"].quantile(0.95)),
        "avg_amount_alerted": float(scored.loc[alerts, "amount"].mean() if alerts.any() else 0),
    }


def feature_drift_proxy(reference: pd.DataFrame, current: pd.DataFrame, feature: str) -> float:
    if feature not in reference.columns or feature not in current.columns:
        raise ValueError(f"Feature not found: {feature}")
    ref_mean = reference[feature].mean()
    current_mean = current[feature].mean()
    denominator = abs(ref_mean) or 1.0
    return float((current_mean - ref_mean) / denominator)

