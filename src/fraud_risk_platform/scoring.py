import pandas as pd

from fraud_risk_platform.config import FEATURE_COLUMNS
from fraud_risk_platform.rules import apply_baseline_rules, explain_rule_hits


def score_transactions(model, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    scored = apply_baseline_rules(df)
    scored["fraud_probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["risk_score"] = (100 * scored["fraud_probability"]).round(1)
    scored["alert"] = scored["fraud_probability"] >= threshold
    scored["review_reason"] = scored.apply(lambda row: "; ".join(explain_rule_hits(row)), axis=1)
    return scored.sort_values("fraud_probability", ascending=False)


def precision_at_k(scored: pd.DataFrame, target_column: str = "is_fraud", k: int = 100) -> float:
    if target_column not in scored.columns or scored.empty:
        return 0.0
    top_k = scored.sort_values("fraud_probability", ascending=False).head(k)
    return float(top_k[target_column].mean()) if len(top_k) else 0.0

