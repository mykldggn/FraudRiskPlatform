import pandas as pd


def apply_baseline_rules(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["rule_high_amount"] = scored["amount"] >= scored["amount"].quantile(0.95)
    scored["rule_velocity"] = scored["transactions_last_24h"] >= 5
    scored["rule_foreign_card_not_present"] = (scored["is_foreign"] == 1) & (scored["is_card_not_present"] == 1)
    scored["rule_far_from_home"] = scored["distance_from_home_km"] >= 75
    scored["rule_score"] = scored[
        ["rule_high_amount", "rule_velocity", "rule_foreign_card_not_present", "rule_far_from_home"]
    ].sum(axis=1)
    return scored


def explain_rule_hits(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row.get("rule_high_amount", False):
        reasons.append("unusually high transaction amount")
    if row.get("rule_velocity", False):
        reasons.append("high transaction velocity in the last 24 hours")
    if row.get("rule_foreign_card_not_present", False):
        reasons.append("foreign card-not-present transaction")
    if row.get("rule_far_from_home", False):
        reasons.append("large distance from typical home location")
    return reasons or ["model score exceeded review threshold"]

