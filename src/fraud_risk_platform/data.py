import numpy as np
import pandas as pd

from fraud_risk_platform.config import FEATURE_COLUMNS, TARGET_COLUMN


def generate_synthetic_transactions(n_rows: int = 5000, fraud_rate: float = 0.025, seed: int = 42) -> pd.DataFrame:
    """Create realistic-enough imbalanced transaction data for local demos."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.binomial(1, fraud_rate, size=n_rows)

    amount = rng.lognormal(mean=3.2 + is_fraud * 0.9, sigma=0.9, size=n_rows).round(2)
    hour = rng.integers(0, 24, size=n_rows)
    merchant_risk = np.clip(rng.beta(2, 8, size=n_rows) + is_fraud * rng.uniform(0.2, 0.55, size=n_rows), 0, 1)
    customer_tenure_days = np.maximum(1, rng.gamma(shape=4, scale=180, size=n_rows) - is_fraud * rng.uniform(0, 250, size=n_rows)).round()
    transactions_last_24h = rng.poisson(1.5 + is_fraud * 3.2, size=n_rows)
    distance_from_home_km = rng.exponential(scale=12 + is_fraud * 48, size=n_rows).round(2)
    is_foreign = rng.binomial(1, 0.08 + is_fraud * 0.32, size=n_rows)
    is_card_not_present = rng.binomial(1, 0.22 + is_fraud * 0.44, size=n_rows)

    df = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:06d}" for i in range(n_rows)],
            "amount": amount,
            "hour": hour,
            "merchant_risk": merchant_risk.round(3),
            "customer_tenure_days": customer_tenure_days.astype(int),
            "transactions_last_24h": transactions_last_24h,
            "distance_from_home_km": distance_from_home_km,
            "is_foreign": is_foreign,
            "is_card_not_present": is_card_not_present,
            TARGET_COLUMN: is_fraud,
        }
    )

    return df[["transaction_id", *FEATURE_COLUMNS, TARGET_COLUMN]]


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")
    return df

