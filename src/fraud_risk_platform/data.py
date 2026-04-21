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


def prepare_public_fraud_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Map public Kaggle-style fraud CSVs into the platform schema."""
    normalized = _normalize_column_names(df)
    output = pd.DataFrame()

    output["transaction_id"] = _first_available(
        normalized,
        ["transaction_id", "trans_num", "transaction_number", "id"],
        default=pd.Series([f"txn_{i:06d}" for i in range(len(normalized))]),
    ).astype(str)
    output["amount"] = pd.to_numeric(_first_available(normalized, ["amount", "amt", "transaction_amount"]), errors="coerce").fillna(0)
    output["hour"] = _derive_hour(normalized)
    output["merchant_risk"] = _derive_merchant_risk(normalized)
    output["customer_tenure_days"] = _derive_customer_tenure_days(normalized)
    output["transactions_last_24h"] = _derive_velocity(normalized)
    output["distance_from_home_km"] = _derive_distance_from_home(normalized)
    output["is_foreign"] = _derive_binary(normalized, ["is_foreign", "foreign_transaction", "international_transaction"], default=0)
    output["is_card_not_present"] = _derive_card_not_present(normalized)
    output[TARGET_COLUMN] = pd.to_numeric(_first_available(normalized, [TARGET_COLUMN, "class", "fraud"]), errors="coerce").fillna(0).astype(int)

    return output[["transaction_id", *FEATURE_COLUMNS, TARGET_COLUMN]]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower().replace(" ", "_") for col in normalized.columns]
    return normalized


def _first_available(df: pd.DataFrame, columns: list[str], default=None):
    for column in columns:
        if column in df.columns:
            return df[column]
    if default is not None:
        return default
    raise ValueError(f"Input data is missing one of these columns: {columns}")


def _derive_hour(df: pd.DataFrame) -> pd.Series:
    if "hour" in df.columns:
        return pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int) % 24
    if "transaction_hour" in df.columns:
        return pd.to_numeric(df["transaction_hour"], errors="coerce").fillna(0).astype(int) % 24
    if "trans_date_trans_time" in df.columns:
        return pd.to_datetime(df["trans_date_trans_time"], errors="coerce").dt.hour.fillna(0).astype(int)
    if "time" in df.columns:
        return (pd.to_numeric(df["time"], errors="coerce").fillna(0) // 3600).astype(int) % 24
    return pd.Series(np.zeros(len(df), dtype=int))


def _derive_merchant_risk(df: pd.DataFrame) -> pd.Series:
    if "merchant_risk" in df.columns:
        return pd.to_numeric(df["merchant_risk"], errors="coerce").fillna(0.5).clip(0, 1)
    if "device_trust_score" in df.columns:
        trust = pd.to_numeric(df["device_trust_score"], errors="coerce").fillna(50).clip(0, 100)
        return (1 - trust / 100).clip(0, 1)
    if "location_mismatch" in df.columns:
        return pd.to_numeric(df["location_mismatch"], errors="coerce").fillna(0).clip(0, 1)
    if "merchant_category" in df.columns:
        codes = pd.factorize(df["merchant_category"].astype(str))[0]
        denominator = max(codes.max(), 1)
        return pd.Series(codes / denominator).clip(0, 1)
    if "category" in df.columns:
        codes = pd.factorize(df["category"].astype(str))[0]
        denominator = max(codes.max(), 1)
        return pd.Series(codes / denominator).clip(0, 1)
    return pd.Series(np.full(len(df), 0.5))


def _derive_customer_tenure_days(df: pd.DataFrame) -> pd.Series:
    if "customer_tenure_days" in df.columns:
        return pd.to_numeric(df["customer_tenure_days"], errors="coerce").fillna(365).clip(lower=1).astype(int)
    if "account_age_days" in df.columns:
        return pd.to_numeric(df["account_age_days"], errors="coerce").fillna(365).clip(lower=1).astype(int)
    if "cardholder_age" in df.columns:
        return (pd.to_numeric(df["cardholder_age"], errors="coerce").fillna(35).clip(lower=18) * 365).astype(int)
    if "customer_age" in df.columns:
        return (pd.to_numeric(df["customer_age"], errors="coerce").fillna(35).clip(lower=18) * 365).astype(int)
    if "dob" in df.columns:
        birth_year = pd.to_datetime(df["dob"], errors="coerce").dt.year
        age = (pd.Timestamp.today().year - birth_year).fillna(35).clip(lower=18)
        return (age * 365).astype(int)
    return pd.Series(np.full(len(df), 365, dtype=int))


def _derive_velocity(df: pd.DataFrame) -> pd.Series:
    if "transactions_last_24h" in df.columns:
        return pd.to_numeric(df["transactions_last_24h"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    if "velocity_last_24h" in df.columns:
        return pd.to_numeric(df["velocity_last_24h"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    if {"cc_num", "trans_date_trans_time"}.issubset(df.columns):
        ordered = df[["cc_num", "trans_date_trans_time"]].copy()
        ordered["trans_date_trans_time"] = pd.to_datetime(ordered["trans_date_trans_time"], errors="coerce")
        return ordered.groupby("cc_num").cumcount().clip(upper=24).astype(int)
    return pd.Series(np.ones(len(df), dtype=int))


def _derive_distance_from_home(df: pd.DataFrame) -> pd.Series:
    if "distance_from_home_km" in df.columns:
        return pd.to_numeric(df["distance_from_home_km"], errors="coerce").fillna(0).clip(lower=0)
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(df.columns):
        return _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]).fillna(0).clip(lower=0)
    if "location_mismatch" in df.columns:
        mismatch = pd.to_numeric(df["location_mismatch"], errors="coerce").fillna(0)
        return mismatch.map(lambda value: 75 if value else 5)
    return pd.Series(np.full(len(df), 5.0))


def _derive_binary(df: pd.DataFrame, columns: list[str], default: int = 0) -> pd.Series:
    for column in columns:
        if column in df.columns:
            values = df[column]
            if values.dtype == object:
                return values.astype(str).str.lower().isin(["1", "true", "yes", "y", "foreign", "online"]).astype(int)
            return pd.to_numeric(values, errors="coerce").fillna(default).clip(0, 1).astype(int)
    return pd.Series(np.full(len(df), default, dtype=int))


def _derive_card_not_present(df: pd.DataFrame) -> pd.Series:
    for column in ["is_card_not_present", "card_not_present", "online_transaction"]:
        if column in df.columns:
            return _derive_binary(df, [column])
    for column in ["transaction_type", "payment_method"]:
        if column in df.columns:
            return df[column].astype(str).str.lower().str.contains("online|web|mobile|digital").astype(int)
    if "location_mismatch" in df.columns:
        return _derive_binary(df, ["location_mismatch"])
    return pd.Series(np.zeros(len(df), dtype=int))


def _haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(6371 * 2 * np.arcsin(np.sqrt(a)))
