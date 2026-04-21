import pandas as pd

from fraud_risk_platform.config import FEATURE_COLUMNS
from fraud_risk_platform.data import prepare_public_fraud_dataset


def test_prepare_public_fraud_dataset_maps_kaggle_like_columns():
    raw = pd.DataFrame(
        {
            "transaction_id": ["a", "b"],
            "amount": [100.0, 250.5],
            "transaction_hour": [8, 22],
            "foreign_transaction": [0, 1],
            "location_mismatch": [0, 1],
            "device_trust_score": [92, 35],
            "velocity_last_24h": [1, 7],
            "cardholder_age": [32, 47],
            "is_fraud": [0, 1],
        }
    )

    prepared = prepare_public_fraud_dataset(raw)

    assert list(prepared.columns) == ["transaction_id", *FEATURE_COLUMNS, "is_fraud"]
    assert prepared.loc[1, "hour"] == 22
    assert prepared.loc[1, "is_foreign"] == 1
    assert prepared.loc[1, "transactions_last_24h"] == 7
    assert prepared.loc[1, "merchant_risk"] > prepared.loc[0, "merchant_risk"]

