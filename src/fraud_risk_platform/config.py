from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

SAMPLE_DATA_PATH = PROCESSED_DATA_DIR / "sample_transactions.csv"
SCORED_DATA_PATH = PROCESSED_DATA_DIR / "scored_transactions.csv"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "merchant_risk",
    "customer_tenure_days",
    "transactions_last_24h",
    "distance_from_home_km",
    "is_foreign",
    "is_card_not_present",
]

TARGET_COLUMN = "is_fraud"

