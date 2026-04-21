import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import MODEL_PATH, SAMPLE_DATA_PATH
from fraud_risk_platform.data import load_transactions
from fraud_risk_platform.model import save_model, train_model


def main() -> None:
    df = load_transactions(SAMPLE_DATA_PATH)
    model, metrics = train_model(df)
    save_model(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    for key, value in metrics.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
