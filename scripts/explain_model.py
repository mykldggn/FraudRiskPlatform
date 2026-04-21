import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import MODEL_PATH, PROCESSED_DATA_DIR, SAMPLE_DATA_PATH
from fraud_risk_platform.data import load_transactions
from fraud_risk_platform.explain import global_feature_attribution
from fraud_risk_platform.model import load_model


def main() -> None:
    model = load_model(MODEL_PATH)
    df = load_transactions(SAMPLE_DATA_PATH)
    attribution = global_feature_attribution(model, df)
    output = PROCESSED_DATA_DIR / "feature_attribution.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    attribution.to_csv(output, index=False)
    print(attribution.head(10).to_string(index=False))
    print(f"Wrote feature attribution to {output}")


if __name__ == "__main__":
    main()
