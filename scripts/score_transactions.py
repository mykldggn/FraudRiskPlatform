import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import MODEL_PATH, SAMPLE_DATA_PATH, SCORED_DATA_PATH
from fraud_risk_platform.data import load_transactions
from fraud_risk_platform.model import load_model
from fraud_risk_platform.scoring import precision_at_k, score_transactions


def main() -> None:
    model = load_model(MODEL_PATH)
    df = load_transactions(SAMPLE_DATA_PATH)
    scored = score_transactions(model, df, threshold=0.5)
    SCORED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_DATA_PATH, index=False)
    print(f"Wrote scored transactions to {SCORED_DATA_PATH}")
    print(f"precision_at_100: {precision_at_k(scored, k=100):.4f}")
    print(f"alert_rate: {scored['alert'].mean():.4f}")


if __name__ == "__main__":
    main()
