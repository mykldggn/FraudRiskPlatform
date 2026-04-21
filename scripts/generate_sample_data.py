import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import SAMPLE_DATA_PATH
from fraud_risk_platform.data import generate_synthetic_transactions


def main() -> None:
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_transactions()
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {len(df):,} sample transactions to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
