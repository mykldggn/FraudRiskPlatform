import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import PROCESSED_DATA_DIR, SAMPLE_DATA_PATH
from fraud_risk_platform.data import load_transactions
from fraud_risk_platform.model import benchmark_models


def main() -> None:
    df = load_transactions(SAMPLE_DATA_PATH)
    results = benchmark_models(df)
    output = PROCESSED_DATA_DIR / "model_benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Wrote benchmark results to {output}")


if __name__ == "__main__":
    main()
