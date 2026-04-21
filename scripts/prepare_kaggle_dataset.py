import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import SAMPLE_DATA_PATH
from fraud_risk_platform.data import prepare_public_fraud_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a public Kaggle fraud CSV for this project.")
    parser.add_argument("input_csv", help="Path to the downloaded Kaggle CSV.")
    parser.add_argument("--output", default=SAMPLE_DATA_PATH, help="Output CSV path.")
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    prepared = prepare_public_fraud_dataset(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    print(f"Wrote {len(prepared):,} prepared transactions to {output}")


if __name__ == "__main__":
    main()
