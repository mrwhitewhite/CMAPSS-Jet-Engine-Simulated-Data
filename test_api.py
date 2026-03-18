"""
example API client using test_FD001.txt

Reads the test file, calls the RUL prediction API for a chosen unit, and prints the result.

Usage
-----
python test_api.py                                          # predict unit 1 (default)
python test_api.py --unit 3                             # predict a different unit
python test_api.py --unit 3 --trajectory         # full cycle-by-cycle trajectory
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import requests

# config
DEFAULT_HOST = "http://localhost:8000"
DEFAULT_UNIT = 1
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


# load data
def load_test_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1)
    df.columns = COLS
    return df


# call /predict
def predict(host: str, df: pd.DataFrame, unit: int) -> dict:
    payload = {
        "unit": unit,
        "readings": df.to_dict(orient="records"),
    }
    resp = requests.post(f"{host}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# call /predict/trajectory
def predict_trajectory(host: str, df: pd.DataFrame, unit: int) -> dict:
    payload = {
        "unit": unit,
        "readings": df.to_dict(orient="records"),
    }
    resp = requests.post(f"{host}/predict/trajectory", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# main
def main():
    parser = argparse.ArgumentParser(description="RUL prediction API client.")
    parser.add_argument("--file", type=str, help="Path to test data file")
    parser.add_argument(
        "--unit", type=int, default=DEFAULT_UNIT, help="Engine unit to predict"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="API base URL")
    parser.add_argument(
        "--trajectory", action="store_true", help="Return RUL at every cycle"
    )
    args = parser.parse_args()

    # load
    path = Path(args.file)
    if not path.exists():
        print(f"Data file not found: {path}")
        sys.exit(1)

    df = load_test_file(path)
    units_in_file = sorted(df["unit"].unique())

    if args.unit not in units_in_file:
        print(f"Unit {args.unit} not in file. Available: {units_in_file}")
        sys.exit(1)

    cycles = df[df["unit"] == args.unit]["cycle"].max()
    print(f"Loaded {path.name} — unit {args.unit}, {cycles} cycles")

    # call API
    if args.trajectory:
        result = predict_trajectory(args.host, df, args.unit)
        print(
            f"\nTrajectory for unit {result['unit']} ({result['cycles_observed']} cycles):"
        )
        print(f"  {'cycle':>6}  {'predicted_rul':>13}")
        print(f"  {'─' * 6}  {'─' * 13}")
        for point in result["trajectory"]:
            print(f"  {point['cycle']:>6}  {point['predicted_rul']:>13.2f}")
    else:
        result = predict(args.host, df, args.unit)
        print(f"\nPrediction for unit {result['unit']}:")
        print(f"  Cycles observed : {result['cycles_observed']}")
        print(f"  Predicted RUL   : {result['predicted_rul']} cycles")
        print(f"  RUL cap         : {result['rul_cap']} cycles")


if __name__ == "__main__":
    main()
