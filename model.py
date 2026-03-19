"""
predict.py — RUL deployment script

Input:
    A DataFrame of raw sensor readings for engines, one row per cycle.
    Required columns: unit, cycle, os1, os2, os3, s1…s21
    Rows must be in chronological order (ascending cycle).

Output:
    Predicted RUL (cycles) at the last observed cycle.

Usage:
    python predict.py                  # runs the built-in smoke test
    python predict.py path/to/engine.csv unit-to-be-tested
"""


import argparse
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pyprojroot import here

warnings.filterwarnings("ignore")
MODELS_DIR = here() / "models"
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]

# INIT
_model = xgb.XGBRegressor()
_model.load_model(MODELS_DIR / "model.ubj")

_km = joblib.load(MODELS_DIR / "condition_clusterer.joblib")
_rs = pd.read_parquet(MODELS_DIR / "normalisation_stats.parquet")

with open(MODELS_DIR / "pipeline_config.json") as _f:
    _cfg = json.load(_f)

SCOLS = _cfg["SCOLS"]
FCOLS = _cfg["FCOLS"]
WINDOWS = _cfg["WINDOWS"]
RUL_CAP = _cfg["RUL_CAP"]


# feature engineering
def _assign_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conditions"] = _km.predict(df[["os1", "os2", "os3"]]).astype(int)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for s in SCOLS:
        mu = df["conditions"].map(_rs[f"{s}_mean"])
        sig = df["conditions"].map(_rs[f"{s}_std"]).replace(0, 1e-6)
        df[s] = (df[s] - mu) / sig
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("cycle").copy()
    for w in WINDOWS:
        for s in SCOLS:
            df[f"{s}_rm{w}"] = df[s].rolling(w, min_periods=1).mean()
            df[f"{s}_rs{w}"] = df[s].rolling(w, min_periods=1).std().fillna(0)
    return df


def build_features(engine_history: pd.DataFrame) -> pd.DataFrame:
    df = engine_history.copy().sort_values("cycle").reset_index(drop=True)
    df = _assign_condition(df)
    df = _normalise(df)
    df = _rolling_features(df)
    return df[FCOLS]


# public prediction function
def predict_rul(engine_history: pd.DataFrame) -> float:
    features = build_features(engine_history)
    last_row = features.iloc[[-1]]  # most recent cycle only
    pred = float(_model.predict(last_row)[0])
    return round(max(0.0, min(pred, RUL_CAP)), 2)


def predict_rul_trajectory(engine_history: pd.DataFrame) -> pd.Series:
    features = build_features(engine_history)
    preds = _model.predict(features)
    preds = np.clip(preds, 0, RUL_CAP)
    return pd.Series(preds, index=engine_history["cycle"].values, name="predicted_rul")


# CLI entry point
def _smoke_test():
    data_dir = here() / "data" / "raw"
    test_df = pd.read_csv(data_dir / "test_FD001.txt", sep=r"\s+", header=None).dropna(
        axis=1
    )
    test_df.columns = COLS

    # first engine only
    engine_df = test_df[test_df["unit"] == 1].copy()
    rul = predict_rul(engine_df)
    print(f"[smoke test] Engine 1 / FD001 — predicted RUL: {rul} cycles")

    traj = predict_rul_trajectory(engine_df)
    print(f"[smoke test] Trajectory (last 5 cycles):\n{traj.tail()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict RUL from engine sensor history at specific unit."
    )
    parser.add_argument(
        "csv",
        type=str,
        default=None,
        help="Path to a CSV with columns: unit, cycle, os1, os2, os3, s1…s21",
    )
    parser.add_argument(
        "unit",
        type=int,
        default=None,
        help="Unit number to be predict",
    )
    args = parser.parse_args()

    if args.csv:
        # load csv
        engine_history = pd.read_csv(args.csv, header=None, sep=" ")
        engine_history.dropna(axis=1, how="all", inplace=True)
        engine_history.columns = COLS

        engine_history = engine_history[engine_history["unit"] == args.unit]

        # prediction
        rul = predict_rul(engine_history)
        print(f"Predicted RUL at unit {args.unit}: {rul} cycles")
    else:
        _smoke_test()
