"""
app.py — FastAPI service for RUL prediction

Endpoints
---------
POST /predict              Predict RUL at the last observed cycle for one engine
POST /predict/trajectory   Predict RUL at every cycle for one engine
GET  /health               Liveness check
GET  /docs                 Auto-generated Swagger UI (built-in)

Run
---
[PRODUCTION] fastapi run api.py
[DEBUG] fastapi dev api.py
"""

from typing import Annotated
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import model as predict

app = FastAPI(
    title="CMAPSS RUL Prediction API",
    description="Predicts remaining useful life (cycles) for turbofan engines.",
    version="1.0.0",
)


# SCHEMA
class CycleReading(BaseModel):
    """One cycle's worth of raw sensor readings. May belong to any engine unit."""

    unit: int = Field(..., description="Engine unit number")
    cycle: int = Field(..., description="Operational cycle number")
    os1: float = Field(..., description="Operational setting 1")
    os2: float = Field(..., description="Operational setting 2")
    os3: float = Field(..., description="Operational setting 3")
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    s5: float = 0.0
    s6: float = 0.0
    s7: float = 0.0
    s8: float = 0.0
    s9: float = 0.0
    s10: float = 0.0
    s11: float = 0.0
    s12: float = 0.0
    s13: float = 0.0
    s14: float = 0.0
    s15: float = 0.0
    s16: float = 0.0
    s17: float = 0.0
    s18: float = 0.0
    s19: float = 0.0
    s20: float = 0.0
    s21: float = 0.0


class PredictRequest(BaseModel):
    """
    Sensor history for one or more engine units, include the unit to predict.

    - `unit`     — which engine to predict; must be present in `readings`
    - `readings` — full cycle history; may contain multiple units (e.g. a fleet
                   batch export). Only rows matching `unit` are used.
    """

    unit: int = Field(
        ..., description="Engine unit number to predict RUL for", examples=[50]
    )
    readings: Annotated[
        list[CycleReading],
        Field(min_length=1, description="Cycle history — may contain multiple units"),
    ]

    @model_validator(mode="after")
    def unit_present_in_readings(self):
        available = {r.unit for r in self.readings}
        if self.unit not in available:
            raise ValueError(
                f"unit={self.unit} not found in readings. "
                f"Units present: {sorted(available)}"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "unit": 50,
                    "readings": [
                        {
                            "unit": 50,
                            "cycle": 1,
                            "os1": 0.0043,
                            "os2": -0.0001,
                            "os3": 100.0,
                            "s1": 518.67,
                            "s2": 642.47,
                            "s3": 1578.98,
                            "s4": 1397.59,
                            "s5": 14.62,
                            "s6": 21.61,
                            "s7": 553.83,
                            "s8": 2388.02,
                            "s9": 9061.18,
                            "s10": 1.30,
                            "s11": 47.37,
                            "s12": 522.04,
                            "s13": 2388.07,
                            "s14": 8143.03,
                            "s15": 8.4187,
                            "s16": 0.03,
                            "s17": 392,
                            "s18": 2388,
                            "s19": 100.0,
                            "s20": 38.98,
                            "s21": 23.4162,
                        },
                        {
                            "unit": 50,
                            "cycle": 2,
                            "os1": 0.0031,
                            "os2": 0.0002,
                            "os3": 100.0,
                            "s1": 518.67,
                            "s2": 642.38,
                            "s3": 1585.08,
                            "s4": 1400.74,
                            "s5": 14.62,
                            "s6": 21.61,
                            "s7": 554.26,
                            "s8": 2388.05,
                            "s9": 9059.53,
                            "s10": 1.30,
                            "s11": 47.35,
                            "s12": 521.45,
                            "s13": 2388.07,
                            "s14": 8139.32,
                            "s15": 8.4194,
                            "s16": 0.03,
                            "s17": 392,
                            "s18": 2388,
                            "s19": 100.0,
                            "s20": 39.02,
                            "s21": 23.4525,
                        },
                        {
                            "unit": 50,
                            "cycle": 3,
                            "os1": 0.0041,
                            "os2": 0.0000,
                            "os3": 100.0,
                            "s1": 518.67,
                            "s2": 642.38,
                            "s3": 1585.41,
                            "s4": 1400.76,
                            "s5": 14.62,
                            "s6": 21.61,
                            "s7": 553.68,
                            "s8": 2388.05,
                            "s9": 9059.73,
                            "s10": 1.30,
                            "s11": 47.24,
                            "s12": 522.20,
                            "s13": 2388.03,
                            "s14": 8142.16,
                            "s15": 8.4061,
                            "s16": 0.03,
                            "s17": 392,
                            "s18": 2388,
                            "s19": 100.0,
                            "s20": 39.09,
                            "s21": 23.4031,
                        },
                    ],
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    unit: int = Field(..., description="Engine unit number predicted")
    cycles_observed: int = Field(..., description="Number of cycles used for this unit")
    predicted_rul: float = Field(
        ..., description="Predicted remaining useful life (cycles)"
    )
    rul_cap: int = Field(..., description="RUL cap used during training")


class TrajectoryPoint(BaseModel):
    cycle: int
    predicted_rul: float


class TrajectoryResponse(BaseModel):
    unit: int
    cycles_observed: int
    trajectory: list[TrajectoryPoint]


# helpers function
def _extract_unit(request: PredictRequest) -> pd.DataFrame:
    """Convert readings to DataFrame, filter to requested unit, sort by cycle."""
    df = pd.DataFrame([r.model_dump() for r in request.readings])
    return df[df["unit"] == request.unit].sort_values("cycle").reset_index(drop=True)


# Endpoints
@app.get("/health", tags=["Meta"])
def health():
    """confirms model artefacts loaded successfully."""
    return {
        "status": "ok",
        "rul_cap": predict.RUL_CAP,
        "n_features": len(predict.FCOLS),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(request: PredictRequest):
    """
    Predict RUL at the **last observed cycle** for the specified engine unit.

    The `readings` list may contain data for multiple units. Only rows with `unit` matches the top-level `unit` field
    are used, the rest are ignored.
    """
    df = _extract_unit(request)
    try:
        rul = predict.predict_rul(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        unit=request.unit,
        cycles_observed=len(df),
        predicted_rul=rul,
        rul_cap=predict.RUL_CAP,
    )


@app.post("/predict/trajectory", response_model=TrajectoryResponse, tags=["Prediction"])
def predict_trajectory_endpoint(request: PredictRequest):
    """
    Predict RUL at **every cycle** for the specified engine unit.

    Useful for plotting how the estimate evolves as the engine degrades.
    Only rows matching the top-level `unit` field are used.
    """
    df = _extract_unit(request)
    try:
        traj = predict.predict_rul_trajectory(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return TrajectoryResponse(
        unit=request.unit,
        cycles_observed=len(df),
        trajectory=[
            TrajectoryPoint(cycle=int(cycle), predicted_rul=float(rul))  # type: ignore
            for cycle, rul in traj.items()
        ],
    )
