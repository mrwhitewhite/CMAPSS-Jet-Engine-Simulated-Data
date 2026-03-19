# ✈️ CMAPSS Turbofan Engine RUL Prediction

Predictive maintenance system for aircraft engines using the NASA CMAPSS benchmark dataset. Predicts **Remaining Useful Life (RUL)** in flight cycles from raw sensor readings, served through an interactive Streamlit dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cmapss.streamlit.app/)

---

## Live Demo

**Dashboard →** [https://cmapss.streamlit.app/](https://cmapss.streamlit.app/)

Upload any CMAPSS-format test file (`test_FD001.txt` – `test_FD004.txt`) to explore fleet health, sensor trajectories, out-of-distribution condition analysis, and live RUL predictions.

---

## Dataset

[NASA CMAPSS](https://www.nasa.gov/intelligent-systems-division/) — Commercial Modular Aero-Propulsion System Simulation. Four sub-datasets of increasing complexity:

| Dataset | Train engines | Test engines | Conditions | Fault modes |
|---------|:---:|:---:|:---:|:---:|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 (HPC + fan) |
| FD004 | 248 | 249 | 6 | 2 |

Each row is one flight cycle: 3 operational settings + 21 sensor measurements. Training trajectories run to failure; test trajectories are truncated at an unknown point before failure.

---

## Pipeline

### 1. Leakage-free split

Engines are split 80/20 at the engine level **before** any feature engineering. Normalisation statistics are fitted on train engines only and applied to validation and test.

### 2. Feature Engineering

| Feature group | Detail |
|---|---|
| Operating condition | K-means (k=6) on `os1/os2/os3` → `conditions` label |
| Sensor normalisation | Z-score per condition (regime-wise) — removes altitude/Mach confound |
| Rolling mean & std | Windows 5 and 15 cycles |
| Rolling std | Same windows — captures volatility increase near failure |
| Absolute cycle | Engine-reported cycle number, no future information needed |

### 3. Target

RUL is computed as `max_cycle − current_cycle`, clipped at **125 cycles** (piecewise-linear target). This cap stabilises training for early-life cycles where degradation is not yet detectable.

### 4. Model

XGBoost regressor with early stopping on the validation set.

```
n_estimators    5000  (early stopping ~800)
learning_rate   0.02
max_depth       6
subsample       0.80
colsample_bytree 0.80
reg_alpha       0.1
reg_lambda      1.0
```

### 5. Saved Artefacts

```python
model.save_model("models/model.ubj")
joblib.dump(km, "models/condition_clusterer.joblib")
rs.to_parquet("models/normalisation_stats.parquet")
```

The `model.py` module loads these once at import time and exposes two functions usable without re-running the notebook.

---

## Inference

### Python

```python
import pandas as pd
from model import predict_rul, predict_rul_trajectory

engine_history = pd.read_csv("engine_readings.csv", ...)   # unit + cycle + os1-3 + s1-21
rul   = predict_rul(engine_history)                         # float — last cycle
traj  = predict_rul_trajectory(engine_history)              # pd.Series indexed by cycle
```

### CLI

```bash
# Point estimate for unit 3
python test_api.py data/raw/test_FD001.txt --unit 3

# Full cycle trajectory
python test_api.py data/raw/test_FD001.txt --unit 3 --trajectory
```

---

## API

Start the server:

```bash
fastapi run api.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | RUL at last observed cycle |
| `/predict/trajectory` | POST | RUL at every cycle |
| `/docs` | GET | Swagger UI |

**Request body** — the `readings` list may contain data for multiple units (fleet batch export). The `unit` field specifies which engine to predict:

```json
{
  "unit": 50,
  "readings": [
    {
      "unit": 50, "cycle": 1,
      "os1": 0.0043, "os2": -0.0001, "os3": 100.0,
      "s1": 518.67, "s2": 642.47, "s3": 1578.98,
      "s4": 1397.59, "s5": 14.62, "s6": 21.61,
      "s7": 553.83, "s8": 2388.02, "s9": 9061.18,
      "s10": 1.30, "s11": 47.37, "s12": 522.04,
      "s13": 2388.07, "s14": 8143.03, "s15": 8.4187,
      "s16": 0.03, "s17": 392, "s18": 2388,
      "s19": 100.0, "s20": 38.98, "s21": 23.4162
    }
  ]
}
```

---

## Dashboard

```bash
streamlit run dashboard.py
```

Five sections accessible from the sidebar:

| Section | What it shows |
|---|---|
| 🚀 Getting Started | File upload, API host configuration |
| 📊 Fleet Overview | Cycle distribution, per-engine bar chart, fleet RUL predictions |
| 📈 Condition Analysis | **OOD detection** — centroid distance, 3D OS scatter, sensor z-score heatmap |
| ⚙️ Sensor Explorer | Sensor trajectories, fleet last-cycle heatmap, correlation matrix |
| 🔍 Engine Deep-Dive | Fleet RUL bar chart, per-engine RUL trajectory from API |

---

## Setup

```bash
git clone https://github.com/your-username/cmapss-rul
cd cmapss-rul
pip install -r requirements.txt
```

Place CMAPSS data files in `data/raw/`:
```
data/raw/train_FD001.txt  ...  train_FD004.txt
data/raw/test_FD001.txt   ...  test_FD004.txt
data/raw/RUL_FD001.txt    ...  RUL_FD004.txt
```

Data is available from the [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/).