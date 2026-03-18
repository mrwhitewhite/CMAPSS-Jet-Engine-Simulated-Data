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

    st.sidebar.success("Dataset Loaded")
    st.sidebar.title("功能菜单")

# Create sidebar navigation 
    if st.sidebar.button("Overview Dashboard"):
        st.session_state.page = "Overview Dashboard"
    
    if st.sidebar.button("Sensor & RUL Analysis"):
        st.session_state.page = "Sensor & RUL Analysis"
    
    if st.sidebar.button("Condition Charts"):
        st.session_state.page = "Condition Charts"
    
    if st.sidebar.button("xx"):
        st.session_state.page = "xx"

    # Initial page state
    if "page" not in st.session_state:
        st.session_state.page = "home"



    
    
    # Sidebar filters
    engine_list = df["unit"].unique()
    selected_engine = st.sidebar.selectbox("Select Engine Unit", engine_list)

    fd_list = df["fd"].unique()
    selected_fd = st.sidebar.selectbox("Select FD", fd_list)

    sensor_columns = [col for col in df.columns if col.startswith("s")]
    selected_sensor = st.sidebar.selectbox("Select Sensor", sensor_columns)

    # Filter engine
    engine_df = df[(df["unit"] == selected_engine) & (df["fd"] == selected_fd)]

def dataset_overview():
    # ======================
    # Dataset Overview
    # ======================


    fig = px.histogram(
        df,
        x="rul",
        nbins=30,
        title="Distribution of Remaining Useful Life (RUL)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    



    traj = predict_rul_trajectory(engine_df)
    print(f"[smoke test] Trajectory (last 5 cycles):\n{traj.tail()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict RUL from engine sensor history at specific unit."
    )

    st.plotly_chart(fig_engine, use_container_width=True)


def sensor_rul():
    st.header(" Dataset Overview")

    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Engines", df["unit"].nunique())
    col2.metric("Total Engines", df["fd"].nunique())

    col3.metric("Total Cycles", df["cycle"].max())
    col4.metric("Total Sensors", len(sensor_columns))

    st.dataframe(df.head())

    # ======================
    # RUL Trend
    # ======================

    st.header(" RUL Degradation Trend")

    fig_rul = px.line(
        engine_df,
        x="rul",
        y=selected_sensor,
        title=f"Engine {selected_engine} Remaining Useful Life"
    )
    
    

    st.plotly_chart(fig_rul, use_container_width=True)

    # ======================
    # Sensor Monitoring
    # ======================

    st.header("🔧 Sensor Health Monitoring")

    fig_sensor = px.line(
        engine_df,
        x="cycle",
        y=selected_sensor,
        title=f"{selected_sensor} Trend"
    )
    args = parser.parse_args()

    st.plotly_chart(fig_sensor, use_container_width=True)
    

if st.session_state.page == "Overview Dashboard":
        st.title("Overview")
        dataset_overview()
elif st.session_state.page == "Sensor & RUL Analysis":
        st.title("Conditions")
        sensor_rul()
elif st.session_state.page == "Condition Charts":
        st.title("Condition Charts")
elif st.session_state.page == "xx":
        st.title("yy")
