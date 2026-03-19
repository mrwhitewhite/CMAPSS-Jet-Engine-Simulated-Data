"""
dashboard.py — Streamlit dashboard for CMAPSS turbofan engine health monitoring

Sections
--------
Fleet Overview    — cycle distribution, predicted RUL bar chart, fleet health summary
Engine Deep-Dive  — sensor trajectories, RUL trajectory from API, operating conditions
Sensor Explorer   — last-cycle fleet heatmap, correlation matrix, violin distributions

Run
---
streamlit run dashboard.py
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Turbofan Engine Health",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
DROP_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
SCOLS = [c for c in COLS if c.startswith("s") and c not in DROP_SENSORS]
SENSOR_LABELS = {
    "s2": "Sensor 2",
    "s3": "Sensor 3",
    "s4": "Sensor 4",
    "s7": "Sensor 7",
    "s8": "Sensor 8",
    "s9": "Sensor 9",
    "s11": "Sensor 11",
    "s12": "Sensor 12",
    "s13": "Sensor 13",
    "s14": "Sensor 14",
    "s15": "Sensor 15",
    "s17": "Sensor 17",
    "s20": "Sensor 20",
    "s21": "Sensor 21",
}
RUL_CAP = 125

try:
    from pyprojroot import here

    MODELS_DIR = here() / "models"
except Exception:
    MODELS_DIR = Path(__file__).parent / "models"


@st.cache_data
def load_data(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None).dropna(axis=1)
    df.columns = COLS
    return df


def rul_color(rul: float) -> str:
    if rul <= 30:
        return "#e15759"
    if rul <= 70:
        return "#f28e2b"
    return "#59a14f"


def call_predict(host: str, df: pd.DataFrame, unit: int) -> float | None:
    try:
        resp = requests.post(
            f"{host}/predict",
            json={"unit": unit, "readings": df.to_dict(orient="records")},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["predicted_rul"]
    except Exception:
        return None


def call_trajectory(host: str, df: pd.DataFrame, unit: int) -> list[dict] | None:
    try:
        resp = requests.post(
            f"{host}/predict/trajectory",
            json={"unit": unit, "readings": df.to_dict(orient="records")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["trajectory"]
    except Exception:
        return None


def base_layout() -> dict:
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=40, l=10, r=10),
        font=dict(size=12),
    )


@st.cache_resource
def load_artefacts():
    """Load KMeans and normalisation stats once; None if models/ not found."""
    try:
        km = joblib.load(MODELS_DIR / "condition_clusterer.joblib")
        rs = pd.read_parquet(MODELS_DIR / "normalisation_stats.parquet")
        with open(MODELS_DIR / "pipeline_config.json") as f:
            import json

            cfg = json.load(f)
        return km, rs, cfg["SCOLS"]
    except Exception:
        return None, None, None


def compute_ood(df_unit: pd.DataFrame, km, rs, scols: list) -> pd.DataFrame:
    """
    For each cycle compute:
      - assigned condition label
      - distance to nearest centroid  (OS-space OOD proxy)
      - per-sensor z-score vs training normalisation stats
    """
    import warnings

    df = df_unit.copy()
    os_arr = df[["os1", "os2", "os3"]].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["condition"] = km.predict(os_arr)

    # distance to nearest centroid
    dists = np.linalg.norm(os_arr[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    df["dist_to_centroid"] = dists.min(axis=1)
    df["nearest_centroid"] = dists.argmin(axis=1)

    # per-sensor z-scores vs training stats
    for s in scols:
        mu = df["condition"].map(rs[f"{s}_mean"])
        sig = df["condition"].map(rs[f"{s}_std"]).replace(0, 1e-6)
        df[f"{s}_z"] = (df[s].values - mu.values) / sig.values

    return df


if "df_all" not in st.session_state:
    st.session_state.df_all = None
if "selected_unit" not in st.session_state:
    st.session_state.selected_unit = None
if "units" not in st.session_state:
    st.session_state.units = []
if "api_host" not in st.session_state:
    st.session_state.api_host = "https://cmapss-jet-engine-simulated-data.onrender.com/"


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈️ Engine Health")
    st.divider()

    # 添加自定义CSS使按钮等宽且美观
    st.markdown(
        """
    <style>
    div.stButton {
        width: 100%;
        margin-bottom: 8px;
    }

    div.stButton > button {
        width: 100%;
        background-color: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 10px 0px;
        font-weight: 500;
        transition: all 0.3s ease;
        text-align: left;
        padding-left: 15px;
    }
    
    div.stButton > button:hover {
        background-color: rgba(78, 121, 167, 0.1);
        border-color: #4e79a7;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:focus {
        background-color: #4e79a7;
        color: white;
        border-color: #4e79a7;
    }
    
    .stButton button span {
        margin-right: 8px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 初始化session state
    if "page" not in st.session_state:
        st.session_state.page = "Getting Started"

    # 创建导航按钮 - 使用emoji让按钮更直观
    if st.button("🚀 Getting Started", use_container_width=True):
        st.session_state.page = "Getting Started"
        st.rerun()

    # 添加一个小间距
    if st.button("📊 Fleet Overview", use_container_width=True):
        st.session_state.page = "Fleet Overview"
        st.rerun()

    if st.button("📈 Condition Analysis", use_container_width=True):
        st.session_state.page = "Condition Analysis"
        st.rerun()

    if st.button("⚙️ Sensor Explorer", use_container_width=True):
        st.session_state.page = "Sensor Explorer"
        st.rerun()

    if st.button("🔍 Predictive Deep-Dive", use_container_width=True):
        st.session_state.page = "Engine Deep-Dive"
        st.rerun()

    # 显示当前页面（可选）
    st.markdown("---")
    st.caption(f"Current: **{st.session_state.page}**")

    # 更新section变量
    section = st.session_state.page


if section == "Getting Started":
    st.header("Getting Started")
    st.markdown("## ✈️ Turbofan Engine Health Dashboard")
    st.markdown(
        "Upload a CMAPSS test file (e.g. `test_FD001.txt`) below to get started. "
        "The file should be space-separated with 26 columns per the CMAPSS format."
    )
    # st.info(
    #     "**API host** — point to your running FastAPI server "
    #     "(default `http://localhost:8000`). Predictions are fetched on demand."
    # )
    uploaded = st.file_uploader("Upload test data (.txt)", type=["txt"])
    st.session_state.uploaded = uploaded
    api_host = st.text_input("API host", value=st.session_state.api_host)
    st.session_state.api_host = api_host

    if uploaded:
        df_all = load_data(uploaded.read())
        units = sorted(df_all["unit"].unique().tolist())
        st.session_state.df_all = df_all
        st.session_state.units = units
    else:
        df_all = None
        selected_unit = None


df_all = st.session_state.df_all
selected_unit = st.session_state.selected_unit
units = st.session_state.units
api_host = st.session_state.api_host
uploaded = st.session_state.uploaded
# ── no data state ─────────────────────────────────────────────────────────────
if df_all is None:
    # st.markdown("## ✈️ Turbofan Engine Health Dashboard")
    # st.markdown(
    #     "Upload a CMAPSS test file (e.g. `test_FD001.txt`) from the sidebar to get started. "
    #     "The file should be space-separated with 26 columns per the CMAPSS format."
    # )
    # st.info(
    #     "**API host** — point to your running FastAPI server "
    #     "(default `http://localhost:8000`). Predictions are fetched on demand."
    # )
    # st.stop()
    section = "Fleet Overview"
    st.stop()

df_unit = (
    df_all[df_all["unit"] == selected_unit].sort_values("cycle").reset_index(drop=True)
)
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "Fleet Overview":
    df_all = st.session_state.df_all
    selected_unit = st.session_state.selected_unit
    units = st.session_state.units
    api_host = st.session_state.api_host
    uploaded = st.session_state.uploaded

    st.header("Fleet Overview")
    # selected_unit = st.selectbox("Engine unit", units, index=0)
    df_unit = (
        df_all[df_all["unit"] == selected_unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    cycles_per_unit = df_all.groupby("unit")["cycle"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines", len(units))
    c2.metric("Avg cycles observed", f"{cycles_per_unit.mean():.0f}")
    c3.metric("Max cycles", int(cycles_per_unit.max()))
    c4.metric("Min cycles", int(cycles_per_unit.min()))

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Cycle length distribution")
        cpu = cycles_per_unit.reset_index().rename(columns={"cycle": "max_cycle"})
        fig = px.histogram(
            cpu,
            x="max_cycle",
            nbins=20,
            labels={"max_cycle": "Cycles observed"},
            color_discrete_sequence=["#4e79a7"],
        )
        fig.update_layout(yaxis_title="Engines", height=300, **base_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Cycles per engine")
        fig2 = px.bar(
            cpu.sort_values("unit"),
            x="unit",
            y="max_cycle",
            labels={"unit": "Engine unit", "max_cycle": "Cycles"},
            color="max_cycle",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(coloraxis_showscale=False, height=300, **base_layout())
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENGINE DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Engine Deep-Dive":
    st.header(f"Engine Deep-Dive")

    selected_unit = st.selectbox("Engine unit", units, index=0)
    df_unit = (
        df_all[df_all["unit"] == selected_unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    st.subheader("Fleet RUL predictions")
    st.caption("Calls the API once per engine — may take a moment for large fleets.")

    if st.button("▶  Run fleet predictions", type="primary"):
        results = []
        bar = st.progress(0, text="Predicting…")
        for i, u in enumerate(units):
            rul = call_predict(api_host, df_all, u)
            results.append({"unit": u, "predicted_rul": rul if rul is not None else -1})
            bar.progress((i + 1) / len(units), text=f"Unit {u} ({i + 1}/{len(units)})")
            time.sleep(0.01)
        bar.empty()
        st.session_state["fleet_rul"] = pd.DataFrame(results)

    rul_df = st.session_state.get("fleet_rul")
    if rul_df is not None:
        rul_df = rul_df.copy()
        rul_df["color"] = rul_df["predicted_rul"].apply(
            lambda v: rul_color(v) if v >= 0 else "#aaaaaa"
        )
        fig3 = go.Figure(
            go.Bar(
                x=rul_df["unit"],
                y=rul_df["predicted_rul"].clip(lower=0),
                marker_color=rul_df["color"],
                hovertemplate="Unit %{x}<br>Predicted RUL: %{y:.1f} cycles<extra></extra>",
            )
        )
        fig3.add_hline(
            y=30,
            line_dash="dash",
            line_color="#e15759",
            annotation_text="Critical (30)",
            annotation_position="right",
        )
        fig3.add_hline(
            y=70,
            line_dash="dot",
            line_color="#f28e2b",
            annotation_text="Warning (70)",
            annotation_position="right",
        )
        fig3.update_layout(
            xaxis_title="Engine unit",
            yaxis_title="Predicted RUL (cycles)",
            height=380,
            **base_layout(),
        )
        st.plotly_chart(fig3, use_container_width=True)

        valid = rul_df[rul_df["predicted_rul"] >= 0]["predicted_rul"]
        if len(valid):
            n_h = int((valid > 70).sum())
            n_w = int(((valid > 30) & (valid <= 70)).sum())
            n_c = int((valid <= 30).sum())
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("🟢 Healthy", n_h, f"{n_h / len(valid) * 100:.0f}% of fleet")
            gc2.metric("🟡 Warning", n_w, f"{n_w / len(valid) * 100:.0f}% of fleet")
            gc3.metric("🔴 Critical", n_c, f"{n_c / len(valid) * 100:.0f}% of fleet")
    else:
        st.info("Click **Run fleet predictions** to call the API for all engines.")
    st.divider()
    st.subheader("RUL trajectory from API")
    if st.button("▶  Fetch RUL trajectory", type="primary"):
        with st.spinner("Calling /predict/trajectory…"):
            traj = call_trajectory(api_host, df_all, selected_unit)
        if traj is None:
            st.error("API call failed — is the server running at the configured host?")
        else:
            st.session_state[f"traj_{selected_unit}"] = pd.DataFrame(traj)

    traj_df = st.session_state.get(f"traj_{selected_unit}")
    if traj_df is not None:
        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=traj_df["cycle"],
                y=traj_df["predicted_rul"],
                mode="lines",
                name="Predicted RUL",
                line=dict(color="#4e79a7", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(78,121,167,0.10)",
            )
        )
        fig_t.add_hline(
            y=30,
            line_dash="dash",
            line_color="#e15759",
            annotation_text="Critical",
            annotation_position="right",
        )
        fig_t.add_hline(
            y=70,
            line_dash="dot",
            line_color="#f28e2b",
            annotation_text="Warning",
            annotation_position="right",
        )
        fig_t.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Predicted RUL (cycles)",
            yaxis=dict(range=[0, RUL_CAP + 5]),
            height=340,
            **base_layout(),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        last_rul = traj_df["predicted_rul"].iloc[-1]
        st.metric(
            "Predicted RUL at last observed cycle",
            f"{last_rul:.1f} cycles",
            delta="⚠ Schedule maintenance" if last_rul <= 70 else "✓ Within safe range",
            delta_color="inverse" if last_rul <= 70 else "off",
        )
    else:
        st.info("Click **Fetch RUL trajectory** to call the API.")

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SENSOR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONDITION ANALYSIS (OOD)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Condition Analysis":
    st.header(f"Condition Analysis")

    selected_unit = st.selectbox("Engine unit", units, index=0)
    df_unit = (
        df_all[df_all["unit"] == selected_unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    st.caption(
        "Checks whether the engine's operating conditions fall within the "
        "clusters seen during training (K-means on os1/os2/os3). "
        "High centroid distance or large sensor z-scores indicate out-of-distribution (OOD) cycles."
    )

    km, rs, scols = load_artefacts()
    if km is None:
        st.warning(
            "Model artefacts not found. Make sure `models/` contains "
            "`condition_clusterer.joblib` and `normalisation_stats.parquet`."
        )
        st.stop()

    ood_df = compute_ood(df_unit, km, rs, scols)

    # OOD threshold: 99th percentile of all distances in the uploaded file
    all_os = df_all[["os1", "os2", "os3"]].to_numpy()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_dists = np.linalg.norm(
            all_os[:, None, :] - km.cluster_centers_[None, :, :], axis=2
        ).min(axis=1)
    ood_thresh = float(np.percentile(all_dists, 99))
    ood_mask = ood_df["dist_to_centroid"] > ood_thresh
    n_ood = int(ood_mask.sum())
    n_total = len(ood_df)
    pct_ood = n_ood / n_total * 100

    # ── summary cards ─────────────────────────────────────────────────────────
    st.markdown(
        """
    <style>
    .card {
        background-color: #111827;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
    }
    .card-title {
        font-size: 14px;
        color: #9CA3AF;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 28px;
        font-weight: 600;
        color: #F9FAFB;
    }
    .card-delta {
        font-size: 14px;
        margin-top: 6px;
    }
    .positive { color: #10B981; }
    .negative { color: #EF4444; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Cards Layout ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Card 1
    with col1:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">Cycles Analysed</div>
            <div class="card-value">{n_total}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Card 2 (with delta)
    delta_class = "negative" if n_ood > 0 else "positive"
    with col2:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">OOD Cycles</div>
            <div class="card-value">{n_ood}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Card 3
    with col3:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">Conditions Assigned</div>
            <div class="card-value">{ood_df["condition"].nunique()}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Card 4
    with col4:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">OOD Threshold</div>
            <div class="card-value">{ood_thresh:.5f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── distance to centroid over time ─────────────────────────────────────────
    st.subheader("Distance to nearest training centroid over time")
    st.caption(
        "Spikes above the red threshold line indicate cycles outside the training OS space."
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Scatter(
            x=ood_df["cycle"],
            y=ood_df["dist_to_centroid"],
            mode="lines",
            name="Centroid distance",
            line=dict(color="#4e79a7", width=1.5),
        )
    )
    if n_ood > 0:
        fig_dist.add_trace(
            go.Scatter(
                x=ood_df.loc[ood_mask, "cycle"],
                y=ood_df.loc[ood_mask, "dist_to_centroid"],
                mode="markers",
                name="OOD cycle",
                marker=dict(color="#e15759", size=7, symbol="x"),
            )
        )
    fig_dist.add_hline(
        y=ood_thresh,
        line_dash="dash",
        line_color="#e15759",
        annotation_text=f"OOD threshold ({ood_thresh:.5f})",
        annotation_position="top right",
    )
    fig_dist.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Distance to centroid",
        height=280,
        legend=dict(orientation="h", y=1.12),
        **base_layout(),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── 3D OS scatter — engine vs centroids ────────────────────────────────────
    st.subheader("Operating condition space")
    st.caption(
        "Engine cycles plotted against training centroids. OOD cycles shown in red."
    )

    centroids_df = (
        pd.DataFrame(km.cluster_centers_, columns=["os1", "os2", "os3"])
        .reset_index()
        .rename(columns={"index": "condition"})
    )
    centroids_df["label"] = "Centroid " + centroids_df["condition"].astype(str)

    fig_3d = go.Figure()

    # normal cycles
    normal = ood_df[~ood_mask]
    fig_3d.add_trace(
        go.Scatter3d(
            x=normal["os1"],
            y=normal["os2"],
            z=normal["os3"],
            mode="markers",
            marker=dict(size=3, color="#4e79a7", opacity=0.6),
            name="Normal cycles",
        )
    )

    # OOD cycles
    if n_ood > 0:
        ood_pts = ood_df[ood_mask]
        fig_3d.add_trace(
            go.Scatter3d(
                x=ood_pts["os1"],
                y=ood_pts["os2"],
                z=ood_pts["os3"],
                mode="markers",
                marker=dict(size=6, color="#e15759", symbol="x", opacity=0.9),
                name="OOD cycles",
            )
        )

    # training centroids
    fig_3d.add_trace(
        go.Scatter3d(
            x=centroids_df["os1"],
            y=centroids_df["os2"],
            z=centroids_df["os3"],
            mode="markers+text",
            marker=dict(size=10, color="#f28e2b", symbol="diamond", opacity=1),
            text=centroids_df["label"],
            textposition="top center",
            name="Training centroids",
        )
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="os1",
            yaxis_title="os2",
            zaxis_title="os3",
        ),
        height=500,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.05),
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    st.subheader("Operating conditions")
    oc1, oc2, oc3 = st.columns(3)
    for col, os_col, label in [
        (oc1, "os1", "Altitude (os1)"),
        (oc2, "os2", "Mach number (os2)"),
        (oc3, "os3", "TRA (os3)"),
    ]:
        fig_oc = px.scatter(
            df_unit,
            x="cycle",
            y=os_col,
            labels={"cycle": "Cycle", os_col: label},
            color_discrete_sequence=["#76b7b2"],
            opacity=0.55,
        )
        fig_oc.update_traces(marker_size=4)
        fig_oc.update_layout(
            height=200, title=dict(text=label, font_size=12), **base_layout()
        )
        col.plotly_chart(fig_oc, use_container_width=True)

    st.divider()


elif section == "Sensor Explorer":
    st.header("Sensor Explorer")
    selected_unit = st.selectbox("Engine unit", units, index=0)
    df_unit = (
        df_all[df_all["unit"] == selected_unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cycles observed", len(df_unit))
    m2.metric("Last cycle", int(df_unit["cycle"].iloc[-1]))
    m3.metric(
        "T24 (last cycle)",
        f"{df_unit['s2'].iloc[-1]:.2f} °R",
        help="Total temperature at LPC outlet",
    )
    m4.metric(
        "Ps30 (last cycle)",
        f"{df_unit['s11'].iloc[-1]:.2f} psia",
        help="Static pressure at HPC outlet",
    )

    st.divider()

    st.subheader("Sensor trajectories")
    selected_sensors = st.multiselect(
        "Sensors to display",
        options=SCOLS,
        default=["s2"],
        format_func=lambda s: f"{s} — {SENSOR_LABELS.get(s, s)}",
    )
    if selected_sensors:
        fig_s = go.Figure()
        for s in selected_sensors:
            fig_s.add_trace(
                go.Scatter(
                    x=df_unit["cycle"],
                    y=df_unit[s],
                    mode="lines",
                    name=SENSOR_LABELS.get(s, s),
                    line=dict(width=1.8),
                )
            )
        fig_s.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Raw sensor value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=360,
            **base_layout(),
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Select at least one sensor above.")

    st.divider()

    last_cycles = (
        df_all.sort_values("cycle")
        .groupby("unit")
        .last()
        .reset_index()[["unit"] + SCOLS]
    )

    st.subheader("Last-cycle sensor value across all engines")
    heatmap_sensor = st.selectbox(
        "Sensor",
        SCOLS,
        format_func=lambda s: f"{s} — {SENSOR_LABELS.get(s, s)}",
        index=SCOLS.index("s11"),
    )
    col_vals = last_cycles[heatmap_sensor]
    norm_vals = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
    fig_h = go.Figure(
        go.Bar(
            x=last_cycles["unit"],
            y=last_cycles[heatmap_sensor],
            marker=dict(
                color=norm_vals,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Norm.", thickness=12),
            ),
            hovertemplate="Unit %{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig_h.update_layout(
        xaxis_title="Engine unit",
        yaxis_title=SENSOR_LABELS.get(heatmap_sensor, heatmap_sensor),
        height=320,
        **base_layout(),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.divider()

    st.subheader("Sensor correlation matrix (last observed cycle, all engines)")
    corr = last_cycles[SCOLS].rename(columns=SENSOR_LABELS).corr()
    fig_c = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig_c.update_layout(
        height=520,
        coloraxis_colorbar=dict(title="r", thickness=12),
        **base_layout(),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    st.divider()
