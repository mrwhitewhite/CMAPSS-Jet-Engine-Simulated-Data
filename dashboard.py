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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

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
    "s2": "T24 — LPC out temp",
    "s3": "T30 — HPC out temp",
    "s4": "T50 — LPT out temp",
    "s7": "P30 — HPC pressure",
    "s8": "Nf  — fan speed",
    "s9": "Nc  — core speed",
    "s11": "Ps30 — static pressure",
    "s12": "phi  — fuel/Ps30",
    "s13": "NRf  — corr fan speed",
    "s14": "NRc  — corr core speed",
    "s15": "BPR  — bypass ratio",
    "s17": "htBleed",
    "s20": "W31  — HPT coolant bleed",
    "s21": "W32  — LPT coolant bleed",
}
RUL_CAP = 125


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


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈️ Engine Health")
    st.divider()

    uploaded = st.file_uploader("Upload test data (.txt)", type=["txt"])
    api_host = st.text_input("API host", value="http://localhost:8000")

    st.divider()

    if uploaded:
        df_all = load_data(uploaded.read())
        units = sorted(df_all["unit"].unique().tolist())
        selected_unit = st.selectbox("Engine unit", units, index=0)
    else:
        df_all = None
        selected_unit = None

    st.divider()
    section = st.radio(
        "Section",
        ["Fleet Overview", "Engine Deep-Dive", "Sensor Explorer"],
        label_visibility="collapsed",
    )

# ── no data state ─────────────────────────────────────────────────────────────
if df_all is None:
    st.markdown("## ✈️ Turbofan Engine Health Dashboard")
    st.markdown(
        "Upload a CMAPSS test file (e.g. `test_FD001.txt`) from the sidebar to get started. "
        "The file should be space-separated with 26 columns per the CMAPSS format."
    )
    st.info(
        "**API host** — point to your running FastAPI server "
        "(default `http://localhost:8000`). Predictions are fetched on demand."
    )
    st.stop()

df_unit = (
    df_all[df_all["unit"] == selected_unit].sort_values("cycle").reset_index(drop=True)
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "Fleet Overview":
    st.header("Fleet Overview")

    cycles_per_unit = df_all.groupby("unit")["cycle"].max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Engines", len(units))  # type: ignore
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
    st.subheader("Fleet RUL predictions")
    st.caption("Calls the API once per engine — may take a moment for large fleets.")

    if st.button("▶  Run fleet predictions", type="primary"):
        results = []
        bar = st.progress(0, text="Predicting…")
        for i, u in enumerate(units):  # type: ignore
            rul = call_predict(api_host, df_all, u)
            results.append({"unit": u, "predicted_rul": rul if rul is not None else -1})
            bar.progress((i + 1) / len(units), text=f"Unit {u} ({i + 1}/{len(units)})")  # type: ignore
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENGINE DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Engine Deep-Dive":
    st.header(f"Engine Deep-Dive — Unit {selected_unit}")

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
        default=["s2", "s4", "s11", "s14"],
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

    st.subheader("RUL trajectory from API")
    if st.button("▶  Fetch RUL trajectory", type="primary"):
        with st.spinner("Calling /predict/trajectory…"):
            traj = call_trajectory(api_host, df_all, selected_unit)  # type: ignore
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SENSOR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Sensor Explorer":
    st.header("Sensor Explorer")

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
        text_auto=".2f",  # type: ignore
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

    st.subheader("Sensor distribution across all cycles")
    dist_sensor = st.selectbox(
        "Sensor",
        SCOLS,
        format_func=lambda s: f"{s} — {SENSOR_LABELS.get(s, s)}",
        index=SCOLS.index("s4"),
        key="dist_sensor",
    )
    unit_filter = st.multiselect(
        "Filter to specific units (empty = all)",
        options=units,  # type: ignore
        default=[],
    )
    plot_df = df_all if not unit_filter else df_all[df_all["unit"].isin(unit_filter)]
    fig_d = px.violin(
        plot_df,
        x="unit",
        y=dist_sensor,
        box=True,
        points=False,
        labels={
            "unit": "Engine unit",
            dist_sensor: SENSOR_LABELS.get(dist_sensor, dist_sensor),
        },
        color_discrete_sequence=["#4e79a7"],
    )
    fig_d.update_layout(height=380, **base_layout())
    st.plotly_chart(fig_d, use_container_width=True)
