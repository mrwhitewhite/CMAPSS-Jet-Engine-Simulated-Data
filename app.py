import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RUL Engine Dashboard", layout="wide")

st.title("Aircraft Engine Remaining Useful Life Dashboard")

# Upload data
# uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
uploaded_file = r"D:\CMAPSS-Jet-Engine-Simulated-Data\data\processed\train.csv"

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset Loaded")
    st.sidebar.title("功能菜单")

# Create sidebar navigation 
    if st.sidebar.button("🏠 首页"):
        st.session_state.page = "home"
    
    if st.sidebar.button("📊 数据看板"):
        st.session_state.page = "dashboard"
    
    if st.sidebar.button("📈 图表分析"):
        st.session_state.page = "charts"
    
    if st.sidebar.button("⚙️ 设置"):
        st.session_state.page = "settings"

# Initial page state
    if 'page' not in st.session_state:
        st.session_state.page = "home"


    if st.session_state.page == "home":
        st.title("Overview")
    elif st.session_state.page == "dashboard":
        st.title("Conditions")
    elif st.session_state.page == "charts":
        st.title("xx")
    elif st.session_state.page == "settings":
        st.title("yy")
    
    
    # Sidebar filters
    engine_list = df["unit"].unique()
    selected_engine = st.sidebar.selectbox("Select Engine Unit", engine_list)

    fd_list = df["fd"].unique()
    selected_fd = st.sidebar.selectbox("Select FD", fd_list)

    sensor_columns = [col for col in df.columns if col.startswith("s")]
    selected_sensor = st.sidebar.selectbox("Select Sensor", sensor_columns)

    # Filter engine
    engine_df = df[(df["unit"] == selected_engine) & (df["fd"] == selected_fd)]

    # ======================
    # Dataset Overview
    # ======================

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
        x=selected_sensor,
        y="rul",
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

    st.plotly_chart(fig_sensor, use_container_width=True)

    # ======================
    # Rolling Mean Comparison
    # ======================

    rm_col = f"{selected_sensor}_rm5"

    if rm_col in engine_df.columns:

        st.header("📈 Sensor Trend with Rolling Mean")

        fig_rm = px.line(
            engine_df,
            x="cycle",
            y=[selected_sensor, rm_col],
            title="Sensor vs Rolling Mean"
        )

        st.plotly_chart(fig_rm, use_container_width=True)

    # ======================
    # Engine Comparison
    # ======================

    st.header("⚙️ Engine Comparison")

    final_rul = df.groupby("unit")["rul"].min().reset_index()

    fig_engine = px.bar(
        final_rul,
        x="unit",
        y="rul",
        title="Final RUL Distribution Across Engines"
    )

    st.plotly_chart(fig_engine, use_container_width=True)

else:

    st.info("Upload your dataset to start the dashboard.")