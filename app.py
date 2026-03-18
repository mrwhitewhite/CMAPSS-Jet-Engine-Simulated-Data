import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RUL Engine Dashboard", layout="wide")

st.title("Aircraft Engine Remaining Useful Life Dashboard")

# Upload data
# uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
uploaded_file = r"data\processed\train.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)

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

    st.subheader("RUL Distribution")

    fig = px.histogram(
        df,
        x="rul",
        nbins=30,
        title="Distribution of Remaining Useful Life (RUL)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    



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
            title="Sensor vs Rolling Mean",
        )

        st.plotly_chart(fig_rm, use_container_width=True)

    # ======================
    # Engine Comparison
    # ======================

    st.header("⚙️ Engine Comparison")

    final_rul = df.groupby("unit")["rul"].min().reset_index()

    fig_engine = px.bar(
        final_rul, x="unit", y="rul", title="Final RUL Distribution Across Engines"
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
