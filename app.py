"""Multimodal Rainfall Forecasting Portal (v3)."""
import streamlit as st
import traceback

st.set_page_config(
    page_title="Rainfall Forecast Portal",
    page_icon="🌧️",
    layout="wide"
)

# Stage 1: Import application modules
try:
    from core.data import load_config, missing_bundle_files
    from core.paths import BUNDLE

except Exception as e:
    st.title("🌧️ Rainfall Forecast Portal")
    st.error("Application import failed.")
    st.code(traceback.format_exc())
    st.stop()

# Stage 2: Validate model bundle
try:
    missing = missing_bundle_files()

    if missing:
        st.title("🌧️ Rainfall Forecast Portal")
        st.error("Required model bundle files are missing:")
        st.write(missing)
        st.write(f"Expected bundle location: {BUNDLE}")
        st.stop()

except Exception:
    st.error("Bundle validation failed.")
    st.code(traceback.format_exc())
    st.stop()

# Stage 3: Load model configuration
try:
    cfg = load_config()

except Exception:
    st.title("🌧️ Rainfall Forecast Portal")
    st.error("Model configuration loading failed.")
    st.code(traceback.format_exc())
    st.stop()

# Stage 4: Configure application pages
try:
    pages = {
        "Overview": [
            st.Page("views/home.py", title="Home",
                    icon="🏠", default=True)
        ],
        "Forecasts": [
            st.Page("views/live.py", title="Live forecast", icon="🛰️"),
            st.Page("views/track.py", title="Live track record", icon="✅"),
            st.Page("views/explorer.py", title="Forecast explorer", icon="📅"),
            st.Page("views/replay.py", title="Period replay", icon="📈"),
        ],
        "Insights": [
            st.Page("views/regions.py", title="Regional climate", icon="🗺️"),
            st.Page("views/performance.py", title="Model performance", icon="🎯"),
            st.Page("views/about.py", title="Method & data", icon="📘"),
        ],
    }

    nav = st.navigation(pages)

    with st.sidebar:
        st.caption(
            f"Gated multimodal fusion · {cfg['n_models']}-model ensemble "
            f"· trained {cfg['trained_on']} on data to {cfg['latest']} "
            "· NWP + IMD + GPM IMERG + NASA POWER + MODIS"
        )

    nav.run()

except Exception:
    st.error("Application page initialization failed.")
    st.code(traceback.format_exc())
    st.stop()