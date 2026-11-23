"""Multimodal Rainfall Forecasting Portal (v3, operational) — run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Rainfall Forecast Portal", page_icon="🌧️", layout="wide")

from core.data import load_config, missing_bundle_files  # noqa: E402
from core.paths import BUNDLE  # noqa: E402

missing = missing_bundle_files()
if missing:
    st.title("🌧️ Rainfall Forecast Portal")
    st.error("The model bundle is missing: " + ", ".join(missing))
    st.markdown(f"""
1. Run the Colab notebook **Rainfall_operational_model_v3.ipynb** to the end — it downloads
   `rainfall_portal_bundle_v3.zip`.
2. Unzip its contents into: `{BUNDLE}`
3. Refresh this page.
""")
    st.stop()

try:
    cfg = load_config()
except Exception as e:  # wrong bundle version etc.
    st.error(str(e))
    st.stop()

pages = {
    "Overview": [st.Page("views/home.py", title="Home", icon="🏠", default=True)],
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
    st.caption(f"Gated multimodal fusion · {cfg['n_models']}-model ensemble · trained {cfg['trained_on']} on data "
               f"to {cfg['latest']} · NWP + IMD + GPM IMERG + NASA POWER + MODIS")
nav.run()
