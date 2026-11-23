import pandas as pd
import streamlit as st

from core import charts, data
from core.metrics import scores
from core.ui import fmt, pct, show_chart

cfg = data.load_config()
pred = data.load_predictions()
daily = data.load_daily()
test = pred[pred["split"] == "test"]
t0, t1 = pd.Timestamp(cfg["test_start"]), pd.Timestamp(cfg["latest"])

st.title("🌧️ Multimodal Rainfall Forecast Portal")
st.markdown(
    "Day-ahead forecasts of **whether it will rain (≥ 2.5 mm)** and **how much**, all year round, for four "
    "contrasting Indian regions. A deep-learning ensemble combines the **numerical weather forecast** for the "
    "next day with **IMD rain-gauge observations**, **NASA GPM IMERG satellite precipitation**, **NASA POWER "
    "precipitation, cloud & solar data** (CERES) and **MODIS satellite images** — and learns which source to "
    "trust in each region.")

s = scores(test["obs_mm"], test["pred_mm"], test["rain_fc"], cfg["rain_mm"])
n = scores(test["obs_mm"], test["nwp_mm"], test["nwp_rain"], cfg["rain_mm"])
st.subheader(f"Tested on {t0:%d %b %Y} – {t1:%d %b %Y}, days the model never saw")
k = st.columns(5)
k[0].metric("Rain / no-rain correct", pct(s["Accuracy"]), f"{100 * (s['Accuracy'] - n['Accuracy']):+.1f} pts vs raw NWP",
            help="Share of region-days where the rain/no-rain call was right. The delta compares with using the "
                 "weather-model forecast directly.")
k[1].metric("Rainy days caught (POD)", pct(s["POD"]))
k[2].metric("False alarm ratio", pct(s["FAR"]), delta_color="inverse")
k[3].metric("Heidke skill score", fmt(s["HSS"]), f"{s['HSS'] - n['HSS']:+.2f} vs raw NWP",
            help="Skill above random chance (0 = none, 1 = perfect).")
k[4].metric("Rainfall error (RMSE)", f"{s['RMSE']:.2f} mm", f"{s['RMSE'] - n['RMSE']:+.2f} mm vs raw NWP",
            delta_color="inverse")
st.caption(f"{s['n']} region-days. Rain is called when the ensemble's probability ≥ {pct(cfg['threshold'])} "
           "(threshold tuned on the validation period). See *Model performance* for every comparison and its "
           "statistical significance.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Study regions")
    wet = daily.dropna(subset=["imd_mm"]).assign(w=lambda d: d["imd_mm"] >= cfg["rain_mm"]).groupby("region")["w"].mean()
    show_chart(charts.region_map(cfg, data.SHORT, {r: f"wet-day frequency {wet[r]:.0%} (all months)"
                                                   for r in cfg["region_names"]}))
with right:
    st.subheader("Test-period skill per region")
    by_r = data.score_rows(test, cfg).set_index("region")
    by_n = data.score_rows(test, cfg, pred_col="nwp_mm", rain_col="nwp_rain").set_index("region")
    for r in cfg["region_names"]:
        with st.container(border=True):
            st.markdown(f"**{data.LABELS[r]}**  \n{data.REGIME[r]}")
            c = st.columns(3)
            c[0].metric("Correct calls", pct(by_r.at[r, "Accuracy"]),
                        f"{100 * (by_r.at[r, 'Accuracy'] - by_n.at[r, 'Accuracy']):+.1f} pts vs NWP")
            c[1].metric("CSI", fmt(by_r.at[r, "CSI"]))
            c[2].metric("RMSE", f"{by_r.at[r, 'RMSE']:.1f} mm")

st.subheader("Pages")
st.markdown("""
| Page | What it shows |
|---|---|
| 🛰️ **Live forecast** | Rain / no rain for the next IMD day (08:30 → 08:30 IST) in all four regions, from today's data |
| ✅ **Live track record** | Every live forecast made here, checked against IMD once IMD publishes the day |
| 📅 **Forecast explorer** | Any day in the record: forecast vs what happened, the inputs used, which source the model trusted |
| 📈 **Period replay** | A month or season for one region: forecast vs observed, wrong calls, comparison with raw NWP |
| 🗺️ **Regional climate** | How the four rainfall regimes differ through the year; how well each source matches IMD |
| 🎯 **Model performance** | Test scores vs baselines and ablations, significance, probability reliability |
| 📘 **Method & data** | Sources, timing, architecture, limitations |
""")
