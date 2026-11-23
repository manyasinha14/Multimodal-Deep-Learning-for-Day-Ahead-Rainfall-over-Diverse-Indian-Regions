import datetime as dt

import pandas as pd
import streamlit as st

from core import charts, data
from core.metrics import scores
from core.ui import fmt, pct, show_chart, show_table

cfg = data.load_config()
pred = data.load_predictions()

st.title("📈 Period replay")
st.markdown("Replay any stretch of the record for one region: daily forecast vs observed rainfall, every wrong "
            "rain/no-rain call (red ×), and how the model compares with using the raw NWP forecast directly.")

c = st.columns([1.3, 1.6])
region = c[0].selectbox("Region", cfg["region_names"], format_func=lambda r: data.LABELS[r])
sub_all = pred[pred["region"] == region]
dmin, dmax = sub_all["date"].min().date(), sub_all["date"].max().date()
t0 = max(dmin, pd.Timestamp(cfg["test_start"]).date())
rng = c[1].date_input("Period", value=(t0, dmax), min_value=dmin, max_value=dmax, format="DD/MM/YYYY")
if not isinstance(rng, (tuple, list)) or len(rng) != 2:
    st.info("Pick a start and an end date.")
    st.stop()
a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
sub = sub_all[(sub_all["date"] >= a) & (sub_all["date"] <= b)].sort_values("date")
if sub.empty:
    st.warning("No forecasts in that period.")
    st.stop()
splits = sub["split"].value_counts()
if set(splits.index) != {"test"}:
    st.info("Part of this period is training/validation data, where scores are optimistic: "
            + ", ".join(f"{n} days {data.SPLIT_LABEL[s]}" for s, n in splits.items()), icon="ℹ️")

s = scores(sub["obs_mm"], sub["pred_mm"], sub["rain_fc"], cfg["rain_mm"])
nw = scores(sub["obs_mm"], sub["nwp_mm"], sub["nwp_rain"], cfg["rain_mm"])
k = st.columns(5)
k[0].metric("Days", s["n"])
k[1].metric("Correct calls", pct(s["Accuracy"]), f"{100 * (s['Accuracy'] - nw['Accuracy']):+.1f} pts vs raw NWP")
k[2].metric("POD", pct(s["POD"]))
k[3].metric("FAR", pct(s["FAR"]), delta_color="inverse")
k[4].metric("RMSE", f"{s['RMSE']:.2f} mm", f"{s['RMSE'] - nw['RMSE']:+.2f} vs raw NWP", delta_color="inverse")

show_chart(charts.season_chart(sub, cfg["rain_mm"]), key="replay_main")
st.caption("Click *Raw NWP forecast* in the legend to overlay the weather model's own forecast.")

l, r = st.columns([1, 1.4])
with l:
    st.markdown("**Rain / no-rain calls**")
    show_chart(charts.confusion(s), key="replay_cm")
with r:
    st.markdown("**Month by month**")
    rows = []
    for m, g in sub.groupby(sub["date"].dt.to_period("M")):
        ms = scores(g["obs_mm"], g["pred_mm"], g["rain_fc"], cfg["rain_mm"])
        mn = scores(g["obs_mm"], g["nwp_mm"], g["nwp_rain"], cfg["rain_mm"])
        rows.append({"Month": m.strftime("%b %Y"), "Days": ms["n"], "Observed (mm)": round(g["obs_mm"].sum()),
                     "Forecast (mm)": round(g["pred_mm"].sum()), "Rainy days": int(g["rain_obs"].sum()),
                     "Correct calls": pct(ms["Accuracy"]), "Raw NWP correct": pct(mn["Accuracy"]),
                     "RMSE (mm)": fmt(ms["RMSE"])})
    show_table(pd.DataFrame(rows))

out = sub[["date", "split", "obs_mm", "pred_mm", "lo_mm", "hi_mm", "p_rain", "rain_fc", "nwp_mm", "rain_obs", "correct"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
st.download_button("⬇️ Download these forecasts (CSV)", out.to_csv(index=False).encode(),
                   file_name=f"forecasts_{region}_{a:%Y%m%d}_{b:%Y%m%d}.csv", mime="text/csv")
