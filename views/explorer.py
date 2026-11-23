import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core import charts, data
from core.ui import pct, rain_badge, show_chart

cfg = data.load_config()
pred = data.load_predictions()
daily = data.load_daily()
images = data.load_images()

st.title("📅 Forecast explorer")
st.markdown("Pick a region and a day to see what the model forecast from the data available beforehand, "
            "what actually fell, and which data source it relied on.")

c = st.columns([1.4, 1])
region = c[0].selectbox("Region", cfg["region_names"], format_func=lambda r: data.LABELS[r])
sub = pred[pred["region"] == region]
dmin, dmax = sub["date"].min().date(), sub["date"].max().date()
default = pd.Timestamp(cfg["test_start"]).date()
day = c[1].date_input("Target day", value=max(dmin, min(default, dmax)), min_value=dmin, max_value=dmax,
                      format="DD/MM/YYYY")
row = sub[sub["date"] == pd.Timestamp(day)]
if row.empty:
    st.warning(f"No forecast for {day:%d %b %Y} (an input or the IMD observation is missing for that day).")
    st.stop()
row = row.iloc[0]
if row["split"] != "test":
    st.info(f"This day is in the **{data.SPLIT_LABEL[row['split']]}**. For a forecast the model never saw, pick a day "
            f"from {pd.Timestamp(cfg['test_start']):%d %b %Y} onwards.", icon="ℹ️")

k = st.columns(4)
with k[0], st.container(border=True):
    st.markdown("**Model forecast**")
    st.markdown(rain_badge(row["rain_fc"]))
    st.caption(f"P(rain) {pct(row['p_rain'])} · {row['pred_mm']:.1f} mm (range {row['lo_mm']:.1f}–{row['hi_mm']:.1f})")
with k[1], st.container(border=True):
    st.markdown("**What happened (IMD)**")
    st.markdown("🌧️ **Rained**" if row["rain_obs"] else "☀️ **Dry**")
    st.caption(f"{row['obs_mm']:.1f} mm observed")
with k[2], st.container(border=True):
    st.markdown("**Verdict**")
    st.markdown("✅ **Correct call**" if row["correct"] else "❌ **Wrong call**")
    st.caption(("hit" if row["rain_obs"] else "correct no-rain") if row["correct"] else
               ("miss" if row["rain_obs"] else "false alarm"))
with k[3], st.container(border=True):
    st.markdown("**Raw NWP forecast**")
    st.markdown("🌧️ rain" if row["nwp_rain"] else "☀️ no rain")
    st.caption(f"{row['nwp_mm']:.1f} mm · {'correct' if row['nwp_rain'] == row['rain_obs'] else 'wrong'}")

d = daily[daily["region"] == region].set_index("date")
t = pd.Timestamp(row["date"])
a, b = st.columns([1.7, 1])
with a:
    st.markdown("**Inputs on the calendar and the forecast** — each source at its publication delay")
    show_chart(charts.timeline_chart(d, t, cfg["lag"], cfg["window"], row["pred_mm"], row["lo_mm"], row["hi_mm"],
                                     obs=row["obs_mm"]), key="exp_tl")
with b:
    st.markdown("**Which data source the model relied on**")
    gn = data.gate_names(cfg)
    show_chart(charts.gates_chart([row[f"gate_{g}"] for g in gn], gn, data.MODALITY_NAMES), key="exp_g")
    img = images.get(f"{region}__{row['img_date']}")
    if img is not None and img.max() > 0:
        st.image(Image.fromarray(img).resize((200, 200), Image.NEAREST),
                 caption=f"MODIS true-colour, {pd.Timestamp(str(row['img_date'])):%d %b %Y}", width=200)
    else:
        st.caption("No MODIS image for the input day (passed to the model as blank, as in training).")

st.markdown("**The surrounding three weeks**")
ctx = sub[(sub["date"] >= t - pd.Timedelta(days=10)) & (sub["date"] <= t + pd.Timedelta(days=10))]
fig = charts.season_chart(ctx, cfg["rain_mm"])
fig.add_vline(x=t, line_dash="dot", line_color=charts.C["ink"])
show_chart(fig, key="exp_ctx")
