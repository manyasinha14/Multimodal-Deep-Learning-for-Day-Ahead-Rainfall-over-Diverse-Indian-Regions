import pandas as pd
import streamlit as st

from core import charts, data
from core.ui import show_chart, show_table

cfg = data.load_config()
daily = data.load_daily()
pred = data.load_predictions()
order = cfg["region_names"]
thr = cfg["rain_mm"]

st.title("🗺️ Regional climate")
st.markdown("The four regions were chosen because their rainfall behaves very differently through the year — "
            "which is exactly what a multimodal model has to cope with.")

d = daily.dropna(subset=["imd_mm"])
rows = []
for r in order:
    x = d[d["region"] == r]
    rows.append({"Region": data.LABELS[r], "Regime": data.REGIME[r],
                 "Mean rain (mm/day)": round(x["imd_mm"].mean(), 2),
                 "Wet days (≥ 2.5 mm)": f"{(x['imd_mm'] >= thr).mean():.0%}",
                 "Heavy days (≥ 20 mm)": f"{(x['imd_mm'] >= 20).mean():.1%}",
                 "Wettest month": pd.Timestamp(2000, int(x.groupby(x['date'].dt.month)['imd_mm'].mean().idxmax()), 1).strftime("%B"),
                 "Wettest day (mm)": round(x["imd_mm"].max(), 1)})
show_table(pd.DataFrame(rows))
st.caption(f"IMD 0.25° gridded rainfall averaged over each region's land cells, {d['date'].min():%b %Y} – "
           f"{d['date'].max():%b %Y} (from {pd.Timestamp(cfg['imd_realtime_from']):%b %Y}: IMD real-time provisional grids).")

t1, t2, t3 = st.tabs(["Seasonal cycle", "Month by month", "Do the sources agree with IMD?"])
with t1:
    metric = st.radio("Show", ["Mean daily rainfall", "Share of wet days"], horizontal=True)
    show_chart(charts.monthly_heatmap(d, order, data.SHORT, "mean" if metric.startswith("Mean") else "wet", thr),
               key="clim")
    st.caption("Kerala is wet from the June onset onwards; the plains and Rajasthan get nearly all rain in July–"
               "September; Chennai's wettest months are October–December (north-east monsoon).")
with t2:
    r = st.selectbox("Region", order, format_func=lambda x: data.LABELS[x], key="tot_region")
    show_chart(charts.monthly_totals(d, r), key="totals")
with t3:
    r = st.selectbox("Region", order, format_func=lambda x: data.LABELS[x], key="src_region")
    p = pred[pred["region"] == r].rename(columns={"obs_mm": "imd_next"})
    x = d[d["region"] == r].sort_values("date").copy()
    x["imd_next"] = x["imd_mm"].shift(-1)            # IMD day D = 08:30 D-1 -> 08:30 D, closest to calendar day D-1
    a, b, c = st.columns(3)
    with c:
        st.markdown("**GPM IMERG satellite precipitation vs IMD**")
        if "imerg_mm" in x:
            show_chart(charts.source_scatter(x, "imerg_mm", "IMERG (UTC day D−1)", charts.C["gpm"]), key="sc_gpm")
            st.caption(f"Spearman ρ = {x['imerg_mm'].corr(x['imd_next'], method='spearman'):.2f} — a true satellite "
                       "measurement, published about 4 hours after each UTC day ends.")
    with a:
        st.markdown("**NWP forecast (issued ~24 h ahead) vs IMD**")
        show_chart(charts.source_scatter(p, "nwp_mm", "NWP forecast", charts.C["nwp"]), key="sc_nwp")
        st.caption(f"Spearman ρ = {p['nwp_mm'].corr(p['imd_next'], method='spearman'):.2f} — the forecast is "
                   "aggregated over the exact IMD window, so this is a fair comparison.")
    with b:
        st.markdown("**NASA POWER precipitation (blended product) vs IMD**")
        show_chart(charts.source_scatter(x, "PRECTOTCORR", "POWER (calendar day D−1)", charts.C["sat"]), key="sc_pw")
        st.caption(f"Spearman ρ = {x['PRECTOTCORR'].corr(x['imd_next'], method='spearman'):.2f} — a good match, but "
                   "POWER is published 2–3 days late, so the live model can only use it as recent history.")
