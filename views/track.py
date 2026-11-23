import pandas as pd
import streamlit as st

from core import charts, data, tracker
from core.metrics import scores
from core.ui import fmt, pct, show_chart, show_table

cfg = data.load_config()
st.title("✅ Live track record")
st.markdown("Every forecast made on the *Live forecast* page is stored here and scored against IMD's rainfall "
            "once IMD publishes that day (usually about a day later). This is the model's real-world accuracy.")

log = tracker.load_log()
if log.empty:
    st.info("No live forecasts yet. Open the **Live forecast** page once a day; each visit stores that day's "
            "forecast for all four regions.")
    st.stop()

log, err = tracker.verify(f"{len(log)}-{log['obs_mm'].notna().sum()}-{pd.Timestamp.now():%Y%m%d%H}")
if err:
    st.warning(f"Could not reach IMD to verify new days: {err}")
v = tracker.verified(log, cfg["rain_mm"])
pending = len(log) - len(v)

k = st.columns(4)
k[0].metric("Forecasts stored", len(log))
k[1].metric("Verified against IMD", len(v))
k[2].metric("Waiting for IMD", pending)
if len(v):
    s = scores(v["obs_mm"], v["pred_mm"], v["rain_fc"], cfg["rain_mm"])
    nw = scores(v["obs_mm"], v["nwp_mm"].astype(float), v["nwp_rain"], cfg["rain_mm"])
    k[3].metric("Correct rain/no-rain calls", pct(s["Accuracy"]),
                f"{100 * (s['Accuracy'] - nw['Accuracy']):+.1f} pts vs raw NWP")
    if len(v) < 30:
        st.caption(f"Only {len(v)} verified region-days so far — live scores are noisy until a few weeks have "
                   "accumulated. The test-period scores on *Model performance* are based on far more days.")
    rows = []
    for r in cfg["region_names"]:
        g = v[v["region"] == r]
        if len(g):
            m = scores(g["obs_mm"], g["pred_mm"], g["rain_fc"], cfg["rain_mm"])
            rows.append({"Region": data.LABELS[r], "Verified days": len(g), "Correct calls": pct(m["Accuracy"]),
                         "POD": fmt(m["POD"]), "FAR": fmt(m["FAR"]), "RMSE (mm)": fmt(m["RMSE"])})
    show_table(pd.DataFrame(rows))
    a, b = st.columns([1, 1.3])
    with a:
        show_chart(charts.confusion(s), key="track_cm")
    with b:
        r = st.selectbox("Region", cfg["region_names"], format_func=lambda x: data.LABELS[x])
        g = v[v["region"] == r].rename(columns={"target_date": "date"}).sort_values("date")
        if len(g):
            show_chart(charts.season_chart(g.assign(nwp_mm=g["nwp_mm"].astype(float)), cfg["rain_mm"]), key="track_ts")

st.subheader("All stored forecasts")
tab = log.sort_values(["target_date", "region"], ascending=[False, True]).copy()
tab["region"] = tab["region"].map(data.SHORT)
tab["target_date"] = pd.to_datetime(tab["target_date"]).dt.strftime("%d %b %Y")
tab["issued_at"] = pd.to_datetime(tab["issued_at"]).dt.strftime("%d %b %H:%M")
tab["result"] = [("✅" if (str(f).lower() in ("true", "1")) == (o >= cfg["rain_mm"]) else "❌") if pd.notna(o) else "⏳"
                 for f, o in zip(tab["rain_fc"], pd.to_numeric(tab["obs_mm"], errors="coerce"))]
show_table(tab[["target_date", "region", "issued_at", "model", "p_rain", "pred_mm", "nwp_mm", "obs_mm", "result"]].round(2))
st.download_button("⬇️ Download the log (CSV)", log.to_csv(index=False).encode(), "forecast_log.csv", "text/csv")
st.caption("The log is stored in the portal's user_data folder on this computer. On Streamlit Community Cloud it "
           "resets whenever the app restarts.")
