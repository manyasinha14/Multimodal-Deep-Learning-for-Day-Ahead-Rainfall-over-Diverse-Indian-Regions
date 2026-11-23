import pandas as pd
import streamlit as st
from PIL import Image

from core import charts, data, live, tracker
from core import features as FT
from core.ui import pct, rain_badge, show_chart, show_table

cfg = data.load_config()
st.title("🛰️ Live forecast")

if "live_refresh" not in st.session_state:
    st.session_state.live_refresh = 0

# ---- which IMD day to show: the past two weeks, today's window and the next window
now = live.now_ist()
nxt = FT.next_imd_day(now)
MODE_TXT = {"forecast": "forecast", "current": "today, running now", "past": "past day"}


def day_label(d):
    w0 = d - pd.Timedelta(days=1)
    m = live.target_mode(d, now)
    return f"{w0:%a %d %b} 08:30 → {d:%a %d %b} 08:30   ({MODE_TXT[m]})"


options = [nxt - pd.Timedelta(days=k) for k in range(0, 16)]
c1, c2, c3 = st.columns([2.2, 1, 2])
target = c1.selectbox("Rain day (IMD 24-hour window)", options, index=0, format_func=day_label,
                      help="IMD counts rain from 08:30 IST one day to 08:30 IST the next, and names the window "
                           "after the day it ends. The first option is the next window that has not started.")
if c2.button("🔄 Fetch latest data", type="primary"):
    st.session_state.live_refresh += 1
key = now.strftime("%Y-%m-%d %H") + f"#{st.session_state.live_refresh}"
res = live.live_forecast(key, target.strftime("%Y-%m-%d"))
mode = res["mode"]
if mode == "forecast":
    tracker.record(res)                       # only genuine forecasts go into the live track record

t, (w0, w1) = res["target"], res["window"]
st.markdown(f"### Rain from {w0:%a %d %b, %H:%M} to {w1:%a %d %b, %H:%M} IST  \n"
            f"IMD day {t:%d %b %Y} · issued {res['issued']:%d %b %H:%M IST}")
if mode == "forecast":
    st.caption("This window has not started yet: a true forecast. It is saved and checked on the *Live track "
               "record* page once IMD publishes the day. A forecast further ahead is not possible: the model needs "
               "IMD and satellite rainfall up to 3 days before the target day, and those days are not published yet.")
elif mode == "current":
    st.info("**This window is running now** (it began at 08:30 today). The forecast below uses only data that was "
            "published before 08:30, and the weather forecast issued about 24 h earlier — exactly as in training. "
            "IMD publishes the observed rainfall about a day after the window ends.", icon="🕗")
else:
    st.info("**Past day** — re-run with only the data that was available before this window began (a hindcast), "
            "next to what IMD observed.", icon="📅")

if "imd_error" in res:
    st.warning(f"IMD real-time data could not be fetched ({res['imd_error']}); forecasts below may be missing.")
fallback = [r for r, o in res["regions"].items() if o.get("ok") and o.get("model") == "no_gpm"]
if fallback:
    st.warning("**GPM IMERG satellite precipitation was not available**, so these regions use the back-up ensemble "
               "trained without IMERG (also evaluated in the notebook): " + ", ".join(data.SHORT[r] for r in fallback)
               + f".  \nReason: {res['regions'][fallback[0]]['fallback_reason']}", icon="🛰️")

cols = st.columns(len(cfg["region_names"]))
for col, r in zip(cols, cfg["region_names"]):
    out = res["regions"][r]
    with col, st.container(border=True):
        st.markdown(f"**{data.LABELS[r]}**")
        if not out["ok"]:
            st.error("No forecast")
            st.caption(out["error"])
            continue
        st.markdown(rain_badge(out["rain"]))
        st.metric("Chance of rain ≥ 2.5 mm", pct(out["p"]))
        st.metric("Expected rainfall", f"{out['mm']:.1f} mm")
        thr = cfg["threshold"] if out["model"] == "full" else cfg["threshold_no_gpm"]
        st.caption(f"model range {out['lo']:.1f}–{out['hi']:.1f} mm · raw NWP {out['nwp_mm']:.1f} mm · "
                   f"rain called at ≥ {pct(thr)}" + (" · back-up model (no IMERG)" if out["model"] != "full" else ""))
        obs = res["observed"].get(r)
        if obs is not None:
            hit = (obs >= cfg["rain_mm"]) == out["rain"]
            st.markdown(f"**IMD observed: {obs:.1f} mm** {'✅ correct' if hit else '❌ wrong'}")
        elif mode != "forecast":
            st.caption("IMD observation not published yet.")

st.caption("The same inputs, delays and code as in training are used — nothing is estimated.")

ok = [r for r in cfg["region_names"] if res["regions"][r]["ok"]]
if ok:
    st.subheader("Why the model says this")
    tabs = st.tabs([data.SHORT[r] for r in ok])
    for tab, r in zip(tabs, ok):
        out = res["regions"][r]
        with tab:
            a, b = st.columns([1.7, 1])
            with a:
                st.markdown("**Inputs on the calendar and the forecast** — each source at its publication delay")
                show_chart(charts.timeline_chart(out["daily"], t, cfg["lag"], cfg["window"], out["mm"], out["lo"],
                                                 out["hi"], obs=res["observed"].get(r)), key=f"live_tl_{r}")
            with b:
                st.markdown("**Which data source the model relied on**")
                show_chart(charts.gates_chart(out["gates"], out["gate_names"], data.MODALITY_NAMES), key=f"live_g_{r}")
                if out["img"] is not None:
                    st.image(Image.fromarray(out["img"]).resize((200, 200), Image.NEAREST),
                             caption=f"MODIS true-colour, {pd.Timestamp(str(out['img_date'])):%d %b}", width=200)
            st.markdown("**Data used**")
            show_table(out["prov"])