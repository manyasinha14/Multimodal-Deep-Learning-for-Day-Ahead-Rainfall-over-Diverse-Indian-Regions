"""Live track record: every live forecast is saved, then verified against IMD once IMD publishes that day."""
import pandas as pd
import streamlit as st

from core import features as FT
from core.paths import USER_DATA

LOG = USER_DATA / "forecast_log.csv"
COLS = ["region", "target_date", "issued_at", "p_rain", "rain_fc", "pred_mm", "lo_mm", "hi_mm", "nwp_mm", "model",
        "obs_mm"]


def load_log():
    if not LOG.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(LOG, parse_dates=["target_date", "issued_at"])
    for c in COLS:
        if c not in df:
            df[c] = pd.NA
    return df[COLS]


def record(res):
    """Store this run's forecasts; one row per region and target day (the latest issue before the window opens)."""
    rows = [{"region": r, "target_date": res["target"], "issued_at": res["issued"].floor("min"), "p_rain": o["p"],
             "rain_fc": o["rain"], "pred_mm": o["mm"], "lo_mm": o["lo"], "hi_mm": o["hi"], "nwp_mm": o["nwp_mm"],
             "model": o.get("model", "full"), "obs_mm": pd.NA} for r, o in res["regions"].items() if o["ok"]]
    if not rows:
        return
    USER_DATA.mkdir(parents=True, exist_ok=True)
    log = pd.concat([load_log(), pd.DataFrame(rows)], ignore_index=True)
    log = log.sort_values("issued_at").drop_duplicates(["region", "target_date"], keep="last")
    log.to_csv(LOG, index=False)


@st.cache_data(ttl=3 * 3600, show_spinner="Checking past forecasts against IMD…")
def verify(log_key: str):
    """Fill in IMD observations for logged target days that IMD has published. log_key busts the cache."""
    from core.live import imd_recent, now_ist

    log = load_log()
    pending = log[log["obs_mm"].isna() & (log["target_date"] < now_ist().normalize())]
    if pending.empty:
        return log, None
    try:
        series, _ = imd_recent(pending["target_date"].min(), now_ist().normalize() - pd.Timedelta(days=1))
    except Exception as e:
        return log, str(e)
    for i, row in pending.iterrows():
        v = series[row["region"]].get(pd.Timestamp(row["target_date"]))
        if v is not None and pd.notna(v):
            log.at[i, "obs_mm"] = float(v)
    log.to_csv(LOG, index=False)
    return log, None


def verified(log, rain_mm=FT.RAIN_MM):
    v = log.dropna(subset=["obs_mm"]).copy()
    v["obs_mm"] = v["obs_mm"].astype(float)
    v["rain_fc"] = v["rain_fc"].astype(str).str.lower().isin(["true", "1"])
    v["rain_obs"] = v["obs_mm"] >= rain_mm
    v["nwp_rain"] = v["nwp_mm"].astype(float) >= rain_mm
    v["correct"] = v["rain_fc"] == v["rain_obs"]
    return v
