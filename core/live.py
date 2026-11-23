"""Live forecast: fetches exactly the inputs the model was trained on, with the same delays, and runs it.

GPM IMERG needs a NASA Earthdata login (secrets, environment or ~/.netrc). If IMERG cannot be fetched, the
portal switches to the fallback ensemble that was trained without IMERG and says so on the page.

Nothing is estimated: NWP is the latest Open-Meteo run, IMD comes from IMD's real-time grids (imdlib),
NASA POWER and MODIS from their APIs. The only gap-filling is features.fill_short_gaps, the same rule used in
training (a short internal gap is interpolated; the newest day may be carried forward by one day).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

from core import data, inference
from core import features as FT
from core.paths import USER_DATA


def now_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def http_get(url, params, timeout=45, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(3 * (attempt + 1))
            continue
        return r
    return r


# ---------------------------------------------------------------- IMD real-time (imdlib)
def imd_recent(start, end):
    """Regional IMD rainfall for all regions from IMD's real-time grids. Returns (dict region->Series, last day)."""
    import imdlib as imd

    file_dir = USER_DATA / "imd_realtime"
    file_dir.mkdir(parents=True, exist_ok=True)
    end, last_err = pd.Timestamp(end), None
    for back in range(10):
        e = end - pd.Timedelta(days=back)
        if e < pd.Timestamp(start):
            break
        try:
            ds = imd.get_real_data("rain", pd.Timestamp(start).strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"),
                                   file_dir=str(file_dir)).get_xarray()
            out = {r: FT.imd_region_series(ds, FT.REGIONS[r]["bbox"]) for r in FT.REGION_NAMES}
            last = max(s.dropna().index.max() for s in out.values())
            return out, last
        except Exception as ex:  # newest day(s) not published yet -> step back
            last_err = ex
    raise RuntimeError(f"IMD real-time data unavailable ({last_err})")


# ---------------------------------------------------------------- GPM IMERG (NASA Earthdata)
def _earthdata_env():
    import os
    try:
        sec = st.secrets.get("earthdata", {})
        if sec.get("username") and sec.get("password"):
            os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"] = sec["username"], sec["password"]
    except Exception:                                     # no secrets.toml -> fine, try env / netrc
        pass


@st.cache_resource(show_spinner=False, ttl=6 * 3600)
def earthdata_session():
    import os
    import earthaccess

    _earthdata_env()
    strategies = (["environment"] if os.environ.get("EARTHDATA_USERNAME") else []) + ["netrc"]
    for strategy in strategies:
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth is not None and auth.authenticated:
                return earthaccess.get_requests_https_session(), earthaccess.search_data
        except Exception:
            continue
    raise RuntimeError("NASA Earthdata login not configured (see README: .streamlit/secrets.toml)")


def imerg_recent(start, end):
    session, search = earthdata_session()
    granules = FT.imerg_granules(search, start, end)
    if not granules:
        raise RuntimeError("no IMERG Early granules found for the input days yet")
    days = pd.date_range(start, end)
    return FT.fetch_imerg(session, granules, days)


# ---------------------------------------------------------------- NWP, POWER
def nwp_live(bbox):
    pts = FT.grid_points(bbox)
    params = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
              "hourly": ",".join(FT.NWP_HOURLY), "past_days": 2, "forecast_days": 3, "timezone": "Asia/Kolkata"}
    r = http_get(FT.OM_FORECAST, params)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    return FT.nwp_daily([FT.hourly_frame(loc) for loc in js])


def nwp_previous(bbox, t):
    """NWP issued ~24 h before each hour (Open-Meteo Previous Runs) for IMD day t - exactly the training input."""
    pts = FT.grid_points(bbox)
    params = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
              "hourly": ",".join(v + "_previous_day1" for v in FT.NWP_HOURLY),
              "start_date": (t - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "end_date": t.strftime("%Y-%m-%d"),
              "timezone": "Asia/Kolkata"}
    r = http_get(FT.OM_PREVIOUS_RUNS, params, timeout=90)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    return FT.nwp_daily([FT.hourly_frame(loc, "_previous_day1") for loc in js])


def target_mode(t, now=None):
    """'forecast' = next window (not started), 'current' = window running now, 'past' = window finished."""
    now = now_ist() if now is None else now
    t = pd.Timestamp(t).normalize()
    w0 = t - pd.Timedelta(days=1) + pd.Timedelta(hours=8, minutes=30)
    if now < w0:
        return "forecast"
    return "current" if now < w0 + pd.Timedelta(days=1) else "past"


def power_live(bbox, start, end):
    pts = FT.grid_points(bbox)
    with ThreadPoolExecutor(max_workers=len(pts)) as ex:
        frames = list(ex.map(lambda p: _safe_power(p, start, end), pts))
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("NASA POWER unavailable")
    return pd.concat(frames).groupby(level=0).mean()


def _safe_power(p, start, end):
    try:
        return FT.power_point(p[0], p[1], start, end, http_get)
    except Exception:
        return None


# ---------------------------------------------------------------- provenance
def _status(series, dates):
    got = [d for d in dates if d in series.index and np.isfinite(series.get(d, np.nan))]
    return f"{len(got)}/{len(dates)} days published" + ("" if len(got) == len(dates) else " (gap filled as in training)")


RAW = {"power_precip_log": "PRECTOTCORR", "imd_log": "imd_mm", "imerg_log": "imerg_mm"}


def diagnose(daily, t, use_gpm=True):
    """Which required input days are missing, and when each source last published - for the error message."""
    t = pd.Timestamp(t)
    need = [("Open-Meteo NWP forecast", "nwp_precip_log", None), ("IMD rainfall", "imd_log", "imd")]
    if use_gpm:
        need.append(("GPM IMERG", "imerg_log", "gpm"))
    need += [(f"NASA POWER {RAW.get(c, c)}", c, g) for g, cols in FT.POWER_GROUPS.items() for c in cols]
    msgs = []
    for label, col, lag in need:
        days = [t] if lag is None else list(pd.date_range(t - pd.Timedelta(days=FT.LAG[lag] + FT.WINDOW - 1),
                                                          t - pd.Timedelta(days=FT.LAG[lag])))
        if daily.reindex(days)[col].isna().any():
            last = daily[col].dropna().index.max()
            msgs.append(f"{label} needed up to {days[-1]:%d %b}, last published "
                        + (f"{last:%d %b}" if pd.notna(last) else "nothing in the last weeks"))
    return "; ".join(msgs) or "see the data table"


def provenance(raw_imd, power, nwp_issue, t, img_ok, cfg, imerg=None, imerg_note=""):
    W, L = cfg["window"], cfg["lag"]
    rng = lambda lag: list(pd.date_range(t - pd.Timedelta(days=lag + W - 1), t - pd.Timedelta(days=lag)))
    rows = [
        ("NWP forecast", f"hours 09:00 {t - pd.Timedelta(days=1):%d %b} – 08:00 {t:%d %b}",
         f"Open-Meteo latest run (fetched {nwp_issue:%d %b %H:%M})"),
        ("IMD observations", f"IMD days {rng(L['imd'])[0]:%d %b} – {rng(L['imd'])[-1]:%d %b}",
         _status(raw_imd, rng(L["imd"]))),
        ("GPM IMERG (Early)", f"UTC days {rng(L['gpm'])[0]:%d %b} – {rng(L['gpm'])[-1]:%d %b}",
         _status(imerg, rng(L["gpm"])) if imerg is not None else f"not used — {imerg_note}"),
        *[(f"NASA POWER {', '.join(RAW.get(c, c) for c in cols)}",
           f"{rng(L[g])[0]:%d %b} – {rng(L[g])[-1]:%d %b}", _status(power[RAW.get(cols[0], cols[0])], rng(L[g])))
          for g, cols in FT.POWER_GROUPS.items()],
        ("MODIS image", f"{t - pd.Timedelta(days=L['img']):%d %b}", "observed" if img_ok else "not available (blank, as in training)"),
    ]
    return pd.DataFrame(rows, columns=["Input", "Days used", "Status"])


# ---------------------------------------------------------------- the forecast
@st.cache_data(ttl=3600, show_spinner="Fetching the latest IMD, satellite and forecast data…")
def live_forecast(now_key: str, target: str = ""):
    """Forecast for IMD day `target` (default: the next window that has not started).

    Every input keeps the delay it had in training, so a forecast for today's or a past window uses only data that
    was published before that window began. For a window that has already started, the NWP input is the run issued
    ~24 h earlier (Open-Meteo Previous Runs), exactly as in training; the next window uses the latest run.
    """
    cfg, models = data.load_config(), data.load_models()
    now = now_ist()
    t = pd.Timestamp(target).normalize() if target else FT.next_imd_day(now)
    mode = target_mode(t, now)
    start = t - pd.Timedelta(days=FT.HISTORY_DAYS)
    yesterday = now.normalize() - pd.Timedelta(days=1)
    out = {"target": t, "issued": now, "mode": mode, "observed": {},
           "window": (t - pd.Timedelta(days=1) + pd.Timedelta(hours=8, minutes=30),
                      t + pd.Timedelta(hours=8, minutes=30)), "regions": {}}
    try:
        imd_series, imd_last = imd_recent(start, min(yesterday, t))       # up to t: also gives the observation
        out["imd_last"] = imd_last
        for r, ser in imd_series.items():
            v = ser.get(t)
            if v is not None and pd.notna(v):
                out["observed"][r] = float(v)
    except Exception as e:
        imd_series, out["imd_error"] = {r: pd.Series(dtype=float) for r in FT.REGION_NAMES}, str(e)
    try:
        imerg = imerg_recent(start, min(yesterday, t - pd.Timedelta(days=cfg["lag"]["gpm"])))
    except Exception as e:
        imerg, out["imerg_error"] = None, f"{type(e).__name__}: {e}"
    for r in FT.REGION_NAMES:
        bbox = FT.REGIONS[r]["bbox"]
        try:
            nwp = None
            if mode != "forecast":
                try:
                    nwp = nwp_previous(bbox, t)
                    if t not in nwp.index:
                        nwp = None
                except Exception:
                    nwp = None
            nwp = nwp if nwp is not None else nwp_live(bbox)
            power = power_live(bbox, start, yesterday)
            im = imerg[r] if imerg is not None and r in imerg else None
            daily = FT.build_daily(imd_series[r], power, nwp, start, t, im)
            ds = FT.make_dataset(daily, r, [t], require_target=False)
            use, note = "full", ""
            if len(ds["date"]) == 0 and "no_gpm" in models:          # IMERG missing -> fallback ensemble
                ds = FT.make_dataset(daily, r, [t], require_target=False, skip=("gpm",))
                use = "no_gpm"
                note = out.get("imerg_error", "IMERG days not yet published")
            if len(ds["date"]) == 0:
                raise RuntimeError("inputs not yet published — " + diagnose(daily, t, use_gpm="no_gpm" not in models))
            arch = cfg["full"] if use == "full" else cfg["full_no_gpm"]
            thr = cfg["threshold"] if use == "full" else cfg["threshold_no_gpm"]
            img_date = ds["img_date"][0]
            img = FT.fetch_modis(bbox, img_date, lambda u, p: requests.get(u, params=p, timeout=20))
            inputs = FT.model_inputs(ds, cfg["scaler"], {f"{r}__{img_date}": img} if img is not None else {})
            mean, prob, gates, members = inference.predict(models[use], inputs, cfg, n_gates=len(arch["modalities"]))
            out["regions"][r] = {
                "ok": True, "mm": float(mean[0]), "p": float(prob[0]), "rain": bool(prob[0] >= thr),
                "lo": float(members[:, 0].min()), "hi": float(members[:, 0].max()), "gates": gates[0],
                "gate_names": arch["modalities"], "model": use, "fallback_reason": note,
                "nwp_mm": float(daily.at[t, "nwp_precip_mm"]), "daily": daily, "img": img, "img_date": img_date,
                "prov": provenance(imd_series[r], power, now, t, img is not None, cfg,
                                   im if use == "full" else None, note)}
        except Exception as e:
            out["regions"][r] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out