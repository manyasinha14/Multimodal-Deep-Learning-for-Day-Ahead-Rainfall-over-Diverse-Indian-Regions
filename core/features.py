"""Shared feature definitions for the operational (v3) rainfall model.

This SAME file is used by the Colab training notebook (written there as rainfall_features.py)
and by the portal (core/features.py), so training inputs and live inputs are built identically.

Timing convention (all times IST)
---------------------------------
IMD rainfall for date t is the 24 h total from 08:30 on t-1 to 08:30 on t.
A forecast for IMD day t is issued before 08:30 on t-1 and uses ONLY data that is already
published at that moment:

  modality            days used (relative to t)            why this lag
  NWP forecast        hours 09:00 t-1 .. 08:00 t           forecast issued ~24 h ahead (Open-Meteo)
  IMD observations    t-7 .. t-3  (IMD days)               IMD real-time grids appear with ~1 day delay
  GPM IMERG (Early)   t-7 .. t-3  (UTC days)               Early daily appears ~4 h after the UTC day ends
  POWER precipitation t-9 .. t-5  (calendar days)          POWER meteorology is 2-3 days late
  POWER cloud & solar t-13 .. t-9 (calendar days)          POWER solar/cloud is 5-7 days late
  MODIS image         t-3        (calendar day)            GIBS near-real-time imagery
"""
import threading
from io import BytesIO

import numpy as np
import pandas as pd

VERSION = 3
REGIONS = {
    "coastal_chennai":  {"bbox": [79.6, 12.6, 80.5, 13.4], "type": "coastal"},
    "plains_delhi_lko": {"bbox": [77.0, 26.5, 81.0, 28.8], "type": "plains"},
    "ghats_kerala":     {"bbox": [76.3,  9.5, 77.3, 11.5], "type": "orographic"},
    "arid_rajasthan":   {"bbox": [70.5, 25.0, 74.5, 27.5], "type": "semi_arid"},
}
REGION_NAMES = list(REGIONS)
GRID_N = 3
WINDOW = 5
RAIN_MM = 2.5
IMG_SIZE = 32
LAG = {"imd": 3, "gpm": 3, "power_precip": 5, "power_sc": 9, "img": 3}
HISTORY_DAYS = max(LAG.values()) + WINDOW + 2        # calendar days of inputs needed before a target day

NWP_HOURLY = ["precipitation", "cloud_cover", "relative_humidity_2m", "temperature_2m",
              "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
NWP_FEATS = ["nwp_precip_log", "nwp_pmax_log", "nwp_wet_frac", "nwp_cloud", "nwp_rh", "nwp_t", "nwp_p",
             "nwp_u", "nwp_v"]
IMD_FEATS = ["imd_log"]
GPM_FEATS = ["imerg_log"]
# NASA POWER variables grouped by their publication delay (lag key -> features); the notebook measures the real
# delays when it trains and may lengthen the lags or drop a variable that is published too late to be useful.
POWER_GROUPS = {"power_precip": ["power_precip_log"], "power_sc": ["CLOUD_AMT", "ALLSKY_SFC_SW_DWN"]}
SAT_FEATS = [f for g in POWER_GROUPS.values() for f in g]
MODALITIES = {"nwp": NWP_FEATS, "imd": IMD_FEATS, "gpm": GPM_FEATS, "sat": SAT_FEATS}
TAB_MODALITIES = list(MODALITIES)                    # tabular modalities fed to the network (+ 'img')
SEQ_MODALITIES = ["imd", "gpm", "sat"]               # sequences (GRU); nwp is a vector (MLP)
CTX_COLS = ["doy_sin", "doy_cos"]
POWER_PARAMS = ["PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT"]



def configure(nwp_feats=None, lag=None, power_groups=None):
    """Apply the settings the model was trained with (stored in the bundle's config.json)."""
    global NWP_FEATS, SAT_FEATS, HISTORY_DAYS
    if nwp_feats is not None:
        missing = [f for f in nwp_feats if f not in ALL_NWP_FEATS]
        if missing:
            raise ValueError(f"unknown NWP features {missing}")
        NWP_FEATS = list(nwp_feats)
        MODALITIES["nwp"] = NWP_FEATS
    if lag is not None:
        LAG.update({k: int(v) for k, v in lag.items()})
    if power_groups is not None:
        POWER_GROUPS.clear()
        POWER_GROUPS.update({k: list(v) for k, v in power_groups.items() if v})
    SAT_FEATS = [f for g in POWER_GROUPS.values() for f in g]
    MODALITIES["sat"] = SAT_FEATS
    HISTORY_DAYS = max(LAG.values()) + WINDOW + 2


ALL_NWP_FEATS = list(NWP_FEATS)
OM_PREVIOUS_RUNS ="https://previous-runs-api.open-meteo.com/v1/forecast"   # training: forecasts issued 24 h before
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"                       # live: latest run
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GIBS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
# NASA GPM IMERG Early Run, daily, 0.1 deg, V07 (GES DISC) - needs a free NASA Earthdata login
IMERG_CONCEPT = "C2723754850-GES_DISC"
IMERG_OPENDAP = "https://opendap.earthdata.nasa.gov/collections/{concept}/granules/{ur}"
_NC_LOCK = threading.Lock()          # netCDF/HDF5 is not thread-safe: file reads are serialised (prevents crashes)


# ---------------------------------------------------------------- geometry & dates
def grid_points(bbox, n=GRID_N):
    min_lon, min_lat, max_lon, max_lat = bbox
    ls, lo = (max_lat - min_lat) / n, (max_lon - min_lon) / n
    lats = [round(min_lat + ls * (i + 0.5), 4) for i in range(n)]
    lons = [round(min_lon + lo * (j + 0.5), 4) for j in range(n)]
    return [(a, b) for a in lats for b in lons]


def imd_day_of_hour(times):
    """IMD day that an hourly value ending at `times` belongs to (09:00 t-1 .. 08:00 t -> t)."""
    return (pd.DatetimeIndex(times) + pd.Timedelta(hours=15)).normalize()


def next_imd_day(now_ist):
    """Target IMD day for a forecast issued at `now_ist`: the next window that has not started yet."""
    now = pd.Timestamp(now_ist)
    start_today = now.normalize() + pd.Timedelta(hours=8, minutes=30)
    window_start = start_today if now < start_today else start_today + pd.Timedelta(days=1)
    return window_start.normalize() + pd.Timedelta(days=1)


def doy_ctx(dates):
    doy = pd.DatetimeIndex(dates).dayofyear.values
    return np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1).astype(np.float32)


# ---------------------------------------------------------------- NWP (Open-Meteo)
def hourly_frame(loc_json, suffix=""):
    """One Open-Meteo location -> hourly DataFrame with the base variable names."""
    h = loc_json["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for v in NWP_HOURLY:
        df[v] = pd.to_numeric(pd.Series(h.get(v + suffix, [np.nan] * len(df))), errors="coerce")
    return df


def nwp_daily(hourly_frames):
    """Average the hourly series of all grid points, then aggregate each IMD window (24 hours)."""
    h = pd.concat(hourly_frames).groupby("time", as_index=False).mean(numeric_only=True)
    spd = h["wind_speed_10m"]
    rad = np.deg2rad(h["wind_direction_10m"])
    h["u"], h["v"] = -spd * np.sin(rad), -spd * np.cos(rad)
    h["imd_day"] = imd_day_of_hour(h["time"])
    g = h.groupby("imd_day")
    p = g["precipitation"]
    out = pd.DataFrame({
        "nwp_precip_mm": p.sum(min_count=24),
        "nwp_pmax_mm": p.max(),
        "nwp_wet_frac": g["precipitation"].apply(lambda s: float((s >= 0.1).mean())),
        "nwp_cloud": g["cloud_cover"].mean(), "nwp_rh": g["relative_humidity_2m"].mean(),
        "nwp_t": g["temperature_2m"].mean(), "nwp_p": g["surface_pressure"].mean(),
        "nwp_u": g["u"].mean(), "nwp_v": g["v"].mean(),
        "_hours": g["precipitation"].count(),
    })
    out = out[out["_hours"] == 24].drop(columns="_hours")
    out["nwp_precip_log"] = np.log1p(out["nwp_precip_mm"].clip(lower=0))
    out["nwp_pmax_log"] = np.log1p(out["nwp_pmax_mm"].clip(lower=0))
    out.index.name = "date"
    return out


# ---------------------------------------------------------------- NASA POWER
def power_point(lat, lon, start, end, http_get, params=tuple(POWER_PARAMS)):
    q = {"parameters": ",".join(params), "community": "AG", "latitude": lat, "longitude": lon,
         "start": pd.Timestamp(start).strftime("%Y%m%d"), "end": pd.Timestamp(end).strftime("%Y%m%d"),
         "format": "JSON"}
    r = http_get(POWER_URL, q)
    if getattr(r, "status_code", 200) == 422 and "CLOUD_AMT" in params:
        return power_point(lat, lon, start, end, http_get, tuple(p for p in params if p != "CLOUD_AMT"))
    r.raise_for_status()
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce").where(lambda x: x > -990)
    for c in POWER_PARAMS:
        if c not in df:
            df[c] = np.nan
    df.index.name = "date"
    return df[POWER_PARAMS]


# ---------------------------------------------------------------- IMD gridded rainfall
def imd_region_series(ds, bbox):
    """Regional mean of an IMD gridded xarray Dataset/DataArray over the land cells in bbox."""
    da = ds if not hasattr(ds, "data_vars") else ds[[v for v in ds.data_vars][0]]
    lat = next(c for c in da.coords if c.lower() in ("lat", "latitude"))
    lon = next(c for c in da.coords if c.lower() in ("lon", "longitude"))
    tim = next(c for c in da.coords if c.lower() in ("time", "date"))
    min_lon, min_lat, max_lon, max_lat = bbox
    box = da.where(da >= 0)
    box = box.where((box[lon] >= min_lon) & (box[lon] <= max_lon) & (box[lat] >= min_lat) & (box[lat] <= max_lat),
                    drop=True)
    s = box.mean(dim=[d for d in box.dims if d != tim], skipna=True).to_series()
    s.index = pd.to_datetime(s.index).normalize()
    s.index.name = "date"
    return s.rename("imd_mm")


# ---------------------------------------------------------------- GPM IMERG Early (NASA, OPeNDAP)
def imerg_date(native_id):
    """UTC day of an IMERG daily granule, e.g. '...3IMERG.20240101-S000000-E235959.V07B.nc4' -> 2024-01-01."""
    import re
    m = re.search(r"3IMERG\.(\d{8})-S", native_id)
    return pd.Timestamp(m.group(1)) if m else None


def imerg_granules(search_data, start, end):
    """{UTC day -> granule UR} for IMERG Early daily granules (search_data = earthaccess.search_data)."""
    res = search_data(concept_id=IMERG_CONCEPT, temporal=(pd.Timestamp(start).strftime("%Y-%m-%d"),
                                                          pd.Timestamp(end).strftime("%Y-%m-%d")))
    out = {}
    for g in res:
        ur = g["meta"]["native-id"]
        d = imerg_date(ur)
        if d is not None:
            out[d] = ur
    return out


def imerg_url(ur):
    from urllib.parse import quote
    return IMERG_OPENDAP.format(concept=IMERG_CONCEPT, ur=quote(ur, safe=""))


def imerg_index_ranges(bbox):
    """Index ranges of the IMERG 0.1 deg grid (cell centres lon -179.95.., lat -89.95..) inside bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    i0 = int(np.ceil((min_lon + 179.95) / 0.1 - 1e-6)); i1 = int(np.floor((max_lon + 179.95) / 0.1 + 1e-6))
    j0 = int(np.ceil((min_lat + 89.95) / 0.1 - 1e-6)); j1 = int(np.floor((max_lat + 89.95) / 0.1 + 1e-6))
    return {"lon": (i0, i1), "lat": (j0, j1), "time": (0, 0)}


def imerg_dims(session, url):
    """Dimension order of 'precipitation' from the DAP2 DDS, e.g. ('time', 'lon', 'lat')."""
    import re
    try:
        r = session.get(url + ".dds", timeout=60)
        m = re.search(r"precipitation\[(\w+) = \d+\]\[(\w+) = \d+\]\[(\w+) = \d+\]", r.text)
        if r.status_code == 200 and m:
            return tuple(m.groups())
    except Exception:
        pass
    return ("time", "lon", "lat")                    # documented IMERG V07 layout


def parse_dap2_ascii(text, var="precipitation"):
    """Values of a 3-D variable from a DAP2 .ascii response -> 2-D array (rows = 2nd dim, cols = 3rd dim)."""
    import re
    rows = []
    pat = re.compile(r"^[\w.]*" + var + r"\[\d+\]\[\d+\]\s*,(.*)$")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            rows.append([float(v) for v in m.group(1).split(",") if v.strip()])
    if not rows:
        raise ValueError("no data rows in OPeNDAP ascii response")
    return np.array(rows, dtype=float)


def _imerg_ascii(session, url, rng, dims):
    ce = "precipitation" + "".join(f"[{rng[d][0]}:1:{rng[d][1]}]" for d in dims)
    r = session.get(url + ".ascii?" + ce, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return parse_dap2_ascii(r.text)


def _imerg_dap4(session, url, rng, dims):
    import os
    import tempfile
    from urllib.parse import quote
    ce4 = "/precipitation" + "".join(f"[{rng[d][0]}:{rng[d][1]}]" for d in dims)
    r = session.get(url + ".dap.nc4?dap4.ce=" + quote(ce4, safe=""), timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    fd, tmp = tempfile.mkstemp(suffix=".nc4")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    try:
        with _NC_LOCK:                                   # one HDF5 read at a time
            import xarray as xr
            with xr.open_dataset(tmp) as ds:
                return np.asarray(ds["precipitation"].values, dtype=float)
    finally:
        os.remove(tmp)


def imerg_route(session, ur, dims=("time", "lon", "lat")):
    """Find which OPeNDAP response works ('ascii' preferred, else 'dap4'); raises with the server's answer if neither."""
    url, rng = imerg_url(ur), imerg_index_ranges(REGIONS[REGION_NAMES[0]]["bbox"])
    errors = []
    for route, fn in [("ascii", _imerg_ascii), ("dap4", _imerg_dap4)]:
        try:
            fn(session, url, rng, dims)
            return route
        except Exception as e:
            errors.append(f"{route}: {type(e).__name__}: {e}")
    raise RuntimeError("IMERG download failed on both routes -> " + " | ".join(errors))


def imerg_region_mean(session, ur, bbox, dims=("time", "lon", "lat"), route="ascii"):
    """Regional mean IMERG precipitation (mm/day) for one granule."""
    url, rng = imerg_url(ur), imerg_index_ranges(bbox)
    vals = (_imerg_ascii if route == "ascii" else _imerg_dap4)(session, url, rng, dims)
    vals = np.where(vals >= 0, vals, np.nan)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def fetch_imerg(session, granules, days, regions=None, max_workers=4, dims=None, route=None):
    """DataFrame (UTC day x region) of regional mean IMERG Early precipitation for the requested days."""
    from concurrent.futures import ThreadPoolExecutor
    regions = regions or REGION_NAMES
    days = [pd.Timestamp(d) for d in days if pd.Timestamp(d) in granules]
    if not days:
        return pd.DataFrame(columns=regions, dtype=float)
    dims = dims or imerg_dims(session, imerg_url(granules[days[0]]))
    route = route or imerg_route(session, granules[days[0]], dims)
    jobs = [(d, r) for d in days for r in regions]

    def one(job):
        d, r = job
        for attempt in range(2):
            try:
                return imerg_region_mean(session, granules[d], REGIONS[r]["bbox"], dims, route)
            except Exception:
                continue
        return np.nan
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        vals = list(ex.map(one, jobs))
    df = pd.DataFrame([(d, r, v) for (d, r), v in zip(jobs, vals)], columns=["date", "region", "v"])
    out = df.pivot(index="date", columns="region", values="v").reindex(columns=regions)
    out.index.name = "date"
    return out


# ---------------------------------------------------------------- MODIS (NASA GIBS)
def fetch_modis(bbox, date, http_get, size=IMG_SIZE):
    from PIL import Image
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
              "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor", "SRS": "EPSG:4326",
              "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}", "WIDTH": size, "HEIGHT": size,
              "FORMAT": "image/png", "TIME": pd.Timestamp(str(date)).strftime("%Y-%m-%d")}
    try:
        r = http_get(GIBS_URL, params)
        r.raise_for_status()
        img = np.array(Image.open(BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        return img if img.shape == (size, size, 3) else None
    except Exception:
        return None


# ---------------------------------------------------------------- daily frame & samples
def fill_short_gaps(s, limit=3):
    return s.interpolate(limit=limit, limit_area="inside").ffill(limit=1)


def build_daily(imd_mm, power, nwp, start, end, imerg_mm=None):
    """Calendar-daily frame for one region. imd_mm: Series (IMD day -> mm), power: DataFrame (date ->
    POWER_PARAMS), nwp: DataFrame from nwp_daily (IMD day -> NWP features), imerg_mm: Series (UTC day -> mm)."""
    idx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D", name="date")
    d = pd.DataFrame(index=idx)
    d["imd_mm"] = imd_mm.reindex(idx) if imd_mm is not None else np.nan
    d["imd_log"] = fill_short_gaps(np.log1p(d["imd_mm"].clip(lower=0)))           # input only; target stays raw
    d["imerg_mm"] = imerg_mm.reindex(idx) if imerg_mm is not None else np.nan
    d["imerg_log"] = fill_short_gaps(np.log1p(d["imerg_mm"].clip(lower=0)))
    pw = power.reindex(idx) if power is not None else pd.DataFrame(index=idx, columns=POWER_PARAMS, dtype=float)
    d["PRECTOTCORR"] = pw["PRECTOTCORR"]
    d["power_precip_log"] = fill_short_gaps(np.log1p(pw["PRECTOTCORR"].clip(lower=0)))
    d["CLOUD_AMT"] = fill_short_gaps(pw["CLOUD_AMT"])
    d["ALLSKY_SFC_SW_DWN"] = fill_short_gaps(pw["ALLSKY_SFC_SW_DWN"])
    nw = nwp.reindex(idx) if nwp is not None else pd.DataFrame(index=idx)
    for c in NWP_FEATS + ["nwp_precip_mm"]:
        d[c] = nw[c] if c in nw else np.nan
    return d


def sample_for(d, t, skip=()):
    """Inputs for target IMD day t from a daily frame d (None when anything required is missing).
    skip: modalities the model in use does not need (their arrays are zero-filled)."""
    t = pd.Timestamp(t).normalize()
    W = WINDOW

    def days(lag):
        return pd.date_range(t - pd.Timedelta(days=lag + W - 1), t - pd.Timedelta(days=lag), freq="D")

    try:
        imd = d.loc[days(LAG["imd"]), IMD_FEATS].to_numpy(np.float32)
        gpm = d.loc[days(LAG["gpm"]), GPM_FEATS].to_numpy(np.float32)
        sat = np.concatenate([d.loc[days(LAG[g]), cols].to_numpy(np.float32) for g, cols in POWER_GROUPS.items()],
                             axis=1)
        nwp = d.loc[t, NWP_FEATS].to_numpy(np.float32)
    except KeyError:
        return None
    arrs = {"nwp": nwp, "imd": imd, "gpm": gpm, "sat": sat}
    for m, a in arrs.items():
        if np.isnan(a).any():
            if m not in skip:
                return None
            arrs[m] = np.zeros_like(a)
    return {**arrs, "img_date": (t - pd.Timedelta(days=LAG["img"])).strftime("%Y-%m-%d"), "ctx": doy_ctx([t])[0]}


def make_dataset(d, region, dates, require_target=True, skip=()):
    rid = REGION_NAMES.index(region)
    rows = {k: [] for k in ["nwp", "imd", "gpm", "sat", "img_date", "ctx", "y", "date", "nwp_mm", "imd_last_mm",
                            "imerg_last_mm"]}
    for t in pd.DatetimeIndex(dates):
        s = sample_for(d, t, skip)
        y = d["imd_mm"].get(t, np.nan)
        if s is None or (require_target and not np.isfinite(y)):
            continue
        for k in ["nwp", "imd", "gpm", "sat", "img_date", "ctx"]:
            rows[k].append(s[k])
        rows["y"].append(y)
        rows["date"].append(t)
        rows["nwp_mm"].append(d.at[t, "nwp_precip_mm"])
        rows["imd_last_mm"].append(d["imd_mm"].get(t - pd.Timedelta(days=LAG["imd"]), np.nan))
        rows["imerg_last_mm"].append(d["imerg_mm"].get(t - pd.Timedelta(days=LAG["gpm"]), np.nan))
    n = len(rows["date"])
    out = {"nwp": np.array(rows["nwp"], np.float32).reshape(n, len(NWP_FEATS)),
           "imd": np.array(rows["imd"], np.float32).reshape(n, WINDOW, len(IMD_FEATS)),
           "gpm": np.array(rows["gpm"], np.float32).reshape(n, WINDOW, len(GPM_FEATS)),
           "sat": np.array(rows["sat"], np.float32).reshape(n, WINDOW, len(SAT_FEATS)),
           "ctx": np.array(rows["ctx"], np.float32).reshape(n, len(CTX_COLS)),
           "img_date": np.array(rows["img_date"]), "y": np.array(rows["y"], float),
           "date": pd.DatetimeIndex(rows["date"]), "nwp_mm": np.array(rows["nwp_mm"], float),
           "imd_last_mm": np.array(rows["imd_last_mm"], float),
           "imerg_last_mm": np.array(rows["imerg_last_mm"], float), "rid": np.full(n, rid, np.int64),
           "region": np.array([region] * n)}
    return out


def concat(datasets):
    keys = datasets[0].keys()
    return {k: (pd.DatetimeIndex(np.concatenate([np.asarray(d[k]) for d in datasets])) if k == "date"
                else np.concatenate([d[k] for d in datasets])) for k in keys}


def subset(ds, mask):
    return {k: (v[mask] if not isinstance(v, pd.DatetimeIndex) else v[mask]) for k, v in ds.items()}


# ---------------------------------------------------------------- scaling & model inputs
def fit_scaler(ds):
    sc = {}
    for m in TAB_MODALITIES:
        x = ds[m].reshape(-1, ds[m].shape[-1])
        sc[m] = {"mean": x.mean(0).tolist(), "std": (x.std(0) + 1e-6).tolist()}
    return sc


def model_inputs(ds, scaler, images, blank=None):
    """Scaled arrays ready for the network. images: dict 'region__YYYY-MM-DD' -> uint8 (32,32,3)."""
    blank = blank if blank is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    out = {m: ((ds[m] - np.array(scaler[m]["mean"], np.float32)) / np.array(scaler[m]["std"], np.float32)).astype(np.float32)
           for m in TAB_MODALITIES}
    raw = np.stack([images.get(f"{r}__{d}", blank) for r, d in zip(ds["region"], ds["img_date"])]) if len(ds["region"]) \
        else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    out["img"] = raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    out["rid"], out["ctx"] = ds["rid"].astype(np.int64), ds["ctx"].astype(np.float32)
    return out
