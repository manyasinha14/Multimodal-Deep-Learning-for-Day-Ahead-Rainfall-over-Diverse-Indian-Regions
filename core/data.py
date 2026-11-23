"""Bundle loading and shared lookups for all pages."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from core import features as FT
from core import inference
from core.metrics import scores
from core.paths import BUNDLE, REQUIRED

LABELS = {"coastal_chennai": "Coastal · Chennai", "plains_delhi_lko": "Plains · Delhi–Lucknow",
          "ghats_kerala": "Orographic · Kerala Western Ghats", "arid_rajasthan": "Semi-arid · Rajasthan"}
SHORT = {"coastal_chennai": "Chennai", "plains_delhi_lko": "Delhi–Lucknow",
         "ghats_kerala": "Kerala Ghats", "arid_rajasthan": "Rajasthan"}
REGIME = {"coastal_chennai": "Coastal; most rain in the Oct–Dec north-east monsoon",
          "plains_delhi_lko": "Indo-Gangetic plains; monsoon-trough rainfall",
          "ghats_kerala": "Orographic; monsoon flow against the Western Ghats",
          "arid_rajasthan": "Semi-arid; sparse, episodic rainfall"}
MODALITY_NAMES = {"nwp": "NWP forecast (Open-Meteo)", "imd": "IMD observations",
                  "gpm": "GPM IMERG satellite precip.", "sat": "NASA POWER precip. & cloud",
                  "img": "MODIS imagery"}
SPLIT_LABEL = {"train": "training period (in-sample)", "val": "validation period", "test": "test period (unseen)"}


def missing_bundle_files():
    return [f for f in REQUIRED if not (BUNDLE / f).exists()]


@st.cache_resource(show_spinner=False)
def load_config():
    cfg = json.loads((BUNDLE / "config.json").read_text())
    if cfg.get("version") != FT.VERSION:
        raise RuntimeError(f"Bundle version {cfg.get('version')} does not match the portal (v{FT.VERSION}). "
                           "Re-export the bundle with the v3 notebook.")
    FT.configure(cfg["nwp_feats"], cfg.get("lag"), cfg.get("power_groups"))
    return cfg


@st.cache_resource(show_spinner=False)
def load_predictions():
    p = pd.read_csv(BUNDLE / "predictions.csv", parse_dates=["date"])
    cfg = load_config()
    p["rain_obs"] = p["obs_mm"] >= cfg["rain_mm"]
    p["rain_fc"] = p["p_rain"] >= cfg["threshold"]
    p["correct"] = p["rain_obs"] == p["rain_fc"]
    p["nwp_rain"] = p["nwp_mm"] >= cfg["rain_mm"]
    return p.sort_values(["region", "date"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_daily():
    d = pd.read_csv(BUNDLE / "daily.csv", parse_dates=["date"])
    return d.sort_values(["region", "date"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_images():
    with np.load(BUNDLE / "modis_images.npz") as z:
        return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner=False)
def load_models():
    return inference.load_models(load_config())


@st.cache_resource(show_spinner=False)
def load_csv(name, **kw):
    p = BUNDLE / name
    return pd.read_csv(p, **kw) if p.exists() else None


def gate_names(cfg):
    return cfg["full"]["modalities"]


def score_rows(df, cfg, by="region", pred_col="pred_mm", rain_col="rain_fc"):
    rows = []
    for key, g in df.groupby(by):
        rows.append({by: key, **scores(g["obs_mm"], g[pred_col], g[rain_col], cfg["rain_mm"])})
    return pd.DataFrame(rows)


def period_text(cfg):
    return (f"Training {pd.Timestamp(cfg['data_start']):%b %Y} – {pd.Timestamp(cfg['train_end']):%b %Y} · "
            f"validation to {pd.Timestamp(cfg['val_end']):%b %Y} · "
            f"test {pd.Timestamp(cfg['test_start']):%d %b %Y} – {pd.Timestamp(cfg['latest']):%d %b %Y}")
