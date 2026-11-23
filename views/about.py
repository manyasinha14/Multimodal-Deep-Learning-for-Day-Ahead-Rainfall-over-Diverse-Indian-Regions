import pandas as pd
import streamlit as st

from core import data
from core.paths import ASSETS

cfg = data.load_config()
L = cfg["lag"]

st.title("📘 Method & data")

st.header("The idea")
st.markdown(
    "Each data source sees a different part of the weather: the **numerical weather prediction (NWP)** model "
    "simulates the coming day, **IMD rain gauges** say what has really been falling, **GPM IMERG** measures "
    "precipitation from a satellite constellation, **NASA POWER / CERES** adds precipitation, cloud and radiation, "
    "and **MODIS images** show cloud structure. Every source gets its own encoder, and a "
    "**gating network** learns, for each forecast, how much to trust each one — conditioned on the region and the "
    "time of year. The network predicts the rainfall amount and the probability of a rainy day (≥ 2.5 mm) together.")

st.header("Data and timing")
st.markdown(f"""
IMD rainfall for day **t** is the total from **08:30 IST on t−1 to 08:30 IST on t**. A forecast for day t is made
before that window opens and uses only data already published at that moment:

| Source | What | Days used for target day t | Why that delay |
|---|---|---|---|
| **Open-Meteo NWP** (ECMWF/GFS best match) | precipitation, cloud, humidity, temperature, pressure, wind | forecast for 09:00 t−1 → 08:00 t | training uses forecasts issued 24 h before (Previous Runs archive); live uses the latest run |
| **IMD gridded rainfall** (0.25°) | rain-gauge rainfall | t−{L['imd'] + cfg['window'] - 1} … t−{L['imd']} | real-time grids appear about a day late |
| **NASA GPM IMERG Early Run** (0.1°, V07) | satellite precipitation (microwave + infrared constellation) | UTC days t−{L['gpm'] + cfg['window'] - 1} … t−{L['gpm']} | the daily Early product appears ~4 h after the UTC day ends |
| **NASA POWER** | blended (reanalysis + gauge/satellite-corrected) precipitation | t−{L['power_precip'] + cfg['window'] - 1} … t−{L['power_precip']} | published 2–3 days late |
| **NASA POWER / CERES** | cloud amount, all-sky solar radiation | t−{L['power_sc'] + cfg['window'] - 1} … t−{L['power_sc']} | published 5–7 days late |
| **MODIS Terra** (NASA GIBS) | true-colour image, 32 × 32 | t−{L['img']} | near-real-time |

Point sources are averaged over a 3 × 3 grid inside each region so they describe the same area as the IMD mean.
Because training uses the same delays, **the live forecast receives exactly the kind of inputs the model learned
from** — nothing is estimated.
""")

arch = ASSETS / "architecture.png"
if arch.exists():
    st.image(str(arch), caption="The gated fusion design (shown for the research version; the operational model "
                                "replaces the ERA5 branch with the NWP-forecast branch).")

st.header("Training and testing")
st.markdown(f"""
* {data.period_text(cfg)} — split strictly by date, never shuffled.
* {cfg['n_models']} networks trained with different random seeds; the portal shows their mean and range.
* Rain is called when the ensemble probability ≥ {cfg['threshold']:.0%} (tuned on the validation period).
* The deployed ensemble was then retrained on all data up to {pd.Timestamp(cfg['latest']):%d %b %Y} for
  {cfg['epochs_final']} epochs, so it has seen the most recent weather. Historical pages show the out-of-sample
  forecasts of the evaluation ensemble instead.
* IMD data from {pd.Timestamp(cfg['imd_realtime_from']):%b %Y} are IMD's provisional real-time grids, which IMD may
  revise.
""")

st.header("Limitations")
st.markdown("""
* **Short record.** The NWP forecast archive starts in 2024, so the model has seen only about two years of each
  season; unusual years can surprise it.
* **Regional means.** It forecasts the average rainfall of each box, not rainfall at a specific town.
* **Heavy rain is under-forecast** in amount (the probability of rain is the better guide on such days).
* **Day-ahead only.** One lead time (the next IMD day).
* **Depends on five public services.** If IMERG is unreachable (or no Earthdata login is configured) the portal
  switches to a back-up ensemble trained without IMERG and says so; if another service is down, that region shows an
  error rather than a guess.
* **IMERG calibration change.** NASA changed the Early-run calibration on 1 March 2026 (V07B → V07C); the model was
  trained mostly on V07B.
""")

st.header("Retraining")
st.markdown("Re-run the Colab notebook *Rainfall_operational_model_v3.ipynb* to learn from newer days, and replace "
            "the files in `bundle/` with the new export.")
