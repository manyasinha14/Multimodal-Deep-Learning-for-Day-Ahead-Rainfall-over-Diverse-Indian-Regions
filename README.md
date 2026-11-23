# Rainfall Forecast Portal (v3 — operational)

A standalone, multi-page web app that forecasts, every day of the year, **whether it will rain (≥ 2.5 mm)** and
**how much** in four Indian regions (Chennai, Delhi–Lucknow, Kerala Western Ghats, Rajasthan) for the next IMD
rainfall day (08:30 → 08:30 IST).

The model is trained in Colab on data from **January 2024 up to the latest day IMD has published**, using only
sources that are available in real time, with their real publication delays. Five modalities: NWP forecast
(Open-Meteo), IMD rain-gauge grids, **NASA GPM IMERG** satellite precipitation, NASA POWER precipitation / cloud /
solar (CERES) and MODIS imagery. The live page therefore feeds the
model exactly the kind of inputs it learned from — nothing is estimated.

| Page | What it does |
|---|---|
| 🏠 Home | Test-period skill (unseen days) vs the raw weather forecast, the regions, per-region scores |
| 🛰️ Live forecast | Rain / no rain for the next IMD day in all four regions, with the inputs used and their dates |
| ✅ Live track record | Every live forecast, automatically checked against IMD once IMD publishes the day |
| 📅 Forecast explorer | Any day in the record: forecast vs observed, inputs on a calendar, modality reliance, MODIS image |
| 📈 Period replay | Any date range for a region: forecasts, wrong calls, month-by-month table, CSV download |
| 🗺️ Regional climate | Seasonal cycle of the four regimes, monthly totals, agreement of NWP and satellite with IMD |
| 🎯 Model performance | Baselines & ablations, bootstrap significance, probability reliability, gate weights |
| 📘 Method & data | Sources, timing, training, limitations |

## 0 · NASA Earthdata account (for GPM IMERG)

1. Create a free account at <https://urs.earthdata.nasa.gov>.
2. In your profile: **Applications → Authorized Apps → Approve More Applications** → approve
   **NASA GESDISC DATA ARCHIVE** (without this, IMERG requests fail with EULA / 401 errors).
3. For the portal, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and enter your username and
   password (or set `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, or use a `~/.netrc` entry for
   `urs.earthdata.nasa.gov`). Without it the portal still runs, using the back-up model trained without IMERG.

## 1 · Train the model (Colab, once — then whenever you want newer data)

1. Upload `Rainfall_operational_model_v3.ipynb` to Colab, choose a GPU runtime, **Run all**.
   The first run downloads ~3 years of data (20–40 min); later runs only fetch new days.
2. Read the results in sections 9 and 12. The key line is **“Deployed ensemble vs the raw NWP forecast”**.
3. The last cell downloads **`rainfall_portal_bundle_v3.zip`**. Unzip its files into this app's `bundle/` folder.

## 2 · Run the portal

* **Windows:** double-click `run_portal.bat` (first run installs packages, a few minutes).
* **macOS / Linux:** `./run_portal.sh`
* **Any OS, manually:** `pip install -r requirements.txt` then `streamlit run app.py`

It opens at <http://localhost:8501>. Python 3.10–3.12 recommended. The computer needs internet access for the
live page (IMD, Open-Meteo, NASA POWER, NASA GIBS).

## 3 · Building a live track record

Open the **Live forecast** page once a day (or more). Each visit stores that day's forecasts in
`user_data/forecast_log.csv`; the **Live track record** page checks them against IMD a day or two later. After a
few weeks you have a genuine, independently verified accuracy figure.

## Optional: public link (Streamlit Community Cloud)

Push this folder including `bundle/` to GitHub, then deploy `app.py` on <https://share.streamlit.io>. Replace the
`torch` line in `requirements.txt` with the CPU build:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

The live track-record log is reset whenever a Cloud app restarts; run the portal locally to keep it.

## Data and timing (for target IMD day t)

| Source | Days used | Why |
|---|---|---|
| Open-Meteo NWP forecast | 09:00 on t−1 → 08:00 on t | trained on forecasts issued 24 h ahead; live uses the latest run |
| IMD gridded rainfall (real-time grids via `imdlib`) | t−7 … t−3 | published about a day late |
| NASA GPM IMERG Early Run, daily 0.1° (cloud OPeNDAP subsets) | UTC days t−7 … t−3 | published ~4 h after the UTC day ends |
| NASA POWER precipitation (blended product) | t−9 … t−5 | 2–3 days late |
| NASA POWER / CERES cloud & solar | t−13 … t−9 | 5–7 days late |
| MODIS true-colour (NASA GIBS) | t−3 | near real time |

`core/features.py` and `core/model.py` are byte-for-byte the code the notebook trains with (the notebook writes
them as `rainfall_features.py` / `rainfall_model.py` and also copies them into the bundle).
