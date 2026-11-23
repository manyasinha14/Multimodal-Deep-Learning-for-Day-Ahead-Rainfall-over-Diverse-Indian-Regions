from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "bundle"
ASSETS = ROOT / "assets"
REQUIRED = ["config.json", "model.pt", "predictions.csv", "daily.csv", "modis_images.npz"]
USER_DATA = ROOT / "user_data"
