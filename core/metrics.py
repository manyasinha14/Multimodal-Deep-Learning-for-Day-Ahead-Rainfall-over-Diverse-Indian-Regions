"""Verification metrics (same definitions as the paper)."""
import numpy as np


def contingency(obs_rain, fc_rain):
    o, f = np.asarray(obs_rain, bool), np.asarray(fc_rain, bool)
    return {"hits": int(np.sum(o & f)), "misses": int(np.sum(o & ~f)),
            "false_alarms": int(np.sum(~o & f)), "correct_negatives": int(np.sum(~o & ~f))}


def scores(y_true, y_pred, rain_fc, threshold_mm=2.5):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ct = contingency(y_true >= threshold_mm, rain_fc)
    h, m, f, c = ct["hits"], ct["misses"], ct["false_alarms"], ct["correct_negatives"]
    n = h + m + f + c
    exp = ((h + m) * (h + f) + (c + m) * (c + f)) / n if n else np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1] if n > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    return {
        "n": n,
        "Accuracy": (h + c) / n if n else np.nan,
        "POD": h / (h + m) if h + m else np.nan,
        "FAR": f / (h + f) if h + f else 0.0,
        "CSI": h / (h + m + f) if h + m + f else np.nan,
        "HSS": (h + c - exp) / (n - exp) if n and n != exp else np.nan,
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if n else np.nan,
        "MAE": float(np.mean(np.abs(y_true - y_pred))) if n else np.nan,
        "r": float(r),
        **ct,
    }
