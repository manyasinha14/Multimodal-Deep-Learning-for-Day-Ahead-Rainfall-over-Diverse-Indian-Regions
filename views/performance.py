import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts, data
from core.metrics import scores
from core.ui import fmt, pct, show_chart, show_table

cfg = data.load_config()
pred = data.load_predictions()
test = pred[pred["split"] == "test"]
FULL = "Full multimodal (gated) [ensemble]"

st.title("🎯 Model performance")
st.markdown(f"All numbers are for the **test period {pd.Timestamp(cfg['test_start']):%d %b %Y} – "
            f"{pd.Timestamp(cfg['latest']):%d %b %Y}**, which the model never saw. {data.period_text(cfg)}.")

t1, t2, t3, t4 = st.tabs(["Against the raw forecast", "All baselines & ablations", "Probability reliability",
                          "Which data source matters where"])

with t1:
    rows = []
    for key, g in [("All regions", test)] + [(data.LABELS[r], test[test["region"] == r]) for r in cfg["region_names"]]:
        m = scores(g["obs_mm"], g["pred_mm"], g["rain_fc"], cfg["rain_mm"])
        n = scores(g["obs_mm"], g["nwp_mm"], g["nwp_rain"], cfg["rain_mm"])
        rows.append({"Region": key, "Days": m["n"], "Correct calls": pct(m["Accuracy"]),
                     "Correct (raw NWP)": pct(n["Accuracy"]), "HSS": fmt(m["HSS"]), "HSS (raw NWP)": fmt(n["HSS"]),
                     "CSI": fmt(m["CSI"]), "RMSE (mm)": fmt(m["RMSE"]), "RMSE raw NWP (mm)": fmt(n["RMSE"])})
    show_table(pd.DataFrame(rows))
    st.caption("Raw NWP = the Open-Meteo forecast for the same IMD window used directly (rain if ≥ 2.5 mm). The model "
               "is only worth deploying where it beats this.")
    s = scores(test["obs_mm"], test["pred_mm"], test["rain_fc"], cfg["rain_mm"])
    a, b = st.columns([1, 1.4])
    with a:
        st.markdown("**Rain / no-rain calls, all regions**")
        show_chart(charts.confusion(s), key="perf_cm")
    with b:
        st.markdown("**Forecast vs observed daily rainfall**")
        fig = go.Figure(go.Scattergl(x=test["obs_mm"], y=test["pred_mm"], mode="markers",
                                     marker=dict(size=5, color=charts.C["fc"], opacity=0.45),
                                     text=[data.SHORT[r] for r in test["region"]],
                                     hovertemplate="%{text}: observed %{x:.1f} → forecast %{y:.1f} mm<extra></extra>"))
        mx = float(max(test["obs_mm"].max(), test["pred_mm"].max(), 1))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color=charts.C["muted"]),
                                 hoverinfo="skip"))
        fig.update_layout(height=300, showlegend=False, margin=dict(l=8, r=8, t=16, b=8),
                          xaxis_title="observed (mm)", yaxis_title="forecast (mm)")
        show_chart(fig, key="perf_scatter")

with t2:
    res = data.load_csv("results_test.csv")
    if res is None:
        st.info("results_test.csv is missing from the bundle.")
    else:
        allr = res[res["region"] == "ALL"]
        ens = allr[allr["model"].str.endswith("[ensemble]") | ~allr["model"].isin(
            [m.replace(" [ensemble]", "") for m in allr["model"] if m.endswith("[ensemble]")])]
        agg = ens.groupby("model", sort=False)[["RMSE", "Accuracy", "CSI", "HSS", "r"]].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index().fillna(0)
        agg["model"] = agg["model"].str.replace(" [ensemble]", "", regex=False)
        metric = st.radio("Metric", ["Accuracy", "HSS", "CSI", "RMSE"], horizontal=True)
        show_chart(charts.model_bar(agg.rename(columns={f"{metric}_mean": metric}), metric,
                                    FULL.replace(" [ensemble]", ""), "lower" if metric == "RMSE" else "higher",
                                    err=f"{metric}_std"), key="perf_models")
        st.caption("Neural models: 5-seed ensembles. Gradient boosting: mean ± std over 5 seeds. Each ablation removes "
                   "one data source from the full model.")
        boot = data.load_csv("significance_bootstrap.csv")
        if boot is not None:
            ci = boot["95% CI"].astype(str).str.findall(r"-?\d+\.?\d*")
            bt = pd.DataFrame({"compared to": boot["compared to"],
                               "delta": boot.filter(like="RMSE").iloc[:, 0].astype(float),
                               "ci_lo": ci.str[0].astype(float), "ci_hi": ci.str[1].astype(float)})
            st.markdown("**Is the full model significantly better?** Paired bootstrap of the test days (RMSE)")
            show_chart(charts.forest(bt), key="perf_forest")
            sig = bt[bt["ci_lo"] > 0]["compared to"].tolist()
            st.caption(f"Violet = 95% interval entirely right of 0 ({len(sig)} of {len(bt)}): "
                       + (", ".join(sig) if sig else "none") + ".")

with t3:
    st.markdown("When the model says *70 % chance of rain*, does it rain on about 70 % of those days?")
    show_chart(charts.reliability(test["p_rain"], test["rain_obs"]), key="perf_rel")
    st.caption("Marker size = number of days in the bin. Above the diagonal = under-confident; below = over-confident.")

with t4:
    gn = data.gate_names(cfg)
    g = test.groupby("region")[[f"gate_{k}" for k in gn]].mean().reindex(cfg["region_names"])
    g.columns = gn
    show_chart(charts.gates_by_region(g, gn, data.MODALITY_NAMES, data.LABELS), key="perf_gates")
    top = "; ".join(f"{data.SHORT[r]}: {data.MODALITY_NAMES[g.loc[r].idxmax()]} ({g.loc[r].max():.2f})" for r in g.index)
    st.caption(f"Mean gate weight on the test period. Largest per region — {top}.")
    split = test.assign(kind=np.where(test["rain_obs"], "Rainy days", "Dry days")).groupby("kind")[[f"gate_{k}" for k in gn]].mean()
    split.columns = [data.MODALITY_NAMES[k] for k in gn]
    show_table(split.round(3).reset_index().rename(columns={"kind": ""}))
