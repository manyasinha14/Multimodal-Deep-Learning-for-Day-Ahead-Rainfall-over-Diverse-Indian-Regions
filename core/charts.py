"""Plotly figures. Colours follow one rule: a colour always means the same thing.
Modalities: IMD blue, NASA POWER orange, NWP forecast aqua, MODIS yellow, GPM IMERG magenta. Model forecast violet,
observed grey. Regions are always axis categories, never a hue."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

C = {"imd": "#2a78d6", "sat": "#eb6834", "nwp": "#1baf7a", "met": "#1baf7a", "img": "#eda100", "gpm": "#e87ba4",
     "fc": "#6250d6", "fc_band": "rgba(98,80,214,0.18)", "obs": "#a8a69c", "ink": "#3d3c38",
     "good": "#0ca30c", "bad": "#d03b3b", "muted": "#8a897f", "full": "#6250d6"}


def _layout(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(l=8, r=8, t=48 if title else 16, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    return fig


def region_map(cfg, labels, extra_text=None):
    fig = go.Figure()
    for r in cfg["region_names"]:
        x0, y0, x1, y1 = cfg["regions"][r]["bbox"]
        txt = labels[r] + (f"<br>{extra_text[r]}" if extra_text else "")
        fig.add_trace(go.Scattergeo(lon=[x0, x1, x1, x0, x0], lat=[y0, y0, y1, y1, y0], mode="lines",
                                    fill="toself", fillcolor="rgba(98,80,214,0.25)",
                                    line=dict(color=C["fc"], width=2), hoverinfo="text", text=txt,
                                    showlegend=False))
        fig.add_trace(go.Scattergeo(lon=[(x0 + x1) / 2], lat=[y1 + 0.6], mode="text", text=[labels[r]],
                                    textfont=dict(size=12, color=C["ink"]), hoverinfo="skip", showlegend=False))
    fig.update_geos(projection_type="mercator", lonaxis_range=[67, 90], lataxis_range=[7, 31],
                    showcountries=True, countrycolor="#b9b7ae", showland=True, landcolor="#f3f2ee",
                    showocean=True, oceancolor="#e6eef8", showlakes=False, resolution=50)
    return _layout(fig, height=430)


def timeline_chart(daily, t, lag, window, pred, lo, hi, obs=None):
    """What the model saw, placed on the calendar: each source's 5-day window at its publication delay,
    the NWP forecast for the target window, and the model's forecast (+ the observation when known)."""
    t = pd.Timestamp(t)
    rng = lambda L: pd.date_range(t - pd.Timedelta(days=L + window - 1), t - pd.Timedelta(days=L))
    fig = go.Figure()
    series = [("IMD observed (inputs)", "imd_mm", lag["imd"], "imd", -1),
              ("GPM IMERG, UTC day (inputs)", "imerg_mm", lag.get("gpm", 3), "gpm", 0),
              ("NASA POWER precip. (inputs)", "PRECTOTCORR", lag["power_precip"], "sat", 1)]
    for name, col, L, key, shift in series:
        if col not in daily:
            continue
        d = rng(L)
        vals = daily.reindex(d)[col].to_numpy(float)
        d = d + pd.Timedelta(hours=8 * shift)                          # side-by-side bars on shared days
        fig.add_trace(go.Bar(x=d, y=np.nan_to_num(vals), name=name, marker_color=C[key], width=0.27 * 86400000,
                             hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra>" + name + "</extra>"))
    nwp = float(daily.at[t, "nwp_precip_mm"]) if t in daily.index else np.nan
    fig.add_trace(go.Bar(x=[t], y=[nwp], name="NWP forecast for target window", marker_color=C["nwp"], width=0.35 * 86400000,
                         offset=-0.4 * 86400000, hovertemplate="NWP forecast %{y:.1f} mm<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t + pd.Timedelta(hours=4)], y=[pred], mode="markers", name="Model forecast (ensemble)",
                             marker=dict(size=15, color=C["fc"], line=dict(width=2, color="white")),
                             error_y=dict(type="data", symmetric=False, array=[hi - pred], arrayminus=[pred - lo],
                                          color=C["fc"], thickness=2, width=8),
                             hovertemplate=f"model forecast %{{y:.1f}} mm (range {lo:.1f}–{hi:.1f})<extra></extra>"))
    if obs is not None and np.isfinite(obs):
        fig.add_trace(go.Scatter(x=[t + pd.Timedelta(hours=10)], y=[obs], mode="markers", name="Observed (IMD)",
                                 marker=dict(size=12, symbol="diamond", color=C["ink"]),
                                 hovertemplate="observed %{y:.1f} mm<extra></extra>"))
    fig.add_vrect(x0=t - pd.Timedelta(hours=12), x1=t + pd.Timedelta(hours=14), fillcolor="#f0efe9", line_width=0,
                  layer="below")
    fig.update_xaxes(range=[t - pd.Timedelta(days=lag["power_precip"] + window - 0.5), t + pd.Timedelta(days=0.8)],
                     tickformat="%d %b")
    fig.update_yaxes(title="rainfall (mm/day)", rangemode="tozero")
    return _layout(fig, height=360, barmode="overlay", bargap=0.3)


def gates_chart(gates, names, labels):
    keys = list(names)
    fig = go.Figure(go.Bar(x=list(gates), y=[labels[k] for k in keys], orientation="h",
                           marker_color=[C[k] for k in keys], text=[f"{g:.2f}" for g in gates],
                           textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="gate weight (weights sum to 1)")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=250, showlegend=False)


def season_chart(df, threshold):
    d = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d["obs_mm"], name="Observed (IMD)", marker_color=C["obs"],
                         hovertemplate="%{x|%d %b}: %{y:.1f} mm observed<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["hi_mm"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["lo_mm"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=C["fc_band"], name="Ensemble range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["pred_mm"], mode="lines", name="Forecast (ensemble mean)",
                             line=dict(color=C["fc"], width=2),
                             hovertemplate="%{x|%d %b}: %{y:.1f} mm forecast<extra></extra>"))
    if "nwp_mm" in d:
        fig.add_trace(go.Scatter(x=d["date"], y=d["nwp_mm"], mode="lines", name="Raw NWP forecast",
                                 line=dict(color=C["nwp"], width=1.2, dash="dot"), visible="legendonly",
                                 hovertemplate="%{x|%d %b}: %{y:.1f} mm raw NWP<extra></extra>"))
    wrong = d[~d["correct"]]
    fig.add_trace(go.Scatter(x=wrong["date"], y=np.full(len(wrong), -1.5), mode="markers",
                             name="Rain/no-rain call wrong", marker=dict(symbol="x", size=7, color=C["bad"]),
                             customdata=np.stack([wrong["p_rain"] * 100], 1) if len(wrong) else None,
                             hovertemplate="%{x|%d %b}: wrong call (P(rain) %{customdata[0]:.0f}%)<extra></extra>"))
    fig.update_yaxes(title="rainfall (mm/day)")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, height=420)


def monthly_heatmap(daily, region_order, labels, value="mean", threshold=2.5):
    d = daily.dropna(subset=["imd_mm"]).copy()
    d["month"] = d["date"].dt.month
    g = d.groupby(["region", "month"])["imd_mm"]
    tab = (g.mean() if value == "mean" else g.apply(lambda s: (s >= threshold).mean() * 100)).unstack().reindex(region_order)
    tab = tab.reindex(columns=range(1, 13))
    months = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    unit = "mm/day" if value == "mean" else "% wet days"
    fig = go.Figure(go.Heatmap(z=tab.values, x=months, y=[labels[r] for r in tab.index],
                               colorscale=[[0, "#f3f7fd"], [0.5, "#5598e7"], [1, "#0d366b"]],
                               text=np.round(tab.values, 1), texttemplate="%{text}",
                               colorbar=dict(title=unit), hovertemplate="%{y}, %{x}: %{z:.1f} " + unit + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=300)


def monthly_totals(daily, region):
    d = daily[daily["region"] == region].dropna(subset=["imd_mm"]).set_index("date")["imd_mm"]
    tot = d.resample("MS").sum(min_count=20)
    fig = go.Figure(go.Bar(x=tot.index, y=tot.values, marker_color=C["imd"],
                           hovertemplate="%{x|%b %Y}: %{y:.0f} mm<extra></extra>"))
    fig.update_yaxes(title="monthly total (mm, regional mean)")
    fig.update_xaxes(tickformat="%b %Y")
    return _layout(fig, height=320, showlegend=False)


def source_scatter(df, xcol, xlabel, color):
    d = df.dropna(subset=[xcol, "imd_next"])
    fig = go.Figure(go.Scattergl(x=d[xcol], y=d["imd_next"], mode="markers",
                                 marker=dict(size=5, color=color, opacity=0.45),
                                 hovertemplate=xlabel + " %{x:.1f} mm → IMD %{y:.1f} mm<extra></extra>"))
    mx = float(np.nanmax([d[xcol].max(), d["imd_next"].max(), 1]))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dot", color=C["muted"]),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=f"{xlabel} (mm)")
    fig.update_yaxes(title="IMD observed rainfall (mm)")
    return _layout(fig, height=340, showlegend=False)


def model_bar(df, metric, highlight, better="lower", err=None):
    d = df.sort_values(metric, ascending=(better == "lower"))
    cols = [C["full"] if m == highlight else C["obs"] for m in d["model"]]
    fig = go.Figure(go.Bar(x=d[metric], y=d["model"], orientation="h", marker_color=cols,
                           error_x=dict(type="data", array=d[err]) if err is not None else None,
                           text=[f"{v:.3f}" for v in d[metric]], textposition="outside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"{metric} ({'lower' if better == 'lower' else 'higher'} is better)")
    return _layout(fig, height=40 + 34 * len(d), showlegend=False)


def forest(boot):
    d = boot.iloc[::-1]
    lo = d["ci_lo"].to_numpy(); hi = d["ci_hi"].to_numpy(); mid = d["delta"].to_numpy()
    sig = lo > 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid, y=d["compared to"], mode="markers",
                             marker=dict(size=11, color=[C["fc"] if s else C["muted"] for s in sig]),
                             error_x=dict(type="data", symmetric=False, array=hi - mid, arrayminus=mid - lo,
                                          color="#8a897f", thickness=2, width=0),
                             hovertemplate="%{y}: ΔRMSE %{x:.3f} mm<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color=C["ink"])
    fig.update_xaxes(title="ΔRMSE = RMSE(other) − RMSE(full model), mm/day  (right of 0 = full model better)")
    return _layout(fig, height=60 + 34 * len(d), showlegend=False)


def gates_by_region(gdf, names, labels, region_labels):
    fig = go.Figure()
    for k in names:
        fig.add_trace(go.Bar(y=[region_labels[r] for r in gdf.index], x=gdf[k], orientation="h",
                             name=labels[k], marker_color=C[k], text=[f"{v:.2f}" if v >= 0.08 else "" for v in gdf[k]],
                             textposition="inside", hovertemplate="%{y}<br>" + labels[k] + ": %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="mean gate weight")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=300, barmode="stack")


def reliability(p, o, bins=10):
    p, o = np.asarray(p), np.asarray(o, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [(p[idx == b].mean(), o[idx == b].mean(), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]
    mp, mo, n = map(np.array, zip(*rows))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=C["muted"]),
                             name="perfect reliability", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mp, y=mo, mode="lines+markers", name="ensemble",
                             line=dict(color=C["fc"], width=2), marker=dict(size=6 + 14 * n / n.max(), color=C["fc"]),
                             customdata=n, hovertemplate="forecast %{x:.0%} → observed %{y:.0%} (%{customdata} days)<extra></extra>"))
    fig.update_xaxes(title="forecast probability of rain", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="observed frequency of rain", range=[0, 1], tickformat=".0%")
    return _layout(fig, height=360)


def confusion(ct):
    z = [[ct["hits"], ct["misses"]], [ct["false_alarms"], ct["correct_negatives"]]]
    txt = [[f"Hits<br>{ct['hits']}", f"Misses<br>{ct['misses']}"],
           [f"False alarms<br>{ct['false_alarms']}", f"Correct no-rain<br>{ct['correct_negatives']}"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Rain forecast", "No rain forecast"], y=["Rain observed", "No rain observed"],
                               text=txt, texttemplate="%{text}", colorscale=[[0, "#eef3fb"], [1, "#256abf"]],
                               showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=260)
