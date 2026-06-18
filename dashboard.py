"""
Sri Lanka Climate Indices Dashboard
- Top tabs: Temperature | Precipitation
- Radio: Grid resolution (25 km / 12.5 km)
- Radio: Climate zone filter
- Slider: pick grid by number
- Map: click a grid point to load plots
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

RV_COLS = ["rv20TXx", "rv20TXn", "rv20TNx", "rv20TNn"]

# Drought timescale variants present in each resolution
DROUGHT_SCALES = {
    "SPI" : {"25": [1,3,6,12,24], "12.5": [3]},
    "SPEI": {"25": [1,3,6,12],    "12.5": [3]},
    "EDDI": {"25": [1,3,6,12],    "12.5": [3]},
}

# ── Light theme tokens ────────────────────────────────────────────────────────
BG        = "#f8f9fa"   # page background
SURFACE   = "#ffffff"   # cards / header / panels
SURFACE2  = "#dde3ea"   # subtle secondary surface (slider panel)
BORDER    = "#dee2e6"   # borders
TEXT      = "#212529"   # primary text
TEXT2     = "#6c757d"   # secondary / muted text
ACCENT    = "#2f6eb5"   # accent blue (radio highlight, tab selected)
ACCENT_BG = "#e7f0fb"   # very light blue tint for selected tab
SEL_GREEN = "#2d6a4f"   # selected-grid label

# ── Zone & data colours ───────────────────────────────────────────────────────
ZONE_COLORS = {
    "Wet zone":          "#2a9d8f",
    "Dry zone":          "#c9a227",
    "Intermediate zone": "#e07c3b",
    "Arid zone":         "#c0392b",
}
ALL_ZONES = ["All zones", "No Climate Zones"] + list(ZONE_COLORS.keys())

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

WARM   = "#c0392b"
COOL   = "#2980b9"
WARM2  = "#e07c3b"
COOL2  = "#16a085"
PURPLE = "#7209b7"
GREEN  = "#2d6a4f"
BLUE   = "#2980b9"
TEAL   = "#06d6a0"
NAVY   = "#1a3a6b"
SKY    = "#48cae4"

# ── Index metadata ────────────────────────────────────────────────────────────
TEMP_ANN_GROUPS = {
    "Temperature Extremes": {
        "cols"  : ["TXx","TXn","TNx","TNn"],
        "labels": ["TXx – Monthly max of Tmax","TXn – Monthly min of Tmax",
                   "TNx – Monthly max of Tmin","TNn – Monthly min of Tmin"],
        "colors": [WARM, COOL, WARM2, COOL2],
        "yunits": "°C",
    },
    "Percentile Indices": {
        "cols"  : ["TX90p","TX10p","TN90p","TN10p"],
        "labels": ["TX90p – % days Tmax > 90th pct","TX10p – % days Tmax < 10th pct",
                   "TN90p – % days Tmin > 90th pct","TN10p – % days Tmin < 10th pct"],
        "colors": [WARM, COOL, WARM2, COOL2],
        "yunits": "%",
    },
    "WSDI & DTR": {
        "cols"  : ["WSDI","DTR"],
        "labels": ["WSDI – Warm spell duration (days)","DTR – Mean diurnal temp range (°C)"],
        "colors": [PURPLE, GREEN],
        "yunits": None,
    },
    "Return Values (20-yr)": {
        "cols"  : ["rv20TXx","rv20TXn","rv20TNx","rv20TNn"],
        "labels": ["20TXx – 20-yr return of monthly max Tmax",
                   "20TXn – 20-yr return of monthly min Tmax",
                   "20TNx – 20-yr return of monthly max Tmin",
                   "20TNn – 20-yr return of monthly min Tmin"],
        "colors": [WARM, COOL, WARM2, COOL2],
        "yunits": "°C",
        "bar": True,
    },
}

TEMP_HEATMAP_SPECS = [
    ("TXx",   "TXx – Monthly Max of Tmax (°C)",     "YlOrRd"),
    ("TXn",   "TXn – Monthly Min of Tmax (°C)",     "Blues"),
    ("TNx",   "TNx – Monthly Max of Tmin (°C)",     "Oranges"),
    ("TNn",   "TNn – Monthly Min of Tmin (°C)",     "PuBu"),
    ("TX90p", "TX90p – % Days Tmax > 90th Pct",     "Reds"),
    ("TX10p", "TX10p – % Days Tmax < 10th Pct",     "Blues_r"),
    ("TN90p", "TN90p – % Days Tmin > 90th Pct",     "OrRd"),
    ("TN10p", "TN10p – % Days Tmin < 10th Pct",     "GnBu"),
    ("DTR",   "DTR – Mean Diurnal Temp Range (°C)", "Viridis"),
]

PRECIP_ANN_GROUPS = {
    "Rainfall Extremes": {
        "cols"  : ["Rx1day","Rx5day"],
        "labels": ["Rx1day – Max 1-day precipitation (mm)",
                   "Rx5day – Max 5-day precipitation (mm)"],
        "colors": [BLUE, NAVY],
        "yunits": "mm",
    },
    "Rainfall Threshold Days": {
        "cols"  : ["R5mm","R10mm","R20mm","R50mm"],
        "labels": ["R5mm – Days ≥ 5 mm","R10mm – Days ≥ 10 mm",
                   "R20mm – Days ≥ 20 mm","R50mm – Days ≥ 50 mm"],
        "colors": [SKY, BLUE, TEAL, NAVY],
        "yunits": "days",
    },
    "Dry & Wet Spells": {
        "cols"  : ["CDD","CWD"],
        "labels": ["CDD – Max consecutive dry days","CWD – Max consecutive wet days"],
        "colors": [WARM2, TEAL],
        "yunits": "days",
    },
    "Rainfall Percentiles & Intensity": {
        "cols"  : ["R95p","R99p","SDII"],
        "labels": ["R95p – Very heavy rainfall total (mm)",
                   "R99p – Extremely heavy rainfall total (mm)",
                   "SDII – Simple daily intensity index (mm/day)"],
        "colors": [BLUE, NAVY, TEAL],
        "yunits": None,
    },
}

# ── Climate zone polygons ─────────────────────────────────────────────────────
cz_raw = gpd.read_file(os.path.join(BASE, "Shape_Files", "Climate Zone",
                                    "Climate_Zone.shp")).to_crs(epsg=4326)

def hex_to_rgba(hex_color, alpha=0.18):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def polygon_to_traces(geom, name, color):
    traces = []
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        lons, lats = poly.exterior.xy
        traces.append(go.Scattermapbox(
            lon=list(lons) + [None], lat=list(lats) + [None],
            mode="lines", fill="toself",
            fillcolor=hex_to_rgba(color, 0.20),
            line=dict(color=color, width=1.5),
            name=name, hoverinfo="skip", showlegend=False,
        ))
    return traces

ZONE_TRACES = []
for _, zrow in cz_raw.iterrows():
    ZONE_TRACES.extend(polygon_to_traces(zrow.geometry, zrow["Name"],
                                         ZONE_COLORS[zrow["Name"]]))

# ── Load grid data ────────────────────────────────────────────────────────────
def load_data(res):
    if res == "25":
        pts   = gpd.read_file(os.path.join(BASE, "Shape_Files", "25 Grid",
                                           "25_grid_points.shp"))
        pts   = pts.rename(columns={"OBJECTID": "Grid"})
        t_ann = pd.read_csv(os.path.join(BASE, "temp_indices_annual_25Grid.csv"))
        t_mon = pd.read_csv(os.path.join(BASE, "temp_indices_monthly_25Grid.csv"))
        p_ann = pd.read_csv(os.path.join(BASE, "rainfall_indices_CHIRPS_25Grid.csv"))
    else:
        pts   = gpd.read_file(os.path.join(BASE, "Shape_Files", "12_5 Grid",
                                           "12_5_grid_points.shp"))
        pts   = pts.rename(columns={"ID": "Grid"})
        t_ann = pd.read_csv(os.path.join(BASE, "temp_indices_annual_12.5Grid.csv"))
        t_mon = pd.read_csv(os.path.join(BASE, "temp_indices_monthly_12.5Grid.csv"))
        p_ann = pd.read_csv(os.path.join(BASE, "rainfall_indices_CHIRPS_12.5Grid.csv"))

    pts_wgs = pts.to_crs(epsg=4326)
    joined  = gpd.sjoin(pts_wgs, cz_raw[["Name","geometry"]],
                        how="left", predicate="within")
    pts["Zone"] = joined["Name"].values
    pts["Zone"] = pts["Zone"].fillna("Unknown")
    pts["Lon"]  = pts_wgs.geometry.x.values
    pts["Lat"]  = pts_wgs.geometry.y.values

    for c in RV_COLS:
        if c in t_ann.columns:
            t_ann[c] = pd.to_numeric(t_ann[c], errors="coerce")
            t_ann[c] = t_ann[c].where(t_ann[c].abs() < 1e6, other=np.nan)

    d_mon = pd.read_csv(os.path.join(BASE, f"drought_indices_{res}Grid.csv"))

    return pts, t_ann, t_mon, p_ann, d_mon

DATA = {"25": load_data("25"), "12.5": load_data("12.5")}

# ── Helpers ───────────────────────────────────────────────────────────────────
def filtered_pts(res, zone):
    pts = DATA[res][0]
    if zone in ("All zones", "No Climate Zones"):
        return pts
    return pts[pts["Zone"] == zone].reset_index(drop=True)

def slider_marks(grid_ids):
    ids = sorted(grid_ids)
    if not ids:
        return {}, 0, 0
    step  = max(1, len(ids) // 10)
    marks = {int(ids[i]): str(ids[i]) for i in range(0, len(ids), step)}
    marks[int(ids[-1])] = str(ids[-1])
    return marks, int(ids[0]), int(ids[-1])

def make_map(res, zone, selected_grid=None):
    pts_f     = filtered_pts(res, zone)
    marker_sz = 7 if res == "12.5" else 10
    colors = [("#c0392b" if g == selected_grid else ZONE_COLORS.get(z, "#2f6eb5"))
              for g, z in zip(pts_f["Grid"], pts_f["Zone"])]
    sizes  = [15 if g == selected_grid else marker_sz for g in pts_f["Grid"]]
    show_zones = (zone != "No Climate Zones")

    fig = go.Figure()
    if show_zones:
        for tr in ZONE_TRACES:
            fig.add_trace(tr)
    fig.add_trace(go.Scattermapbox(
        lat=pts_f["Lat"], lon=pts_f["Lon"], mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.9),
        text=[f"Grid {g} — {z}" for g, z in zip(pts_f["Grid"], pts_f["Zone"])],
        hovertemplate="<b>%{text}</b><br>Lat: %{lat:.3f}<br>Lon: %{lon:.3f}<extra></extra>",
        customdata=pts_f["Grid"].values, name="Grid points", showlegend=False,
    ))
    if show_zones:
        shown = set()
        for tr in ZONE_TRACES:
            if tr.name not in shown:
                fig.add_trace(go.Scattermapbox(
                    lat=[None], lon=[None], mode="markers",
                    marker=dict(size=12, color=ZONE_COLORS.get(tr.name, "#888")),
                    name=tr.name, showlegend=True,
                ))
                shown.add(tr.name)
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=7.8731, lon=80.7718), zoom=6),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=SURFACE,
        legend=dict(bgcolor=SURFACE, font=dict(color=TEXT, size=11),
                    x=0.01, y=0.99, bordercolor=BORDER, borderwidth=1),
    )
    return fig

def _plot_layout(title, grid_id, height=None):
    """Shared light-theme layout kwargs for charts."""
    layout = dict(
        title=dict(text=f"{title} — Grid {grid_id}",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10),
                    bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1),
    )
    if height:
        layout["height"] = height
    return layout

def _line_subplots(gdata, group, meta, grid_id, show_trend=False):
    cols   = [c for c in meta["cols"] if c in gdata.columns]
    labels = [l for c,l in zip(meta["cols"], meta["labels"]) if c in gdata.columns]
    colors = [cl for c,cl in zip(meta["cols"], meta["colors"]) if c in gdata.columns]
    if not cols:
        return None
    nr = max(1, (len(cols)+1)//2)
    nc = min(2, len(cols))
    fig = make_subplots(rows=nr, cols=nc, shared_xaxes=True, vertical_spacing=0.14)
    for i, (col, lbl, clr) in enumerate(zip(cols, labels, colors)):
        r, c = divmod(i, 2)
        valid = gdata[["Year", col]].dropna()
        fig.add_trace(
            go.Scatter(x=valid["Year"], y=valid[col], mode="lines+markers",
                       name=lbl, line=dict(color=clr, width=2), marker=dict(size=4),
                       hovertemplate=f"Year: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>",
                       legendgroup=lbl),
            row=r+1, col=c+1)
        if show_trend and len(valid) >= 2:
            m, b  = np.polyfit(valid["Year"], valid[col], 1)
            trend = m * valid["Year"] + b
            fig.add_trace(
                go.Scatter(x=valid["Year"], y=trend, mode="lines",
                           name=f"{col} trend", line=dict(color="#555", width=1.5,
                           dash="dash"), showlegend=False, legendgroup=lbl,
                           hovertemplate=f"Year: %{{x}}<br>Trend: %{{y:.2f}}<extra></extra>"),
                row=r+1, col=c+1)
        fig.update_yaxes(title_text=meta["yunits"] or col, row=r+1, col=c+1,
                         title_font=dict(size=10), gridcolor=BORDER,
                         linecolor=BORDER, zerolinecolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_layout(**_plot_layout(group, grid_id,
                                     height=300 if len(cols) <= 2 else 400))
    return fig

def make_temp_annual_plots(res, grid_id, show_trend=False):
    _, t_ann, _, _, _ = DATA[res]
    gdata = t_ann[t_ann["Grid"] == grid_id].sort_values("Year")
    figs  = []
    for group, meta in TEMP_ANN_GROUPS.items():
        if meta.get("bar"):
            cols   = [c for c in meta["cols"] if c in gdata.columns]
            labels = [l for c,l in zip(meta["cols"], meta["labels"]) if c in gdata.columns]
            colors = [cl for c,cl in zip(meta["cols"], meta["colors"]) if c in gdata.columns]
            vals   = [gdata[c].dropna().iloc[0] if not gdata[c].isna().all() else np.nan
                      for c in cols]
            fig = go.Figure([go.Bar(x=labels, y=vals, marker_color=colors,
                                    hovertemplate="%{x}<br>%{y:.2f} °C<extra></extra>")])
            fig.update_layout(**_plot_layout(group, grid_id))
            fig.update_layout(yaxis_title="°C", xaxis=dict(tickfont=dict(size=10)),
                              showlegend=False)
        else:
            fig = _line_subplots(gdata, group, meta, grid_id, show_trend)
        if fig:
            figs.append(fig)
    return figs

def make_precip_annual_plots(res, grid_id, show_trend=False):
    _, _, _, p_ann, _ = DATA[res]
    gdata = p_ann[p_ann["Grid"] == grid_id].sort_values("Year")
    return [fig for group, meta in PRECIP_ANN_GROUPS.items()
            for fig in [_line_subplots(gdata, group, meta, grid_id, show_trend)]
            if fig is not None]

def make_monthly_heatmap(res, grid_id, col, title, colorscale):
    _, _, t_mon, _, _ = DATA[res]
    gdata = t_mon[t_mon["Grid"] == grid_id].copy()
    if gdata.empty or col not in gdata.columns:
        return None
    pivot = gdata.pivot(index="Year", columns="Month", values=col)
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=colorscale,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Value: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT), tickcolor=TEXT),
    ))
    fig.update_layout(
        title=dict(text=f"{title} — Grid {grid_id}", font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, autorange="reversed"),
        height=340, margin=dict(t=40, b=40),
    )
    return fig

# ── Drought plot functions ────────────────────────────────────────────────────
def make_drought_heatmap(d_mon, grid_id, col, title):
    """Year × Month heatmap with diverging RdBu scale (red=dry, blue=wet)."""
    gdata = d_mon[d_mon["Grid"] == grid_id].copy()
    if gdata.empty or col not in gdata.columns:
        return None
    pivot = gdata.pivot(index="Year", columns="Month", values=col)
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
    # Centre the colorscale at 0 (neutral)
    abs_max = pivot.stack().abs().quantile(0.99)
    abs_max = abs_max if abs_max > 0 else 1
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-abs_max, zmax=abs_max,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Value: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT), tickcolor=TEXT,
                      title=dict(text=col.split("_")[0], font=dict(color=TEXT))),
    ))
    fig.update_layout(
        title=dict(text=f"{title} — Grid {grid_id}", font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, autorange="reversed"),
        height=340, margin=dict(t=40, b=40),
    )
    return fig

def make_drought_pdsi_plot(d_mon, grid_id):
    """Annual mean PDSI time series with drought category bands."""
    gdata = d_mon[d_mon["Grid"] == grid_id].copy()
    if gdata.empty or "PDSI" not in gdata.columns:
        return None
    annual = gdata.groupby("Year")["PDSI"].mean().reset_index()
    fig = go.Figure()
    # Shaded drought bands
    for ylo, yhi, clr, lbl in [
        (-8, -3,   "rgba(180,0,0,0.10)",   "Extreme drought"),
        (-3, -2,   "rgba(220,80,0,0.10)",  "Severe drought"),
        (-2, -1,   "rgba(255,165,0,0.10)", "Moderate drought"),
        (-1,  0,   "rgba(255,220,0,0.08)", "Mild drought"),
        ( 0,  1,   "rgba(200,230,200,0.08)","Near normal"),
        ( 1,  8,   "rgba(0,100,200,0.08)", "Wet"),
    ]:
        fig.add_hrect(y0=ylo, y1=yhi, fillcolor=clr, line_width=0,
                      annotation_text=lbl, annotation_position="right",
                      annotation=dict(font_size=9, font_color=TEXT2))
    fig.add_trace(go.Scatter(
        x=annual["Year"], y=annual["PDSI"], mode="lines+markers",
        name="PDSI (annual mean)", line=dict(color=ACCENT, width=2),
        marker=dict(size=5),
        hovertemplate="Year: %{x}<br>PDSI: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT2, line_width=1)
    fig.update_layout(
        **_plot_layout("Palmer Drought Severity Index (PDSI)", grid_id, height=340),
        yaxis_title="PDSI", xaxis_title="Year",
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        showlegend=False,
    )
    return fig

def make_drought_plots(res, grid_id, drought_inner_tab):
    """Return a list of figures for the selected drought inner tab."""
    _, _, _, _, d_mon = DATA[res]
    figs = []

    if drought_inner_tab == "drought-pdsi":
        fig = make_drought_pdsi_plot(d_mon, grid_id)
        if fig:
            figs.append(fig)
        # Also show PDSI heatmap
        fig = make_drought_heatmap(d_mon, grid_id, "PDSI",
                                   "PDSI – Palmer Drought Severity Index")
        if fig:
            figs.append(fig)
        return figs

    # SPI / SPEI / EDDI tabs
    index_key = drought_inner_tab.split("-")[1].upper()   # "spi" → "SPI"
    scales    = DROUGHT_SCALES.get(index_key, {}).get(res, [])
    scale_labels = {
        "SPI" : "Standardised Precipitation Index",
        "SPEI": "Standardised Precipitation-Evapotranspiration Index",
        "EDDI": "Evaporative Demand Drought Index",
    }
    for s in scales:
        col   = f"{index_key}_{s}"
        title = f"{index_key}-{s}  ({scale_labels.get(index_key, index_key)}, {s}-month)"
        fig   = make_drought_heatmap(d_mon, grid_id, col, title)
        if fig:
            figs.append(fig)
    return figs

# ── Zone-average helpers ──────────────────────────────────────────────────────
def _zone_grids(res, zone):
    pts = DATA[res][0]
    return pts[pts["Zone"] == zone]["Grid"].tolist()

def _plot_layout_zone(title, zone_name, height=None):
    layout = dict(
        title=dict(text=f"{title} — {zone_name} (zone average)",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG, font=dict(color=TEXT),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10),
                    bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1),
    )
    if height:
        layout["height"] = height
    return layout

def _zone_line_subplots(mean_df, std_df, group, meta, zone_name, show_trend=False):
    cols   = [c for c in meta["cols"] if c in mean_df.columns]
    labels = [l for c,l in zip(meta["cols"], meta["labels"]) if c in mean_df.columns]
    colors = [cl for c,cl in zip(meta["cols"], meta["colors"]) if c in mean_df.columns]
    if not cols:
        return None
    nr = max(1, (len(cols)+1)//2)
    nc = min(2, len(cols))
    fig = make_subplots(rows=nr, cols=nc, shared_xaxes=True, vertical_spacing=0.14)
    for i, (col, lbl, clr) in enumerate(zip(cols, labels, colors)):
        r, c = divmod(i, 2)
        valid_m = mean_df[["Year", col]].dropna()
        if valid_m.empty:
            continue
        # ±1 SD shaded band
        if std_df is not None and col in std_df.columns:
            merged = valid_m.merge(std_df[["Year", col]].rename(columns={col: "_s"}), on="Year")
            upper = merged[col] + merged["_s"]
            lower = merged[col] - merged["_s"]
            fig.add_trace(go.Scatter(
                x=pd.concat([merged["Year"], merged["Year"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(clr, 0.15),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip", legendgroup=lbl,
            ), row=r+1, col=c+1)
        fig.add_trace(
            go.Scatter(x=valid_m["Year"], y=valid_m[col], mode="lines+markers",
                       name=lbl, line=dict(color=clr, width=2), marker=dict(size=4),
                       hovertemplate=f"Year: %{{x}}<br>Mean {col}: %{{y:.2f}}<extra></extra>",
                       legendgroup=lbl),
            row=r+1, col=c+1)
        if show_trend and len(valid_m) >= 2:
            m, b  = np.polyfit(valid_m["Year"], valid_m[col], 1)
            trend = m * valid_m["Year"] + b
            fig.add_trace(
                go.Scatter(x=valid_m["Year"], y=trend, mode="lines",
                           name=f"{col} trend",
                           line=dict(color="#555", width=1.5, dash="dash"),
                           showlegend=False, legendgroup=lbl,
                           hovertemplate=f"Year: %{{x}}<br>Trend: %{{y:.2f}}<extra></extra>"),
                row=r+1, col=c+1)
        fig.update_yaxes(title_text=meta["yunits"] or col, row=r+1, col=c+1,
                         title_font=dict(size=10), gridcolor=BORDER,
                         linecolor=BORDER, zerolinecolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_layout(**_plot_layout_zone(group, zone_name,
                                          height=300 if len(cols) <= 2 else 400))
    return fig

def make_temp_annual_plots_zone(res, zone, show_trend=False):
    _, t_ann, _, _, _ = DATA[res]
    grids    = _zone_grids(res, zone)
    zdf      = t_ann[t_ann["Grid"].isin(grids)]
    key_cols = [c for g in TEMP_ANN_GROUPS.values() for c in g["cols"] if c in zdf.columns]
    mean_df  = zdf.groupby("Year")[key_cols].mean().reset_index()
    std_df   = zdf.groupby("Year")[key_cols].std().reset_index()
    figs = []
    for group, meta in TEMP_ANN_GROUPS.items():
        if meta.get("bar"):
            cols   = [c for c in meta["cols"] if c in mean_df.columns]
            labels = [l for c,l in zip(meta["cols"], meta["labels"]) if c in mean_df.columns]
            colors = [cl for c,cl in zip(meta["cols"], meta["colors"]) if c in mean_df.columns]
            vals   = [mean_df[c].mean() if c in mean_df.columns else np.nan for c in cols]
            errs   = [std_df[c].mean() if c in std_df.columns else 0 for c in cols]
            fig = go.Figure([go.Bar(x=labels, y=vals, marker_color=colors,
                                    error_y=dict(type="data", array=errs, visible=True),
                                    hovertemplate="%{x}<br>Mean: %{y:.2f} °C<extra></extra>")])
            fig.update_layout(**_plot_layout_zone(group, zone))
            fig.update_layout(yaxis_title="°C", xaxis=dict(tickfont=dict(size=10)),
                              showlegend=False)
        else:
            fig = _zone_line_subplots(mean_df, std_df, group, meta, zone, show_trend)
        if fig:
            figs.append(fig)
    return figs

def make_precip_annual_plots_zone(res, zone, show_trend=False):
    _, _, _, p_ann, _ = DATA[res]
    grids    = _zone_grids(res, zone)
    zdf      = p_ann[p_ann["Grid"].isin(grids)]
    key_cols = [c for g in PRECIP_ANN_GROUPS.values() for c in g["cols"] if c in zdf.columns]
    mean_df  = zdf.groupby("Year")[key_cols].mean().reset_index()
    std_df   = zdf.groupby("Year")[key_cols].std().reset_index()
    return [fig for group, meta in PRECIP_ANN_GROUPS.items()
            for fig in [_zone_line_subplots(mean_df, std_df, group, meta, zone, show_trend)]
            if fig is not None]

def make_monthly_heatmap_zone(res, zone, col, title, colorscale):
    _, _, t_mon, _, _ = DATA[res]
    grids = _zone_grids(res, zone)
    zdf   = t_mon[t_mon["Grid"].isin(grids)]
    if zdf.empty or col not in zdf.columns:
        return None
    avg   = zdf.groupby(["Year", "Month"])[col].mean().reset_index()
    pivot = avg.pivot(index="Year", columns="Month", values=col)
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=colorscale,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Avg: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT), tickcolor=TEXT),
    ))
    fig.update_layout(
        title=dict(text=f"{title} — {zone} (zone average)", font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, autorange="reversed"),
        height=340, margin=dict(t=40, b=40),
    )
    return fig

def make_drought_heatmap_zone(d_mon, zone_grids, col, title, zone_name):
    zdf = d_mon[d_mon["Grid"].isin(zone_grids)].copy()
    if zdf.empty or col not in zdf.columns:
        return None
    avg   = zdf.groupby(["Year", "Month"])[col].mean().reset_index()
    pivot = avg.pivot(index="Year", columns="Month", values=col)
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
    abs_max = pivot.stack().abs().quantile(0.99)
    abs_max = abs_max if abs_max > 0 else 1
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-abs_max, zmax=abs_max,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Avg: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT), tickcolor=TEXT,
                      title=dict(text=col.split("_")[0], font=dict(color=TEXT))),
    ))
    fig.update_layout(
        title=dict(text=f"{title} — {zone_name} (zone average)", font=dict(color=TEXT, size=13)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, autorange="reversed"),
        height=340, margin=dict(t=40, b=40),
    )
    return fig

def make_drought_pdsi_plot_zone(d_mon, zone_grids, zone_name):
    zdf = d_mon[d_mon["Grid"].isin(zone_grids)].copy()
    if zdf.empty or "PDSI" not in zdf.columns:
        return None
    annual = zdf.groupby("Year")["PDSI"].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    for ylo, yhi, clr, lbl in [
        (-8, -3,   "rgba(180,0,0,0.10)",    "Extreme drought"),
        (-3, -2,   "rgba(220,80,0,0.10)",   "Severe drought"),
        (-2, -1,   "rgba(255,165,0,0.10)",  "Moderate drought"),
        (-1,  0,   "rgba(255,220,0,0.08)",  "Mild drought"),
        ( 0,  1,   "rgba(200,230,200,0.08)","Near normal"),
        ( 1,  8,   "rgba(0,100,200,0.08)",  "Wet"),
    ]:
        fig.add_hrect(y0=ylo, y1=yhi, fillcolor=clr, line_width=0,
                      annotation_text=lbl, annotation_position="right",
                      annotation=dict(font_size=9, font_color=TEXT2))
    # SD band
    upper = annual["mean"] + annual["std"]
    lower = annual["mean"] - annual["std"]
    fig.add_trace(go.Scatter(
        x=pd.concat([annual["Year"], annual["Year"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(ACCENT, 0.15),
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=annual["Year"], y=annual["mean"], mode="lines+markers",
        name="PDSI (zone mean)", line=dict(color=ACCENT, width=2), marker=dict(size=5),
        hovertemplate="Year: %{x}<br>Mean PDSI: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT2, line_width=1)
    fig.update_layout(
        **_plot_layout_zone("Palmer Drought Severity Index (PDSI)", zone_name, height=340),
        yaxis_title="PDSI", xaxis_title="Year",
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        showlegend=False,
    )
    return fig

def make_drought_plots_zone(res, zone, drought_inner_tab):
    _, _, _, _, d_mon = DATA[res]
    zone_grids = _zone_grids(res, zone)
    figs = []
    if drought_inner_tab == "drought-pdsi":
        fig = make_drought_pdsi_plot_zone(d_mon, zone_grids, zone)
        if fig:
            figs.append(fig)
        fig = make_drought_heatmap_zone(d_mon, zone_grids, "PDSI",
                                        "PDSI – Palmer Drought Severity Index", zone)
        if fig:
            figs.append(fig)
        return figs
    index_key = drought_inner_tab.split("-")[1].upper()
    scales = DROUGHT_SCALES.get(index_key, {}).get(res, [])
    scale_labels = {
        "SPI" : "Standardised Precipitation Index",
        "SPEI": "Standardised Precipitation-Evapotranspiration Index",
        "EDDI": "Evaporative Demand Drought Index",
    }
    for s in scales:
        col   = f"{index_key}_{s}"
        title = f"{index_key}-{s}  ({scale_labels.get(index_key, index_key)}, {s}-month)"
        fig   = make_drought_heatmap_zone(d_mon, zone_grids, col, title, zone)
        if fig:
            figs.append(fig)
    return figs

# ── Layout helpers ────────────────────────────────────────────────────────────
def radio_group(label, id_, options, value):
    return html.Div(
        style={"display":"flex","flexDirection":"column","gap":"4px"},
        children=[
            html.Label(label, style={"color":TEXT2,"fontSize":"0.72rem",
                                     "fontWeight":"700","letterSpacing":"0.06em",
                                     "textTransform":"uppercase"}),
            dcc.RadioItems(
                id=id_, options=options, value=value, inline=True,
                style={"display":"flex","gap":"16px","alignItems":"center","flexWrap":"wrap"},
                inputStyle={"accentColor":ACCENT,"marginRight":"5px"},
                labelStyle={"color":TEXT,"fontSize":"0.86rem"},
            ),
        ],
    )

def tab_style(selected=False):
    base = {"fontSize":"0.95rem","fontWeight":"600","padding":"10px 26px",
            "borderRadius":"6px 6px 0 0","border":f"1px solid {BORDER}",
            "borderBottom":"none"}
    if selected:
        return {**base,"color":ACCENT,"backgroundColor":SURFACE,
                "borderColor":BORDER,"borderBottom":f"2px solid {ACCENT}"}
    return {**base,"color":TEXT2,"backgroundColor":SURFACE2}

def inner_tab_style(selected=False):
    base = {"fontSize":"0.84rem","padding":"6px 16px",
            "border":f"1px solid {BORDER}","borderBottom":"none"}
    if selected:
        return {**base,"color":ACCENT,"backgroundColor":SURFACE,"fontWeight":"600",
                "borderBottom":f"2px solid {ACCENT}"}
    return {**base,"color":TEXT2,"backgroundColor":SURFACE2}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Sri Lanka Climate Indices")
server = app.server

init_ids = sorted(DATA["25"][0]["Grid"].tolist())
init_marks, init_min, init_max = slider_marks(init_ids)

app.layout = html.Div(
    style={"backgroundColor":BG,"minHeight":"100vh",
           "fontFamily":"'Segoe UI', Arial, sans-serif"},
    children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(
        style={"backgroundColor":SURFACE,"borderBottom":f"1px solid {BORDER}",
               "boxShadow":"0 1px 4px rgba(0,0,0,0.08)","padding":"10px 24px 12px"},
        children=[

            # Title — centred
            html.H2("Sri Lanka Climate Indices Explorer",
                    style={"color":TEXT,"margin":"0 0 10px 0","fontSize":"1.3rem",
                           "fontWeight":"700","textAlign":"center"}),

            # Controls row
            html.Div(
                style={"display":"flex","alignItems":"center","gap":"16px",
                       "flexWrap":"wrap","justifyContent":"center"},
                children=[

                    # Grid Resolution — bordered box
                    html.Div(
                        style={"border":"1.5px solid #000","borderRadius":"6px",
                               "padding":"6px 12px"},
                        children=[
                            html.Label("GRID RESOLUTION",
                                       style={"color":TEXT2,"fontSize":"0.68rem",
                                              "fontWeight":"700","letterSpacing":"0.06em",
                                              "display":"block","marginBottom":"4px"}),
                            dcc.RadioItems(
                                id="grid-res",
                                options=[{"label":"25 km  (107 pts)","value":"25"},
                                         {"label":"12.5 km  (423 pts)","value":"12.5"}],
                                value="25", inline=True,
                                style={"display":"flex","gap":"14px"},
                                inputStyle={"accentColor":ACCENT,"marginRight":"4px"},
                                labelStyle={"color":TEXT,"fontSize":"0.86rem"},
                            ),
                        ],
                    ),

                    # Climate Zone — bordered box
                    html.Div(
                        style={"border":"1.5px solid #000","borderRadius":"6px",
                               "padding":"6px 12px"},
                        children=[
                            html.Label("CLIMATE ZONE",
                                       style={"color":TEXT2,"fontSize":"0.68rem",
                                              "fontWeight":"700","letterSpacing":"0.06em",
                                              "display":"block","marginBottom":"4px"}),
                            dcc.RadioItems(
                                id="zone-filter",
                                options=[{"label":z,"value":z} for z in ALL_ZONES],
                                value="All zones", inline=True,
                                style={"display":"flex","gap":"14px","flexWrap":"wrap"},
                                inputStyle={"accentColor":ACCENT,"marginRight":"4px"},
                                labelStyle={"color":TEXT,"fontSize":"0.86rem"},
                            ),
                        ],
                    ),

                    # Trend line toggle — bordered box
                    html.Div(
                        style={"border":"1.5px solid #000","borderRadius":"6px",
                               "padding":"6px 12px","display":"flex",
                               "alignItems":"center","gap":"8px","cursor":"pointer"},
                        children=[
                            dcc.Checklist(
                                id="trend-toggle",
                                options=[{"label":"","value":"show"}],
                                value=[],
                                inputStyle={"accentColor":ACCENT,"width":"15px",
                                            "height":"15px","cursor":"pointer",
                                            "marginRight":"4px"},
                            ),
                            html.Label("Show trend lines",
                                       style={"color":TEXT,"fontSize":"0.86rem",
                                              "cursor":"pointer","userSelect":"none",
                                              "fontWeight":"500","marginBottom":"0"}),
                        ],
                    ),
                ],
            ),
        ],
    ),

    # ── Top-level variable tabs ───────────────────────────────────────────────
    html.Div(
        style={"backgroundColor":SURFACE2,"padding":"8px 24px 0",
               "borderBottom":f"1px solid {BORDER}"},
        children=[
            dcc.Tabs(
                id="variable-tab", value="temp",
                colors={"border":BORDER,"primary":ACCENT,"background":SURFACE2},
                children=[
                    dcc.Tab(label="🌡  Temperature", value="temp",
                            style=tab_style(False), selected_style=tab_style(True)),
                    dcc.Tab(label="🌧  Precipitation", value="precip",
                            style=tab_style(False), selected_style=tab_style(True)),
                    dcc.Tab(label="🌵  Drought", value="drought",
                            style=tab_style(False), selected_style=tab_style(True)),
                ],
            ),
        ],
    ),

    # ── Body ─────────────────────────────────────────────────────────────────
    html.Div(
        style={"display":"flex","gap":"12px","padding":"12px",
               "height":"calc(100vh - 136px)"},
        children=[

        # Left – map + slider
        html.Div(
            style={"flex":"0 0 40%","display":"flex","flexDirection":"column","gap":"8px"},
            children=[
                html.Div(id="selected-label",
                         style={"color":SEL_GREEN,"fontSize":"0.86rem","fontWeight":"600",
                                "padding":"2px","minHeight":"20px"}),
                dcc.Graph(id="map", figure=make_map("25","All zones"),
                          style={"flex":"1","borderRadius":"8px","overflow":"hidden",
                                 "border":f"1px solid {BORDER}"},
                          config={"scrollZoom":True}),
                html.Div(
                    style={"backgroundColor":SURFACE,"borderRadius":"8px",
                           "padding":"12px 18px 8px",
                           "border":f"1px solid {BORDER}","boxShadow":"0 1px 3px rgba(0,0,0,0.06)"},
                    children=[
                        html.Label(id="slider-label", children="Grid selector",
                                   style={"color":TEXT2,"fontSize":"0.74rem","fontWeight":"700",
                                          "letterSpacing":"0.06em","textTransform":"uppercase"}),
                        dcc.Slider(id="grid-slider", min=init_min, max=init_max,
                                   step=None, marks=init_marks, value=init_min,
                                   tooltip={"placement":"bottom","always_visible":True},
                                   updatemode="drag"),
                    ],
                ),
            ],
        ),

        # Right – plot area
        html.Div(
            style={"flex":"1","overflowY":"auto","display":"flex",
                   "flexDirection":"column","gap":"10px"},
            children=[
                html.Div(id="plot-placeholder",
                         style={"color":TEXT2,"textAlign":"center",
                                "marginTop":"120px","fontSize":"1rem"},
                         children="← Click a grid point or move the slider to load plots"),

                html.Div(id="tabs-container", style={"display":"none"}, children=[

                    # Temperature inner tabs
                    html.Div(id="temp-tabs-wrapper", children=[
                        dcc.Tabs(id="temp-tabs", value="temp-annual",
                                 colors={"border":BORDER,"primary":ACCENT,"background":SURFACE2},
                                 children=[
                            dcc.Tab(label="Annual Indices", value="temp-annual",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                            dcc.Tab(label="Monthly Heatmaps", value="temp-monthly",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                        ]),
                    ]),

                    # Precipitation inner tabs
                    html.Div(id="precip-tabs-wrapper", children=[
                        dcc.Tabs(id="precip-tabs", value="precip-annual",
                                 colors={"border":BORDER,"primary":ACCENT,"background":SURFACE2},
                                 children=[
                            dcc.Tab(label="Annual Indices", value="precip-annual",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                        ]),
                    ]),

                    # Drought inner tabs
                    html.Div(id="drought-tabs-wrapper", children=[
                        dcc.Tabs(id="drought-tabs", value="drought-spi",
                                 colors={"border":BORDER,"primary":"#c0392b","background":SURFACE2},
                                 children=[
                            dcc.Tab(label="SPI", value="drought-spi",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                            dcc.Tab(label="SPEI", value="drought-spei",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                            dcc.Tab(label="EDDI", value="drought-eddi",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                            dcc.Tab(label="PDSI", value="drought-pdsi",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                        ]),
                    ]),

                    html.Div(id="tab-content"),
                ]),
            ],
        ),
    ]),

    dcc.Store(id="selected-grid"),
    dcc.Store(id="slider-valid-ids"),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("grid-slider",     "min"),
    Output("grid-slider",     "max"),
    Output("grid-slider",     "marks"),
    Output("grid-slider",     "value"),
    Output("slider-valid-ids","data"),
    Output("slider-label",    "children"),
    Input("grid-res",    "value"),
    Input("zone-filter", "value"),
)
def update_slider(res, zone):
    pts_f = filtered_pts(res, zone)
    ids   = sorted(pts_f["Grid"].tolist())
    marks, mn, mx = slider_marks(ids)
    n = len(ids)
    return mn, mx, marks, mn, ids, f"Grid selector  ({n} grid{'s' if n!=1 else ''} in view)"


@app.callback(
    Output("temp-tabs-wrapper",   "style"),
    Output("precip-tabs-wrapper", "style"),
    Output("drought-tabs-wrapper","style"),
    Input("variable-tab", "value"),
)
def toggle_inner_tabs(var_tab):
    show, hide = {"display":"block"}, {"display":"none"}
    return (
        show if var_tab == "temp"   else hide,
        show if var_tab == "precip" else hide,
        show if var_tab == "drought" else hide,
    )


@app.callback(
    Output("selected-grid",    "data"),
    Output("map",              "figure"),
    Output("selected-label",   "children"),
    Output("plot-placeholder", "style"),
    Output("tabs-container",   "style"),
    Output("grid-slider",      "value", allow_duplicate=True),
    Input("map",          "clickData"),
    Input("grid-slider",  "value"),
    Input("grid-res",     "value"),
    Input("zone-filter",  "value"),
    State("selected-grid","data"),
    State("slider-valid-ids","data"),
    prevent_initial_call=True,
)
def on_interaction(click_data, slider_val, res, zone, current_grid, valid_ids):
    triggered = ctx.triggered_id
    ph  = {"color":TEXT2,"textAlign":"center","marginTop":"120px","fontSize":"1rem"}
    hid = {"display":"none"}
    sho = {"display":"block"}

    if triggered in ("grid-res", "zone-filter"):
        pts_f    = filtered_pts(res, zone)
        first_id = int(sorted(pts_f["Grid"].tolist())[0]) if len(pts_f) else 1
        if zone not in ("All zones", "No Climate Zones"):
            n     = len(pts_f)
            label = f"Zone average  →  {zone}  ({n} grid point{'s' if n != 1 else ''})"
            return None, make_map(res, zone, None), label, hid, sho, first_id
        return None, make_map(res, zone, None), "", ph, hid, first_id

    if triggered == "map" and click_data:
        grid_id = int(click_data["points"][0]["customdata"])
        pts_f   = filtered_pts(res, zone)
        row     = pts_f[pts_f["Grid"] == grid_id]
        if row.empty:
            return (current_grid, make_map(res, zone, current_grid), "",
                    ph if current_grid is None else hid,
                    hid if current_grid is None else sho, slider_val)
        row   = row.iloc[0]
        label = (f"Selected  →  Grid {grid_id}  ({row['Zone']})  "
                 f"Lat {row['Lat']:.3f}  Lon {row['Lon']:.3f}")
        return grid_id, make_map(res, zone, grid_id), label, hid, sho, int(grid_id)

    if triggered == "grid-slider" and slider_val is not None:
        grid_id = min(valid_ids, key=lambda x: abs(x - slider_val)) if valid_ids else slider_val
        pts_f   = filtered_pts(res, zone)
        row     = pts_f[pts_f["Grid"] == grid_id]
        if row.empty:
            return (current_grid, make_map(res, zone, current_grid), "",
                    ph if current_grid is None else hid,
                    hid if current_grid is None else sho, slider_val)
        row   = row.iloc[0]
        label = (f"Selected  →  Grid {grid_id}  ({row['Zone']})  "
                 f"Lat {row['Lat']:.3f}  Lon {row['Lon']:.3f}")
        return grid_id, make_map(res, zone, grid_id), label, hid, sho, int(grid_id)

    return current_grid, make_map(res, zone, current_grid), "", ph, hid, slider_val


@app.callback(
    Output("tab-content", "children"),
    Input("variable-tab",   "value"),
    Input("temp-tabs",      "value"),
    Input("precip-tabs",    "value"),
    Input("drought-tabs",   "value"),
    Input("selected-grid",  "data"),
    Input("trend-toggle",   "value"),
    State("grid-res",       "value"),
    State("zone-filter",    "value"),
)
def render_plots(var_tab, temp_tab, precip_tab, drought_tab, grid_id, trend_val, res, zone):
    show_trend  = "show" in (trend_val or [])
    zone_mode   = (grid_id is None and zone not in ("All zones", "No Climate Zones"))

    if grid_id is None and not zone_mode:
        return []

    def graph(fig):
        return dcc.Graph(figure=fig, config={"displayModeBar":False},
                         style={"borderRadius":"8px","border":f"1px solid {BORDER}",
                                "backgroundColor":SURFACE})

    # ── Zone-average mode ─────────────────────────────────────────────────────
    if zone_mode:
        if var_tab == "temp":
            if temp_tab == "temp-annual":
                return [graph(f) for f in make_temp_annual_plots_zone(res, zone, show_trend)]
            _, _, t_mon, _, _ = DATA[res]
            return [graph(fig) for col, title, cs in TEMP_HEATMAP_SPECS
                    if col in t_mon.columns
                    for fig in [make_monthly_heatmap_zone(res, zone, col, title, cs)]
                    if fig is not None]
        if var_tab == "precip":
            return [graph(f) for f in make_precip_annual_plots_zone(res, zone, show_trend)]
        return [graph(f) for f in make_drought_plots_zone(res, zone, drought_tab)]

    # ── Individual grid mode ──────────────────────────────────────────────────
    if var_tab == "temp":
        if temp_tab == "temp-annual":
            return [graph(f) for f in make_temp_annual_plots(res, grid_id, show_trend)]
        _, _, t_mon, _, _ = DATA[res]
        return [graph(fig) for col, title, cs in TEMP_HEATMAP_SPECS
                if col in t_mon.columns
                for fig in [make_monthly_heatmap(res, grid_id, col, title, cs)]
                if fig is not None]

    if var_tab == "precip":
        return [graph(f) for f in make_precip_annual_plots(res, grid_id, show_trend)]

    return [graph(f) for f in make_drought_plots(res, grid_id, drought_tab)]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
