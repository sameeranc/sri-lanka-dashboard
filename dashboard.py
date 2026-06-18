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

ZONE_COLORS = {
    "Wet zone":          "#2a9d8f",
    "Dry zone":          "#e9c46a",
    "Intermediate zone": "#f4a261",
    "Arid zone":         "#e76f51",
}
ALL_ZONES = ["All zones", "No Climate Zones"] + list(ZONE_COLORS.keys())

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Colour palette ────────────────────────────────────────────────────────────
WARM   = "#e63946"
COOL   = "#457b9d"
WARM2  = "#f4a261"
COOL2  = "#2a9d8f"
PURPLE = "#7209b7"
GREEN  = "#2d6a4f"
BLUE   = "#4cc9f0"
TEAL   = "#06d6a0"
NAVY   = "#023e8a"
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
            fillcolor=hex_to_rgba(color, 0.18),
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
        pts = gpd.read_file(os.path.join(BASE, "Shape_Files", "25 Grid",
                                         "25_grid_points.shp"))
        pts = pts.rename(columns={"OBJECTID": "Grid"})
        t_ann = pd.read_csv(os.path.join(BASE, "temp_indices_annual_25Grid.csv"))
        t_mon = pd.read_csv(os.path.join(BASE, "temp_indices_monthly_25Grid.csv"))
        p_ann = pd.read_csv(os.path.join(BASE, "rainfall_indices_CHIRPS_25Grid.csv"))
    else:
        pts = gpd.read_file(os.path.join(BASE, "Shape_Files", "12_5 Grid",
                                         "12_5_grid_points.shp"))
        pts = pts.rename(columns={"ID": "Grid"})
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

    return pts, t_ann, t_mon, p_ann

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
    colors = [("#e63946" if g == selected_grid else ZONE_COLORS.get(z, "#457b9d"))
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
        paper_bgcolor="#1e1e2e",
        legend=dict(bgcolor="#181825", font=dict(color="#cdd6f4", size=11),
                    x=0.01, y=0.99, bordercolor="#313244", borderwidth=1),
    )
    return fig

def _line_subplots(gdata, group, meta, grid_id):
    cols   = [c for c in meta["cols"] if c in gdata.columns]
    labels = [l for c,l in zip(meta["cols"], meta["labels"]) if c in gdata.columns]
    colors = [cl for c,cl in zip(meta["cols"], meta["colors"]) if c in gdata.columns]
    if not cols:
        return None
    nr = max(1, (len(cols)+1)//2)
    nc = min(2, len(cols))
    fig = make_subplots(rows=nr, cols=nc, shared_xaxes=True, vertical_spacing=0.12)
    for i, (col, lbl, clr) in enumerate(zip(cols, labels, colors)):
        r, c = divmod(i, 2)
        fig.add_trace(
            go.Scatter(x=gdata["Year"], y=gdata[col], mode="lines+markers",
                       name=lbl, line=dict(color=clr, width=1.5), marker=dict(size=4),
                       hovertemplate=f"Year: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>"),
            row=r+1, col=c+1)
        fig.update_yaxes(title_text=meta["yunits"] or col, row=r+1, col=c+1,
                         title_font=dict(size=10), gridcolor="#313244")
    fig.update_xaxes(gridcolor="#313244")
    fig.update_layout(
        title=dict(text=f"{group} — Grid {grid_id}", font=dict(color="#cdd6f4")),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"), showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        height=320 if len(cols) <= 2 else 420,
    )
    return fig

def make_temp_annual_plots(res, grid_id):
    _, t_ann, _, _ = DATA[res]
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
            fig.update_layout(
                title=dict(text=f"{group} — Grid {grid_id}", font=dict(color="#cdd6f4")),
                yaxis_title="°C", paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                font=dict(color="#cdd6f4"), xaxis=dict(tickfont=dict(size=10)),
            )
        else:
            fig = _line_subplots(gdata, group, meta, grid_id)
        if fig:
            figs.append(fig)
    return figs

def make_precip_annual_plots(res, grid_id):
    _, _, _, p_ann = DATA[res]
    gdata = p_ann[p_ann["Grid"] == grid_id].sort_values("Year")
    figs  = []
    for group, meta in PRECIP_ANN_GROUPS.items():
        fig = _line_subplots(gdata, group, meta, grid_id)
        if fig:
            figs.append(fig)
    return figs

def make_monthly_heatmap(res, grid_id, col, title, colorscale):
    _, _, t_mon, _ = DATA[res]
    gdata = t_mon[t_mon["Grid"] == grid_id].copy()
    if gdata.empty or col not in gdata.columns:
        return None
    pivot = gdata.pivot(index="Year", columns="Month", values=col)
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=colorscale,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Value: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#cdd6f4")),
    ))
    fig.update_layout(
        title=dict(text=f"{title} — Grid {grid_id}", font=dict(color="#cdd6f4")),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        xaxis=dict(gridcolor="#313244"),
        yaxis=dict(gridcolor="#313244", autorange="reversed"),
        height=350, margin=dict(t=40, b=40),
    )
    return fig

# ── Shared style helpers ──────────────────────────────────────────────────────
def radio_group(label, id_, options, value):
    return html.Div(
        style={"display":"flex","flexDirection":"column","gap":"4px"},
        children=[
            html.Label(label, style={"color":"#a6adc8","fontSize":"0.75rem",
                                     "fontWeight":"600","letterSpacing":"0.05em"}),
            dcc.RadioItems(
                id=id_, options=options, value=value, inline=True,
                style={"display":"flex","gap":"18px","alignItems":"center"},
                inputStyle={"accentColor":"#89b4fa","marginRight":"5px"},
                labelStyle={"color":"#cdd6f4","fontSize":"0.88rem"},
            ),
        ],
    )

def tab_style(selected=False):
    base = {"fontSize":"1rem","fontWeight":"bold","padding":"10px 28px",
            "borderRadius":"6px 6px 0 0","border":"none"}
    if selected:
        return {**base,"color":"#1e1e2e","backgroundColor":"#89b4fa"}
    return {**base,"color":"#cdd6f4","backgroundColor":"#313244"}

def inner_tab_style(selected=False):
    base = {"fontSize":"0.88rem","padding":"7px 18px"}
    if selected:
        return {**base,"color":"#89b4fa","backgroundColor":"#1e1e2e","fontWeight":"bold"}
    return {**base,"color":"#cdd6f4","backgroundColor":"#181825"}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Sri Lanka Climate Indices")
server = app.server

init_ids = sorted(DATA["25"][0]["Grid"].tolist())
init_marks, init_min, init_max = slider_marks(init_ids)

app.layout = html.Div(
    style={"backgroundColor":"#1e1e2e","minHeight":"100vh",
           "fontFamily":"'Segoe UI', sans-serif"},
    children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(
        style={"backgroundColor":"#181825","padding":"10px 24px",
               "borderBottom":"2px solid #313244",
               "display":"flex","alignItems":"center","gap":"40px","flexWrap":"wrap"},
        children=[
            html.Div([
                html.H2("Sri Lanka Climate Indices Explorer",
                        style={"color":"#cdd6f4","margin":0,"fontSize":"1.3rem"}),
                html.P("Click a grid point or use the slider to select a grid.",
                       style={"color":"#6c7086","margin":"2px 0 0","fontSize":"0.8rem"}),
            ]),
            radio_group("Grid Resolution", "grid-res",
                        [{"label":"25 km  (107 pts)","value":"25"},
                         {"label":"12.5 km  (423 pts)","value":"12.5"}], "25"),
            radio_group("Climate Zone", "zone-filter",
                        [{"label":z,"value":z} for z in ALL_ZONES], "All zones"),
        ],
    ),

    # ── Top-level tabs: Temperature | Precipitation ───────────────────────────
    html.Div(
        style={"backgroundColor":"#181825","padding":"10px 24px 0",
               "borderBottom":"2px solid #313244"},
        children=[
            dcc.Tabs(
                id="variable-tab", value="temp",
                colors={"border":"#313244","primary":"#89b4fa","background":"#181825"},
                children=[
                    dcc.Tab(label="🌡  Temperature", value="temp",
                            style=tab_style(False),
                            selected_style=tab_style(True)),
                    dcc.Tab(label="🌧  Precipitation", value="precip",
                            style=tab_style(False),
                            selected_style=tab_style(True)),
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
                         style={"color":"#a6e3a1","fontSize":"0.88rem",
                                "padding":"2px","minHeight":"20px"}),
                dcc.Graph(id="map", figure=make_map("25","All zones"),
                          style={"flex":"1","borderRadius":"8px","overflow":"hidden"},
                          config={"scrollZoom":True}),
                html.Div(
                    style={"backgroundColor":"#181825","borderRadius":"8px",
                           "padding":"12px 18px 6px"},
                    children=[
                        html.Label(id="slider-label", children="Grid selector",
                                   style={"color":"#a6adc8","fontSize":"0.78rem",
                                          "fontWeight":"600","letterSpacing":"0.05em"}),
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
                         style={"color":"#6c7086","textAlign":"center",
                                "marginTop":"120px","fontSize":"1rem"},
                         children="← Click a grid point or move the slider to load plots"),

                html.Div(id="tabs-container", style={"display":"none"}, children=[

                    # Temperature inner tabs
                    html.Div(id="temp-tabs-wrapper", children=[
                        dcc.Tabs(id="temp-tabs", value="temp-annual",
                                 colors={"border":"#313244","primary":"#89b4fa",
                                          "background":"#181825"},
                                 children=[
                            dcc.Tab(label="Annual Indices", value="temp-annual",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                            dcc.Tab(label="Monthly Heatmaps", value="temp-monthly",
                                    style=inner_tab_style(False),
                                    selected_style=inner_tab_style(True)),
                        ]),
                    ]),

                    # Precipitation inner tabs (annual only)
                    html.Div(id="precip-tabs-wrapper", children=[
                        dcc.Tabs(id="precip-tabs", value="precip-annual",
                                 colors={"border":"#313244","primary":"#4cc9f0",
                                          "background":"#181825"},
                                 children=[
                            dcc.Tab(label="Annual Indices", value="precip-annual",
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


# ── Callback 1: slider range on res/zone change ───────────────────────────────
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
    n     = len(ids)
    return mn, mx, marks, mn, ids, f"Grid selector  ({n} grid{'s' if n!=1 else ''} in view)"


# ── Callback 2: show/hide correct inner tab wrapper on variable-tab change ────
@app.callback(
    Output("temp-tabs-wrapper",   "style"),
    Output("precip-tabs-wrapper", "style"),
    Input("variable-tab", "value"),
)
def toggle_inner_tabs(var_tab):
    show = {"display":"block"}
    hide = {"display":"none"}
    if var_tab == "temp":
        return show, hide
    return hide, show


# ── Callback 3: map click / slider / res / zone → selected grid ──────────────
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

    ph_style   = {"color":"#6c7086","textAlign":"center","marginTop":"120px","fontSize":"1rem"}
    tabs_hide  = {"display":"none"}
    tabs_show  = {"display":"block"}

    if triggered in ("grid-res", "zone-filter"):
        pts_f    = filtered_pts(res, zone)
        first_id = int(sorted(pts_f["Grid"].tolist())[0]) if len(pts_f) else 1
        return None, make_map(res, zone, None), "", ph_style, tabs_hide, first_id

    if triggered == "map" and click_data:
        grid_id = int(click_data["points"][0]["customdata"])
        pts_f   = filtered_pts(res, zone)
        row     = pts_f[pts_f["Grid"] == grid_id]
        if row.empty:
            show_ph = ph_style if current_grid is None else {"display":"none"}
            show_tb = tabs_hide if current_grid is None else tabs_show
            return current_grid, make_map(res, zone, current_grid), "", show_ph, show_tb, slider_val
        row   = row.iloc[0]
        label = (f"Selected  →  Grid {grid_id}  ({row['Zone']})  "
                 f"Lat {row['Lat']:.3f}  Lon {row['Lon']:.3f}")
        return grid_id, make_map(res, zone, grid_id), label, {"display":"none"}, tabs_show, int(grid_id)

    if triggered == "grid-slider" and slider_val is not None:
        grid_id = min(valid_ids, key=lambda x: abs(x - slider_val)) if valid_ids else slider_val
        pts_f   = filtered_pts(res, zone)
        row     = pts_f[pts_f["Grid"] == grid_id]
        if row.empty:
            show_ph = ph_style if current_grid is None else {"display":"none"}
            show_tb = tabs_hide if current_grid is None else tabs_show
            return current_grid, make_map(res, zone, current_grid), "", show_ph, show_tb, slider_val
        row   = row.iloc[0]
        label = (f"Selected  →  Grid {grid_id}  ({row['Zone']})  "
                 f"Lat {row['Lat']:.3f}  Lon {row['Lon']:.3f}")
        return grid_id, make_map(res, zone, grid_id), label, {"display":"none"}, tabs_show, int(grid_id)

    return current_grid, make_map(res, zone, current_grid), "", ph_style, tabs_hide, slider_val


# ── Callback 4: render plots ──────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("variable-tab",  "value"),
    Input("temp-tabs",     "value"),
    Input("precip-tabs",   "value"),
    Input("selected-grid", "data"),
    State("grid-res",      "value"),
)
def render_plots(var_tab, temp_tab, precip_tab, grid_id, res):
    if grid_id is None:
        return []

    def graph(fig):
        return dcc.Graph(figure=fig, config={"displayModeBar":False},
                         style={"borderRadius":"8px"})

    if var_tab == "temp":
        if temp_tab == "temp-annual":
            return [graph(f) for f in make_temp_annual_plots(res, grid_id)]
        # monthly heatmaps
        _, _, t_mon, _ = DATA[res]
        return [graph(fig) for col, title, cs in TEMP_HEATMAP_SPECS
                if col in t_mon.columns
                for fig in [make_monthly_heatmap(res, grid_id, col, title, cs)]
                if fig is not None]

    # Precipitation (annual only)
    return [graph(f) for f in make_precip_annual_plots(res, grid_id)]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
