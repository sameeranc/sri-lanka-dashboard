---
title: Sri Lanka Climate Indices Dashboard
emoji: 🌡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Sri Lanka Climate Indices Dashboard

An interactive dashboard to explore gridded climate indices for Sri Lanka.

## Features

- **Grid resolution**: Switch between 25 km (107 grid points) and 12.5 km (423 grid points)
- **Climate zone filter**: Filter grids by Wet / Dry / Intermediate / Arid zone
- **Map**: Click any grid point to load its plots; zone boundaries overlaid
- **Grid slider**: Select a grid number directly from the slider
- **Annual indices**: TXx, TXn, TNx, TNn, TX90p, TX10p, TN90p, TN10p, WSDI, DTR, 20-year return values
- **Monthly heatmaps**: Year × Month heatmaps for all indices

## Indices reference

| Index | Description |
|---|---|
| TXx / TXn | Monthly max / min of daily maximum temperature |
| TNx / TNn | Monthly max / min of daily minimum temperature |
| TX90p / TX10p | % days Tmax above 90th / below 10th percentile (1981–2010 baseline) |
| TN90p / TN10p | % days Tmin above 90th / below 10th percentile (1981–2010 baseline) |
| WSDI | Warm Spell Duration Index (days in spells ≥6 consecutive TX > 90th pct) |
| DTR | Mean Diurnal Temperature Range |
| 20TXx / 20TXn / 20TNx / 20TNn | 20-year return values via GEV distribution |
