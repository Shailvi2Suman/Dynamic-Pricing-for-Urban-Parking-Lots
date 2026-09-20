"""
Bokeh visualisation of priced output.

Produces a standalone HTML dashboard: for a chosen lot, the three model prices
over time plus the occupancy rate that drives them. This is the "make the data
visible and useful" layer the problem statement requires.
"""
from __future__ import annotations

import pathlib

import pandas as pd
from bokeh.layouts import column
from bokeh.models import HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import ColumnDataSource, figure, output_file, save

from . import config as C


def build_dashboard(priced: pd.DataFrame, lot: str | None = None,
                    out_path: str | pathlib.Path = "dashboard.html") -> pathlib.Path:
    """Render a two-panel Bokeh dashboard for one lot and save to HTML."""
    if lot is None:
        lot = sorted(priced[C.LOT_COL].unique())[0]
    d = priced[priced[C.LOT_COL] == lot].copy()
    d["occ_rate"] = d["Occupancy"] / d["Capacity"]
    src = ColumnDataSource(d)
    colors = Category10[3]

    p1 = figure(height=340, width=880, x_axis_type="datetime",
                title=f"Dynamic price by model — {lot}",
                x_axis_label="time", y_axis_label="price (USD)")
    for col, color, label in zip(
        ["Price_Model1", "Price_Model2", "Price_Model3"], colors,
        ["Model 1 · baseline", "Model 2 · demand", "Model 3 · competitive"],
    ):
        if col in d.columns:
            p1.line("Timestamp", col, source=src, color=color, line_width=2,
                    legend_label=label)
    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"
    p1.add_tools(HoverTool(
        tooltips=[("time", "@Timestamp{%F %H:%M}"), ("demand $", "@Price_Model2{0.00}")],
        formatters={"@Timestamp": "datetime"}, mode="vline"))

    p2 = figure(height=200, width=880, x_axis_type="datetime",
                title="Occupancy rate (the driver)", x_range=p1.x_range,
                x_axis_label="time", y_axis_label="occupancy")
    p2.varea("Timestamp", 0, "occ_rate", source=src, alpha=0.25, color=colors[1])
    p2.line("Timestamp", "occ_rate", source=src, color=colors[1], line_width=1.5)

    out_path = pathlib.Path(out_path)
    output_file(out_path, title=f"Parking pricing — {lot}")
    save(column(p1, p2))
    return out_path
