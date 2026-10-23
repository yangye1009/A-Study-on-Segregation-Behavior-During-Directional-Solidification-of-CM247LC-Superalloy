#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPACING_MM = 5.0
LIQUIDUS_C = 1370.9


def read_procast_table(path: Path) -> pd.DataFrame:
    """Read a ProCAST TSV/CSV export containing a time column followed by P1...Pn."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    # The first line of the original file is the software export note; data start on the next line.
    raw = pd.read_csv(path, sep=None, engine="python", skiprows=1, header=None)
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.shape[1] < 3:
        raise ValueError(f"{path}: a time column and at least two probe columns are required")
    raw.columns = ["time"] + [f"P{i}" for i in range(1, raw.shape[1])]
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if raw["time"].duplicated().any():
        raise ValueError(f"{path}: duplicate time records; interpolation would be unreliable")
    if raw.iloc[:, 1:].isna().any().any():
        raise ValueError(f"{path}: probe temperature/solid-fraction data contain missing or non-numeric values")
    if len(raw) < 3:
        raise ValueError(f"{path}: too few valid time records")
    return raw


def first_in_range(time: np.ndarray, values: np.ndarray, low: float, high: float):
    """Return the indices (in time order) of records lying in the open interval (low, high); callers use the earliest ones."""
    idx = np.flatnonzero((values > low) & (values < high))
    if not len(idx):
        raise ValueError(f"no samples found within ({low}, {high})")
    return idx


def temperature_at(time: np.ndarray, temp: np.ndarray, target: float) -> float:
    """Return the original value on the same time grid, otherwise interpolate linearly; extrapolation is not allowed."""
    if target < time[0] or target > time[-1]:
        raise ValueError(f"time {target:g} s is outside the range of the temperature data")
    return float(np.interp(target, time, temp))


def compute_parameters(temp: pd.DataFrame, solid: pd.DataFrame,
                       spacing_mm: float = SPACING_MM,
                       liquidus_c: float = LIQUIDUS_C,
                       solid_fraction_threshold: float = 0.05) -> pd.DataFrame:
    """Compute the quantities of the four original functions and align results of different lengths by probe."""
    points = [col for col in temp if col != "time"]
    if not all(col in solid for col in points):
        raise ValueError("probe columns of the temperature file and the solid-fraction file do not match")
    t = temp["time"].to_numpy(dtype=float)
    ts = solid["time"].to_numpy(dtype=float)
    n = len(points)
    if n < 3:
        raise ValueError("at least 3 probes are needed for a gradient spanning two probe intervals")
    v = np.full(n, np.nan)
    delta_t = np.full(n, np.nan)
    t_r = np.full(n, np.nan)
    t_g = np.full(n, np.nan)
    ref_temp_g = np.full(n, np.nan)

    for j, label in enumerate(points):
        y = temp[label].to_numpy(dtype=float)
        chosen = first_in_range(t, y, 1200., 1400.)
        if len(chosen) < 2:
            raise ValueError(f"{label}: fewer than two valid records within 1200–1400 °C")
        i0, i1 = chosen[:2]
        v[j] = -(y[i1] - y[i0]) / (t[i1] - t[i0])  # K/s
        t_r[j] = t[i0]                            # s

        g0 = first_in_range(t, y, 600., 1400.)[0]
        t_g[j] = t[g0]
        ref_temp_g[j] = y[g0]

        sf = solid[label].to_numpy(dtype=float)
        valid_sf = np.flatnonzero(sf >= solid_fraction_threshold)
        if len(valid_sf) == 0:
            raise ValueError(f"{label}: solid fraction never reaches {solid_fraction_threshold:g}")
        time_sf = ts[valid_sf[0]]
        delta_t[j] = liquidus_c - temperature_at(t, y, time_sf)

    # R follows the original program, 60*5/Δt; time in the temperature file is in seconds.
    dt_r = np.diff(t_r)
    if np.any(dt_r <= 0):
        raise ValueError("non-positive time difference between adjacent probes entering the high-temperature window; check the probe ordering")
    r = 60.0 * spacing_mm / dt_r

    # G uses the temperature difference of P_i between t_i and t_{i+2}; the spatial denominator must be 2*5 = 10 mm.
    g = np.array([
        (ref_temp_g[i] - temperature_at(t, temp[points[i]].to_numpy(dtype=float), t_g[i+2]))
        / (2.0 * spacing_mm)
        for i in range(n-2)
    ], dtype=float)

    result = pd.DataFrame({
        "point": points,
        "position_mm": np.arange(n, dtype=float) * spacing_mm,
        "cooling_rate_K_s": v,
        "undercooling_K": delta_t,
        "interface_speed_mm_min": np.r_[r, np.nan],
        "temperature_gradient_K_mm": np.r_[g, np.nan, np.nan],
        "R_reference_time_s": t_r,
        "G_reference_time_s": t_g,
    })
    return result


def _plot_stage(ax1, ax2, label, table, x, mask):
    """Plot the four quantities of one axis, keeping the twin-y-axis layout of the original thesis figure."""
    dash = "-" if label == "2" else "--"
    for name, c, marker, title in [
        ("cooling_rate_K_s", "black" if label == "2" else "tab:orange", "o", "V"),
        ("interface_speed_mm_min", "navy" if label == "1" else "tab:blue", ".", "R"),
        ("undercooling_K", "limegreen" if label == "2" else "crimson", "x", "ΔT"),
    ]:
        vals = table[name].to_numpy()
        ok = mask & np.isfinite(vals)
        ax1.plot(x[ok], vals[ok], linestyle=dash, marker=marker, markersize=3,
                 linewidth=1.2, color=c, label=f"{title}{label}")
    vals = table["temperature_gradient_K_mm"].to_numpy()
    ok = mask & np.isfinite(vals)
    ax2.plot(x[ok], vals[ok], "-.", marker="<" if label == "1" else ">",
             markersize=3, linewidth=1.3, color="tab:purple" if label == "1" else "tab:green",
             label=f"G{label}")


def draw_figures(axis1, axis2, output: Path, x_mode: str = "nominal-fs"):
    """Draw the initial and steady stages; the old 5 mm denominator of the original G is not mixed in."""
    if len(axis1) != len(axis2):
        raise ValueError("the two axes must have the same number of probes")
    n = len(axis1)
    if x_mode == "nominal-fs":
        x = np.linspace(0.0, 0.9, n)
        x_label = "Nominal solid fraction (linear index mapping)"
        intervals = [(0.0, 0.2, "a"), (0.2, 0.9, "b")]
    else:
        x = axis2["position_mm"].to_numpy()
        x_label = "Axial position (mm)"
        cut = x[min(8, n-1)]
        intervals = [(0.0, cut, "a"), (cut, x[-1], "b")]
    for lo, hi, code in intervals:
        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=180)
        ax2 = ax1.twinx()
        mask = (x >= lo - 1e-12) & (x <= hi + 1e-12)
        _plot_stage(ax1, ax2, "1", axis1, x, mask)
        _plot_stage(ax1, ax2, "2", axis2, x, mask)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("ΔT (K), V (K/s), R (mm/min)")
        ax2.set_ylabel("G (K/mm)")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)
        ax1.set_xlim(lo, hi)
        ax1.set_title(f"Local thermal parameters ({'initial' if code=='a' else 'steady'} stage)")
        fig.tight_layout()
        fig.savefig(output / f"Fig4_24{code}_recalculated.png", dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp1", type=Path, required=True, help="temperature data of reference axis 1 (TSV/CSV)")
    p.add_argument("--temp2", type=Path, required=True, help="temperature data of centerline axis 2 (TSV/CSV)")
    p.add_argument("--solid1", type=Path, required=True, help="solid-fraction data of axis 1")
    p.add_argument("--solid2", type=Path, required=True, help="solid-fraction data of axis 2")
    p.add_argument("--spacing", type=float, default=5.0, help="axial distance between adjacent probes, mm")
    p.add_argument("--liquidus", type=float, default=1370.9, help="liquidus temperature, °C")
    p.add_argument("--fs-threshold", type=float, default=0.05, help="solid-fraction threshold used to define the undercooling")
    p.add_argument("--x-mode", choices=["nominal-fs", "position"], default="nominal-fs")
    p.add_argument("--output", type=Path, default=Path("results/01_thermal"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    axis1 = compute_parameters(read_procast_table(a.temp1), read_procast_table(a.solid1),
                               a.spacing, a.liquidus, a.fs_threshold)
    axis2 = compute_parameters(read_procast_table(a.temp2), read_procast_table(a.solid2),
                               a.spacing, a.liquidus, a.fs_threshold)
    axis1.to_csv(a.output / "axis1_thermal_parameters.csv", index=False, encoding="utf-8-sig")
    axis2.to_csv(a.output / "axis2_thermal_parameters.csv", index=False, encoding="utf-8-sig")
    draw_figures(axis1, axis2, a.output, a.x_mode)
    if a.x_mode == "nominal-fs":
        warnings.warn("The x-axis is the linear mapping of the original script, not the true solid fraction read point by point from the simulation files")
    print(f"Done. Results are in {a.output.resolve()}; use the Axis 2 (centerline) data primarily.")


if __name__ == "__main__":
    main()
