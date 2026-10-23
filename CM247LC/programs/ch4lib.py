# -*- coding: utf-8 -*-
"""ch4lib —  shared data layer for the Chapter-4 supplement.

- thermal_raw_output.txt
- EDS: da2.csv, da4.csv, da7.csv
- JMatPro: liquid.dat / gamma.dat / gammap.dat
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for d in (RESULTS, FIGS):
    d.mkdir(exist_ok=True)

ELEMENTS = ["Al", "Co", "Cr", "Hf", "Mo", "Ta", "Ti", "W", "Ni"]
GROUP_A = ["Ta", "Hf", "Ti"]
T_LIQ = 1370.9          # °C
DX = 5.0                # mm
EDS_X0 = 1.0            # mm
N_DERIVED = 37          # x_i = 5 i, i = 0..36 (0–180 mm)
STEADY_X = 70.0         # mm

K0_TABLE44 = {"W": 1.232, "Co": 1.097, "Ta": 0.570, "Ti": 0.756, "Al": 1.056,
              "Cr": 1.012, "Mo": 0.764, "Hf": 0.107}
M_TABLE32 = {"Al": 80.9, "Co": 81.5, "Cr": -106, "Hf": -16, "Mo": -200,
             "Ta": -27, "Ti": -274, "W": 37.8}
DL_TABLE32 = {"Al": 2.10, "Co": 1.83, "Cr": 1.59, "Hf": 1.37, "Mo": 1.60,
              "Ta": 1.68, "Ti": 2.10, "W": 2.51}          # 1e-9 m2/s
Z_NUM = {"Al": 13, "Co": 27, "Cr": 24, "Hf": 72, "Mo": 42, "Ta": 73, "Ti": 22, "W": 74, "Ni": 28}
KE_TABLE43_34 = {"Ni": 1.003, "W": 1.057, "Ta": 0.836, "Al": 1.019, "Co": 1.015,
                 "Cr": 1.005, "Hf": 0.964, "Mo": 0.975, "Ti": 0.881}
NOMINAL = {"Al": 5.6, "Co": 9.2, "Cr": 8.1, "Hf": 1.4, "Mo": 0.5, "Ta": 3.2, "Ti": 0.7, "W": 9.5, "Ni": 61.5}


def load_thermal() -> dict[str, pd.DataFrame]:
    """Return {'axis2': df, 'axis1': df} with columns x_mm, V (K/s), R (mm/min), G (K/mm), dT (K).
    """
    txt = (DATA / "thermal_raw_output.txt").read_text()
    arrays = [np.array(ast.literal_eval(line.strip())) for line in txt.splitlines() if line.strip()]
    assert len(arrays) == 8 and all(len(a) == N_DERIVED for a in arrays)
    x = DX * np.arange(N_DERIVED)
    out = {}
    for name, (v, r, g, dt), dt_valid in (("axis2", arrays[0:4], True), ("axis1", arrays[4:8], False)):
        df = pd.DataFrame({"i": np.arange(N_DERIVED), "x_mm": x,
                           "V_K_s": v, "R_mm_min": r, "G_K_mm": g / 2.0, "dT_K": dt})
        df.loc[df.i == N_DERIVED - 1, "R_mm_min"] = np.nan
        df.loc[df.i >= N_DERIVED - 2, "G_K_mm"] = np.nan
        if not dt_valid:
            df["dT_K"] = np.nan
        df["G_center_mm"] = df.x_mm + DX
        out[name] = df
    return out


def arrival_times_from_R(df: pd.DataFrame) -> np.ndarray:
    """Arrival times (s) from R = 60*5/(t_{i+1}-t_i), with t_0 = 0."""
    dt = 60 * DX / df.R_mm_min.to_numpy()[:-1]
    return np.concatenate([[0.0], np.cumsum(dt)])


def load_eds(name: str = "da2") -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv", index_col=0)
    df.index.name = "j"
    return df


def eds_positions(n: int = 34, x0: float = EDS_X0) -> np.ndarray:
    return x0 + DX * np.arange(n)


def pair_with_G(eds: pd.DataFrame, th: pd.DataFrame, offset: int = -1, x0: float = EDS_X0) -> pd.DataFrame:
    """Pair EDS point j with derived point i = j + offset (default offset = -1)."""
    j = np.arange(len(eds)); i = j + offset
    ok = (i >= 0) & (i < N_DERIVED)
    out = eds.copy()
    out["x_mm"] = eds_positions(len(eds), x0)
    out["i_paired"] = i
    for col, src in (("G_K_mm", "G_K_mm"), ("R_mm_min", "R_mm_min"), ("V_K_s", "V_K_s")):
        arr = np.full(len(eds), np.nan); arr[ok] = th[src].to_numpy()[i[ok]]; out[col] = arr
    out["misreg_mm"] = out.x_mm - (DX * i + DX)
    out["S"] = out[GROUP_A].sum(axis=1)
    return out


def load_jmatpro(name: str) -> pd.DataFrame:
    rows = []
    for line in (DATA / f"{name}.dat").read_text().splitlines():
        parts = [p.strip().strip("'") for p in line.rstrip("\t\r\n").split("\t")]
        rows.append(parts)
    hdr = rows[0][1:]
    header = ["T_C"] + [h if h in ("GAMMA_PRIME", "LIQUID", "GAMMA") else h.capitalize() for h in hdr]
    width = len(header)
    body = [r + [""] * (width - len(r)) for r in rows[1:]]
    df = pd.DataFrame([r[:width] for r in body], columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def detrend_linear(y, x=None):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


def acf(y, maxlag):
    y = np.asarray(y, float) - np.mean(y)
    n = len(y); den = np.sum(y * y)
    return np.array([np.sum(y[:n - k] * y[k:]) / den for k in range(maxlag + 1)])


def bartlett_neff(a, b, maxlag=None):
    n = len(a); maxlag = n // 2 if maxlag is None else maxlag
    ra, rb = acf(a, maxlag), acf(b, maxlag)
    return n / max(1 + 2 * np.sum(ra[1:] * rb[1:]), 1e-6)


def pearson_p(r, n):
    from scipy import stats
    if n <= 2 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return 2 * stats.t.sf(abs(t), n - 2)


def save_table(df, name, index=True):
    df.to_csv(RESULTS / f"{name}.csv", index=index, encoding="utf-8-sig", float_format="%.6g")
