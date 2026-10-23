#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program 3: correction of measured compositions, approximate γ-phase composition and Nv/Md/Bo phase-stability indices.

Note: the γ-phase partitioning correction keeps the empirical formula of the original script; the thesis must state
its source, assumptions and independent validation. It is not a CALPHAD phase-equilibrium calculation and cannot by
itself prove TCP precipitation.
This version fixes the original indexing error in which at[16]/at[18] were taken as Ta/Hf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ELEMENTS = ("Cr", "Ti", "Mo", "Al", "Co", "B", "Zr", "C", "Si", "Mn", "Fe",
            "Ru", "V", "W", "Ta", "Nb", "Hf", "Re", "Ni")
ATOMIC_MASS = {
    "Cr": 52.00, "Ti": 47.90, "Mo": 95.94, "Al": 26.98, "Co": 58.93,
    "B": 10.81, "Zr": 91.22, "C": 12.01, "Si": 28.09, "Mn": 54.94,
    "Fe": 55.85, "Ru": 101.07, "V": 50.94, "W": 183.85,
    "Ta": 180.95, "Nb": 92.91, "Hf": 178.49, "Re": 186.21, "Ni": 58.71,
}

# Empirical electron-vacancy numbers / d-orbital energy levels / bond orders, kept item by item from the original program.
NV = {"Cr":4.66, "Ti":6.66, "Mo":4.66, "Al":7.66, "Co":1.71,
      "Zr":6.66, "W":4.66, "Ta":5.66, "Hf":6.66, "Ni":0.66}
MD = {"Cr":1.142, "Ti":2.271, "Mo":1.550, "Al":1.900, "Co":0.777,
      "Zr":2.944, "W":1.655, "Ta":2.224, "Hf":3.020, "Ni":0.717}
BO = {"Cr":1.278, "Ti":1.098, "Mo":1.611, "Al":0.589, "Co":0.697,
      "Zr":1.479, "W":1.730, "Ta":1.670, "Hf":1.518, "Ni":0.514}
CORRECTION_FACTOR = {
    "Ni": 0.9449838187702265, "W": 1.003157894736842,
    "Ta": 1.3925, "Al": 1.2732142857142859,
    "Co": 1.007608695652174, "Cr": 1.0462962962962963,
    "Hf": 0.9221428571428572, "Mo": 1.348, "Ti": 1.3542857142857143,
}
FITTED_PARAMS = {  # (log intercept A, exponent b): taken from the original script, not fitted by this program
    "Ta": (1.2307148,-0.232271394), "Cr": (2.142897573,0.005770819),
    "Co": (2.242986164,0.015668588), "Al": (1.990646669,0.026305367),
    "Ti": (-0.301835848,-0.219577817), "Hf": (-0.298932961,-0.425501808),
    "W": (2.312092152,0.059362565), "Mo": (-0.523970931,-0.120862468),
    "Ni": (4.0807237,0.012521441),
}
DEFAULT_TRACE_WT = {"B":0.015, "C":0.07, "Zr":0.007}
GAMMA_ELEMENTS = ("Ni", "Cr", "Co", "Mo", "W", "Zr")


def _safe_div(x: float, y: float, meaning: str) -> float:
    if abs(y) < 1e-15:
        raise ValueError(f"zero denominator ({meaning}) in the γ-phase correction model")
    return x/y


def prepare_composition(csv_path: Path, mode: str = "measured") -> pd.DataFrame:
    """Read element wt% data. 'measured' keeps the corrected measured composition used in the original workflow."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    for e, factor in CORRECTION_FACTOR.items():
        if e not in df:
            raise ValueError(f"{csv_path}: required column {e} is missing")
        df[e] = pd.to_numeric(df[e], errors="raise")
        df[e] = (df[e]-(0.092 if e=="Ni" else 0)) / factor
    if mode == "fitted":
        fs = np.linspace(0, 0.9, len(df))  # nominal solid-fraction mapping of the original script
        for e, (a, b) in FITTED_PARAMS.items():
            df[e] = np.exp(a) * np.power(1-fs, b) / CORRECTION_FACTOR[e]
    for e, value in DEFAULT_TRACE_WT.items():
        if e not in df:
            df[e] = value
    for e in ELEMENTS:
        if e not in df:
            df[e] = 0.0  # 'Nan' in the original code is treated here as "element not provided" = 0
        df[e] = pd.to_numeric(df[e], errors="raise").fillna(0.)
    if (df[list(ELEMENTS)] < 0).any().any():
        raise ValueError("negative element mass fractions; check the correction factors and the raw compositions")
    return df


def to_atomic_fraction(row: pd.Series) -> dict[str, float]:
    """Mass fraction -> atomic fraction; any consistent mass-fraction unit is accepted."""
    amount = np.array([float(row.get(e, 0.0))/ATOMIC_MASS[e] for e in ELEMENTS], dtype=float)
    total = amount.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mass fractions cannot be converted to valid atomic fractions")
    return dict(zip(ELEMENTS, amount/total))


def estimate_gamma_matrix(at: dict[str, float]) -> dict[str, float]:
    """Original empirical γ-phase partitioning model, kept verbatim, with guards against zero denominators and negative compositions."""
    a = at
    mo_w = a["Mo"] + a["W"]
    carbon_group = a["Ti"] + a["Zr"] + a["Ta"] + a["Hf"]
    zirconium_group = a["Ti"] + a["Zr"] + a["V"] + a["Ta"] + a["Nb"] + a["Hf"]
    cr = 0.97*a["Cr"] - 0.375*a["B"]
    co = a["Co"] - a["C"]
    mo = a["Mo"] - 0.75*a["B"] - 1.5*a["C"]*_safe_div(a["Mo"], mo_w, "Mo+W")
    zr = a["Zr"] - 0.5*a["C"]*_safe_div(a["Zr"], zirconium_group, "Zr-group term")
    w  = a["W"] - 1.5*a["C"]*_safe_div(a["W"], mo_w, "Mo+W")
    ni = a["Ni"] - 0.15*a["B"] - 0.5*a["C"] - 3 * (
        a["Al"]+0.03*a["Cr"]+a["Ti"]-0.225*a["B"]
        - 0.5*a["C"]*_safe_div(a["Ti"],carbon_group,"Ti+Zr+Ta+Hf")
        + a["Ta"] - 0.5*a["C"]*_safe_div(a["Ta"],carbon_group,"Ti+Zr+Ta+Hf")
        + a["Hf"] - 0.5*a["C"]*_safe_div(a["Hf"],carbon_group,"Ti+Zr+Ta+Hf")
    )
    raw = {"Ni":ni,"Cr":cr,"Co":co,"Mo":mo,"W":w,"Zr":zr}
    if any(v < -1e-12 for v in raw.values()):
        raise ValueError(f"the empirical model gives a negative γ-phase composition: {raw}; check model applicability")
    all_atomic = sum(raw.values())
    if all_atomic <= 0:
        raise ValueError("sum of the γ-phase components is non-positive")
    return {e:max(raw[e],0.0)/all_atomic for e in GAMMA_ELEMENTS}


def weighted_parameter(composition: dict[str, float], coefficients: dict[str, float]) -> float:
    """Note: trace elements not listed in the NV/MD/BO dicts have zero weight in the original model."""
    return sum(value*coefficients.get(e, 0.0) for e, value in composition.items())


def calculate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (_, record) in enumerate(df.iterrows()):
        atomic = to_atomic_fraction(record)
        gamma = estimate_gamma_matrix(atomic)
        values = {"point_index":i, "position_cm":17.0*i/(len(df)-1) if len(df)>1 else 0.0,
                  "Nv_gamma":weighted_parameter(gamma,NV),
                  "Md_gamma":weighted_parameter(gamma,MD),
                  "Bo_gamma":weighted_parameter(gamma,BO),
                  "Nv_alloy":weighted_parameter(atomic,NV),
                  "Md_alloy":weighted_parameter(atomic,MD),
                  "Bo_alloy":weighted_parameter(atomic,BO)}
        values.update({f"{e}_gamma_at_pct":100*gamma[e] for e in GAMMA_ELEMENTS})
        values.update({f"{e}_alloy_at_pct":100*atomic[e] for e in CORRECTION_FACTOR})
        rows.append(values)
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, output: Path, nv_c: float|None, md_c: float|None):
    x = results["position_cm"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    for ax, col, threshold, label in [
        (axes[0],"Nv_gamma",nv_c,"Average electron vacancy number, Nv(γ)"),
        (axes[1],"Md_gamma",md_c,"Average d-orbital energy level, Md(γ)"),
    ]:
        ax.plot(x,results[col],"o-",markersize=3,linewidth=1)
        if threshold is not None:
            ax.axhline(threshold,linestyle="--",color="gray",label=f"Reference threshold = {threshold:g}")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=.4,linestyle=":")
    axes[-1].set_xlabel("Axial position (cm; linear mapping)")
    fig.tight_layout()
    fig.savefig(output/"gamma_phase_stability.png",dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5,4.8))
    for e in ("Ni","Cr","Co","W","Mo"):
        ax.plot(x,results[f"{e}_gamma_at_pct"],".-",label=e,linewidth=1)
    ax.set_xlabel("Axial position (cm; linear mapping)")
    ax.set_ylabel("Estimated γ-phase composition (at%)")
    ax.grid(alpha=.4,linestyle=":")
    ax.legend(ncol=5,fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"estimated_gamma_composition.png",dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True,help="raw CSV of element mass fractions, e.g. mapping1216/data717.csv")
    p.add_argument("--source",choices=["measured","fitted"],default="measured",
                   help="measured = corrected measured compositions as in the original code; fitted = regenerate the curves from the fixed fit coefficients of the original program")
    p.add_argument("--nv-critical",type=float,default=None,help="optional reference threshold, e.g. 2.32 (verify its applicability)")
    p.add_argument("--md-critical",type=float,default=None,help="optional reference threshold, e.g. 0.925 (verify its applicability)")
    p.add_argument("--output",type=Path,default=Path("results/03_stability"))
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    compositions = prepare_composition(args.data,args.source)
    results = calculate_table(compositions)
    results.to_csv(args.output/"phase_stability_parameters.csv",index=False,encoding="utf-8-sig")
    plot_results(results,args.output,args.nv_critical,args.md_critical)
    print(f"Computed {len(results)} positions; output: {args.output.resolve()}")
    print("Note: the γ-phase correction is the empirical model of the original script; verify its physical applicability independently.")


if __name__ == "__main__":
    main()
