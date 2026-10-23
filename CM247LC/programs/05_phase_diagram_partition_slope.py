#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program 5: inter-phase partition coefficients k(T) and T-C slope fitting from phase-diagram exports.

Purpose: 1) ratio Cs/Cl of the solid- and liquid-phase element mass fractions at the same temperature;
         2) optional linear regression T = a + m*C on a temperature-concentration table.
Check the phase-diagram export files: solid/liquid correspondence, temperature alignment, phase identity and
temperature range. Scheil and lever rule are different solidification models; this program does not automatically
interpret the ratio as the equilibrium k0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ["T(C)","Ni","Al","Co","Cr","Hf","Mo","Ta","Ti","W","Zr","B","C"]
PLOT_ELEMENTS = ["W","Cr","Co","Al","Mo","Ti","Ta","Hf"]


def read_phase_table(path: Path) -> pd.DataFrame:
    """Read a purely numeric 13-column text file, as in the original program."""
    matrix=np.loadtxt(path,ndmin=2)
    if matrix.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: contains missing, infinite or non-numeric values")
    return pd.DataFrame(matrix,columns=COLUMNS)


def calculate_partition(solid:pd.DataFrame,liquid:pd.DataFrame,
                        liquidus:float=1370.9043,temperature_tol:float=1e-3) -> pd.DataFrame:
    """Compute Cs/Cl row by row at equal temperature; the temperature column itself is not divided."""
    if len(solid)!=len(liquid):
        raise ValueError("the solid and liquid tables have different numbers of points; they cannot be divided row by row")
    sT=solid["T(C)"].to_numpy()
    lT=liquid["T(C)"].to_numpy()
    if not np.allclose(sT,lT,atol=temperature_tol,rtol=0):
        raise ValueError("temperatures of corresponding rows differ between the solid and liquid tables; interpolate onto a common temperature grid first")
    result=pd.DataFrame({"T_C":lT,"delta_T_K":liquidus-lT})
    for element in COLUMNS[1:]:
        cs=solid[element].to_numpy(dtype=float)
        cl=liquid[element].to_numpy(dtype=float)
        if np.any(cl<=0):
            raise ValueError(f"{element}: liquid concentration contains zero/negative values; the partition coefficient cannot be computed")
        result[f"k_{element}"]=cs/cl
    return result


def fit_temperature_slopes(table:pd.DataFrame,
                           elements:list[str]|None=None,
                           max_undercooling:float|None=None,
                           liquidus:float=1370.9043)->pd.DataFrame:
    """Fit T = intercept + slope*C; with a restricted window this approximates the local liquidus slope."""
    data=table.copy()
    if max_undercooling is not None:
        if max_undercooling<=0:
            raise ValueError("--max-undercooling must be positive")
        deltaT=liquidus-data["T(C)"]
        data=data.loc[(deltaT>=0)&(deltaT<=max_undercooling)]
    if len(data)<3:
        raise ValueError("fewer than 3 temperature records are available for the linear regression")
    names=elements or COLUMNS[1:]
    out=[]
    for e in names:
        if e not in data.columns:
            raise ValueError(f"element column {e} not found in the data table")
        x=data[e].to_numpy(dtype=float)
        y=data["T(C)"].to_numpy(dtype=float)
        if np.std(x)==0:
            out.append({"element":e,"slope_K_per_wt_pct":np.nan,"intercept_C":np.nan,
                        "R2":np.nan,"n_points":len(x)})
            continue
        slope,intercept=np.polyfit(x,y,deg=1)
        pred=intercept+slope*x
        sst=((y-y.mean())**2).sum()
        r2=1-((y-pred)**2).sum()/sst if sst>0 else np.nan
        out.append({"element":e,"slope_K_per_wt_pct":slope,"intercept_C":intercept,
                    "R2":r2,"n_points":len(x)})
    return pd.DataFrame(out)


def plot_partition(result:pd.DataFrame,output:Path):
    fig,ax=plt.subplots(figsize=(8,5),dpi=150)
    for e in PLOT_ELEMENTS:
        ax.plot(result["delta_T_K"],result[f"k_{e}"],label=e,linewidth=1.5)
    ax.axhline(1,ls="--",color="gray",label="k = 1")
    ax.set_xlabel("Undercooling relative to the liquidus, ΔT (K)")
    ax.set_ylabel("Phase concentration ratio, Cs/Cl")
    ax.grid(alpha=.35,linestyle=":")
    ax.legend(ncol=3,fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"partition_vs_undercooling.png",dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solid",type=Path,required=True,help="solid-phase composition export, e.g. leverscheil/scheil2.txt")
    p.add_argument("--liquid",type=Path,required=True,help="liquid-phase composition export, e.g. leverscheil/scheil1.txt")
    p.add_argument("--slope-table",type=Path,default=None,
                   help="optional temperature-composition table (confirm it belongs to the intended phase), e.g. sm.txt")
    p.add_argument("--slope-elements",default=None,help="optional comma-separated list of elements")
    p.add_argument("--max-undercooling",type=float,default=None,
                   help="optional: fit only within this undercooling range below the liquidus; default is the whole table")
    p.add_argument("--liquidus",type=float,default=1370.9043)
    p.add_argument("--temperature-tol",type=float,default=1e-3)
    p.add_argument("--report-row",type=int,default=18,help="0-based row reported by the original script; skipped if out of range")
    p.add_argument("--output",type=Path,default=Path("results/05_phase_diagram"))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    solid=read_phase_table(a.solid)
    liquid=read_phase_table(a.liquid)
    k=calculate_partition(solid,liquid,a.liquidus,a.temperature_tol)
    k.to_csv(a.output/"partition_coefficients_vs_temperature.csv",index=False,encoding="utf-8-sig")
    plot_partition(k,a.output)
    if 0<=a.report_row<len(k):
        print(f"Partition coefficients at row {a.report_row} (0-based index):\n{k.iloc[a.report_row].round(3).to_string()}")
    if a.slope_table is not None:
        slopes=fit_temperature_slopes(read_phase_table(a.slope_table),
            [v.strip() for v in a.slope_elements.split(",") if v.strip()] if a.slope_elements else None,
            a.max_undercooling,a.liquidus)
        slopes.to_csv(a.output/"temperature_composition_slopes.csv",index=False,encoding="utf-8-sig")
        print("Interpretation of the slopes depends on the actual phase in --slope-table and on the fitted temperature range")
    print(f"Phase-diagram data processed: {a.output.resolve()}")


if __name__=="__main__":
    main()
