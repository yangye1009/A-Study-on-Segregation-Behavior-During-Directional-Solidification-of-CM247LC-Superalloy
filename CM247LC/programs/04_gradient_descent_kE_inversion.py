#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program 4: log-linearised Scheil form + hand-written gradient descent to invert the effective partition coefficient k_E.

    Cs = k_E*C0*(1-fs)^(k_E-1)
    ln(Cs) = A + b*ln(1-fs), k_E = b+1, C0 = exp(A)/k_E

Supports batch fitting column by column and automatically checks the difference between the gradient-descent
solution and the analytical least-squares solution.
The original 10/17/34-point inputs use the linear solid-fraction ranges given in the thesis code; this is a model
assumption, not the true solid fraction read point by point from the simulation data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS_DEFAULTS = {10:(0.05,0.85),17:(0.025,0.875),34:(0.,0.9)}


def cost_function(X: np.ndarray,y: np.ndarray,theta: np.ndarray) -> float:
    """J(theta)=||Xtheta-y||²/(2*n)."""
    residual = X @ theta - y
    return float(np.dot(residual,residual)/(2*len(y)))


def gradient_descent(X: np.ndarray,y: np.ndarray,alpha: float=0.01,
                     max_iter: int=4000,tolerance: float=1e-10):
    """Vectorised gradient descent without the global lm variable; the loss history is recorded to check convergence."""
    if alpha <= 0 or max_iter < 1:
        raise ValueError("learning rate must be > 0 and the number of iterations >= 1")
    theta = np.zeros(X.shape[1],dtype=float)
    history = [cost_function(X,y,theta)]
    for iteration in range(max_iter):
        residual = X@theta-y
        gradient = X.T@residual/len(y)
        theta -= alpha*gradient
        if iteration % 10 == 9 or iteration == max_iter-1:
            current = cost_function(X,y,theta)
            if not np.isfinite(current):
                raise FloatingPointError("gradient descent diverged; reduce the learning rate")
            history.append(current)
            if np.linalg.norm(gradient,ord=np.inf) < tolerance:
                break
    return theta, np.asarray(history), iteration+1


def r2(y_true:np.ndarray,y_pred:np.ndarray)->float:
    sst = np.sum((y_true-y_true.mean())**2)
    if sst <= 0:
        return float("nan")
    return float(1-np.sum((y_true-y_pred)**2)/sst)


def fit_element(composition: np.ndarray,fs:np.ndarray,alpha:float=0.01,
                max_iter:int=4000,tolerance:float=1e-10):
    """Fit one element and return the parameters, fitted curve and iteration information."""
    c = np.asarray(composition,dtype=float).reshape(-1)
    f = np.asarray(fs,dtype=float).reshape(-1)
    if len(c)!=len(f) or len(c)<3:
        raise ValueError("composition and solid fraction must have equal length and at least 3 points")
    if not np.isfinite(c).all() or np.any(c<=0):
        raise ValueError("log fitting requires all compositions to be finite and positive")
    if not np.isfinite(f).all() or np.any((f<0)|(f>=1)):
        raise ValueError("solid fraction must lie in [0,1)")
    X = np.column_stack([np.ones(len(f)),np.log1p(-f)])
    y = np.log(c)
    theta,history,iterations = gradient_descent(X,y,alpha,max_iter,tolerance)
    # The analytical least-squares solution is an independent numerical check; it does not replace the gradient-descent result.
    ols = np.linalg.lstsq(X,y,rcond=None)[0]
    A,b = map(float,theta)
    k_E = b+1.
    if k_E<=0:
        raise ValueError(f"inverted k_E={k_E:.5g} <= 0, outside the validity range of the current model")
    c0 = float(np.exp(A)/k_E)
    fitted = np.exp(X@theta)
    result = {
        "n_points":len(c),"ln_intercept_A":A,"slope_b":b,"k_E":k_E,
        "nominal_C0_wt_pct":c0,"R2_original":r2(c,fitted),
        "R2_log":r2(y,np.log(fitted)),"cost":history[-1],
        "iterations":iterations,
        "max_abs_difference_from_OLS":float(np.max(np.abs(theta-ols))),
        "OLS_k_E_reference":float(1+ols[1]),
        "final_gradient_inf_norm":float(np.linalg.norm(X.T@(X@theta-y)/len(y),np.inf)),
    }
    return result,fitted,history


def infer_elements(df:pd.DataFrame)->list[str]:
    """Exclude the index, position, time and direction columns of the export."""
    ignore={"time","direction","position","position_cm","position_mm",
            "fs","solid_fraction","Unnamed: 0","index","spotID","areaID"}
    return [e for e in df.columns if e not in ignore and
            pd.api.types.is_numeric_dtype(df[e])]


def run(data:Path,output:Path,elements:str|None=None,
        fs_start:float|None=None,fs_end:float|None=None,
        alpha:float=.01,iterations:int=4000,tol:float=1e-10):
    df = pd.read_csv(data)
    n=len(df)
    if fs_start is None or fs_end is None:
        if n not in FS_DEFAULTS:
            raise ValueError(f"no preset solid-fraction mapping for {n} points; provide both --fs-start and --fs-end")
        start,end=FS_DEFAULTS[n]
        start = fs_start if fs_start is not None else start
        end = fs_end if fs_end is not None else end
    else:
        start,end=fs_start,fs_end
    if not (0<=start<end<1):
        raise ValueError("the solid-fraction range must satisfy 0 <= fs-start < fs-end < 1")
    f = np.linspace(start,end,n)
    names = [s.strip() for s in elements.split(",") if s.strip()] if elements else infer_elements(df)
    if not names:
        raise ValueError("no element columns available for fitting")
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    failures=[]
    for element in names:
        try:
            if element not in df:
                raise KeyError(f"column {element} not found in the data table")
            c = pd.to_numeric(df[element],errors="raise").to_numpy(dtype=float)
            res, fitted, history=fit_element(c,f,alpha,iterations,tol)
            res["element"]=element
            results.append(res)
            pd.DataFrame({"nominal_solid_fraction":f,"measured_wt_pct":c,
                          "fitted_wt_pct":fitted}).to_csv(
                output/f"fit_{element}.csv",index=False,encoding="utf-8-sig")
            fig,ax=plt.subplots(figsize=(5,3.3),dpi=160)
            ax.plot(f,c,"o",markersize=3,label="Measured")
            ax.plot(f,fitted,"-",linewidth=1.5,label="Fitted")
            ax.axhline(res["nominal_C0_wt_pct"],ls=":",color="gray",label="Fitted C0")
            ax.set_xlabel("Nominal solid fraction (linear mapping)")
            ax.set_ylabel(f"{element} (wt%)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(output/f"fit_{element}.png",dpi=300)
            plt.close(fig)
            if res["max_abs_difference_from_OLS"] > 1e-3:
                warnings.warn(f"{element}: gradient descent has not fully converged (deviation from the analytical OLS solution {res['max_abs_difference_from_OLS']:.3g}); consider more iterations")
        except (ValueError,KeyError,FloatingPointError) as exc:
            failures.append((element,str(exc)))
    if not results:
        raise RuntimeError("no element was fitted successfully: "+repr(failures))
    pd.DataFrame(results).set_index("element").to_csv(output/"effective_partition_coefficients.csv",encoding="utf-8-sig")
    if failures:
        (output/"fit_failures.txt").write_text("\n".join(f"{k}: {v}" for k,v in failures),encoding="utf-8")
    print(f"{len(names)} element(s) specified/detected, {len(results)} fitted successfully; output: {output.resolve()}")
    if failures:
        print(f"{len(failures)} element(s) could not be fitted; see fit_failures.txt")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True,help="e.g. mapping1216/da2.csv (34 points)")
    p.add_argument("--elements",default=None,help="comma-separated, e.g. Al,Co,Cr,Hf,Mo,Ta,Ti,W,Ni; auto-detected if omitted")
    p.add_argument("--fs-start",type=float,default=None)
    p.add_argument("--fs-end",type=float,default=None)
    p.add_argument("--alpha",type=float,default=.01)
    p.add_argument("--iters",type=int,default=4000)
    p.add_argument("--tol",type=float,default=1e-10)
    p.add_argument("--output",type=Path,default=Path("results/04_partition_fit"))
    a=p.parse_args()
    run(a.data,a.output,a.elements,a.fs_start,a.fs_end,a.alpha,a.iters,a.tol)


if __name__=="__main__":
    main()
