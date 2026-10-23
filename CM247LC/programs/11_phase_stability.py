# -*- coding: utf-8 -*-
"""11  γ-matrix composition and Nv, Md (§4.3.6, Figs 4.20–4.21).

OUTPUT:
  results/B5_gamma_phase_stability.csv    
  results/11_correction_factors.csv       
  results/11_summary.csv                  Nv, Md 
  figures/F18_gamma_composition.png       FIGURE 4.20
  figures/F19_phase_stability.png         FIGURE 4.21
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ch4lib as L

MASS = {"Cr": 52.00, "Ti": 47.90, "Mo": 95.94, "Al": 26.98, "Co": 58.93, "B": 10.81, "Zr": 91.22, "C": 12.01,
        "W": 183.85, "Ta": 180.95, "Hf": 178.49, "Ni": 58.71}
NV = {"Cr": 4.66, "Ti": 6.66, "Mo": 4.66, "Al": 7.66, "Co": 1.71, "Zr": 6.66, "W": 4.66, "Ta": 5.66, "Hf": 6.66, "Ni": 0.66}
MD = {"Cr": 1.142, "Ti": 2.271, "Mo": 1.550, "Al": 1.900, "Co": 0.777, "Zr": 2.944, "W": 1.655, "Ta": 2.224, "Hf": 3.020, "Ni": 0.717}
CORRECTION_FACTOR = {"Ni": 0.9449838187702265, "W": 1.003157894736842, "Ta": 1.3925, "Al": 1.2732142857142859,
                     "Co": 1.007608695652174, "Cr": 1.0462962962962963, "Hf": 0.9221428571428572, "Mo": 1.348, "Ti": 1.3542857142857143}
NI_OFFSET = 0.092
TRACE = {"B": 0.015, "C": 0.07, "Zr": 0.007}
NV_C, MD_C = 2.32, 0.925


def atomic(row):
    amt = {e: row[e] / MASS[e] for e in MASS}
    s = sum(amt.values()); return {e: v / s for e, v in amt.items()}


def gamma_matrix(a):
    mo_w = a["Mo"] + a["W"]; cg = a["Ti"] + a["Zr"] + a["Ta"] + a["Hf"]
    cr = 0.97 * a["Cr"] - 0.375 * a["B"]; co = a["Co"] - a["C"]
    mo = a["Mo"] - 0.75 * a["B"] - 1.5 * a["C"] * a["Mo"] / mo_w
    zr = a["Zr"] - 0.5 * a["C"] * a["Zr"] / cg
    w = a["W"] - 1.5 * a["C"] * a["W"] / mo_w
    ni = a["Ni"] - 0.15 * a["B"] - 0.5 * a["C"] - 3 * (a["Al"] + 0.03 * a["Cr"] + a["Ti"] - 0.225 * a["B"]
                                                      - 0.5 * a["C"] * a["Ti"] / cg + a["Ta"] - 0.5 * a["C"] * a["Ta"] / cg + a["Hf"] - 0.5 * a["C"] * a["Hf"] / cg)
    raw = {"Ni": ni, "Cr": cr, "Co": co, "Mo": mo, "W": w, "Zr": zr}
    if min(raw.values()) < -1e-12:
        return None
    s = sum(raw.values()); return {e: v / s for e, v in raw.items()}


def run(eds, corrected):
    rows = []
    for j, r in eds.iterrows():
        row = {e: ((r[e] - (NI_OFFSET if e == "Ni" else 0)) / CORRECTION_FACTOR[e] if corrected else r[e]) for e in L.ELEMENTS}
        row.update(TRACE); a = atomic(row); g = gamma_matrix(a)
        rec = dict(j=j, x_mm=1 + 5 * j, Ni_alloy_at=100 * a["Ni"])
        if g is None:
            rec.update(valid=False)
        else:
            rec.update(valid=True, Nv_gamma=sum(g[e] * NV.get(e, 0) for e in g), Md_gamma=sum(g[e] * MD.get(e, 0) for e in g),
                       **{f"{e}_gamma_at": 100 * g[e] for e in ("Ni", "Cr", "Co", "W", "Mo")})
        rows.append(rec)
    return pd.DataFrame(rows)


eds = L.load_eds("da2")
corr = run(eds, True); raw = run(eds, False)
L.save_table(corr.round(4), "B5_gamma_phase_stability", index=False)
cf = pd.DataFrame({"correction_factor_program03": pd.Series(CORRECTION_FACTOR), "da2_mean/nominal": {e: eds[e].mean() / L.NOMINAL[e] for e in L.ELEMENTS}})
L.save_table(cf.round(4), "11_correction_factors")
summ = pd.DataFrame([dict(scheme="corrected (program 03 factors)", n_valid=int(corr.valid.sum()), Nv_min=corr.Nv_gamma.min(), Nv_max=corr.Nv_gamma.max(),
                          n_Nv_above_2_32=int((corr.Nv_gamma > NV_C).sum()), Md_min=corr.Md_gamma.min(), Md_max=corr.Md_gamma.max(),
                          n_Md_above_0_925=int((corr.Md_gamma > MD_C).sum()), Ni_gamma_mean_at=corr.Ni_gamma_at.mean(),
                          CrCo_gamma_mean_at=(corr.Cr_gamma_at + corr.Co_gamma_at).mean(), W_gamma_mean_at=corr.W_gamma_at.mean()),
                     dict(scheme="uncorrected measured wt%", n_valid=int(raw.valid.sum()), Nv_min=np.nan, Nv_max=np.nan, n_Nv_above_2_32=np.nan,
                          Md_min=np.nan, Md_max=np.nan, n_Md_above_0_925=np.nan, Ni_gamma_mean_at=np.nan, CrCo_gamma_mean_at=np.nan, W_gamma_mean_at=np.nan)])
L.save_table(summ.round(3), "11_summary", index=False)

x = corr.x_mm
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
axes[0, 0].plot(x, corr.Ni_gamma_at, "o-", ms=3); axes[0, 0].set_ylabel("Ni in γ /at%"); axes[0, 0].set_title("(a) Ni", loc="left", fontsize=10)
axes[0, 1].plot(x, corr.Cr_gamma_at, "o-", ms=3, label="Cr"); axes[0, 1].plot(x, corr.Co_gamma_at, "s-", ms=3, label="Co"); axes[0, 1].legend(); axes[0, 1].set_ylabel("at%"); axes[0, 1].set_title("(b) Co, Cr", loc="left", fontsize=10)
axes[1, 0].plot(x, corr.W_gamma_at, "o-", ms=3); axes[1, 0].set_ylabel("W in γ /at%"); axes[1, 0].set_title("(c) W", loc="left", fontsize=10)
axes[1, 1].plot(x, corr.Mo_gamma_at, "o-", ms=3); axes[1, 1].set_ylabel("Mo in γ /at%"); axes[1, 1].set_title("(d) Mo", loc="left", fontsize=10)
for ax in axes[1]: ax.set_xlabel("x /mm (nominal, 1 + 5j)")
fig.suptitle("Estimated γ-matrix composition (empirical partitioning model; element-wise corrected EDS input)"); fig.tight_layout()
fig.savefig(L.FIGS / "F18_gamma_composition.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(x, corr.Nv_gamma, "o-", ms=3); axes[0].axhline(NV_C, ls="--", color="r", lw=.8, label="N̄$_{v,c}$ = 2.32"); axes[0].set_ylabel("N̄$_{v,γ}$"); axes[0].legend(fontsize=8); axes[0].set_title("(a)", loc="left")
axes[1].plot(x, corr.Md_gamma, "o-", ms=3); axes[1].axhline(MD_C, ls="--", color="r", lw=.8, label="M̄$_{d,c}$ = 0.925"); axes[1].set_ylabel("M̄$_{d,γ}$ /eV"); axes[1].legend(fontsize=8); axes[1].set_title("(b)", loc="left")
for ax in axes: ax.set_xlabel("x /mm (nominal, 1 + 5j)")
fig.tight_layout(); fig.savefig(L.FIGS / "F19_phase_stability.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("11 done")
