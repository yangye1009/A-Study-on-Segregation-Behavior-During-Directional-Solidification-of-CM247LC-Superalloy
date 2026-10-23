# -*- coding: utf-8 -*-
"""09 Scheil log-linear fits and BPS inversion (§4.3.3, §4.4.3.1).

MODEL: ln C = A + b ln(1−fs), kE = b + 1, C0 = exp(A)/kE
α = 0.01, 4000 STEPS

OUTPUT:
  results/T4_3_kE_fits.csv               CI, p, R²
  results/09_kE_17groups_fulllength.csv  17groups kE
  results/T4_7_Pc_inversion.csv          
  results/09_Rdelta_consistency.csv      Rdelta_consistency
  figures/F14_BPS_curve.png              FIGURE 4.29
  figures/F15_scheil_fits_CI.png         Ta, W, Co _scheil_fits_CI
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import ch4lib as L

E = L.ELEMENTS
d2, d4, d7 = L.load_eds("da2"), L.load_eds("da4"), L.load_eds("da7")


def fit(df, fs0, fs1):
    n = len(df); f = np.linspace(fs0, fs1, n); X = np.log1p(-f); rows = {}
    tcrit = stats.t.ppf(.975, n - 2)
    for e in E:
        y = np.log(df[e].to_numpy()); res = stats.linregress(X, y); kE = res.slope + 1
        rows[e] = dict(kE=kE, C0=np.exp(res.intercept) / kE, R2_log=res.rvalue ** 2, p_slope=res.pvalue,
                       kE_lo95=res.slope - tcrit * res.stderr + 1, kE_hi95=res.slope + tcrit * res.stderr + 1,
                       cost=np.mean((y - (res.intercept + res.slope * X)) ** 2) / 2)
    return pd.DataFrame(rows).T


f34, f17, f10 = fit(d2, 0, .9), fit(d4, .025, .875), fit(d7, .05, .85)
t43 = pd.DataFrame({"kE_10": f10.kE, "kE_17": f17.kE, "kE_34": f34.kE, "C0_10": f10.C0, "C0_17": f17.C0, "C0_34": f34.C0,
                    "kE_34_lo95": f34.kE_lo95, "kE_34_hi95": f34.kE_hi95, "p_slope_34": f34.p_slope, "R2_34": f34.R2_log,
                    "p_slope_17": f17.p_slope, "p_slope_10": f10.p_slope})
L.save_table(t43.round(4), "T4_3_kE_fits")

# ---- DATA
A = d2[E].to_numpy()
d4_recon = (A[:-2] + A[2:]) / 2
prov = pd.DataFrame([dict(dataset="da4 (17 points)", relation="row k = mean(da2 row k, row k+2), k = 0..16; uses da2 rows 0-18 only",
                          max_abs_diff_wt=float(np.abs(d4_recon[:17] - d4[E].to_numpy()).max()),
                          direction_range=f"{d4.direction.iloc[0]}–{d4.direction.iloc[-1]}", covered_x_mm="6–91 (nominal)"),
                     dict(dataset="da7 (10 points)", relation="smoothed subset of da2 rows ~0-10 (exact kernel not recoverable)",
                          max_abs_diff_wt=np.nan, direction_range=f"{d7.direction.iloc[0]}–{d7.direction.iloc[-1]}", covered_x_mm="≈6–50 (nominal)")])
L.save_table(prov, "09_data_provenance", index=False)
pairs = pd.DataFrame(A.reshape(17, 2, -1).mean(axis=1), columns=E)
L.save_table(fit(pairs, 0, .9)[["kE", "C0", "R2_log", "p_slope"]].round(4), "09_kE_17groups_fulllength")

# ---- TABLE 4.7
liq = L.load_jmatpro("liquid"); gam = L.load_jmatpro("gamma")
g0 = gam.iloc[0]; l0 = liq[liq.T_C == g0.T_C].iloc[0]
k0_liq = {e: float(g0[e] / l0[e]) for e in L.K0_TABLE44}


def Pc(kE, k0):
    x = kE * (1 - k0) / (k0 * (1 - kE))
    return np.log(x) if x > 0 else np.nan


R0 = 41e-6  # m/s
rows = []
for e in ("W", "Co", "Ta", "Ti", "Al", "Cr", "Mo", "Hf"):
    k0 = L.K0_TABLE44[e]; kE = (f10.kE[e], f17.kE[e], f34.kE[e])
    pcs = [Pc(k, k0) for k in kE]
    rows.append(dict(element=e, k0_T4_4=k0, k0_liquidus=k0_liq[e],
                     kE_10=kE[0], kE_17=kE[1], kE_34=kE[2],
                     Pc_10=pcs[0], Pc_17=pcs[1], Pc_34=pcs[2],
                     Pc_34_k0_liquidus=Pc(kE[2], k0_liq[e]),
                     PcDL_34_1e9=pcs[2] * L.DL_TABLE32[e],
                     delta_um_34=pcs[2] * L.DL_TABLE32[e] * 1e-9 / R0 * 1e6,
                     sensitivity_dPc_dkE=abs(1 / (kE[2] * (1 - kE[2]))),
                     Pc_34_lo95=Pc(f34.kE_lo95[e], k0) if k0 < 1 else Pc(f34.kE_hi95[e], k0),
                     Pc_34_hi95=((Pc(f34.kE_hi95[e], k0) if f34.kE_hi95[e] < 1 else np.inf) if k0 < 1 else (Pc(f34.kE_lo95[e], k0) if f34.kE_lo95[e] > 1 else np.inf))))
t47 = pd.DataFrame(rows)
# Co --- 1.015
t47.loc[t47.element == "Co", "note"] = f"thesis used kE=1.020 -> Pc=1.51; actual 34-pt kE={f34.kE['Co']:.4f} -> Pc={Pc(f34.kE['Co'], 1.097):.2f}"
L.save_table(t47.round(4), "T4_7_Pc_inversion", index=False)

four = ["W", "Co", "Ta", "Ti"]
rows = []
for lab, kset in (("Table 4.4 k0", L.K0_TABLE44), ("liquidus k0 (JMatPro)", k0_liq)):
    v = np.array([Pc(f34.kE[e], kset[e]) * L.DL_TABLE32[e] for e in four])
    rows.append(dict(k0_set=lab, Rdelta_mean_1e9=v.mean(), Rdelta_sd=v.std(ddof=1), scatter_pct=100 * v.std(ddof=1) / v.mean(),
                     delta_um=v.mean() * 1e-9 / R0 * 1e6, delta_sd_um=v.std(ddof=1) * 1e-9 / R0 * 1e6))
v = np.array([Pc(k, L.K0_TABLE44[e]) * L.DL_TABLE32[e] for e, k in zip(four, (1.057, 1.020, .836, .881))])
rows.append(dict(k0_set="thesis Table 4.7 values (Co rounded to 1.020)", Rdelta_mean_1e9=v.mean(), Rdelta_sd=v.std(ddof=1), scatter_pct=100 * v.std(ddof=1) / v.mean(),
                 delta_um=v.mean() * 1e-9 / R0 * 1e6, delta_sd_um=v.std(ddof=1) * 1e-9 / R0 * 1e6))
L.save_table(pd.DataFrame(rows).round(3), "09_Rdelta_consistency", index=False)

# ----  F14
fig, ax = plt.subplots(figsize=(6.4, 4.4))
pc = np.linspace(0, 6, 300)
for e, c in zip(("W", "Co", "Ta", "Ti", "Mo", "Hf"), plt.cm.tab10.colors):
    k0 = L.K0_TABLE44[e]; kE = k0 / (k0 + (1 - k0) * np.exp(-pc))
    ax.plot(pc, kE, color=c, lw=1.2, label=f"{e} (k$_0$={k0:.3f})")
    p34 = Pc(f34.kE[e], k0); ax.plot(p34, f34.kE[e], "o", color=c, ms=6)
ax.axhline(1, color="k", lw=.5); ax.set_xlabel("Effective Péclet number $P_c = R\\delta/D_L$"); ax.set_ylabel("$k_E$")
ax.set_title("BPS curves $k_E(P_c)$ with 34-point inversion points", fontsize=10); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(L.FIGS / "F14_BPS_curve.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ----  F15
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
f = np.linspace(0, .9, 34); X = np.log1p(-f)
for ax, e in zip(axes, ("Ta", "W", "Co")):
    y = np.log(d2[e].to_numpy()); res = stats.linregress(X, y)
    yhat = res.intercept + res.slope * X; n = len(y); s = np.sqrt(np.sum((y - yhat) ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (X - X.mean()) ** 2 / np.sum((X - X.mean()) ** 2)); tc = stats.t.ppf(.975, n - 2)
    ax.plot(f, d2[e], "o", ms=3, label="measured"); ax.plot(f, np.exp(yhat), "-", label=f"fit k$_E$={res.slope+1:.3f} [{res.slope-tc*res.stderr+1:.3f}, {res.slope+tc*res.stderr+1:.3f}]")
    ax.fill_between(f, np.exp(yhat - tc * se), np.exp(yhat + tc * se), alpha=.2)
    ax.set_xlabel("Nominal solid fraction (linear mapping)"); ax.set_ylabel(f"{e} /wt%"); ax.legend(fontsize=7); ax.set_title(f"p(slope≠0) = {res.pvalue:.3f}", fontsize=9)
fig.tight_layout(); fig.savefig(L.FIGS / "F15_scheil_fits_CI.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("09 done")
