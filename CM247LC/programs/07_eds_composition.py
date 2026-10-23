# -*- coding: utf-8 -*-
"""07 EDS composition statistics (34 points, da2.csv).

OUTPUT:
  results/B2_eds_34points.csv              x_j = 1 + 5 j
  results/T4_8_amplitude_vs_criteria.csv   k 
  results/T4_9_pearson_matrix.csv          T4.9 
  results/T4_10_pool_vs_matrix.csv         T4.10 
  results/07_closure_dilution.csv          §4.4.4
  results/07_partial_correlation.csv       W–Ta (partial_correlation)
  results/07_trend_spearman.csv            Spearman
  figures/F05_pearson_heatmap.png          pearson_heatmap
  figures/F07_trend_spearman.png           Spearman 
  figures/F08_composition_vs_position.png  composition_vs_position
  figures/F09_CV_vs_k0.png                 CV_vs_|k0−1| 
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import ch4lib as L

E = L.ELEMENTS
eds = L.load_eds("da2")
eds["x_mm"] = L.eds_positions(len(eds))
eds["S"] = eds[L.GROUP_A].sum(axis=1)
L.save_table(eds[["x_mm"] + E + ["S", "direction"]], "B2_eds_34points")

# ---- T 4.8
E8 = ["Hf", "Ta", "Ti", "Mo", "Al", "W", "Co", "Cr"]
cv = 100 * eds[E8].std(ddof=1) / eds[E8].mean()
liq = L.load_jmatpro("liquid"); gam = L.load_jmatpro("gamma")
g0 = gam.iloc[0]; l0 = liq[liq.T_C == g0.T_C].iloc[0]
k0_liq = {e: float(g0[e] / l0[e]) for e in E8}
kP = {"Al": 0.935, "Co": 1.14, "Cr": 0.81, "Hf": 0.095, "Mo": 0.639, "Ta": 0.826, "Ti": 0.808, "W": 1.03}  # Table 3.2 k_P
t48 = pd.DataFrame({"CV_pct": cv, "Z": [L.Z_NUM[e] for e in E8],
                    "abs_k0_minus_1_T4.4": [abs(L.K0_TABLE44[e] - 1) for e in E8],
                    "abs_k0_minus_1_liquidus": [abs(k0_liq[e] - 1) for e in E8],
                    "abs_kP_minus_1_ProCAST": [abs(kP[e] - 1) for e in E8],
                    "abs_m_K_per_wt": [abs(L.M_TABLE32[e]) for e in E8],
                    "DL_1e-9_m2_s": [L.DL_TABLE32[e] for e in E8]}, index=E8)
t48["CV_scheil_random_area"] = [abs(1 - L.K0_TABLE44[e]) / np.sqrt(2 * L.K0_TABLE44[e] - 1) * 100 if L.K0_TABLE44[e] > .5 else np.nan for e in E8]
sp = {c: stats.spearmanr(cv, t48[c] if c != "DL_1e-9_m2_s" else 1 / t48[c])[0] for c in t48.columns[1:-1]}
t48.loc["Spearman(CV, .)"] = [np.nan] + [sp[c] for c in t48.columns[1:-1]] + [np.nan]
L.save_table(t48.round(3), "T4_8_amplitude_vs_criteria")

# ---- T 4.9
corr = eds[E + ["S"]].corr()
L.save_table(corr.round(3), "T4_9_pearson_matrix")

# ---- T 4.10
pool = eds.Hf > 1.5
cols = ["Hf", "Ta", "Ti", "Al", "W", "Co", "Cr", "Ni", "Mo"]
t410 = pd.DataFrame({"pool_frames_Hf>1.5": eds.loc[pool, cols].mean(), "matrix_frames_Hf<=1.5": eds.loc[~pool, cols].mean()})
t410["difference"] = t410.iloc[:, 0] - t410.iloc[:, 1]
t410.loc["n"] = [pool.sum(), (~pool).sum(), np.nan]
L.save_table(t410.round(3), "T4_10_pool_vs_matrix")

# ---- closure_dilution
M = ["Ni", "Cr", "Co", "W", "Al", "Mo"]
wsum = eds[M].mean().sum(); sS = eds.S.std(ddof=1)
dil = pd.DataFrame({"mean_wt": eds[M].mean(), "sd_predicted_dilution": eds[M].mean() / wsum * sS,
                    "sd_measured": eds[M].std(ddof=1)})
dil["predicted/measured"] = dil.sd_predicted_dilution / dil.sd_measured
dil["r_with_S"] = [corr.loc[e, "S"] for e in M]
L.save_table(dil.round(3), "07_closure_dilution")

# ---- partial_correlation
def pcorr(a, b, c):
    rab, rac, rbc = np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1], np.corrcoef(b, c)[0, 1]
    return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
pc = pd.DataFrame([dict(pair="W-Ta", r=corr.loc["W", "Ta"], partial_r_given_Ni=pcorr(eds.W, eds.Ta, eds.Ni)),
                   dict(pair="W-S", r=corr.loc["W", "S"], partial_r_given_Ni=pcorr(eds.W, eds.S, eds.Ni)),
                   dict(pair="Al-S", r=corr.loc["Al", "S"], partial_r_given_Ni=pcorr(eds.Al, eds.S, eds.Ni))])
L.save_table(pc.round(3), "07_partial_correlation", index=False)

# ---- "ascending", "descending"
def mm(x): x = np.asarray(x, float); return (x - x.min()) / (x.max() - x.min())
def deng(ref, comp, rho=.5):
    d = np.abs(ref - comp)
    if d.max() == 0: return 1.0
    return float(np.mean((d.min() + rho * d.max()) / (d + rho * d.max())))
seqs = {e: mm(eds[e]) for e in E}; seqs["ascending"] = mm(np.arange(len(eds))); seqs["descending"] = 1 - seqs["ascending"]
names = E + ["ascending", "descending"]
gra = pd.DataFrame([[deng(seqs[a], seqs[b]) for b in names] for a in names], index=names, columns=names)
L.save_table(gra.round(3), "07_grey_relational")

# ---- trend_spearman
trend = pd.DataFrame([dict(element=e, spearman_vs_position=stats.spearmanr(eds.x_mm, eds[e])[0],
                           p=stats.spearmanr(eds.x_mm, eds[e])[1],
                           kendall_tau=stats.kendalltau(eds.x_mm, eds[e])[0],
                           GRA_ascending=gra.loc[e, "ascending"], GRA_descending=gra.loc[e, "descending"]) for e in E])
L.save_table(trend.round(3), "07_trend_spearman", index=False)

# ---- FIGURE
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr))); ax.set_xticklabels(corr.columns); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if abs(corr.values[i, j]) > .6 else "k")
fig.colorbar(im, ax=ax, label="Pearson r"); ax.set_title("Pearson correlation, 34 points (S = Ta+Hf+Ti)", fontsize=10)
fig.tight_layout(); fig.savefig(L.FIGS / "F05_pearson_heatmap.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(gra.values, cmap="viridis", vmin=.5, vmax=1)
ax.set_xticks(range(len(gra))); ax.set_yticks(range(len(gra))); ax.set_xticklabels(gra.columns, rotation=45, ha="right"); ax.set_yticklabels(gra.index)
for i in range(len(gra)):
    for j in range(len(gra)):
        ax.text(j, i, f"{gra.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if gra.values[i, j] < .75 else "k")
fig.colorbar(im, ax=ax, label="Deng grey relational grade (ρ = 0.5)"); ax.set_title("Deng grey relational grades, 34 points (audit only)", fontsize=10)
fig.tight_layout(); fig.savefig(L.FIGS / "F06_grey_relational_heatmap.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
cols_ = ["tab:red" if p < .05 else ("tab:orange" if p < .1 else "0.6") for p in trend.p]
ax.bar(trend.element, trend.spearman_vs_position, color=cols_); ax.axhline(0, color="k", lw=.6)
ax.set_ylabel("Spearman ρ (composition vs x)"); ax.set_title("Monotonic axial trend of each element (red p<0.05, orange p<0.1)")
for e, r, p in zip(trend.element, trend.spearman_vs_position, trend.p):
    ax.text(e, r + (0.02 if r >= 0 else -0.06), f"p={p:.2f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(L.FIGS / "F07_trend_spearman.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
for ax, group, title in ((axes[0], ["Ta", "Hf", "Ti"], "(a) Rejecters (k<1)"), (axes[1], ["Co", "Cr", "Al", "W", "Ni"], "(b) Absorbers / matrix elements")):
    for e in group:
        ax.plot(eds.x_mm, eds[e] / eds[e].mean(), "o-", ms=3, lw=1, label=e)
    ax.axhline(1, color="k", lw=.5); ax.set_ylabel("C / C̄"); ax.legend(ncol=5, fontsize=8); ax.set_title(title, loc="left", fontsize=10)
    ax.axvspan(0, 46, color="0.92", zorder=0)
axes[-1].set_xlabel("Distance from chill face $x_j = 1 + 5j$ /mm (nominal; kerf and shrinkage not corrected)")
fig.suptitle("Normalised axial composition on the true position axis (grey = excluded from the steady-stage correlation)")
fig.tight_layout(); fig.savefig(L.FIGS / "F08_composition_vs_position.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.loglog(t48.loc[E8, "abs_k0_minus_1_T4.4"], t48.loc[E8, "CV_pct"], "o")
for e in E8:
    ax.annotate(e, (t48.loc[e, "abs_k0_minus_1_T4.4"], t48.loc[e, "CV_pct"]), textcoords="offset points", xytext=(4, 3), fontsize=8)
kk = np.linspace(.52, 1.3, 200); ax.plot(np.abs(kk - 1), 100 * np.abs(1 - kk) / np.sqrt(2 * kk - 1), "--", color="0.5", label="Scheil random-area CV")
ax.set_xlabel("|k$_0$ − 1| (Table 4.4)"); ax.set_ylabel("Measured CV /%"); ax.legend(fontsize=8)
ax.set_title(f"CV vs |k0−1|: Spearman {sp['abs_k0_minus_1_T4.4']:.2f}", fontsize=10)
fig.tight_layout(); fig.savefig(L.FIGS / "F09_CV_vs_k0.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("07 done")
