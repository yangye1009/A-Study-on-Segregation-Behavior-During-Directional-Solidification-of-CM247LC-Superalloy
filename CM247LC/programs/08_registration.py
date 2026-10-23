# -*- coding: utf-8 -*-
"""08 / Composition–G registration and correlation (§4.4.5).

OUTPUT:
  results/B3_paired_series.csv             EDS + G, R, Tdot
  results/T4_12_correlation_with_G.csv     TABLE 4.12 
  results/08_window_sensitivity.csv        window_sensitivity (x ≥ 46 / 51 / 66 / 71 / 76 mm)
  results/08_registration_scan.csv         r(S,G) FIGURE 4.30
  results/08_peaks.csv                     S & G peaks
  results/08_pool_frame_G_test.csv         pool_frame Mann–Whitney 
  results/08_axis1_control.csv             axis1
  results/08_position_mapping_sensitivity.csv  position_mapping_sensitivity
  results/08_R_registered.csv              R 
  results/08_closure_apparent_kE.csv       closure_apparent_kE
  figures/F10_registration_cosine_fit.png  FIGURE 4.30
  figures/F11_S_vs_G_aligned.png           
  figures/F12_position_mapping_sensitivity.png 
  figures/F13_axis1_control.png            
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import ch4lib as L

E = L.ELEMENTS + ["S"]
th = L.load_thermal(); a2, a1 = th["axis2"], th["axis1"]
eds = L.load_eds("da2")
P = L.pair_with_G(eds, a2, offset=-1)
P1 = L.pair_with_G(eds, a1, offset=-1)
L.save_table(P.round(4), "B3_paired_series")

JMIN_PAPER = 9          # x >= 46 mm, n = 25 (TABLE 4.12)
JMIN_T411 = 14          # x >= 71 mm, n = 20 (TABLE 4.11)


def ma_resid(y, w=5):
    s = pd.Series(np.asarray(y, float)); return (s - s.rolling(w, center=True).mean()).to_numpy()


def corr_table(P, jmin):
    m = ((P.index >= jmin) & P.G_K_mm.notna()).to_numpy()
    g = P.G_K_mm.to_numpy()[m]; rows = []
    for e in E:
        c = P[e].to_numpy()[m]
        cd, gd = L.detrend_linear(c), L.detrend_linear(g)
        r_lin = np.corrcoef(cd, gd)[0, 1]
        r_dif = np.corrcoef(np.diff(c), np.diff(g))[0, 1]
        a, b = ma_resid(c), ma_resid(g); ok = ~np.isnan(a) & ~np.isnan(b)
        r_ma = np.corrcoef(a[ok], b[ok])[0, 1]
        n = len(c); neff = L.bartlett_neff(cd, gd)
        rows.append(dict(element=e, n=n, r_linear=r_lin, r_difference=r_dif, r_moving_avg=r_ma, n_moving_avg=int(ok.sum()),
                         p_uncorrected=L.pearson_p(r_lin, n), n_eff_Bartlett=neff, p_corrected=L.pearson_p(r_lin, neff),
                         R2_pct=100 * r_lin ** 2))
    return pd.DataFrame(rows)


t412 = corr_table(P, JMIN_PAPER)
L.save_table(t412.round(4), "T4_12_correlation_with_G", index=False)
rows = []
for jmin in (8, 9, 10, 13, 14, 15):
    t = corr_table(P, jmin).set_index("element")
    rows.append(dict(jmin=jmin, x_min_mm=1 + 5 * jmin, n=int(t.loc["Ta", "n"]), **{f"r_{e}": t.loc[e, "r_linear"] for e in ("Ta", "Ti", "Hf", "S", "Ni", "Co", "Cr", "Al", "W", "Mo")}))
L.save_table(pd.DataFrame(rows).round(3), "08_window_sensitivity", index=False)

# ---- FIG 4.30
scan = []
for off in (-3, -2, -1, 0, 1, 2):
    Q = L.pair_with_G(eds, a2, offset=off)
    m = ((Q.index >= JMIN_PAPER) & Q.G_K_mm.notna()).to_numpy()
    r = np.corrcoef(L.detrend_linear(Q.S.to_numpy()[m]), L.detrend_linear(Q.G_K_mm.to_numpy()[m]))[0, 1]
    scan.append(dict(offset_index=off, misregistration_G_centre_minus_EDS_mm=-(4 + 5 * off), n=int(m.sum()), r_S_G=r))
scan = pd.DataFrame(scan)
f = lambda x, A, lam, x0: A * np.cos(2 * np.pi * (x - x0) / lam)
popt, _ = curve_fit(f, scan.misregistration_G_centre_minus_EDS_mm, scan.r_S_G, p0=[.6, 18, 0])
scan["cosine_model"] = f(scan.misregistration_G_centre_minus_EDS_mm, *popt)
scan.attrs = dict(A=popt[0], lam=popt[1], x0=popt[2])
rms = np.sqrt(np.mean((scan.cosine_model - scan.r_S_G) ** 2))
scan = pd.concat([scan, pd.DataFrame([dict(offset_index="fit", misregistration_G_centre_minus_EDS_mm=f"A={popt[0]:.2f}, lambda={popt[1]:.1f} mm, x0={popt[2]:.1f} mm, rms={rms:.3f}")])])
L.save_table(scan.round(3), "08_registration_scan", index=False)

# ---- peaks
m = ((P.index >= JMIN_PAPER) & P.G_K_mm.notna()).to_numpy()
xs = P.x_mm.to_numpy()[m]; sd = L.detrend_linear(P.S.to_numpy()[m]); gd = L.detrend_linear(P.G_K_mm.to_numpy()[m])
pk = lambda y: [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]
peaks = pd.DataFrame([dict(series="S (EDS x_j, mm)", peaks=";".join(f"{xs[i]:.0f}" for i in pk(sd))),
                      dict(series="G paired (interval centre, mm)", peaks=";".join(f"{xs[i]-1:.0f}" for i in pk(gd)))])
L.save_table(peaks, "08_peaks", index=False)

# ---- pool_frame_G_test
d = P[m]; gdd = L.detrend_linear(d.G_K_mm); pool = (d.Hf > 1.5).to_numpy()
gd1 = L.detrend_linear(P1[m].G_K_mm)
pf = pd.DataFrame([dict(axis="axis 2 (centreline)", n_pool=int(pool.sum()), n_matrix=int((~pool).sum()),
                        mean_dG_pool=gdd[pool].mean(), mean_dG_matrix=gdd[~pool].mean(), diff=gdd[pool].mean() - gdd[~pool].mean(),
                        sd_dG=gdd.std(ddof=1), p_one_sided_MW=stats.mannwhitneyu(gdd[pool], gdd[~pool], alternative="greater").pvalue),
                   dict(axis="axis 1 (control)", n_pool=int(pool.sum()), n_matrix=int((~pool).sum()),
                        mean_dG_pool=gd1[pool].mean(), mean_dG_matrix=gd1[~pool].mean(), diff=gd1[pool].mean() - gd1[~pool].mean(),
                        sd_dG=gd1.std(ddof=1), p_one_sided_MW=stats.mannwhitneyu(gd1[pool], gd1[~pool], alternative="greater").pvalue)])
L.save_table(pf.round(4), "08_pool_frame_G_test", index=False)

# ---- axis1_control
c1 = corr_table(P1, JMIN_PAPER).set_index("element")
ctrl = pd.DataFrame({"r_linear_axis2": t412.set_index("element").r_linear, "r_linear_axis1": c1.r_linear, "p_corr_axis1": c1.p_corrected})
L.save_table(ctrl.round(3), "08_axis1_control")

# ---- position_mapping_sensitivity
gc = a2.G_center_mm.to_numpy(); gv = a2.G_K_mm.to_numpy(); okg = ~np.isnan(gv)
def r_mapped(x0=L.EDS_X0, shrink=0.0, kerf=0.0, elem="S", jmin=JMIN_PAPER):
    j = np.arange(34); seg = j // 5
    x = (x0 + 5 * j + kerf * seg) / (1 - shrink)          # REALITY → MODEL
    gi = np.interp(x, gc[okg], gv[okg]); mm_ = j >= jmin
    c = (eds[L.GROUP_A].sum(axis=1) if elem == "S" else eds[elem]).to_numpy()
    return np.corrcoef(L.detrend_linear(c[mm_]), L.detrend_linear(gi[mm_]))[0, 1]
rows = [dict(case="paired index rule (reference)", x0_mm=1, shrink_pct=0, kerf_mm=0, r_S=t412.set_index("element").loc["S", "r_linear"], r_Ta=t412.set_index("element").loc["Ta", "r_linear"])]
for x0 in (-2, -1, 0, 1, 2, 3, 4, 5):
    rows.append(dict(case="first-point offset", x0_mm=x0, shrink_pct=0, kerf_mm=0, r_S=r_mapped(x0), r_Ta=r_mapped(x0, elem="Ta")))
for sh, kf in ((0, .3), (.01, 0), (.01, .2), (.02, .3), (.03, .3)):
    rows.append(dict(case="shrinkage + kerf (6 kerfs, 5 points per segment)", x0_mm=1, shrink_pct=100 * sh, kerf_mm=kf, r_S=r_mapped(1, sh, kf), r_Ta=r_mapped(1, sh, kf, "Ta")))
sens = pd.DataFrame(rows); L.save_table(sens.round(3), "08_position_mapping_sensitivity", index=False)

# ---- R 
# R_i -- x_i+2.5
# §4.4.3.3  r(Ta,R) = -0.39, r(Ni,R) = +0.38 (x >= 71 mm) 
PR = L.pair_with_G(eds, a2, offset=0)
rows = []
for jmin, lab in ((JMIN_PAPER, "x>=46 mm (n=25)"), (JMIN_T411, "x>=71 mm (n=20)")):
    mm_ = ((PR.index >= jmin) & PR.R_mm_min.notna()).to_numpy(); r = PR.R_mm_min.to_numpy()[mm_]
    rows.append(dict(window=lab, R_pairing="i=j (R interval centre 1.5 mm from EDS point)",
                     **{f"r_{e}_R": np.corrcoef(L.detrend_linear(PR[e].to_numpy()[mm_]), L.detrend_linear(r))[0, 1] for e in ("Ta", "Ni", "S")}))
L.save_table(pd.DataFrame(rows).round(3), "08_R_registered", index=False)

# ---- closure_apparent kE
syn = eds[L.ELEMENTS].copy()
for e in ("Al", "Co", "Cr", "Mo", "W", "Ni"):
    syn[e] = eds[e].mean()
syn = syn.div(syn.sum(axis=1), axis=0) * 100
X = np.log1p(-np.linspace(0, .9, 34))
cl = pd.DataFrame([dict(element=e, apparent_kE_closure_only=stats.linregress(X, np.log(syn[e]))[0] + 1, kE_measured_34pt=stats.linregress(X, np.log(eds[e]))[0] + 1)
                   for e in ("Ni", "Co", "Cr", "Al", "W")])
L.save_table(cl.round(4), "08_closure_apparent_kE", index=False)

# ---- FIGURE
fig, ax = plt.subplots(figsize=(6.2, 4))
sc = scan.iloc[:6]; xx = np.linspace(-16, 16, 300)
ax.plot(xx, f(xx, *popt), "-", color="0.5", label=f"cosine fit: λ = {popt[1]:.1f} mm, peak at {popt[2]:+.1f} mm")
ax.plot(sc.misregistration_G_centre_minus_EDS_mm.astype(float), sc.r_S_G, "o", color="tab:red", label="measured r(S, G), x ≥ 46 mm")
ax.axvline(1, ls=":", color="k", lw=.8); ax.text(1.3, -0.7, "geometric\nregistration\n(+1 mm)", fontsize=7)
ax.axhline(0, color="k", lw=.5); ax.set_xlabel("Misregistration Δx = (G interval centre) − (EDS position) /mm"); ax.set_ylabel("r(S, G), linear detrending")
ax.legend(fontsize=8); ax.set_ylim(-0.8, 0.8)
fig.tight_layout(); fig.savefig(L.FIGS / "F10_registration_cosine_fit.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(xs, sd / P.S[m].mean(), "o-", ms=3, label="S = Ta+Hf+Ti (detrended, /mean)")
ax.plot(xs, gd / P.G_K_mm[m].mean(), "s--", ms=3, label="G paired (detrended, /mean)")
ax.axhline(0, color="k", lw=.5); ax.set_xlabel("EDS position $x_j$ /mm"); ax.set_ylabel("Relative fluctuation")
r_s = t412.set_index("element").loc["S", "r_linear"]; ax.set_title(f"Registered series, x ≥ 46 mm: r = {r_s:+.2f}", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(L.FIGS / "F11_S_vs_G_aligned.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
s1 = sens[sens.case == "first-point offset"]
axes[0].plot(s1.x0_mm, s1.r_S, "o-", label="S"); axes[0].plot(s1.x0_mm, s1.r_Ta, "s--", label="Ta"); axes[0].axvline(1, ls=":", color="k")
axes[0].set_xlabel("Assumed position of EDS point 0 /mm"); axes[0].set_ylabel("r with G (x ≥ 46 mm)"); axes[0].legend(); axes[0].set_title("(a) First-point offset", fontsize=10)
shr = np.linspace(0, .03, 13); kerfs = (0, .2, .3, .5)
for kf in kerfs:
    axes[1].plot(100 * shr, [r_mapped(1, s, kf) for s in shr], "o-", ms=3, label=f"kerf {kf} mm")
axes[1].set_xlabel("Assumed shrinkage /%"); axes[1].set_ylabel("r(S, G)"); axes[1].legend(fontsize=8); axes[1].set_title("(b) Shrinkage and kerf (6 kerfs)", fontsize=10)
for ax in axes: ax.axhline(0, color="k", lw=.5)
fig.tight_layout(); fig.savefig(L.FIGS / "F12_position_mapping_sensitivity.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
ii = np.arange(len(ctrl)); w = .38
ax.bar(ii - w / 2, ctrl.r_linear_axis2, w, label="G of axis 2 (centreline)"); ax.bar(ii + w / 2, ctrl.r_linear_axis1, w, label="G of axis 1 (5 mm off-axis, control)")
ax.set_xticks(ii); ax.set_xticklabels(ctrl.index); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("r (linear detrending, x ≥ 46 mm)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(L.FIGS / "F13_axis1_control.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("08 done")
