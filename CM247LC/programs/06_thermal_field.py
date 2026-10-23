# -*- coding: utf-8 -*-
""" Thermal-field derived quantities.
input: data/thermal_raw_output.txt.
output:
  results/B1_thermal_axis2.csv, B1_thermal_axis1.csv    V, R, G, ΔT
  results/T4_11_steady_stats.csv                      (x ≥ 70 mm)
  results/06_G_autocorr_spectrum.csv                  
  results/06_RG_consistency.csv                       
  results/06_axis_coherence.csv                       
  results/06_arrival_time_noise.csv                   
  figures/F01_thermal_vs_position.png                 
  figures/F02_G_position_steady.png                  
  figures/F03_G_autocorr_spectrum.png                 
  figures/F04_axis_coherence.png                      
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import ch4lib as L

th = L.load_thermal()
a2, a1 = th["axis2"], th["axis1"]
L.save_table(a2.round(5), "B1_thermal_axis2", index=False)
L.save_table(a1.round(5), "B1_thermal_axis1", index=False)

# ---- TABLE 4.11
rows = []
st = a2[a2.x_mm >= L.STEADY_X]
for col, name in (("V_K_s", "Cooling rate Tdot /(K s^-1)"), ("R_mm_min", "Interface velocity R /(mm min^-1)"),
                  ("G_K_mm", "Temperature gradient G /(K mm^-1)"), ("dT_K", "Undercooling dT /K")):
    s = st[col].dropna()
    rows.append(dict(quantity=name, n=len(s), mean=s.mean(), sd=s.std(ddof=1), cv_pct=100 * s.std(ddof=1) / s.mean(),
                     min=s.min(), max=s.max(), ext_lo_pct=100 * (s.min() / s.mean() - 1), ext_hi_pct=100 * (s.max() / s.mean() - 1)))
s = a2[a2.x_mm > L.STEADY_X].V_K_s
rows.append(dict(quantity="Cooling rate, x>70 mm (transient point removed)", n=len(s), mean=s.mean(), sd=s.std(ddof=1),
                 cv_pct=100 * s.std(ddof=1) / s.mean(), min=s.min(), max=s.max(),
                 ext_lo_pct=100 * (s.min() / s.mean() - 1), ext_hi_pct=100 * (s.max() / s.mean() - 1)))
t411 = pd.DataFrame(rows)
L.save_table(t411.round(4), "T4_11_steady_stats", index=False)

# ---- G (i = 14..34)
g = st.G_K_mm.dropna().to_numpy(); xg = st.x_mm[st.G_K_mm.notna()].to_numpy() + L.DX
gd = L.detrend_linear(g)
ac = L.acf(gd, 4)
pad = 4096
spec = np.abs(rfft(gd, pad)) ** 2; fr = rfftfreq(pad, d=L.DX)
k = np.argmax(spec[1:]) + 1; lam_spec = 1 / fr[k]
peaks = [xg[i] for i in range(1, len(gd) - 1) if gd[i] > gd[i - 1] and gd[i] > gd[i + 1]]
# (x_center ≥ 45 mm)
w = (a2.x_mm >= 40) & a2.G_K_mm.notna()
gdw = L.detrend_linear(a2.G_K_mm[w].to_numpy()); xw = a2.x_mm[w].to_numpy() + L.DX
peaks_w = [xw[i] for i in range(1, len(gdw) - 1) if gdw[i] > gdw[i - 1] and gdw[i] > gdw[i + 1]]
acf_df = pd.DataFrame([dict(item="acf lag 5 mm", value=ac[1]), dict(item="acf lag 10 mm", value=ac[2]),
                       dict(item="acf lag 15 mm", value=ac[3]), dict(item="acf lag 20 mm", value=ac[4]),
                       dict(item="spectral peak wavelength /mm (zero-padded periodogram)", value=lam_spec),
                       dict(item="mean detrended peak spacing /mm (steady)", value=np.mean(np.diff(peaks)) if len(peaks) > 1 else np.nan),
                       dict(item="G peaks, steady window (centres, mm)", value=";".join(f"{p:.0f}" for p in peaks)),
                       dict(item="G peaks, pairing window x_c>=45 mm (centres, mm)", value=";".join(f"{p:.0f}" for p in peaks_w))])
L.save_table(acf_df, "06_G_autocorr_spectrum", index=False)

# ---- R–G 
ok = st.R_mm_min.notna() & st.G_K_mm.notna()
rd = L.detrend_linear(st.R_mm_min[ok]); gd2 = L.detrend_linear(st.G_K_mm[ok])
r_RG = np.corrcoef(rd, gd2)[0, 1]
slope = np.polyfit(-gd2 / st.G_K_mm[ok].mean(), rd / st.R_mm_min[ok].mean(), 1)[0]
rr, gg = a2.R_mm_min.to_numpy(), a2.G_K_mm.to_numpy(); m = (a2.x_mm >= L.STEADY_X).to_numpy()
ok2 = m[:-1] & ~np.isnan(rr[:-1]) & ~np.isnan(gg[1:])      # R_i (centre x_i+2.5) vs G_{i+1} (centre x_i+10): 7.5 mm apart
r_shift = np.corrcoef(L.detrend_linear(rr[:-1][ok2]), L.detrend_linear(gg[1:][ok2]))[0, 1]
cvV = 100 * s.std(ddof=1) / s.mean(); cvG = t411.loc[2, "cv_pct"]; cvR = t411.loc[1, "cv_pct"]
rg = pd.DataFrame([dict(item="r(R,G), detrended, steady, centres 2.5 mm apart", value=r_RG),
                   dict(item="regression slope dR/R0 vs -dG/G0", value=slope),
                   dict(item="slope corrected for misregistration cos(2*pi*2.5/19.3)", value=slope / np.cos(2 * np.pi * 2.5 / 19.3)),
                   dict(item="r(R_i, G_i+1) (centres 7.5 mm apart, about 0.4 lambda)", value=r_shift),
                   dict(item="sqrt(CV_V^2 + CV_G^2) /% (predicted CV_R)", value=np.hypot(cvV, cvG)),
                   dict(item="measured CV_R /%", value=cvR)])
L.save_table(rg.round(4), "06_RG_consistency", index=False)

# ---- axis1&2
rows = []
for col in ("V_K_s", "R_mm_min", "G_K_mm"):
    okc = (a2.x_mm >= L.STEADY_X) & a1[col].notna() & a2[col].notna()
    r = np.corrcoef(L.detrend_linear(a1[col][okc]), L.detrend_linear(a2[col][okc]))[0, 1]
    okd = okc & (a2.x_mm > L.STEADY_X)
    r2 = np.corrcoef(L.detrend_linear(a1[col][okd]), L.detrend_linear(a2[col][okd]))[0, 1]
    rows.append(dict(quantity=col, mean_axis1=a1[col][okc].mean(), mean_axis2=a2[col][okc].mean(),
                     mean_diff_pct=100 * (a1[col][okc].mean() / a2[col][okc].mean() - 1),
                     r_axes_x_ge_70=r, n=int(okc.sum()), p_x_ge_70=L.pearson_p(r, okc.sum()),
                     r_axes_x_gt_70=r2, n_gt=int(okd.sum()), p_x_gt_70=L.pearson_p(r2, okd.sum())))
L.save_table(pd.DataFrame(rows).round(4), "06_axis_coherence", index=False)

# ---- arrival_time
t = L.arrival_times_from_R(a2); x = a2.x_mm.to_numpy()
mt = (x >= L.STEADY_X) & (x <= 175)
p = np.polyfit(x[mt], t[mt], 1); res = t[mt] - np.polyval(p, x[mt])
d2 = (t[2:] - t[:-2]); xd = x[:-2]; md = xd >= L.STEADY_X
acd = L.acf(L.detrend_linear(d2[md]), 4)
V = a2.V_K_s.to_numpy(); G = a2.G_K_mm.to_numpy()
Gapp = np.full_like(G, np.nan); Gapp[:-2] = V[:-2] * d2 / (2 * L.DX)
oka = m & ~np.isnan(G) & ~np.isnan(Gapp)
r_app = np.corrcoef(G[oka], Gapp[oka])[0, 1]
r_dt = np.corrcoef(L.detrend_linear(G[oka]), L.detrend_linear(d2[oka[:-2]]))[0, 1]
noise = pd.DataFrame([dict(item="arrival-time fit slope /(s mm^-1), x=70-175 mm", value=p[0]),
                      dict(item="implied mean R /(mm min^-1)", value=60 / p[0]),
                      dict(item="arrival-time residual s.d. /s", value=res.std(ddof=1)),
                      dict(item="residual acf lag1/lag2/lag3", value=";".join(f"{v:.2f}" for v in L.acf(res, 3)[1:])),
                      dict(item="acf of 2-step arrival-time difference, lag 5/10/15/20 mm", value=";".join(f"{v:.2f}" for v in acd[1:])),
                      dict(item="theory: white-noise 2-step difference acf at lag 10 mm", value=-0.5),
                      dict(item="r(G, Tdot*(t_i+2 - t_i)/10 mm)", value=r_app),
                      dict(item="R^2 of G explained by arrival-time differences", value=r_app ** 2),
                      dict(item="r(detrended G, detrended t_i+2 - t_i)", value=r_dt)])
L.save_table(noise, "06_arrival_time_noise", index=False)

# ---- fig 4.24
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, df, title in ((axes[0], a2, "Axis 2 (centreline)"), (axes[1], a1, "Axis 1 (5 mm off-axis)")):
    ax.plot(df.x_mm, df.R_mm_min, "s--", ms=3, label="R /(mm min$^{-1}$)")
    ax.plot(df.x_mm, df.V_K_s, "o-", ms=3, label="$\\dot T$ /(K s$^{-1}$)")
    if df.dT_K.notna().any():
        ax.plot(df.x_mm, df.dT_K, "x:", ms=4, label="$\\Delta T$ /K")
    ax.set_yscale("log"); ax.set_xlabel("Distance from chill face $x$ /mm"); ax.set_title(title)
    ax2 = ax.twinx(); ax2.plot(df.G_center_mm, df.G_K_mm, "^-.", color="tab:green", ms=3, label="G /(K mm$^{-1}$)")
    ax2.set_ylabel("G /(K mm$^{-1}$)", color="tab:green"); ax2.set_ylim(0, 45)
    ax.axvspan(0, L.STEADY_X, color="0.9", zorder=0); ax.set_ylim(0.1, 100)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
axes[0].set_ylabel("R, $\\dot T$, $\\Delta T$ (log scale)")
fig.suptitle("Thermal quantities versus true axial position (G corrected: 10 mm interval; grey = transient stage)")
fig.tight_layout(); fig.savefig(L.FIGS / "F01_thermal_vs_position.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ---- fig 4.26
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4))
axes[0].plot(a2.G_center_mm, a2.G_K_mm, "o-", ms=3, label="Axis 2"); axes[0].plot(a1.G_center_mm, a1.G_K_mm, "s--", ms=3, alpha=.6, label="Axis 1")
axes[0].set_xlabel("Interval centre $x_i+5$ /mm"); axes[0].set_ylabel("G /(K mm$^{-1}$)"); axes[0].legend(); axes[0].set_title("(a) Full length")
stm = a2.G_center_mm >= 75
axes[1].plot(a2.G_center_mm[stm], a2.G_K_mm[stm], "o-", ms=4, label="Axis 2"); axes[1].plot(a1.G_center_mm[stm], a1.G_K_mm[stm], "s--", ms=3, alpha=.6, label="Axis 1")
axes[1].axhline(t411.loc[2, "mean"], color="k", lw=.8, ls=":"); axes[1].set_title("(b) Steady stage, x$_c$ ≥ 75 mm")
axes[1].set_xlabel("Interval centre $x_i+5$ /mm"); axes[1].set_ylabel("G /(K mm$^{-1}$)"); axes[1].legend()
fig.tight_layout(); fig.savefig(L.FIGS / "F02_G_position_steady.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ----  F03
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
lags = np.arange(0, 7) * L.DX; axes[0].bar(lags, L.acf(gd, 6), width=3); axes[0].axhline(-0.5, ls="--", color="r", lw=.8, label="white-noise 2-step difference: −0.5 at 10 mm")
axes[0].set_xlabel("Lag /mm"); axes[0].set_ylabel("Autocorrelation of detrended G"); axes[0].legend(fontsize=8)
sel = (fr > 1 / 60); axes[1].plot(1 / fr[sel], spec[sel] / spec[sel].max()); axes[1].axvline(lam_spec, color="r", ls="--", lw=.8, label=f"peak {lam_spec:.1f} mm")
axes[1].set_xlabel("Wavelength /mm"); axes[1].set_ylabel("Normalised power"); axes[1].legend()
fig.suptitle("Steady-stage G (centreline, x ≥ 70 mm): autocorrelation and periodogram"); fig.tight_layout()
fig.savefig(L.FIGS / "F03_G_autocorr_spectrum.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ----  F04
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for ax, col, lab in zip(axes, ("V_K_s", "R_mm_min", "G_K_mm"), ("$\\dot T$", "R", "G")):
    okc = (a2.x_mm >= L.STEADY_X) & a1[col].notna() & a2[col].notna()
    xx = a2.x_mm[okc] + (L.DX if col == "G_K_mm" else 0)
    ax.plot(xx, L.detrend_linear(a2[col][okc]) / a2[col][okc].mean(), "o-", ms=3, label="Axis 2")
    ax.plot(xx, L.detrend_linear(a1[col][okc]) / a1[col][okc].mean(), "s--", ms=3, label="Axis 1")
    r = np.corrcoef(L.detrend_linear(a1[col][okc]), L.detrend_linear(a2[col][okc]))[0, 1]
    ax.set_ylabel(f"δ{lab}/{lab}$_0$"); ax.set_title(f"r(axis1, axis2) = {r:+.2f}", fontsize=9, loc="right"); ax.axhline(0, color="k", lw=.5)
axes[0].legend(fontsize=8); axes[-1].set_xlabel("x /mm")
fig.suptitle("Detrended steady-stage fluctuations on the two axes"); fig.tight_layout()
fig.savefig(L.FIGS / "F04_axis_coherence.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("06 done")
