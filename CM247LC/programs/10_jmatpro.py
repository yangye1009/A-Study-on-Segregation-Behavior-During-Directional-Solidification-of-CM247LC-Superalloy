# -*- coding: utf-8 -*-
"""10  JMatPro Scheil-path data (§4.3.5, §4.4.3).
INPUT: liquid.dat, gamma.dat, gammap.dat, Al1.dat / Al2.dat
OUTPUT:
  results/B4_jmatpro_k_vs_T.csv     k_i(T) = C_γ/C_L 
  results/10_k0_liquidus.csv         
  results/10_Al_path.csv            
  figures/F16_k_vs_T.png             
  figures/F17_Al_phases.png          
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ch4lib as L

liq, gam, gp = L.load_jmatpro("liquid"), L.load_jmatpro("gamma"), L.load_jmatpro("gammap")
E8 = ["Al", "Co", "Cr", "Hf", "Mo", "Ta", "Ti", "W"]
m = gam.merge(liq, on="T_C", suffixes=("_g", "_l"))
k = pd.DataFrame({"T_C": m.T_C, "dT_below_liquidus_K": L.T_LIQ - m.T_C})
for e in E8:
    k[f"k_{e}"] = m[f"{e}_g"] / m[f"{e}_l"]
L.save_table(k.round(4), "B4_jmatpro_k_vs_T", index=False)

k0 = k.iloc[0]
kP = {"Al": 0.935, "Co": 1.14, "Cr": 0.81, "Hf": 0.095, "Mo": 0.639, "Ta": 0.826, "Ti": 0.808, "W": 1.03}
#  Scheil 1370.9–1256 °C
sub = k[(k.T_C <= L.T_LIQ) & (k.T_C >= 1256)]
t = pd.DataFrame({"k0_liquidus_1370.9C": [k0[f"k_{e}"] for e in E8],
                  "k_mean_1370.9-1256C": [sub[f"k_{e}"].mean() for e in E8],
                  "k_at_1256C": [sub[f"k_{e}"].iloc[-1] for e in E8],
                  "Table_4.4_kJ": [L.K0_TABLE44[e] for e in E8],
                  "Table_3.2_kP_ProCAST": [kP[e] for e in E8],
                  "T_where_k_crosses_1_C": [next((float(T) for T, kk in zip(k.T_C, k[f"k_{e}"]) if (kk - 1) * (k0[f"k_{e}"] - 1) < 0), np.nan) for e in E8]},
                 index=E8)
L.save_table(t.round(3), "10_k0_liquidus")

al = L.load_jmatpro("Al1")
imax = al.LIQUID.idxmax()
alp = pd.DataFrame([dict(item="nominal Al /wt%", value=5.6),
                    dict(item="gamma Al at liquidus /wt%", value=al.GAMMA.dropna().iloc[0]),
                    dict(item="k_Al at liquidus", value=al.GAMMA.dropna().iloc[0] / 5.6),
                    dict(item="liquid Al maximum /wt%", value=al.LIQUID.max()),
                    dict(item="T of liquid Al maximum /C", value=al.T_C[imax]),
                    dict(item="first T with gamma-prime /C", value=al.T_C[al.GAMMA_PRIME.first_valid_index()]),
                    dict(item="liquid Al at gamma-prime onset /wt%", value=al.LIQUID[al.GAMMA_PRIME.first_valid_index()]),
                    dict(item="liquid Al at last liquid row /wt%", value=al.LIQUID.dropna().iloc[-1]),
                    dict(item="T of last liquid row /C", value=al.T_C[al.LIQUID.last_valid_index()]),
                    dict(item="T where k_Al crosses 1 /C", value=t.loc["Al", "T_where_k_crosses_1_C"])])
L.save_table(alp.round(4), "10_Al_path", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for e in E8:
    ax = axes[0] if L.K0_TABLE44[e] > .5 else axes[1]
    ax.plot(k.T_C, k[f"k_{e}"], label=e)
for ax in axes:
    ax.axhline(1, color="k", lw=.5); ax.invert_xaxis(); ax.set_xlabel("Temperature /°C"); ax.set_ylabel("k = C$_γ$ / C$_L$"); ax.legend(fontsize=8)
    ax.axvline(1256, ls=":", color="0.5"); ax.text(1256, ax.get_ylim()[1] * .98, "γ′ onset", fontsize=7, va="top")
axes[0].set_title("(a) Al, Co, Cr, Mo, Ta, Ti, W", fontsize=10); axes[1].set_title("(b) Hf", fontsize=10)
fig.suptitle("JMatPro (Scheil-Gulliver) partition coefficients along the solidification path"); fig.tight_layout()
fig.savefig(L.FIGS / "F16_k_vs_T.png", dpi=250, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4))
for col, lab in (("LIQUID", "liquid"), ("GAMMA", "γ"), ("GAMMA_PRIME", "γ′")):
    ax.plot(al.T_C, al[col], label=lab)
ax.axhline(5.6, color="k", lw=.5, ls=":"); ax.invert_xaxis(); ax.set_xlabel("Temperature /°C"); ax.set_ylabel("Al /wt%"); ax.legend()
ax.set_title("Al in each phase (JMatPro Scheil): liquid rises to 6.33 wt% at 1290 °C, then falls; k_Al crosses 1 at ≈1290 °C", fontsize=8)
fig.tight_layout(); fig.savefig(L.FIGS / "F17_Al_phases.png", dpi=250, bbox_inches="tight"); plt.close(fig)
print("10 done")
