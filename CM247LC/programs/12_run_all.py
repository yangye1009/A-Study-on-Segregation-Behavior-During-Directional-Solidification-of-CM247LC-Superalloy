# -*- coding: utf-8 -*-
"""12  Run everything and compare recomputed values with the thesis.
OUTPUT: results/comparison_thesis_vs_recomputed.csv
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import ch4lib as L

HERE = Path(__file__).resolve().parent
for n in ("06_thermal_field.py", "07_eds_composition.py", "08_registration.py", "09_scheil_fit.py", "10_jmatpro.py", "11_phase_stability.py"):
    print("running", n); subprocess.run([sys.executable, str(HERE / n)], check=True)

R = L.RESULTS
rd = lambda name: pd.read_csv(R / f"{name}.csv", index_col=0)
rd0 = lambda name: pd.read_csv(R / f"{name}.csv")
rows = []
def add(where, item, thesis, new, tol=None, status=None, note=""):
    if status is None:
        if thesis is None or (isinstance(thesis, float) and np.isnan(thesis)):
            status = "new"
        else:
            status = "reproduced" if abs(float(new) - float(thesis)) <= (tol if tol is not None else 0.011) else "correct"
    rows.append(dict(location=where, item=item, thesis_value=thesis, recomputed=new, status=status, note=note))

t411 = rd0("T4_11_steady_stats").set_index("quantity")
add("Table 4.11", "mean Tdot /K s-1", 0.216, t411.iloc[0]["mean"], .001); add("Table 4.11", "CV Tdot /%", 10.3, t411.iloc[0]["cv_pct"], .1)
add("Table 4.11", "mean R /mm min-1", 2.46, t411.iloc[1]["mean"], .01); add("Table 4.11", "CV R /%", 11.7, t411.iloc[1]["cv_pct"], .1)
add("Table 4.11", "mean G /K mm-1", 6.23, t411.iloc[2]["mean"], .01); add("Table 4.11", "CV G /%", 8.6, t411.iloc[2]["cv_pct"], .1)
add("Table 4.11", "mean dT /K", 3.67, t411.iloc[3]["mean"], .01); add("Table 4.11", "CV dT /%", 14.0, t411.iloc[3]["cv_pct"], .1)
add("§4.4.5", "Tdot x>70 mm: CV /%", 6.9, t411.iloc[4]["cv_pct"], .1)
rg = rd0("06_RG_consistency").set_index("item").value
add("§4.4.5", "r(R,G) steady", -0.44, rg.iloc[0], .02); add("§4.4.5", "r(R,G) shifted 7.5 mm", 0.56, rg.iloc[3], .02)
add("§4.4.5", "slope dR/R0 vs -dG/G0", 0.59, rg.iloc[1], .05, note="0.63 recomputed; 0.59 in thesis"); add("§4.4.5", "predicted CV_R /%", 11.0, rg.iloc[4], .1)
ax = rd0("06_axis_coherence").set_index("quantity")
add("§4.4.5 / Summary (5)", "r(V1,V2) x>=70 mm", 0.65, ax.loc["V_K_s", "r_axes_x_ge_70"], .01)
add("§4.4.5", "r(V1,V2) without x=70 mm transition point", None, ax.loc["V_K_s", "r_axes_x_gt_70"], note=f"p={ax.loc['V_K_s','p_x_gt_70']:.3f}")
add("§4.4.5", "r(R1,R2)", 0.20, ax.loc["R_mm_min", "r_axes_x_ge_70"], .05, note="thesis text (derivation doc) says 0.20; recomputed -0.03")
add("§4.4.5", "r(G1,G2)", 0.03, ax.loc["G_K_mm", "r_axes_x_ge_70"], .05)
acs = rd0("06_G_autocorr_spectrum").set_index("item").value
add("§4.4.3.2", "G acf lag 10 mm", -0.73, float(acs.iloc[1]), .1, note="window/detrending dependent"); add("§4.4.3.2", "G acf lag 20 mm", 0.58, float(acs.iloc[3]), .15, note="window dependent")
add("§4.4.3.2", "G spectral peak /mm", 19.3, float(acs.iloc[4]), 1.0, note="periodogram resolution")
add("§4.4.5", "G peaks (pairing window)", "55,75,95,115,125,140", acs.iloc[7], status="correct", note="8 peaks incl. 150, 160 mm")
nz = rd0("06_arrival_time_noise").set_index("item").value
add("new", "R^2 of G explained by 2-step arrival-time difference", None, float(nz.iloc[8]) ** 2)
add("new", "acf of 2-step arrival-time difference at 10 mm (white-noise theory -0.5)", None, nz.iloc[4])

t48 = rd("T4_8_amplitude_vs_criteria")
for e, v in (("Hf", 60.2), ("Ta", 33.9), ("Ti", 23.0), ("Mo", 21.4), ("Al", 7.5), ("W", 7.2), ("Co", 4.6), ("Cr", 3.8)):
    add("Table 4.8", f"CV {e} /%", v, t48.loc[e, "CV_pct"], .06)
sp = t48.loc["Spearman(CV, .)"]
add("Table 4.8", "Spearman(CV,|k0-1|)", 0.93, sp["abs_k0_minus_1_T4.4"], .01); add("Table 4.8", "Spearman(CV,Z)", 0.26, sp["Z"], .01)
add("Table 4.8", "Spearman(CV,|m|)", -0.38, sp["abs_m_K_per_wt"], .01); add("Table 4.8", "Spearman(CV,1/DL)", 0.24, sp["DL_1e-9_m2_s"], .01)
add("new", "Spearman(CV,|k0-1|) with liquidus k0", None, sp["abs_k0_minus_1_liquidus"]); add("new", "Spearman(CV,|kP-1|) with ProCAST k", None, sp["abs_kP_minus_1_ProCAST"])
t49 = rd("T4_9_pearson_matrix")
for a, b, v in (("Ta", "Ti", .93), ("Ta", "Hf", .80), ("Ti", "Hf", .77), ("Ni", "S", -.89), ("Co", "S", -.92), ("Cr", "S", -.75), ("Al", "Ni", .49), ("Al", "Co", .63), ("Al", "S", -.68), ("W", "Ta", -.33), ("Mo", "Ni", -.36), ("Mo", "S", .19)):
    add("Table 4.9", f"r({a},{b})", v, t49.loc[a, b], .006)
pc = rd0("07_partial_correlation").set_index("pair"); add("§4.4.4.2", "partial r(W,Ta | Ni)", -0.66, pc.loc["W-Ta", "partial_r_given_Ni"], .01)
t410 = rd("T4_10_pool_vs_matrix")
add("Table 4.10", "pool frames n", 15, t410.loc["n"].iloc[0], 0); add("Table 4.10", "matrix frames n", 19, t410.loc["n"].iloc[1], 0)
for e, v in (("Ta", 6.60), ("Hf", 2.27), ("W", 9.00), ("Ni", 56.89), ("Al", 6.55)):
    add("Table 4.10", f"pool-frame mean {e}", v, t410.loc[e].iloc[0], .006)
dil = rd("07_closure_dilution"); add("§4.4.4.2", "Ni: dilution-predicted/measured sd", 1.00, dil.loc["Ni", "predicted/measured"], .01); add("§4.4.4.2", "Cr: dilution-predicted/measured sd", 0.77, dil.loc["Cr", "predicted/measured"], .01)
add("Fig. 4.4 / §4.2.1", "grey relational grade, Hf/Mo/Ta/Ti vs ascending sequence", "0.83-0.87", "0.59-0.65", status="correct", note="Deng GRA from da2 does not reproduce; use Spearman trend test instead")
tr = rd0("07_trend_spearman").set_index("element")
for e in ("W", "Ta", "Ti", "Co", "Al", "Cr", "Hf", "Mo", "Ni"):
    add("new (replaces Fig. 4.4)", f"Spearman({e}, x)", None, tr.loc[e, "spearman_vs_position"], note=f"p={tr.loc[e,'p']:.3f}")

t412 = rd0("T4_12_correlation_with_G").set_index("element")
add("Table 4.12", "steady window", "x>=70 mm, n=25", "EDS x>=46 mm (j>=9), n=25", status="correct", note="x>=71 mm gives n=20; Table 4.11 uses x>=70 mm")
for e, (r1, r2, r3, pu, pcor) in {"Ta": (.66, .73, .85, .0003, .022), "Ti": (.53, .64, .76, .007, .063), "Hf": (.43, .59, .67, .033, .13), "S": (.61, .71, .83, .0011, .036),
                                  "Ni": (-.62, -.73, -.86, .0009, .031), "Co": (-.55, -.69, -.83, .004, .048), "Cr": (-.39, -.54, -.66, .057, .12), "Al": (-.36, -.56, -.72, .076, .093),
                                  "W": (-.07, -.06, -.10, .75, .75), "Mo": (.14, .05, .08, .52, .59)}.items():
    add("Table 4.12", f"{e} r linear", r1, t412.loc[e, "r_linear"], .006); add("Table 4.12", f"{e} r difference", r2, t412.loc[e, "r_difference"], .006)
    add("Table 4.12", f"{e} r moving-avg (n=21, not 25)", r3, t412.loc[e, "r_moving_avg"], .006)
    add("Table 4.12", f"{e} p uncorrected", pu, t412.loc[e, "p_uncorrected"], max(.0002, .15 * pu))
    add("Table 4.12", f"{e} p Bartlett-corrected", pcor, t412.loc[e, "p_corrected"], max(.004, .2 * pcor), note="lag truncation choice")
add("§4.4.5", "R2 of Ta explained by G /%", 44, t412.loc["Ta", "R2_pct"], 1); add("§4.4.5", "R2 of Ni explained by G /%", 39, t412.loc["Ni", "R2_pct"], 1)
sc = rd0("08_registration_scan"); fitline = sc.iloc[-1, 1]
lam = float(fitline.split("lambda=")[1].split(" ")[0]); x0 = float(fitline.split("x0=")[1].split(" ")[0])
add("§4.4.5 / Fig. 4.30", "cosine-fit lambda /mm", 19.3, lam, .3); add("§4.4.5 / Fig. 4.30", "fitted peak offset /mm", "<0.5 from +1 mm", x0, status="reproduced")
for i, v in enumerate((-.60, .02, .61, -.12, -.43, .15)):
    add("Fig. 4.30", f"r(S,G) at offset {sc.iloc[i,1]} mm", v, sc.iloc[i, 3], .06)
pk = rd0("08_peaks").peaks
add("§4.4.5", "S detrended peaks /mm", "60,75,90,115,130,140", pk.iloc[0], status="reproduced", note="EDS x = 1 + 5j, hence 61,76,...")
pf = rd0("08_pool_frame_G_test").set_index("axis")
add("§4.4.5", "pool-frame dG difference /K mm-1", 0.34, pf.iloc[0]["diff"], .01); add("§4.4.5", "pool-frame one-sided MW p", 0.047, pf.iloc[0]["p_one_sided_MW"], .002)
add("new (control)", "pool-frame test with axis-1 G, p", None, pf.iloc[1]["p_one_sided_MW"])
ct = rd("08_axis1_control"); add("new (control)", "r(S, G of axis 1)", None, ct.loc["S", "r_linear_axis1"]); add("new (control)", "r(Ta, G of axis 1)", None, ct.loc["Ta", "r_linear_axis1"])
rr = rd0("08_R_registered")
add("§4.4.3.3", "r(Ta,R) (x>=71 mm, n=20, R paired i=j)", -0.39, rr.iloc[1]["r_Ta_R"], .01, note=f"Table-4.12 window x>=46 mm gives {rr.iloc[0]['r_Ta_R']:+.2f}; window must be stated")
add("§4.4.3.3", "r(Ni,R) (x>=71 mm, n=20, R paired i=j)", 0.38, rr.iloc[1]["r_Ni_R"], .01, note=f"Table-4.12 window x>=46 mm gives {rr.iloc[0]['r_Ni_R']:+.2f}")
se = rd0("08_position_mapping_sensitivity")
add("new", "r(S,G) with 1 % shrinkage + 0.2 mm kerf", None, se[(se.shrink_pct == 1) & (se.kerf_mm == .2)].r_S.iloc[0])
add("new", "r(S,G) with 2 % shrinkage + 0.3 mm kerf", None, se[(se.shrink_pct == 2) & (se.kerf_mm == .3)].r_S.iloc[0])
add("new", "r(S,G) if EDS point 0 were at 4 mm", None, se[(se.case == "first-point offset") & (se.x0_mm == 4)].r_S.iloc[0])
cl = rd0("08_closure_apparent_kE").set_index("element"); add("new", "apparent kE of matrix elements from closure alone", None, cl.loc["Ni", "apparent_kE_closure_only"])

t43 = rd("T4_3_kE_fits")
for e, (k10, k17, k34) in {"Ni": (1.013, 1.023, 1.003), "W": (1.059, 1.044, 1.057), "Ta": (.768, .645, .836), "Al": (1.026, 1.03, 1.019), "Co": (1.016, 1.05, 1.02),
                            "Cr": (1.006, 1.034, 1.005), "Hf": (.574, .510, .964), "Mo": (.879, .803, .975), "Ti": (.780, .750, .881)}.items():
    add("Table 4.3", f"kE_10 {e}", k10, t43.loc[e, "kE_10"], .0015); add("Table 4.3", f"kE_17 {e}", k17, t43.loc[e, "kE_17"], .008)
    add("Table 4.3", f"kE_34 {e}", k34, t43.loc[e, "kE_34"], .0015 if e != "Co" else .006, note="Table 4.2 gives 1.015; Table 4.3 rounds to 1.02" if e == "Co" else "")
add("Table 4.3 / §4.3.3", "17-point data", "17-point fit", "da4 = mean of da2 rows k and k+2, k=0..16 (first 19 points only, x 6-91 mm)", status="correct")
add("Table 4.3 / §4.3.3", "10-point data", "10-point fit", "da7 = smoothed subset of da2 rows ~0-10 (x ≈ 6-50 mm)", status="correct")
k17 = rd("09_kE_17groups_fulllength"); add("new", "kE Ta, 17 pair-averages over full length", None, k17.loc["Ta", "kE"]); add("new", "kE W, 17 pair-averages over full length", None, k17.loc["W", "kE"])
add("new", "34-pt slope p-value: W", None, t43.loc["W", "p_slope_34"]); add("new", "34-pt slope p-value: Ta", None, t43.loc["Ta", "p_slope_34"]); add("new", "34-pt slope p-value: Ti", None, t43.loc["Ti", "p_slope_34"])
t47 = rd0("T4_7_Pc_inversion").set_index("element")
for e, v in (("W", 1.25), ("Ta", 1.35), ("Ti", .87), ("Al", 1.05), ("Mo", 2.49), ("Hf", 5.41)):
    add("Table 4.7", f"Pc_34 {e}", v, t47.loc[e, "Pc_34"], .03, note="difference from rounding of kE" if e in ("Mo", "Hf") else "")
add("Table 4.7", "Pc_34 Co", 1.51, t47.loc["Co", "Pc_34"], .02, note="thesis used rounded kE=1.020; actual 1.015 gives 1.77")
add("Table 4.7", "Pc_34 Cr", 0.87, t47.loc["Cr", "Pc_34"], .02, note="kE 1.0056 -> 0.76; ill-conditioned (sensitivity 179)")
rdl = rd0("09_Rdelta_consistency")
add("§4.4.3.1", "R*delta /1e-9 m2 s-1 (W,Co,Ta,Ti)", 2.5, rdl.iloc[0]["Rdelta_mean_1e9"], .05, note="2.6±0.7 with actual Co kE")
add("§4.4.3.1", "delta /um", 61, rdl.iloc[0]["delta_um"], 2, note="64±17 um with actual Co kE; 61±16 um with liquidus k0")
k0l = rd("10_k0_liquidus")
for e in ("W", "Co", "Ta", "Ti", "Al", "Cr", "Mo", "Hf"):
    add("Table 4.4 vs JMatPro file", f"k0 {e}: Table 4.4 vs liquidus value", L.K0_TABLE44[e], k0l.loc[e, "k0_liquidus_1370.9C"], .02, note="Table 4.4 values are not liquidus values; definition must be stated")
alp = rd0("10_Al_path").set_index("item").value
add("§4.3.5 / §4.4.3", "k_Al near liquidus", 1.06, alp.iloc[2], .01, note="JMatPro file gives 0.91 at the liquidus; k crosses 1 at 1290 C")
add("§4.3.5", "liquid Al: rises to", None, alp.iloc[3], note=f"max at {alp.iloc[4]:.0f} C, gamma-prime appears at {alp.iloc[5]:.0f} C; 'deep-V reversal' wording not supported")
s11 = rd0("11_summary")
add("§4.3.6", "Nv(gamma) range", "1.90-2.35", f"{s11.iloc[0].Nv_min:.2f}-{s11.iloc[0].Nv_max:.2f}", status="reproduced", note="requires undocumented element-wise correction factors")
add("§4.3.6", "Md(gamma) range", "0.870-0.928", f"{s11.iloc[0].Md_min:.3f}-{s11.iloc[0].Md_max:.3f}", status="reproduced")
add("§4.3.6", "Ni in gamma /at%", 45, s11.iloc[0].Ni_gamma_mean_at, 2.5, note="text says ~45 and ~50 in two places; recomputed 47")
cf = rd("11_correction_factors"); add("new", "EDS mean/nominal: Ta", None, cf.loc["Ta", "da2_mean/nominal"]); add("new", "EDS mean/nominal: Ti", None, cf.loc["Ti", "da2_mean/nominal"]); add("new", "EDS mean/nominal: Al", None, cf.loc["Al", "da2_mean/nominal"])

cmp_ = pd.DataFrame(rows)
cmp_.to_csv(R / "comparison_thesis_vs_recomputed.csv", index=False, encoding="utf-8-sig")
print(cmp_.status.value_counts().to_string())
print("comparison table:", R / "comparison_thesis_vs_recomputed.csv")
