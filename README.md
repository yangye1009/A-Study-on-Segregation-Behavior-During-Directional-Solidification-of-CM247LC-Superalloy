# CM247LC Directional Solidification — Chapter 4 Computational Analysis

**Master's thesis (completed in 2023) · 2026 supplementary analysis and reproducibility package**

This repository documents the computational analysis of Chapter 4 of *A Study on Segregation Behavior During Directional Solidification of CM247LC Superalloy*. It covers ProCAST thermal-field post-processing, SEM–EDS compositional analysis, Scheil-based partition-coefficient estimates, JMatPro thermodynamic calculations, γ-phase stability, and the quantitative comparison of composition fluctuations with the simulated temperature field.

> **Version and provenance.** The master's thesis was completed in **2023**. The analyses described below as **2026 supplementary work** were performed **after graduation** and must not be attributed to the original 2023 thesis. They revisit the **original 2023 experimental and simulated data**; **no new experiments were performed in 2026**. The supplementary material is not a new degree submission.

## Research background and chronology

| Year | Research record | Status |
|---|---|---|
| **2023** | Master's thesis on CM247LC directional solidification, segregation, and associated experimental/computational analyses. In Chapter 4, the relationship between composition fluctuations and temperature-field fluctuations was proposed through the visual comparison of normalized curves (Fig. 4.25) and the general growth-layer-theory response (Eq. 4.14; refs. [84, 85]), **without a point-by-point quantitative test**. | Original completed thesis |
| **2024** | First-author journal article on microstructural evolution of the **same directionally solidified specimen**, including mesoscale W depletion in large-area scans (thesis ref. [88]). This observation motivated a later re-examination of the Chapter 4 interpretation. | Published paper |
| **2026** | Supplementary physical analysis, spatial registration, quantitative correlations, reproducibility audit, corrections, and an explicit limitations section using archived 2023 data. | Post-graduation supplementary analysis |

The central question of the 2026 analysis is **whether the measured axial composition fluctuations can be quantitatively related to the simulated local temperature-field fluctuations**, and how much of the observed agreement can be supported by the data and physical assumptions.

## What was added in 2026

The following items distinguish the supplementary analysis from the completed 2023 thesis. Thesis section, equation, table and figure numbers refer to the **supplemented manuscript**, not to the original deposited 2023 version.

| Addition | Scientific or methodological content | Location in supplemented manuscript |
|---|---|---|
| **Solute-boundary-layer interpretation** | Static Burton–Prim–Slichter (BPS) effect; inversion of an effective Péclet number and solute-exchange-layer scale; quasi-static criterion for the solute boundary layer; interpretation of the predicted sign of the composition–temperature-gradient alignment. | §§4.4.3.1–4.4.3.3; Table 4.7 |
| **Element-dependent fluctuation analysis** | Element grouping, the dilution effect of compositional closure (normalization to a fixed total), and an amplitude relationship based on segregation/partitioning behavior. | §4.4.4; Tables 4.8–4.10 |
| **Physical spatial registration** | Explicit pairing of measured EDS compositions and derived temperature-gradient values from their sampling geometry and finite-difference definition, rather than optimizing the pairing to maximize correlation. | §4.4.5; Eq. 4.24 |
| **Statistical tests and sensitivity checks** | Point-by-point Pearson correlations under **three detrending schemes**; significance adjusted using the **Bartlett effective sample size** to account for serial/spatial autocorrelation; registration-offset scan and cosine fit. | §4.4.5; Table 4.12; Fig. 4.30 |
| **Documented uncertainties** | Discussion of missing flow-field evidence, potential numerical differentiation artifacts, spatial-registration sensitivity, and limitations of EDS sampling and partition-coefficient inference. | §4.4.6 |

The 2026 additions include **Tables 4.7–4.12**, **Figs. 4.26–4.30**, and supporting **Appendices B and C** (data, calculations, and programs). Reference numbers **[88]–[94]** and **[95]** (Burton–Prim–Slichter, 1953) were added at this stage; references **[1]–[87]** belong to the 2023 original.

### Registration and correlation methodology

The thermal gradient is constructed over a **10 mm interval**. For thermal index `i`, where `x_i = 5i mm`, its spatial location is the **interval centre**, `x_i + 5 mm`, rather than the left endpoint. The nominal EDS measurement positions are `x_j = 1 + 5j mm` for `j = 0, ..., 33`. These definitions prescribe the pairing

```text
Thermal gradient:  G_i over [x_i, x_i + 10 mm], located at x_i + 5 mm
EDS composition:   C_j at x_j = 1 + 5j mm
Fixed pairing:     j ↔ i = j − 1   (nominal 1 mm positional offset)
```

This fixed rule is derived from how the thermal differences and EDS measurements were constructed; **the correlation was not used to select the primary registration**. Registration scans were subsequently used as a **sensitivity analysis**, not as proof of a uniquely determined spatial alignment.

At each prescribed pairing, the analysis computes Pearson correlations using three detrending schemes. Correlation significance is adjusted for autocorrelation with a Bartlett-based effective sample size. Because this correction is approximate and does not by itself address every possible model-selection or multiple-testing issue, the resulting *p*-values are interpreted cautiously.

- **Primary composition–G comparison:** Table 4.12, EDS positions `x ≥ 46 mm` (`j ≥ 9`, nominal `n = 25`).
- **Steady thermal-field summary:** Table 4.11, `x ≥ 70 mm`.
- **Reference/control:** centreline is Axis 2; Axis 1 lies 5 mm off-axis.

### Main quantitative indications — and their evidential limits

- At the **predefined nominal registration**, the composition–gradient correlations are **statistically suggestive** for selected elements: approximately `r(Ta, G) = +0.66` and `r(Ni, G) = −0.62`, with Bartlett-adjusted *p*-values of about `0.024` and `0.035`, respectively. **These correlations do not establish causation or a specific convection mechanism.**
- The registration scan can be described by an approximately **19.4 mm** cosine wavelength, close to a peak in the computed `G` spectrum. However, the thermal-gradient finite-difference operator can itself generate oscillatory/autocorrelated structure from arrival-time noise; this wavelength must **not** be treated as independently established evidence of a physical oscillation.
- The **BPS-based effective Péclet-number inversion** and exchange-layer estimate offer a possible interpretation of why measured effective partition coefficients tend toward unity. The inferred scale is approximate and **only W shows a clearly significant deviation of the fitted effective partition coefficient from unity** in the 34-point analysis.
- The amplitude analysis finds a strong ordering of elemental fluctuation amplitudes with a partition-coefficient contrast (reported Spearman `ρ ≈ 0.93` for the selected JMatPro coefficient set). **That strength depends on the coefficient source** and should not be presented as universal.
- A proposed **thermosolutal-convection / solute-export mechanism remains a hypothesis**. The available ProCAST simulation uses a thermal-density model; the inferred solutal-density inversion was **not simulated as a solutal-buoyancy term**, and the relevant liquid flow fields were not extracted for direct verification.

For a complete interpretation, consult **§4.4.6 (Limitations)** of the supplemented manuscript rather than relying only on the headline correlations.

## Directory layout

```text
.
├── data/       archived inputs for 06–12 (EDS, JMatPro exports, ProCAST post-processing output)
├── programs/   01–12 and ch4lib.py
├── results/    CSV outputs of 06–12 (created automatically)
├── figures/    PNG figures of 06–12 (created automatically)
└── tests/      unit tests
```

## Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib` and `scipy`. Program 02 also needs `openpyxl` (`.xlsx`/`.xlsm`) or `xlrd` (`.xls`) for reading Excel files.

## Part 1 — Chapter 4 processing workflows (programs 01–05)

| Program | Purpose | Main input | Default output folder |
|---|---|---|---|
| `01_procast_thermal_postprocess.py` | Cooling rate `V`, interface velocity `R`, temperature gradient `G`, and undercooling `ΔT` on two axes; redraws Fig. 4.24 (initial and steady stages) | ProCAST temperature and solid-fraction exports of Axis 1 and Axis 2 (`time, P1…Pn`) | `results/01_thermal/` |
| `02_eds_batch_composition.py` | Batch-reads EDS Excel exports, identifies area/spot IDs, pivots to one row per spot, filters two area groups by a W threshold | Folder of EDS `.xls/.xlsx/.xlsm` files | `results/02_EDS/` |
| `03_gamma_phase_stability.py` | Corrected composition → atomic fraction → empirical γ-matrix composition → `Nv`, `Md`, `Bo` | CSV of element wt% | `results/03_stability/` |
| `04_gradient_descent_kE_inversion.py` | Scheil log-linear fit by gradient descent for the effective partition coefficient `k_E`, checked against the OLS solution | CSV of element wt% (preset `fs` ranges for 10, 17 and 34 points) | `results/04_partition_fit/` |
| `05_phase_diagram_partition_slope.py` | `k(T) = Cs/Cl` from phase-diagram (Scheil / lever) exports; optional `T–C` slope fits | 13-column solid and liquid tables; optional slope table | `results/05_phase_diagram/` |

## Part 2 — 2026 Chapter 4 recomputation and audit (programs 06–12)

### Archived inputs in `data/`

| File | Content |
|---|---|
| `thermal_raw_output.txt` | Printed output of the original ProCAST post-processing program: 8 lines (Axis 2: `V`, `R`, `G`, `ΔT`; Axis 1: `V`, `R`, `G`, `ΔT`), 37 values each |
| `da2.csv`, `da4.csv`, `da7.csv` | EDS compositions (wt%), 34, 17 and 10 points |
| `liquid.dat`, `gamma.dat`, `gammap.dat`, `Al1.dat` | JMatPro Scheil-path exports (phase compositions vs. temperature; Al in each phase) |

### Programs and generated outputs

| Program | Content | Tables | Figures |
|---|---|---|---|
| `06_thermal_field.py` | Thermal quantities on both axes; steady-stage statistics (Table 4.11); `G` autocorrelation and spectrum; `R–G` consistency; coherence of the two axes; arrival-time noise | `B1_thermal_axis1/2`, `T4_11_steady_stats`, `06_*` | F01–F04 |
| `07_eds_composition.py` | 34-point EDS statistics: CV vs segregation criteria (Table 4.8), Pearson matrix (Table 4.9), pool vs matrix frames (Table 4.10), closure dilution, partial correlation, Deng grey relational grades (**audit only**), Spearman trends | `B2_eds_34points`, `T4_8`, `T4_9`, `T4_10`, `07_*` | F05–F09 |
| `08_registration.py` | Composition–`G` registration and correlation (Table 4.12, Fig. 4.30); window and position-mapping sensitivity; pool-frame `G` test; Axis 1 control; `R` correlation; apparent `k_E` from closure | `B3_paired_series`, `T4_12`, `08_*` | F10–F13 |
| `09_scheil_fit.py` | Scheil fits with 95% CI and *p*-values (Table 4.3); provenance of `da4`/`da7`; BPS inversion of effective `Pc` (Table 4.7); `Rδ` consistency | `T4_3`, `T4_7`, `09_*` | F14–F15 |
| `10_jmatpro.py` | `k(T) = Cγ/CL` along the JMatPro Scheil path; liquidus `k0`; Al path | `B4_jmatpro_k_vs_T`, `10_*` | F16–F17 |
| `11_phase_stability.py` | γ-matrix composition and `Nv`, `Md` (Figs. 4.20–4.21) | `B5_gamma_phase_stability`, `11_*` | F18–F19 |
| `12_run_all.py` | Runs 06–11, then writes the audit table | `comparison_thesis_vs_recomputed.csv` | — |
## Conventions

- Derived thermal point `i` lies at `x = 5i mm` (`i = 0…36`, 0–180 mm from the chill face). `R_i` is taken between points `i` and `i+1`; `G` is taken over 10 mm and **centred at `x_i + 5 mm`**.
- EDS point `j` lies at `x = 1 + 5j mm` (`j = 0…33`; nominal; kerf and shrinkage not corrected). `S = Ta + Hf + Ti`.
- Pairing: EDS point `j ↔` derived point `j − 1` for `G` (1 mm nominal misregistration); `j ↔ j` for the `R` correlations of §4.4.3.3.
- Windows: Table 4.12 uses EDS `x ≥ 46 mm` (`j ≥ 9`, `n = 25`); Table 4.11 and the steady stage use `x ≥ 70 mm`.
- Axis 2 (centreline) is the primary axis; Axis 1 (5 mm off-axis) is a reference only.
- Solid fraction in the Scheil fits is a **nominal linear mapping**: 34 points `0–0.9`, 17 points `0.025–0.875`, 10 points `0.05–0.85`.

## Running the analysis

From the package root:

```bash
python -m unittest discover -s tests -v
cd programs && python 12_run_all.py
```

Programs 06–11 can also be run individually from `programs/` because they import `ch4lib`.

The repository documents both the methodology and its assumptions. Successful execution of these commands should be checked in the environment in which the repository is published; **the commands have not been independently run as part of preparing this README**.

## Limitations 

1. **Mechanism remains unresolved.** The simulation did not explicitly incorporate the proposed solutal-buoyancy mechanism, and the liquid flow-field evidence needed to test it is unavailable here.
2. **Computed thermal gradients may contain numerical artifacts.** The differencing of noisy interface-arrival times can contribute to apparent 15–20 mm oscillatory structure; the period is not sufficient evidence of a physical oscillation.
3. **Spatial registration is sensitive to specimen geometry.** Nominal EDS coordinates do not fully account for cutting kerf or shrinkage. The fixed alignment produces suggestive correlations, but modest changes in positional assumptions can weaken them.
4. **Compositional measurements and inferred coefficients have uncertainty.** Individual 100 μm EDS frames sample heterogeneous dendritic/interdendritic material; absolute concentrations may be biased; most inferred `k_E` trends other than W are not statistically robust.
5. **Statistical association is not causation.** Bartlett-adjusted *p*-values address autocorrelation approximately, not all multiplicity, selective-analysis, or model-assumption concerns.
6. **Scope of reproducibility is limited by the archived inputs.** The programs reproduce analyses from retained data and printed solver outputs; they do not recreate new 2026 experimental measurements or the unavailable original flow-field simulation.

See supplemented-thesis **§4.4.6** for detailed caveats, alternative explanations, and proposed follow-up measurements or simulations.

## Research references and attribution

- **Master's thesis (2023):** *A Study on Segregation Behavior During Directional Solidification of CM247LC Superalloy*, Jiangsu University. The thesis was completed in 2023; the additional analyses in this repository are explicitly dated **2026**.
- **Related first-author article (2024; thesis ref. [88]):** Y. Yang, W. T. Xu, and G. P. Zhao, “Microstructure Evolution of Directionally Solidified Ni-based CM247LC Superalloy,” *Special Casting & Nonferrous Alloys*, **44**(8), 1119–1126 (2024). DOI: [10.15980/j.tzzz.2024.08.019](https://doi.org/10.15980/j.tzzz.2024.08.019).
- **BPS reference (added in 2026; thesis ref. [95]):** Burton, Prim, and Slichter (1953). Consult the supplemented manuscript for the full bibliographic entry.
- The additional cited literature in the 2026 supplement is numbered **[88]–[95]** in the supplemented manuscript; **[1]–[87]** belongs to the original 2023 thesis.

If citing the **2026 numerical reanalysis**, please distinguish it from the **2023 degree thesis** and the **2024 peer-reviewed journal publication**. The 2026 supplement is a post-graduation analysis, not a separately published journal article.
