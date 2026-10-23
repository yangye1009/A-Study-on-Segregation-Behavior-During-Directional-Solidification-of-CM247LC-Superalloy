# Chapter 4 data-processing programs 

### Overview

The package holds the data-processing programs for Chapter 4 of the thesis (ProCAST thermal field, EDS segregation,partition coefficients and γ-phase stability of a Ni-based superalloy casting). 

### Directory layout

```
.
├── data/       raw inputs for 06–12 (EDS, JMatPro exports, ProCAST post-processing output)
├── programs/   01–12 and ch4lib.py
├── results/    CSV outputs of 06–12 (created automatically)
├── figures/    PNG figures of 06–12 (created automatically)
└── tests/      unit tests
```

### Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib` and `scipy`. Program 02 also needs `openpyxl` (.xlsx/.xlsm)
or `xlrd` (.xls) for reading Excel files.

### Part 1 —  programs 01–05

| Program | Purpose | Main input | Default output folder |
|---|---|---|---|
| `01_procast_thermal_postprocess.py` | Cooling rate V, interface velocity R, temperature gradient G and undercooling ΔT on two axes; redraws Fig. 4.24 (initial and steady stages) | ProCAST temperature and solid-fraction exports of axis 1 and axis 2 (`time, P1…Pn`) | `results/01_thermal/` |
| `02_eds_batch_composition.py` | Batch-reads EDS Excel exports, identifies area/spot IDs, pivots to one row per spot, filters two area groups by a W threshold | Folder of EDS `.xls/.xlsx/.xlsm` files | `results/02_EDS/` |
| `03_gamma_phase_stability.py` | Corrected composition → atomic fraction → empirical γ-matrix composition → Nv, Md, Bo | CSV of element wt% | `results/03_stability/` |
| `04_gradient_descent_kE_inversion.py` | Scheil log-linear fit by gradient descent for the effective partition coefficient k_E, checked against the OLS solution | CSV of element wt% (preset fs ranges for 10, 17 and 34 points) | `results/04_partition_fit/` |
| `05_phase_diagram_partition_slope.py` | k(T) = Cs/Cl from phase-diagram (Scheil / lever) exports; optional T–C slope fits | 13-column solid and liquid tables; optional slope table | `results/05_phase_diagram/` |

### Part 2 — Chapter 4 recomputation (06–12)

Input data in `data/`:

| File | Content |
|---|---|
| `thermal_raw_output.txt` | Printed output of the original ProCAST post-processing program: 8 lines (axis 2: V, R, G, ΔT; axis 1: V, R, G, ΔT), 37 values each |
| `da2.csv`, `da4.csv`, `da7.csv` | EDS compositions (wt%), 34, 17 and 10 points |
| `liquid.dat`, `gamma.dat`, `gammap.dat`, `Al1.dat` | JMatPro Scheil-path exports (phase compositions vs temperature; Al in each phase) |

Programs and outputs (`results/` and `figures/`):

| Program | Content | Tables | Figures |
|---|---|---|---|
| `06_thermal_field.py` | Thermal quantities on both axes; steady-stage statistics (Table 4.11); G autocorrelation and spectrum; R–G consistency; coherence of the two axes; arrival-time noise | `B1_thermal_axis1/2`, `T4_11_steady_stats`, `06_*` | F01–F04 |
| `07_eds_composition.py` | 34-point EDS statistics: CV vs segregation criteria (Table 4.8), Pearson matrix (Table 4.9), pool vs matrix frames (Table 4.10), closure dilution, partial correlation Spearman trends | `B2_eds_34points`, `T4_8`, `T4_9`, `T4_10`, `07_*` | F05–F09 |
| `08_registration.py` | Composition–G registration and correlation (Table 4.12, Fig. 4.30); window and position-mapping sensitivity; pool-frame G test; axis-1 control; R correlation; apparent k_E from closure | `B3_paired_series`, `T4_12`, `08_*` | F10–F13 |
| `09_scheil_fit.py` | Scheil fits with 95 % CI and p-values (Table 4.3); provenance of da4/da7; BPS inversion of Pc (Table 4.7); Rδ consistency | `T4_3`, `T4_7`, `09_*` | F14–F15 |
| `10_jmatpro.py` | k(T) = Cγ/CL along the JMatPro Scheil path; liquidus k0; Al path | `B4_jmatpro_k_vs_T`, `10_*` | F16–F17 |
| `11_phase_stability.py` | γ-matrix composition and Nv, Md (Figs 4.20–4.21) | `B5_gamma_phase_stability`, `11_*` | F18–F19 |
| `12_run_all.py` | Runs 06–11, then writes the audit table | `comparison_thesis_vs_recomputed.csv` | — |

`B1`–`B5` are the Appendix-B tables and `T4_x` reproduce thesis tables. 

### Conventions

- Derived thermal point i lies at x = 5i mm (i = 0…36, 0–180 mm from the chill face). R_i is taken between points i
  and i+1; G is taken over 10 mm and centred at x_i + 5 mm. 
- EDS point j lies at x = 1 + 5j mm (j = 0…33; nominal, kerf and shrinkage not corrected). S = Ta + Hf + Ti.
- Pairing: EDS point j ↔ derived point j − 1 for G (1 mm misregistration); j ↔ j for the R correlations of §4.4.3.3.
- Windows: Table 4.12 uses EDS x ≥ 46 mm (j ≥ 9, n = 25); Table 4.11 and the steady stage use x ≥ 70 mm.
- Axis 2 (centreline) is the primary axis; axis 1 (5 mm off-axis) is a reference only.
- Solid fraction in the Scheil fits is a nominal linear mapping: 34 points 0–0.9, 17 points 0.025–0.875,
  10 points 0.05–0.85.

### Running

```bash
python -m unittest discover -s tests -v
cd programs && python 12_run_all.py
```

Each of 06–11 can also be run on its own from `programs/` (they import `ch4lib`).
