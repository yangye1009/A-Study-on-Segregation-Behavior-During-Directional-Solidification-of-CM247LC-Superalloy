# -*- coding: utf-8 -*-

import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "programs"))
import ch4lib as L


class TestData(unittest.TestCase):
    def test_thermal_shapes_and_G_correction(self):
        th = L.load_thermal()
        self.assertEqual(len(th["axis2"]), 37)
        self.assertAlmostEqual(th["axis2"].G_K_mm.iloc[0], 85.17282706 / 2, places=6)
        self.assertTrue(np.isnan(th["axis2"].G_K_mm.iloc[-1]))
        self.assertTrue(th["axis1"].dT_K.isna().all())

    def test_table_4_11(self):
        st = L.load_thermal()["axis2"].query("x_mm >= 70")
        self.assertAlmostEqual(st.G_K_mm.mean(), 6.23, places=2)
        self.assertAlmostEqual(st.R_mm_min.mean(), 2.46, places=2)

    def test_eds_closure(self):
        eds = L.load_eds("da2")
        self.assertEqual(len(eds), 34)
        self.assertTrue((abs(eds[L.ELEMENTS].sum(axis=1) - 100) < 0.03).all())

    def test_pairing_misregistration(self):
        P = L.pair_with_G(L.load_eds("da2"), L.load_thermal()["axis2"], offset=-1)
        self.assertTrue((P.misreg_mm.dropna() == 1.0).all())

    def test_table_4_12_Ta(self):
        P = L.pair_with_G(L.load_eds("da2"), L.load_thermal()["axis2"], offset=-1)
        m = ((P.index >= 9) & P.G_K_mm.notna()).to_numpy()
        r = np.corrcoef(L.detrend_linear(P.Ta[m]), L.detrend_linear(P.G_K_mm[m]))[0, 1]
        self.assertEqual(m.sum(), 25); self.assertAlmostEqual(r, 0.66, places=2)

    def test_table_4_8_spearman(self):
        from scipy import stats
        eds = L.load_eds("da2"); E8 = list(L.K0_TABLE44)
        cv = eds[E8].std(ddof=1) / eds[E8].mean()
        self.assertAlmostEqual(stats.spearmanr(cv, [abs(L.K0_TABLE44[e] - 1) for e in E8])[0], 0.93, places=2)

    def test_scheil_fit_W(self):
        from scipy import stats
        eds = L.load_eds("da2"); X = np.log1p(-np.linspace(0, .9, 34))
        self.assertAlmostEqual(stats.linregress(X, np.log(eds.W))[0] + 1, 1.057, places=3)

    def test_da4_is_pair_mean_of_da2(self):
        A = L.load_eds("da2")[L.ELEMENTS].to_numpy(); B = L.load_eds("da4")[L.ELEMENTS].to_numpy()
        self.assertLess(np.abs(((A[:-2] + A[2:]) / 2)[:17] - B).max(), 0.006)

    def test_jmatpro_liquidus_k0(self):
        liq, gam = L.load_jmatpro("liquid"), L.load_jmatpro("gamma")
        g0 = gam.iloc[0]; l0 = liq[liq.T_C == g0.T_C].iloc[0]
        self.assertAlmostEqual(g0.W / l0.W, 1.122, places=3); self.assertLess(g0.Al / l0.Al, 1)

    def test_bartlett_neff_bounds(self):
        rng = np.random.default_rng(0); a = rng.normal(size=25); b = rng.normal(size=25)
        n = L.bartlett_neff(a, b); self.assertTrue(5 < n < 60)


if __name__ == "__main__":
    unittest.main()
