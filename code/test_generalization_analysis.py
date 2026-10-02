# Unit fixtures only. These values are never used in experiment figures.

import unittest
import pandas as pd
import numpy as np
from plot_generalization import paired_cells, summarize, holm, interval, axes, gain_lines, plt
from generalization_study import oracle_xor_error, compose_xor

class PairedAnalysisTests(unittest.TestCase):

    def fixture(self):
        rows, cells = ([], [])
        for seed in range(12):
            for arm, tau, error in [('hard', 0.0, 1.0), ('fixed_tau_0p03', 0.03, 0.8), ('matched_noise__fixed_tau_0p03', 0.03, 1.1), ('public_selected', 0.0, 1.0)]:
                rid = f'{seed}_{arm}'
                rows.append(dict(id=rid, case_id='test', seed=seed, rho=0.003, tau=tau, calibration='global' if tau == 0 else 'analytic', dataset='adult', policy_name='age', n=100, nout=20, reach=0.02, analytic_q=0.2, actual_noise_ratio=0.2, normalized_sigma=0.1, certified=True, l1_2way=error))
                cells.append(dict(result_id=rid, case_id='test', seed=seed, rho=0.003, tau=tau, calibration='global' if tau == 0 else 'analytic', setting=arm, actual_target_rho=0.003))
        return ({'cells': cells, 'cases': [{'dataset': 'adult'}]}, pd.DataFrame(rows))

    def test_paired_merge_and_decomposition(self):
        m, f = self.fixture()
        pairs, expanded = paired_cells(m, f)
        self.assertEqual(len(pairs), 24)
        h = pairs[pairs.setting == 'fixed_tau_0p03']
        np.testing.assert_allclose(h.C, 0.1)
        np.testing.assert_allclose(h.N, 0.3)
        np.testing.assert_allclose(h.G, 0.2)
        h = pairs[pairs.setting == 'public_selected']
        np.testing.assert_array_equal(h.relative_gain, np.zeros(12))
        self.assertTrue(h.selected_hard.all())

    def test_zero_variance_is_not_invented_pvalue(self):
        m, f = self.fixture()
        p, _ = paired_cells(m, f)
        s = summarize(p)
        r = s[s.setting == 'fixed_tau_0p03'].iloc[0]
        self.assertTrue(np.isnan(r.p_raw))
        self.assertEqual(s[s.setting == 'public_selected'].iloc[0].p_raw, 1.0)

    def test_holm_excludes_undefined_tests(self):
        np.testing.assert_allclose(holm([0.01, 0.04, np.nan]), [0.02, 0.04, np.nan], equal_nan=True)

    def test_exact_parity_support_floor_unit_fixtures(self):
        for k, m, expected in [(3, 512, 0.0), (6, 512, 0.0), (10, 512, 1.0)]:
            patterns = np.arange(m) % 2 ** k
            raw = 4 * (patterns[:, None] >> np.arange(k) & 1)
            syn = compose_xor(raw, k)
            self.assertAlmostEqual(oracle_xor_error(syn), expected)
            syn[:, -1] = 1 - syn[:, -1]
            self.assertAlmostEqual(oracle_xor_error(syn), 2.0)

    def test_constant_values_do_not_acquire_roundoff_variance(self):
        mean, lo, hi = interval(np.full(12, 0.13))
        self.assertEqual(mean, lo)
        self.assertEqual(mean, hi)

    def test_nine_policy_curves_have_distinct_styles(self):
        data = pd.DataFrame([dict(group=f'p{i}', rho=rho, relative_gain=0.01 * (i + seed)) for i in range(9) for rho in [0.003, 0.18] for seed in range(2)])
        figure, axis = axes()
        try:
            gain_lines(data, 'group', str, axis)
            styles = [(line.get_color(), line.get_marker(), line.get_linestyle()) for line in axis.get_lines() if line.get_label().startswith('p')]
            self.assertEqual(len(styles), 9)
            self.assertEqual(len(set(styles)), 9)
        finally:
            plt.close(figure)
if __name__ == '__main__':
    unittest.main()
