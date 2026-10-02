# Implement test interaction transfer.

import math
import unittest
import numpy as np
from interaction_transfer import folded_normal, workload, marginal_loss, ISO, conditions
from data import schema_for
from metrics import marginal_errors
from core import reach

class TransferTests(unittest.TestCase):

    def test_folded_normal_limits(self):
        np.testing.assert_allclose(folded_normal([0, 2], [1, 0]), [math.sqrt(2 / math.pi), 2])

    def test_workload_matches_fidelity(self):
        rng = np.random.default_rng(66)
        for dataset in ['adult', 'bank', 'xor3']:
            s = schema_for(dataset)
            c = s.random(31, rng)
            reference = s.random(55, rng)
            value = marginal_loss(workload(s, c, reference), np.ones(len(c)) / len(c), 0.0)
            expected = marginal_errors(reference, c, s)['hoerr_bits' if dataset == 'xor3' else 'l1_2way']
            self.assertAlmostEqual(value, expected)

    def test_equal_ratio(self):
        for dataset, pairs in ISO.items():
            s = schema_for(dataset)
            ratios = [reach(s, 'age', theta)[0] / tau for tau, theta in pairs]
            np.testing.assert_allclose(ratios, ratios[0])

    def test_conditions_unique(self):
        for d in ISO:
            c = conditions(d)
            self.assertEqual(len(c), len(set(c)))

    def test_normalized_noise_transfer(self):
        n_target, n_pilot, rho = (26048, 3000, 0.0006)
        rho_pilot = rho * (n_target / n_pilot) ** 2
        self.assertAlmostEqual(math.sqrt(8 / rho_pilot) / n_pilot, math.sqrt(8 / rho) / n_target)

    def test_censor_status_is_not_just_root_absence(self):
        from analyze_transfer import root_status
        self.assertNotEqual(root_status([0.1, 1], [1, 2]), root_status([0.1, 1], [-1, -2]))

    def test_frozen_choice_does_not_consult_target_error(self):
        import pandas as pd
        from analyze_transfer import choice
        frame = pd.DataFrame({'pilot_scaled_gain': [0.1, -0.2], 'mean': [1.0, -1.0]})
        before = choice(frame, 'Scaled pilot')
        frame['mean'] = -frame['mean']
        np.testing.assert_array_equal(before, choice(frame, 'Scaled pilot'))
if __name__ == '__main__':
    unittest.main()
