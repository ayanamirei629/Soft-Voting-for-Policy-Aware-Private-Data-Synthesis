# Implement test crossover.

import unittest
import numpy as np
from crossover_study import mse_crossover, cfg_for

class CrossoverTests(unittest.TestCase):

    def test_mse_identity(self):
        a = np.array([4.0, 2.0, 0.0])
        b = np.array([3.0, 2.0, 1.0])
        q = 0.3
        T = 8
        root = mse_crossover(a, b, T, q)
        for rho in [root / 2, root, root * 2]:
            hard = len(a) * T / rho
            soft = np.sum((a - b) ** 2) + len(a) * q * q * T / rho
            expected = np.sum((a - b) ** 2) * (1 - root / rho)
            self.assertAlmostEqual(soft - hard, expected)

    def test_root_scaling(self):
        a = np.array([3.0, 4.0])
        b = np.array([4.0, 3.0])
        self.assertAlmostEqual(mse_crossover(2 * a, 2 * b, 8, 0.5), mse_crossover(a, b, 8, 0.5) / 4)

    def test_noisefree_is_separate(self):
        c = cfg_for('adult', 800, 0, 'soft', stage='noisefree')
        self.assertEqual(c['phase'], 'crossover_noisefree')
        self.assertEqual(c['rho'], 0)

    def test_fresh_generators(self):
        self.assertEqual(cfg_for('xor3', 800, 0.1, 'soft')['data_seed'], 40)

    def test_sampling_keeps_candidate_counts(self):
        import core
        from data import schema_for
        from sampling_crossover_control import matched_schedule_schema
        schema = schema_for('xor3')
        x = schema.random(30, np.random.default_rng(42))
        cfg = cfg_for('xor3', 800, 0.1, 'soft', selection='sample', stage='sampling')
        cfg['nout'] = 12
        _, history, _, _ = core.evolve(x, matched_schedule_schema(schema), cfg)
        self.assertEqual([h['m_candidates'] for h in history], [12, 12, 12, 24, 24, 24, 24, 24])
if __name__ == '__main__':
    unittest.main()
