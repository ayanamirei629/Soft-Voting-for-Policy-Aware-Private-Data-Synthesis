# Implement test benefit study.

import math
import unittest
import numpy as np
from benefit_study import config, certify_hard, RHOS, TAUS
from core import bounds, reach
from data import schema_for

class BenefitStudyTests(unittest.TestCase):

    def test_range_and_scale(self):
        self.assertEqual(len(RHOS) * len(TAUS) * 6, 990)
        self.assertEqual(RHOS[-1] / RHOS[0], 100000000.0)
        for phase in ['screen', 'refine', 'confirm']:
            c = config('adult', 'full', 0.001, 0.03, phase)
            self.assertEqual((c['n'], c['nout'], c['rounds']), (0, 512, 8))

    def test_noise_saving_uses_actual_policy(self):
        schema = schema_for('adult')
        full, _ = reach(schema, 'full', 7)
        local, _ = reach(schema, 'age', 7)
        self.assertAlmostEqual(full, 0.2)
        self.assertAlmostEqual(local, 7 / 73 / 5)
        self.assertGreater(bounds(full, 0.03, 512)[0] / math.sqrt(2), 0.99)
        self.assertLess(bounds(local, 0.03, 512)[0] / math.sqrt(2), 0.32)

    def test_local_crossing_certificate(self):
        s = schema_for('xor3')
        c = config('xor3', 'age', 0.001, 0)
        self.assertTrue(certify_hard(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]), s, c))
        self.assertFalse(certify_hard(np.zeros((4, 4)), s, c))
        self.assertFalse(certify_hard(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]), s, dict(c, theta=0)))

    def test_confirmation_generator_is_separate(self):
        self.assertEqual(config('xor3', 'full', 0.001, 0.03)['data_seed'], 20)
        self.assertEqual(config('xor3', 'full', 0.001, 0.03, 'confirm', 700)['data_seed'], 30)
if __name__ == '__main__':
    unittest.main()
