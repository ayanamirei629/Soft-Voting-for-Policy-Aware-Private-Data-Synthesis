# Implement test extended study.

import math
import unittest
import numpy as np
import core
from data import schema_for
from extended_study import Q_GRID, experiment_configs, tau_for_q
from metrics import _interval_answers, _rectangle_answers, range_query_errors

class ExtendedStudyTests(unittest.TestCase):

    def test_q_parameterization(self):
        for dataset in ['adult', 'bank', 'xor3']:
            theta = 1 if dataset == 'xor3' else 7
            schema = schema_for(dataset)
            r, _ = core.reach(schema, 'age', theta)
            for q in Q_GRID:
                tau = tau_for_q(dataset, 'age', theta, None, q)
                self.assertAlmostEqual(math.tanh(r / (2 * tau)), q, places=12)

    def test_interval_and_rectangle_queries(self):
        hist = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(_interval_answers(hist), [0.2, 0.5, 1.0, 0.3, 0.8, 0.5])
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        answers = _rectangle_answers(matrix)
        self.assertEqual(len(answers), 9)
        self.assertTrue(np.any(np.isclose(answers, 1.0)))

    def test_range_errors_vanish_on_identical_data(self):
        rng = np.random.default_rng(82)
        for dataset in ['adult', 'bank', 'xor3']:
            schema = schema_for(dataset)
            x = schema.random(80, rng)
            errors = range_query_errors(x, x, schema)
            for value in errors.values():
                self.assertAlmostEqual(value, 0.0)

    def test_configs_are_unique_and_cover_roles(self):
        configs = experiment_configs()
        self.assertEqual(len(configs), len({str(sorted(c.items())) for c in configs}))
        roles = {role for cfg in configs for role in cfg['roles']}
        self.assertTrue({'sweet-soft', 'matched-soft', 'graph-soft', 'adjacency-soft'} <= roles)
if __name__ == '__main__':
    unittest.main()
