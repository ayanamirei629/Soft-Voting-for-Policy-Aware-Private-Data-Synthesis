# Implement test core.

import unittest
import numpy as np
from scipy.spatial.distance import cdist
from core import Schema, bounds, calibrate, votes, reach, apd, evolve
from data import xor_schema, schema_for

class Guarantees(unittest.TestCase):

    def test_metric_triangle(self):
        for name in ['adult', 'bank', 'xor3']:
            s = schema_for(name)
            x = s.random(20, np.random.default_rng(5))
            d = s.distance(x, x)
            for j in range(len(x)):
                self.assertTrue(np.all(d <= d[:, j, None] + d[j, None, :] + 1e-12))

    def test_tight_two_candidate_bound(self):
        s = Schema('line', ('x',), {0: (0, 1, 0.25)}, {}, {0: [0.5]})
        for tau in [0.01, 0.1, 1, 100]:
            x = np.array([[0.25], [0.75]])
            c = np.array([[0.0], [1.0]])
            actual = np.linalg.norm(votes(x, c, s, tau)[0] - votes(x, c, s, tau)[1])
            self.assertAlmostEqual(actual, bounds(0.5, tau, 2)[0], places=12)

    def test_bounds_random_pairs(self):
        rng = np.random.default_rng(25)
        s = xor_schema(2)
        x = s.domain()
        c = s.random(24, rng)
        for tau in [0.005, 0.05, 0.5, 5]:
            v = votes(x, c, s, tau)
            dv = cdist(v, v)
            dx = s.distance(x, x)
            self.assertTrue(np.all(dv <= np.sqrt(2) * np.tanh(dx / (2 * tau)) + 1e-12))
            self.assertLessEqual(np.linalg.norm(v, axis=1).max(), bounds(0.1, tau, len(c))[1] + 1e-12)
            self.assertTrue(np.allclose(v.sum(1), 1))

    def test_participation_relations(self):
        for tau in [0, 0.05, 0.5, 50]:
            b = calibrate(0.02, tau, 64, 'B')
            u = calibrate(0.02, tau, 64, 'U')
            both = calibrate(0.02, tau, 64, 'BU')
            self.assertEqual(both, max(b, u))
        self.assertEqual(calibrate(0.02, 0, 64, 'U'), 1.0)

    def test_background_bottleneck_and_determinism(self):
        s = schema_for('adult')
        self.assertEqual(reach(s, 'full', 2)[0], reach(s, 'full', 30)[0])
        x = s.random(80, np.random.default_rng(0))
        cfg = dict(seed=0, nout=20, rounds=3, rho=0.01, tau=0.1, policy='full')
        a = evolve(x, s, {**cfg, 'theta': 2})[0]
        b = evolve(x, s, {**cfg, 'theta': 30})[0]
        self.assertTrue(np.array_equal(a, b))

    def test_apd_identical_marginal_counts(self):
        s = Schema('binary', ('a', 'b'), {}, {0: 1 - np.eye(2), 1: 1 - np.eye(2)}, {})
        a = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertAlmostEqual(apd(a, s), apd(b, s))

    def test_hard_no_crossing(self):
        s = xor_schema(1)
        x = s.domain()
        c = np.array([[0.0, 0.0]])
        self.assertEqual(float(cdist(votes(x, c, s, 0), votes(x, c, s, 0)).max()), 0.0)

    def test_unbounded_is_independent_of_substitution_reach(self):
        s = schema_for('adult')
        x = s.random(90, np.random.default_rng(12))
        cfg = dict(seed=8, nout=25, rounds=4, rho=0.01, tau=0.05, policy='age', adjacency='U')
        self.assertTrue(np.array_equal(evolve(x, s, {**cfg, 'theta': 2})[0], evolve(x, s, {**cfg, 'theta': 30})[0]))

    def test_total_zcdp_budget(self):
        s = schema_for('adult')
        x = s.random(90, np.random.default_rng(12))
        for adj in ['B', 'U', 'BU']:
            _, h, _, _ = evolve(x, s, dict(seed=8, nout=25, rounds=5, rho=0.04, tau=0.1, adjacency=adj))
            self.assertAlmostEqual(sum((v['sensitivity'] ** 2 / (2 * v['sigma'] ** 2) for v in h)), 0.04)

    def test_matched_sigma_replays_same_output(self):
        s = schema_for('adult')
        x = s.random(90, np.random.default_rng(12))
        tau = 0.08
        a = dict(seed=8, nout=25, rounds=5, rho=0.04, tau=tau, policy='age', theta=2)
        da = bounds(reach(s, 'age', 2)[0], tau, 25)[0]
        db = bounds(reach(s, 'age', 20)[0], tau, 25)[0]
        b = {**a, 'theta': 20, 'rho': 0.04 * (db / da) ** 2}
        self.assertTrue(np.array_equal(evolve(x, s, a)[0], evolve(x, s, b)[0]))

    def test_hard_component_certificate(self):
        from hard_certificate import certificate
        s = schema_for('adult')
        c = np.array([[0, 0, 0, 1, 0], [1, 1, 1, 2, 1]], dtype=float)
        self.assertTrue(certificate(c, s))
        self.assertFalse(certificate(np.repeat(c[:1], 5, axis=0), s))
        c[1, 3] = 0
        self.assertFalse(certificate(c, s))
if __name__ == '__main__':
    unittest.main()
