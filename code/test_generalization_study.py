# Implement test generalization study.

import math
import unittest
import numpy as np
import core
from data import schema_for, xor_schema
from generalization_policy import named_policies, policy_reach, dispatch_reach, exact_policy_sensitivity, hard_crossing_witness, graph_for_attribute
from generalization_study import cases, compose_xor, oracle_xor_error

class ExplicitPolicyTests(unittest.TestCase):

    def test_real_reaches(self):
        a = schema_for('adult')
        p = named_policies(a)
        self.assertAlmostEqual(policy_reach(a, p['age'])[0], 7 / 73 / 5)
        self.assertEqual(policy_reach(a, p['age'])[0], policy_reach(a, p['age_secondary'])[0])
        self.assertAlmostEqual(policy_reach(a, p['age_education'])[0], 2 / 15 / 5)
        self.assertEqual(policy_reach(a, p['all_columns'])[0], 0.2)
        b = schema_for('bank')
        p = named_policies(b)
        self.assertEqual(policy_reach(b, p['age'])[0], policy_reach(b, p['age_secondary'])[0])
        self.assertAlmostEqual(policy_reach(b, p['all_predictors'])[0], 0.1)
        self.assertEqual(policy_reach(b, p['all_columns'])[0], 0.2)

    def test_xor_union_maximum(self):
        for k in [3, 6, 10]:
            s = xor_schema(k)
            p = named_policies(s)
            self.assertEqual(policy_reach(s, p['x1'])[0], policy_reach(s, p['all_inputs'])[0])
            self.assertAlmostEqual(policy_reach(s, p['x1'])[0], 1 / 7 / (k + 1))
            self.assertAlmostEqual(policy_reach(s, p['all_columns'])[0], 1 / (k + 1))

    def test_exact_against_bound(self):
        s = xor_schema(2)
        c = s.random(18, np.random.default_rng(9))
        for p in named_policies(s).values():
            r, _ = policy_reach(s, p)
            for tau in [0.003, 0.03, 0.3, 3.0]:
                measured = exact_policy_sensitivity(c, s, tau, p)
                self.assertLessEqual(measured, math.sqrt(2) * math.tanh(r / (2 * tau)) + 1e-12)

    def test_legacy_age_output_unchanged(self):
        original = core.reach
        for d in ['adult', 'bank']:
            s = schema_for(d)
            train = s.random(100, np.random.default_rng(3))
            cfg = dict(seed=5, nout=24, rounds=3, tau=0.03, rho=0.003, policy='age', theta=7, adjacency='B', selection='hybrid')
            old = core.evolve(train, s, cfg)[0]
            try:
                core.reach = dispatch_reach
                new = core.evolve(train, s, {**cfg, 'policy': named_policies(s)['age']})[0]
            finally:
                core.reach = original
            np.testing.assert_array_equal(old, new)

    def test_equal_reach_equal_output(self):
        original = core.reach
        try:
            core.reach = dispatch_reach
            for d in ['adult', 'bank', 'xor3']:
                s = schema_for(d)
                p = named_policies(s)
                left, right = ('x1', 'all_inputs') if d.startswith('xor') else ('age', 'age_secondary')
                train = s.random(100, np.random.default_rng(3))
                cfg = dict(seed=5, nout=24, rounds=3, tau=0.03, rho=0.003, adjacency='B', selection='hybrid')
                a = core.evolve(train, s, {**cfg, 'policy': p[left]})[0]
                b = core.evolve(train, s, {**cfg, 'policy': p[right]})[0]
                np.testing.assert_array_equal(a, b)
        finally:
            core.reach = original

    def test_certificates(self):
        for d in ['adult', 'bank', 'xor3', 'xor10']:
            s = schema_for(d)
            c = s.random(96, np.random.default_rng(19))
            for name, p in named_policies(s).items():
                w = hard_crossing_witness(c, s, p)
                if w is not None:
                    self.assertNotEqual(w['winner_left'], w['winner_right'])
                    self.assertIn(w['attribute'], [v['attribute'] for v in p['attributes']])
                spec = p['attributes'][0]
                j = s.names.index(spec['attribute'])
                fixture = np.zeros((2, s.p))
                if j in s.numeric:
                    lo, hi, step = s.numeric[j]
                    fixture[1, j] = step / (hi - lo)
                else:
                    _, vals, allowed, _ = graph_for_attribute(s, spec)
                    aa, bb = np.argwhere(allowed)[0]
                    fixture[:, j] = [vals[aa], vals[bb]]
                self.assertIsNotNone(hard_crossing_witness(fixture, s, p), (d, name))
                self.assertIsNone(hard_crossing_witness(np.repeat(c[:1], 3, axis=0), s, p))

    def test_nested_inputs_and_oracle(self):
        raw = np.random.default_rng(3).integers(0, 8, size=(100, 10))
        a, b = (compose_xor(raw[:20], 3), compose_xor(raw, 10))
        np.testing.assert_array_equal(a[:, :3], b[:20, :3])
        patterns = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        syn = np.column_stack([patterns, patterns[:, 0] ^ patterns[:, 1]])
        self.assertEqual(oracle_xor_error(syn), 0.0)
        self.assertEqual(oracle_xor_error(np.column_stack([patterns, 1 - syn[:, -1]])), 2.0)

    def test_cases_cover_presets(self):
        all_cases = cases()
        size = [c for c in all_cases if 'size' in c['families']]
        self.assertEqual({c['n'] for c in size}, {3000, 12000, 48000, 100000})
        dim = [c for c in all_cases if 'dimension' in c['families']]
        self.assertEqual({c['dataset'] for c in dim}, {'xor3', 'xor6', 'xor10'})
        self.assertEqual({c['nout'] for c in dim}, {512, 2048})
if __name__ == '__main__':
    unittest.main()
