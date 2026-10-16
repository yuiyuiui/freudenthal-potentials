"""Batch evaluation checked against the separate scalar Bernstein path."""
import numpy as np
from freudenthal import construct_basis


def make(case):
    count = {'small': 16, 'medium': 256, 'large': 1024}[case]
    basis = construct_basis(1, 6)
    points = np.random.default_rng(633).uniform(size=(count, 3))
    scalar = [basis.evaluate(1, p) for p in points]
    expected_value = np.array([p[0] for p in scalar])
    expected_curl = np.array([p[1] for p in scalar])

    def call():
        return basis.evaluate_many(1, points)

    def validate(result):
        if not (np.allclose(result[0], expected_value, atol=2e-12, rtol=2e-12)
                and np.allclose(result[1], expected_curl, atol=2e-11, rtol=2e-12)):
            raise ArithmeticError('Independent scalar-path comparison failed')
        return {'passed': True, 'reference': 'scalar physical Bernstein evaluation',
                'potential_atol': 2e-12, 'curl_atol': 2e-11}

    return call, validate, {'N': 1, 'degree': 6, 'basis_id': 1, 'points': count,
                          'seed': 633, 'arithmetic': 'binary64',
                          'cost_scope': 'evaluate_many only; basis construction is setup'}
