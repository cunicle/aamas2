import math

import pytest

from ptcdiag.theory import fixed_k_sizes, p_valid_one_step, p_valid_schedule, simulate


def test_closed_forms():
    assert p_valid_one_step(2) == 0.5
    assert p_valid_one_step(3) == pytest.approx(6 / 27)
    assert p_valid_schedule(4, [4]) == pytest.approx(math.factorial(4) / 4 ** 4)
    assert p_valid_schedule(4, [1, 1, 1, 1]) == 1.0
    assert p_valid_schedule(4, [2, 2]) == pytest.approx((4 * 3 / 16) * (2 * 1 / 4))
    assert fixed_k_sizes(5, 2) == [2, 2, 1]


@pytest.mark.parametrize("n,k", [(2, 2), (3, 3), (4, 2), (4, 4)])
def test_sampler_matches_formula(n, k):
    got = simulate(n, k, trials=3000, temperature=1.0, seed=1)
    want = p_valid_schedule(n, fixed_k_sizes(n, k))
    assert abs(got - want) < 0.03, (got, want)


def test_greedy_collides_whenever_steps_are_parallel():
    assert simulate(3, 3, trials=5, temperature=0.0, order="confidence") == 0.0
    assert simulate(3, 1, trials=5, temperature=0.0, order="confidence") == 1.0
