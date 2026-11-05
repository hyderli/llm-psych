from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "j24_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_j24.py"
)
j24 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j24)


def test_bootstrap_constant_rates() -> None:
    assert j24.bootstrap_difference_ci([0.75] * 6, [0.25] * 6) == (0.5, 0.5)


def test_bootstrap_matches_independent_direction_resampling() -> None:
    a = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
    b = [0.0, 0.0, 0.1, 0.2, 0.4, 0.5]
    rng = np.random.default_rng(20261104)
    ia = rng.integers(0, 6, size=(10_000, 6))
    ib = rng.integers(0, 6, size=(10_000, 6))
    differences = np.asarray(a)[ia].mean(axis=1) - np.asarray(b)[ib].mean(axis=1)
    expected = np.quantile(differences, [0.025, 0.975], method="linear")
    original_a, original_b = a.copy(), b.copy()
    actual = j24.bootstrap_difference_ci(a, b)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert actual == j24.bootstrap_difference_ci(a, b)
    assert a == original_a and b == original_b


def test_bootstrap_does_not_pair_identical_families() -> None:
    rates = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    low, high = j24.bootstrap_difference_ci(rates, rates)
    assert low < 0 < high


@pytest.mark.parametrize("rates", [[], [float("nan")], [float("inf")]])
def test_bootstrap_rejects_unusable_direction_rates(rates: list[float]) -> None:
    with pytest.raises(ValueError):
        j24.bootstrap_difference_ci(rates, [0.0] * 6)
