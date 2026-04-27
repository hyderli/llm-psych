"""Unit tests for `llm_psych.probes`.

All tests use synthetic random data — no model loading required.
Runs instantly on Mac M5.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_psych.probes import (
    ProbeMeta,
    ProbeResult,
    best_layer,
    derive_steering_vector,
    evaluate,
    fit,
    load,
    save,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_separable(n: int = 200, d: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where class 1 is clearly shifted from class 0."""
    rng = np.random.default_rng(1)
    X0 = rng.standard_normal((n // 2, d)).astype(np.float32)
    X1 = rng.standard_normal((n // 2, d)).astype(np.float32) + 3.0
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
    return X, y


def _make_random(n: int = 200, d: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with no signal — AUC should be near 0.5."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_returns_fitted_estimator(self):
        X, y = _make_separable()
        clf = fit(X, y)
        assert hasattr(clf, "coef_")

    def test_separable_data_predicts_correctly(self):
        X, y = _make_separable()
        clf = fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.95

    def test_respects_C_param(self):
        X, y = _make_separable()
        clf = fit(X, y, C=0.01)
        assert clf.C == 0.01

    def test_default_C_is_1(self):
        X, y = _make_separable()
        clf = fit(X, y)
        assert clf.C == 1.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_probe_result(self):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=50, rng=np.random.default_rng(0))
        assert isinstance(result, ProbeResult)

    def test_auc_high_on_separable(self):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=50, rng=np.random.default_rng(0))
        assert result.auc > 0.95

    def test_auc_near_chance_on_random(self):
        # Train on one half, evaluate on held-out half so in-sample overfit
        # doesn't inflate AUC on pure noise.
        X, y = _make_random(n=400)
        clf = fit(X[:200], y[:200])
        result = evaluate(clf, X[200:], y[200:], n_bootstrap=50, rng=np.random.default_rng(0))
        assert result.auc < 0.70

    def test_ci_ordered(self):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=100, rng=np.random.default_rng(0))
        assert result.ci_low <= result.auc <= result.ci_high

    def test_ci_width_shrinks_with_more_bootstrap(self):
        X, y = _make_separable(n=400)
        clf = fit(X, y)
        r_few = evaluate(clf, X, y, n_bootstrap=50, rng=np.random.default_rng(0))
        r_many = evaluate(clf, X, y, n_bootstrap=500, rng=np.random.default_rng(0))
        width_few = r_few.ci_high - r_few.ci_low
        width_many = r_many.ci_high - r_many.ci_low
        # More bootstrap resamples → tighter CI on the same data
        assert width_many < width_few + 0.05  # generous tolerance

    def test_brier_in_unit_interval(self):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=20, rng=np.random.default_rng(0))
        assert 0.0 <= result.brier <= 1.0

    def test_n_matches_input(self):
        X, y = _make_separable(n=160)
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=20, rng=np.random.default_rng(0))
        assert result.n == 160

    def test_reproducible_with_same_rng(self):
        X, y = _make_separable()
        clf = fit(X, y)
        r1 = evaluate(clf, X, y, n_bootstrap=100, rng=np.random.default_rng(7))
        r2 = evaluate(clf, X, y, n_bootstrap=100, rng=np.random.default_rng(7))
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=20, rng=np.random.default_rng(0))

        meta = ProbeMeta(
            emotion="joy",
            layer=16,
            model_id="test-model",
            model_sha="abc123",
            n_train=len(X),
            n_test=len(X),
            auc=result.auc,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            brier=result.brier,
        )

        jl_path, yaml_path = save(clf, meta, tmp_path)
        assert jl_path.exists()
        assert yaml_path.exists()

        clf2, meta2 = load(tmp_path, emotion="joy", layer=16)

        # Probe predictions should be identical after reload
        np.testing.assert_array_equal(clf.predict(X), clf2.predict(X))

        assert meta2.emotion == "joy"
        assert meta2.layer == 16
        assert meta2.model_sha == "abc123"
        assert meta2.auc == pytest.approx(result.auc)

    def test_filename_convention(self, tmp_path):
        X, y = _make_separable()
        clf = fit(X, y)
        result = evaluate(clf, X, y, n_bootstrap=10, rng=np.random.default_rng(0))
        meta = ProbeMeta(
            emotion="anger",
            layer=24,
            model_id="m",
            model_sha=None,
            n_train=100,
            n_test=50,
            auc=result.auc,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            brier=result.brier,
        )
        jl_path, yaml_path = save(clf, meta, tmp_path)
        assert jl_path.name == "anger_layer24.joblib"
        assert yaml_path.name == "anger_layer24.yaml"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path, emotion="sadness", layer=20)


# ---------------------------------------------------------------------------
# derive_steering_vector
# ---------------------------------------------------------------------------

class TestDeriveSteeringVector:
    def test_shape(self):
        rng = np.random.default_rng(0)
        em = rng.standard_normal((50, 64)).astype(np.float32)
        ne = rng.standard_normal((50, 64)).astype(np.float32)
        v = derive_steering_vector(em, ne)
        assert v.shape == (64,)

    def test_dtype_float32(self):
        rng = np.random.default_rng(0)
        em = rng.standard_normal((50, 64)).astype(np.float16)
        ne = rng.standard_normal((50, 64)).astype(np.float16)
        v = derive_steering_vector(em, ne)
        assert v.dtype == np.float32

    def test_zero_on_identical_distributions(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 32)).astype(np.float32)
        v = derive_steering_vector(X, X)
        np.testing.assert_allclose(v, 0.0, atol=1e-6)

    def test_linearity(self):
        rng = np.random.default_rng(0)
        em = rng.standard_normal((40, 16)).astype(np.float32)
        ne = rng.standard_normal((40, 16)).astype(np.float32)
        v = derive_steering_vector(em, ne)
        expected = em.mean(axis=0) - ne.mean(axis=0)
        np.testing.assert_allclose(v, expected, atol=1e-6)

    def test_direction_sign(self):
        # Emotion mean is purely positive; neutral mean is zero.
        em = np.ones((10, 4), dtype=np.float32)
        ne = np.zeros((10, 4), dtype=np.float32)
        v = derive_steering_vector(em, ne)
        assert (v > 0).all()


# ---------------------------------------------------------------------------
# best_layer
# ---------------------------------------------------------------------------

class TestBestLayer:
    def _make_result(self, auc: float) -> ProbeResult:
        return ProbeResult(auc=auc, ci_low=auc - 0.05, ci_high=auc + 0.05, brier=0.1, n=200)

    def test_returns_highest_auc_layer(self):
        results = {
            16: self._make_result(0.70),
            20: self._make_result(0.85),
            24: self._make_result(0.78),
        }
        assert best_layer(results) == 20

    def test_single_layer(self):
        results = {18: self._make_result(0.75)}
        assert best_layer(results) == 18

    def test_tie_returns_one_of_them(self):
        results = {
            10: self._make_result(0.80),
            20: self._make_result(0.80),
        }
        assert best_layer(results) in (10, 20)
