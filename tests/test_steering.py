"""Tests for the story-method emotion-vector derivation primitives.

Covers the pure-NumPy core of ``src/llm_psych/steering.py``:

* :func:`derive_story_vectors` — cross-emotion-mean centering.
* :func:`fit_neutral_pcs` — orthonormal PC basis from a neutral set.
* :func:`project_out` — orthogonal projection (idempotence, equivalence
  to a direct Gram-Schmidt subtraction).
* :func:`derive_paper_steering_vectors` — end-to-end composition.

These tests do not load any model and run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_psych.steering import (
    derive_paper_steering_vectors,
    derive_story_vectors,
    fit_neutral_pcs,
    project_out,
)


# --------------------------------------------------------------------------
# derive_story_vectors
# --------------------------------------------------------------------------

class TestDeriveStoryVectors:
    def test_shapes_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        acts = {
            "joy": rng.standard_normal((7, 8)),
            "fear": rng.standard_normal((5, 8)),
            "anger": rng.standard_normal((6, 8)),
        }
        out = derive_story_vectors(acts)
        assert set(out) == set(acts)
        for v in out.values():
            assert v.shape == (8,)
            assert v.dtype == np.float32

    def test_grand_mean_is_subtracted(self) -> None:
        """Sum of returned vectors must be ~zero (cross-emotion centering)."""
        rng = np.random.default_rng(1)
        acts = {
            "a": rng.standard_normal((10, 16)),
            "b": rng.standard_normal((10, 16)),
            "c": rng.standard_normal((10, 16)),
        }
        out = derive_story_vectors(acts)
        total = np.sum(np.stack(list(out.values()), axis=0), axis=0)
        assert np.allclose(total, 0.0, atol=1e-5)

    def test_equal_emotion_weighting(self) -> None:
        """Per-emotion sample size must not bias the grand mean."""
        # Two emotions; emotion 'a' has 100 samples all-ones, 'b' has 1
        # sample all-twos. The grand mean over per-emotion means is 1.5,
        # so v_a = 1 - 1.5 = -0.5; v_b = 2 - 1.5 = 0.5, regardless of n.
        acts = {
            "a": np.ones((100, 4)),
            "b": 2.0 * np.ones((1, 4)),
        }
        out = derive_story_vectors(acts)
        assert np.allclose(out["a"], -0.5)
        assert np.allclose(out["b"], 0.5)

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            derive_story_vectors({})

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            derive_story_vectors({"a": np.zeros((4,))})

    def test_hidden_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Hidden-dim mismatch"):
            derive_story_vectors(
                {"a": np.zeros((3, 8)), "b": np.zeros((3, 9))}
            )


# --------------------------------------------------------------------------
# fit_neutral_pcs
# --------------------------------------------------------------------------

class TestFitNeutralPCs:
    def test_orthonormal_rows(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 12))
        B = fit_neutral_pcs(X, var_threshold=0.8)
        gram = B @ B.T
        assert np.allclose(gram, np.eye(B.shape[0]), atol=1e-5)

    def test_explained_variance_meets_threshold(self) -> None:
        rng = np.random.default_rng(3)
        # Construct data with a known low-rank structure: rank-3 signal
        # plus tiny noise.
        n, d = 200, 20
        latent = rng.standard_normal((n, 3))
        loadings = rng.standard_normal((3, d))
        X = latent @ loadings + 0.01 * rng.standard_normal((n, d))

        B = fit_neutral_pcs(X, var_threshold=0.9)
        # Project X onto B and measure retained variance.
        Xc = X - X.mean(axis=0, keepdims=True)
        total_var = np.sum(Xc ** 2)
        proj = Xc @ B.T  # (n, k)
        kept_var = np.sum(proj ** 2)
        assert kept_var / total_var >= 0.9 - 1e-6

    def test_var_threshold_one_returns_full_rank(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 10))
        B = fit_neutral_pcs(X, var_threshold=1.0)
        # With var_threshold=1.0 we need all PCs needed to capture 100%
        # variance. For n=30, d=10 full-rank Gaussian: rank=min(n-1, d)=10.
        assert B.shape == (10, 10)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_threshold_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            fit_neutral_pcs(np.zeros((4, 3)), var_threshold=bad)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_neutral_pcs(np.zeros((5,)))


# --------------------------------------------------------------------------
# project_out
# --------------------------------------------------------------------------

class TestProjectOut:
    def test_removes_basis_direction(self) -> None:
        # Basis = e_0; any vector should lose its 0th component.
        B = np.array([[1.0, 0.0, 0.0]])
        v = np.array([3.0, 4.0, 5.0])
        out = project_out(v, B)
        assert out.shape == (3,)
        assert np.allclose(out, [0.0, 4.0, 5.0])

    def test_2d_input_shape_preserved(self) -> None:
        B = np.array([[1.0, 0.0, 0.0]])
        V = np.array([[3.0, 4.0, 5.0], [1.0, 0.0, 0.0]])
        out = project_out(V, B)
        assert out.shape == (2, 3)
        assert np.allclose(out[0], [0.0, 4.0, 5.0])
        assert np.allclose(out[1], [0.0, 0.0, 0.0])

    def test_idempotent_under_orthonormal_basis(self) -> None:
        rng = np.random.default_rng(5)
        # Build an orthonormal basis via QR.
        A = rng.standard_normal((6, 12))
        Q, _ = np.linalg.qr(A.T)  # (12, 6); columns orthonormal.
        B = Q.T  # (6, 12); rows orthonormal.

        V = rng.standard_normal((10, 12))
        once = project_out(V, B)
        twice = project_out(once, B)
        assert np.allclose(once, twice, atol=1e-5)

    def test_orthogonal_to_basis_after_projection(self) -> None:
        rng = np.random.default_rng(6)
        A = rng.standard_normal((4, 8))
        Q, _ = np.linalg.qr(A.T)
        B = Q.T  # (4, 8); rows orthonormal.
        V = rng.standard_normal((5, 8))
        out = project_out(V, B)
        dots = out @ B.T  # (5, 4); should be ~0.
        assert np.allclose(dots, 0.0, atol=1e-5)

    def test_hidden_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden-dim mismatch"):
            project_out(np.zeros((3, 5)), np.zeros((2, 6)))


# --------------------------------------------------------------------------
# derive_paper_steering_vectors (end-to-end)
# --------------------------------------------------------------------------

class TestDerivePaperSteeringVectors:
    def test_output_keys_and_shape(self) -> None:
        rng = np.random.default_rng(7)
        d = 16
        per_emotion = {
            "joy": rng.standard_normal((8, d)),
            "fear": rng.standard_normal((8, d)),
            "anger": rng.standard_normal((8, d)),
        }
        neutral = rng.standard_normal((40, d))
        out = derive_paper_steering_vectors(
            per_emotion, neutral, var_threshold=0.5
        )
        assert set(out) == set(per_emotion)
        for v in out.values():
            assert v.shape == (d,)
            assert v.dtype == np.float32

    def test_output_orthogonal_to_neutral_pcs(self) -> None:
        rng = np.random.default_rng(8)
        d = 24
        per_emotion = {
            "joy": rng.standard_normal((10, d)),
            "fear": rng.standard_normal((10, d)),
        }
        neutral = rng.standard_normal((60, d))
        var_threshold = 0.5

        out = derive_paper_steering_vectors(
            per_emotion, neutral, var_threshold=var_threshold
        )
        # Reconstruct the same PC basis the function used.
        pcs = fit_neutral_pcs(neutral, var_threshold=var_threshold)

        for v in out.values():
            # v should be orthogonal to every PC.
            dots = pcs @ v.astype(np.float64)
            assert np.allclose(dots, 0.0, atol=1e-4)

    def test_centering_preserved_after_projection(self) -> None:
        """Sum of output vectors stays near zero after PC projection.

        Cross-emotion centering produces vectors that sum to zero across
        emotions. Projecting each by the same linear operator preserves
        that property.
        """
        rng = np.random.default_rng(9)
        d = 20
        per_emotion = {
            f"e{i}": rng.standard_normal((6, d)) for i in range(5)
        }
        neutral = rng.standard_normal((50, d))
        out = derive_paper_steering_vectors(per_emotion, neutral)
        total = np.sum(np.stack(list(out.values()), axis=0), axis=0)
        assert np.allclose(total, 0.0, atol=1e-4)
