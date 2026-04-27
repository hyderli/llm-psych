"""Logistic-regression probes over residual-stream activations.

Implements H1 (linear probe accessibility) from HYPOTHESES.md:
one-vs-rest logistic regression per ``(model, layer, emotion)`` triple,
evaluated with AUC and 1000-bootstrap 95% CIs.

Persistence format matches ``docs/methods.md``:
  ``probes/<model>/<emotion>_layer<L>.joblib``     — fitted estimator
  ``probes/<model>/<emotion>_layer<L>.yaml``       — metadata

Notes
-----
Layer selection is separate from probe fitting: callers fit probes on
every candidate layer (``models.probe_layer_range``), pick the best
layer on a validation slice, then evaluate on the held-out test set.
The entry point in ``scripts/train_probes.py`` (TODO) orchestrates this.
Here we expose the building blocks: ``fit``, ``evaluate``, ``save``,
``load``.

References
----------
Sofroniew et al. 2026. Emotion Concepts and their Function in a Large
Language Model. Transformer Circuits Thread.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Metrics for a single fitted probe on a given dataset split.

    Parameters
    ----------
    auc
        Area under the ROC curve (one-vs-rest).
    ci_low, ci_high
        95% bootstrap CI endpoints (1000 resamples).
    brier
        Brier score (lower = better calibration; > 0.25 flags for review).
    n
        Number of examples evaluated.
    """

    auc: float
    ci_low: float
    ci_high: float
    brier: float
    n: int

    def __str__(self) -> str:
        return (
            f"AUC={self.auc:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] "
            f"Brier={self.brier:.4f} n={self.n}"
        )


@dataclass
class ProbeMeta:
    """Everything needed to reproduce or audit a saved probe.

    Stored alongside the joblib file as a YAML sidecar.
    """

    emotion: str
    layer: int
    model_id: str
    model_sha: str | None
    n_train: int
    n_test: int
    auc: float
    ci_low: float
    ci_high: float
    brier: float
    sklearn_c: float = 1.0


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit a one-vs-rest logistic regression probe.

    Parameters
    ----------
    X_train
        Float array of shape ``(n_samples, hidden_dim)``.  Should be
        float32 or float64 for sklearn; callers must cast from float16.
    y_train
        Integer label array of shape ``(n_samples,)``.  Binary
        (0 = not-emotion, 1 = emotion) for one-vs-rest per emotion.
    C
        L2 regularization inverse strength. Default 1.0 per HYPOTHESES.md.
    max_iter
        Solver iteration limit. 1000 is sufficient for well-conditioned
        activations at 4096-dim; increase if convergence warnings appear.
    random_state
        Seed for the solver's randomness (lbfgs is deterministic, but
        set for completeness).

    Returns
    -------
    LogisticRegression
        Fitted estimator.
    """
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> ProbeResult:
    """Evaluate a probe with AUC + 95% bootstrap CI + Brier score.

    Parameters
    ----------
    clf
        Fitted :class:`~sklearn.linear_model.LogisticRegression`.
    X
        Float array ``(n_samples, hidden_dim)``.
    y
        Binary label array ``(n_samples,)``.
    n_bootstrap
        Number of bootstrap resamples for CI estimation. Default 1000
        per HYPOTHESES.md.
    rng
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    ProbeResult
        AUC, 95% CI, Brier score, and sample count.

    Raises
    ------
    ValueError
        If ``y`` contains only one class (AUC undefined).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    proba = clf.predict_proba(X)
    # For binary probes, positive-class probability is column 1.
    # For multi-class OVR, roc_auc_score handles it with multi_class="ovr".
    if proba.shape[1] == 2:
        pos_proba = proba[:, 1]
        auc = float(roc_auc_score(y, pos_proba))
        brier = float(brier_score_loss(y, pos_proba))
        boot_aucs = _bootstrap_auc_binary(y, pos_proba, n_bootstrap, rng)
    else:
        # Multi-class one-vs-rest AUC
        auc = float(roc_auc_score(y, proba, multi_class="ovr"))
        brier = float(np.mean([
            brier_score_loss((y == c).astype(int), proba[:, i])
            for i, c in enumerate(clf.classes_)
        ]))
        boot_aucs = _bootstrap_auc_multiclass(y, proba, clf.classes_, n_bootstrap, rng)

    ci_low, ci_high = float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))

    if auc > 0.80 and brier > 0.25:
        logger.warning(
            "AUC=%.4f but Brier=%.4f > 0.25 — probe may be overconfident. "
            "Review calibration.",
            auc,
            brier,
        )

    return ProbeResult(auc=auc, ci_low=ci_low, ci_high=ci_high, brier=brier, n=len(y))


def _bootstrap_auc_binary(
    y: np.ndarray,
    pos_proba: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    aucs = np.empty(n_bootstrap)
    n = len(y)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], pos_proba[idx]
        if len(np.unique(y_b)) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(y_b, p_b)
    return aucs[~np.isnan(aucs)]


def _bootstrap_auc_multiclass(
    y: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    aucs = np.empty(n_bootstrap)
    n = len(y)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], proba[idx]
        if len(np.unique(y_b)) < len(classes):
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(y_b, p_b, multi_class="ovr")
    return aucs[~np.isnan(aucs)]


# --------------------------------------------------------------------------
# Layer selection
# --------------------------------------------------------------------------

def best_layer(
    results: dict[int, ProbeResult],
) -> int:
    """Return the layer index with the highest AUC.

    Parameters
    ----------
    results
        Mapping from layer index to :class:`ProbeResult` (e.g. from a
        validation-split evaluation over all candidate layers).

    Returns
    -------
    int
        Layer index with maximum AUC.
    """
    return max(results, key=lambda lyr: results[lyr].auc)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save(
    clf: LogisticRegression,
    meta: ProbeMeta,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist probe to disk in the format specified in ``docs/methods.md``.

    Parameters
    ----------
    clf
        Fitted probe.
    meta
        Metadata to store alongside.
    out_dir
        Directory for output files. Created if absent.

    Returns
    -------
    (joblib_path, yaml_path)
        Paths to the written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{meta.emotion}_layer{meta.layer}"
    jl_path = out_dir / f"{stem}.joblib"
    yaml_path = out_dir / f"{stem}.yaml"

    joblib.dump(clf, jl_path)

    with yaml_path.open("w") as fh:
        yaml.safe_dump(asdict(meta), fh, sort_keys=True)

    logger.info("Saved probe to %s", jl_path)
    return jl_path, yaml_path


def load(
    probe_dir: str | Path,
    emotion: str,
    layer: int,
) -> tuple[LogisticRegression, ProbeMeta]:
    """Load a probe and its metadata from disk.

    Parameters
    ----------
    probe_dir
        Directory containing ``<emotion>_layer<L>.joblib`` and ``.yaml``.
    emotion
        Emotion label (e.g. ``"anger"``).
    layer
        Layer index.

    Returns
    -------
    (clf, meta)
    """
    probe_dir = Path(probe_dir)
    stem = f"{emotion}_layer{layer}"
    clf: LogisticRegression = joblib.load(probe_dir / f"{stem}.joblib")

    with (probe_dir / f"{stem}.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    meta = ProbeMeta(**raw)

    return clf, meta


# --------------------------------------------------------------------------
# Steering vector derivation
# --------------------------------------------------------------------------

def derive_steering_vector(
    emotion_activations: np.ndarray,
    neutral_activations: np.ndarray,
) -> np.ndarray:
    """Compute a CAA steering vector (mean difference).

    ``v = mean(emotion_train) − mean(neutral_train)``

    Per ``docs/methods.md`` steering-vector derivation. Reuses the same
    train-set activations extracted for probe fitting — no second forward pass.

    Parameters
    ----------
    emotion_activations
        Float array ``(n_emotion, hidden_dim)`` from the emotion prompt set.
    neutral_activations
        Float array ``(n_neutral, hidden_dim)`` from the neutral prompt set.

    Returns
    -------
    np.ndarray
        Direction vector of shape ``(hidden_dim,)``, float32.
    """
    v = emotion_activations.mean(axis=0) - neutral_activations.mean(axis=0)
    return v.astype(np.float32)
