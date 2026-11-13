from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "j25_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_j25.py"
)
j25 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j25)


def _write(tmp: Path, rows: list[tuple[str, str, list[int], bool]]) -> None:
    """rows: (condition, item, B values, gate_flagged) -> payloads + scores."""
    pay, sc = [], []
    for c, item, vals, flagged in rows:
        for i, v in enumerate(vals):
            sid = f"{c}#{i}"
            if item == "B":
                pay.append({"sid": sid, "condition": c, "gate_flags": ["x"] if flagged else [],
                            "full": "text"})
            sc.append({"sid": sid, "item": item, "value": v})
    (tmp / "payloads.jsonl").write_text("\n".join(json.dumps(r) for r in pay))
    (tmp / "judge_scores.jsonl").write_text("\n".join(json.dumps(r) for r in sc))


def test_condition_names_match_sweep_convention() -> None:
    assert j25.CLASSES["pool"][0] == "j24_pool1_L16_a0.25"
    assert j25.CLASSES["coh"][5] == "j25_coh6_L16_a0.25"
    assert j25.W90 == "j25_w90_L16_a0.2413" and j25.W90CTL == "j25_w90ctl_L16_a0.2413"
    assert j25.UNSTEERED == "j24_full_L16_a0"


def test_rates_skip_gate_flagged_and_missing(tmp_path: Path) -> None:
    c = j25.CLASSES["coh"][0]
    _write(tmp_path, [(c, "B", [1, 0, None, 1], False), ("other", "B", [1], True)])
    pay, sc = j25.load(tmp_path)
    assert j25.rate(pay, sc, "B", c) == (2, 3)          # missing score stays missing
    assert j25.rate(pay, sc, "B", "other") == (0, 0)    # gate-flagged excluded


def test_primary_separated_classes_hit_min_p() -> None:
    cr = {"pool": [0.9, 0.85, 0.8, 0.95, 0.9, 0.88], "coh": [0.0, 0.05, 0.02, 0.0, 0.1, 0.03]}
    r = j25.primary(cr)
    assert r["p"] == pytest.approx(2 / 924)
    assert r["diff"] == pytest.approx(np.mean(cr["pool"]) - np.mean(cr["coh"]))
    assert r["ci"][0] <= r["diff"] <= r["ci"][1]


def test_primary_identical_classes_not_significant() -> None:
    v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    r = j25.primary({"pool": v, "coh": list(v)})
    assert r["diff"] == 0 and r["p"] == 1.0


def test_primary_refuses_empty_direction() -> None:
    with pytest.raises(ValueError):
        j25.primary({"pool": [float("nan")] + [0.5] * 5, "coh": [0.1] * 6})


def test_bootstrap_uses_j25_seed() -> None:
    a, b = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9], [0.0, 0.0, 0.1, 0.2, 0.4, 0.5]
    rng = np.random.default_rng(20261107)
    ia, ib = rng.integers(0, 6, size=(10_000, 6)), rng.integers(0, 6, size=(10_000, 6))
    d = np.asarray(a)[ia].mean(axis=1) - np.asarray(b)[ib].mean(axis=1)
    np.testing.assert_allclose(j25._bootstrap_ci(a, b),
                               np.quantile(d, [0.025, 0.975], method="linear"), atol=1e-15)


# --- amendment 3 -------------------------------------------------------------

def test_amendment3_condition_names() -> None:
    assert j25.W90CLASSES["w90"][0] == "j25_w90_L16_a0.2413"
    assert j25.W90CLASSES["w90"][5] == "j25_w90_6_L16_a0.2413"
    assert j25.W90CLASSES["w90ctl"][5] == "j25_w90ctl_6_L16_a0.2413"
    assert j25.V_ONLY == "j24_full_L16_a0.0654"
    assert j25.POOL_LOW[0] == "j24_pool1_L16_a0.15"
    assert j25.FULL_LOW == "j24_full_L16_a0.15"


def test_holm_known_example() -> None:
    adj = j25.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_dissociation_row() -> None:
    assert j25.dissociation_row(True, True, 0.4, 0.1, 0.01) == "double dissociation"
    assert (j25.dissociation_row(True, True, 0.1, 0.4, 0.01)
            == "not a double dissociation")          # coh <= unsteered
    assert (j25.dissociation_row(True, True, 0.4, 0.1, 0.2)
            == "not a double dissociation")          # Fisher p >= 0.05
    assert j25.dissociation_row(True, False, 0.4, 0.1, 0.01) == "H25c fails"
    assert j25.dissociation_row(False, True, 0.4, 0.1, 0.01) == "other"
    assert j25.dissociation_row(False, False, 0.4, 0.1, 0.01) == "other"


def test_interaction_arithmetic() -> None:
    # I = both - w_only - v_only + none
    assert j25.interaction(0.8, 0.3, 0.2, 0.1) == pytest.approx(0.4)
    assert j25.interaction(0.5, 0.5, 0.2, 0.2) == pytest.approx(0.0)


def test_interaction_ci_contains_estimate() -> None:
    pool = [[1] * 30 + [0] * 10 for _ in range(6)]
    w90 = [[0] * 40 for _ in range(6)]
    v = [0] * 40
    none = [0] * 40
    est = j25.interaction(np.mean([np.mean(x) for x in pool]), 0.0, 0.0, 0.0)
    lo, hi = j25.interaction_ci(pool, w90, v, none)
    assert lo <= est <= hi and est == pytest.approx(0.75)


def test_sign_flip_perfect_separation() -> None:
    assert j25.sign_flip_p([0.5, 0.4, 0.6, 0.3, 0.45, 0.55]) == pytest.approx(2 / 64)
    assert j25.sign_flip_p([0.1, -0.2, 0.15, -0.1, 0.2, -0.15]) > 0.5


def test_coherent_dose_selection() -> None:
    assert j25.coherent_dose({0.15: 0.05, 0.25: 0.30}) == 0.15
    assert j25.coherent_dose({0.15: 0.20, 0.25: 0.05}) == 0.25
    assert j25.coherent_dose({0.15: 0.05, 0.25: 0.10}) == 0.25   # <= 10% qualifies
    assert j25.coherent_dose({0.15: 0.20, 0.25: 0.30}) is None
