"""Unit tests for scripts/analyze_j27.py's registered J27 decision rules.

All tests run on (positives, scored) count fixtures — no payloads, tokenizer
or model needed. Covers the Fisher-based tests (H27a's pooled+per-draw
clause, the directional Fisher tests, H27e), capability acceptance, the
damage override, and the amendment-3 joint reading matrix including the
override precedence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


a = _load("analyze_j27", "scripts/analyze_j27.py")
run = _load("run_j27", "scripts/run_j27.py")
chk = _load("check_j27_manipulation", "scripts/check_j27_manipulation.py")


def kn(k: int, n: int = 100) -> tuple[int, int]:
    return (k, n)


# --------------------------------------------------------------------------
# Fisher mechanics
# --------------------------------------------------------------------------

def test_fisher_two_sided_strong_and_weak():
    assert a.fisher(kn(5), kn(80)) < 0.001          # clearly different
    assert a.fisher(kn(50), kn(50)) > 0.5           # identical rates


def test_collapsed_needs_both_clauses():
    # rate below AND Fisher p < 0.05
    assert a.collapsed(kn(5), kn(80))
    # rate below but not significant at n=100 each
    assert not a.collapsed(kn(45), kn(52))
    # significant difference the WRONG way is not a collapse
    assert not a.collapsed(kn(80), kn(5))


# --------------------------------------------------------------------------
# H27a: pooled Fisher + per-draw point-estimate clause (amendment 2)
# --------------------------------------------------------------------------

def test_h27a_pass_when_below_pool_and_every_draw():
    r = a.h27a(kn(5), kn(80), kn(85), kn(78))
    assert r["p"] < 0.001
    assert all(r["below_each_draw"].values())
    assert r["pass"]


def test_h27a_per_draw_clause_kills_a_pooled_pass():
    # Pooled R still far above B(3) (so the Fisher half would pass), but draw
    # R2 realised BELOW B(3) -> the point-estimate clause fails the primary.
    r = a.h27a(kn(40), kn(95), kn(30), kn(95))
    assert r["below_each_draw"][6] is True
    assert r["below_each_draw"][7] is False      # the per-draw violation
    assert r["below_each_draw"][8] is True
    assert not r["pass"]


def test_h27a_fails_when_not_below_pool():
    r = a.h27a(kn(80), kn(5), kn(6), kn(7))
    assert not r["pass"]


def test_h27a_fails_when_pooled_insignificant():
    r = a.h27a(kn(48), kn(50), kn(52), kn(51))
    assert not r["pass"]
    assert all(r["below_each_draw"].values())     # point estimates below,
    # but the pooled Fisher is not significant -> still fails.


# --------------------------------------------------------------------------
# Directional Fisher tests: H27b/c/d share fisher_down; H27e is its own clause
# --------------------------------------------------------------------------

def test_fisher_down_direction_matters():
    assert a.fisher_down(kn(5), kn(80))["pass"]
    assert not a.fisher_down(kn(80), kn(5))["pass"]
    assert not a.fisher_down(kn(50), kn(50))["pass"]


def test_h27e_needs_fisher_and_below_reference():
    # B(14) below B(15) significantly AND below B(13): pass
    assert a.h27e(kn(5), kn(80), kn(75))["pass"]
    # Fisher passes vs B(15) but B(14) is NOT below B(13): fail
    assert not a.h27e(kn(70), kn(90), kn(60))["pass"]
    # Below both but not significant: fail
    assert not a.h27e(kn(40), kn(48), kn(50))["pass"]


# --------------------------------------------------------------------------
# Capability acceptance and the identification floor
# --------------------------------------------------------------------------

def test_capability_identified_and_accepted():
    cap = a.capability(kn(80), kn(60))            # 0.8 and 0.6 >= 0.4
    assert cap["identified"] and cap["accepted"]


def test_capability_rejected_below_half():
    cap = a.capability(kn(80), kn(30))            # 0.3 < 0.5 x 0.8
    assert cap["identified"] and not cap["accepted"]


def test_capability_not_identified_below_floor():
    cap = a.capability(kn(20), kn(20))            # B(10) = 0.2 < 0.3
    assert not cap["identified"]


def test_capability_floor_boundary():
    assert a.capability(kn(30), kn(30))["identified"]       # 0.3 exactly
    assert not a.capability(kn(29), kn(29))["identified"]


# --------------------------------------------------------------------------
# Damage override: every R draw AND C must collapse vs B(2)
# --------------------------------------------------------------------------

def test_damage_override_all_collapse():
    d = a.damage_override(kn(80), kn(5), kn(4), kn(6), kn(3))
    assert d["damage"]


def test_damage_override_one_surviving_draw_clears_it():
    d = a.damage_override(kn(80), kn(5), kn(4), kn(78), kn(3))
    assert not d["damage"]                        # R2 ~= B(2) -> no override


def test_damage_override_c_survives():
    d = a.damage_override(kn(80), kn(70), kn(4), kn(6), kn(3))
    assert not d["damage"]


# --------------------------------------------------------------------------
# Amendment-3 joint reading matrix + override precedence
# --------------------------------------------------------------------------

def test_joint_reading_all_four_rows():
    assert "same word channel" in a.joint_reading(True, True)
    assert "TWO ROUTES" in a.joint_reading(False, True)
    assert "manipulation failed" in a.joint_reading(False, False)
    assert "not predicted" in a.joint_reading(True, False)


def _cap(identified=True, accepted=True):
    return {"identified": identified, "accepted": accepted}


def test_final_reading_override_precedence():
    # joint matrix reached only when capability OK and no damage
    row, note = a.final_reading(_cap(), {"damage": False}, True, True)
    assert row == "joint" and "same word channel" in note
    # capability not identified beats everything
    row, _ = a.final_reading(_cap(identified=False), {"damage": True}, True, True)
    assert row == "not_identified"
    # capability fail beats damage
    row, _ = a.final_reading(_cap(accepted=False), {"damage": True}, True, True)
    assert row == "capability_fail"
    # damage beats the joint matrix
    row, _ = a.final_reading(_cap(), {"damage": True}, True, True)
    assert row == "damage"


# --------------------------------------------------------------------------
# Pooled counts
# --------------------------------------------------------------------------

def test_pooled_sums_counts():
    pay = {
        "s1": {"sid": "s1", "condition": a.CONDS[6], "gate_flags": []},
        "s2": {"sid": "s2", "condition": a.CONDS[7], "gate_flags": []},
        "s3": {"sid": "s3", "condition": a.CONDS[7], "gate_flags": ["very_short"]},
        "s4": {"sid": "s4", "condition": a.CONDS[8], "gate_flags": []},
    }
    sc = {"B": {"s1": 1, "s2": 0, "s4": 1}}       # s3 gate-flagged, s2 scored 0
    k, n = a.pooled(pay, sc, "B", [6, 7, 8])
    assert (k, n) == (2, 3)                        # s1 + s4 positives of s1,s2,s4


# --------------------------------------------------------------------------
# Amendment-6 wiring: runner conditions, analysis CONDS and basis dirs agree
# --------------------------------------------------------------------------

def test_amendment6_condition_map_matches_runner():
    names = {c[0]: c[1] for c in run.CONDITIONS}
    assert a.CONDS == names                        # same order, same stems
    assert len(names) == 16                        # 14 scenario + 2 probe


def test_amendment6_spaces_dirs_agree():
    assert run.SPACES_BY_SET == chk.SPACES_BY_SET
    for cond in run.CONDITIONS:
        abl_set = cond[4]
        if abl_set is not None:
            assert abl_set in run.SPACES_BY_SET
            d = run.SPACES_BY_SET[abl_set]
            want = ("subspaces_pclean" if abl_set in
                    ("pclean", "c", "r1", "r2", "r3") else "subspaces_clean")
            assert d == want
    assert chk.RANK_SET == "pclean" and chk.MASS_SET == "pclean"
