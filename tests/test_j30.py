"""Tests for J30's registered decision functions (pure logic, no model)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a30", ROOT / "scripts" / "analyze_j30.py")
a30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a30)


def test_lower_requires_direction_significance_and_ratio():
    assert a30.lower((10, 100), (80, 100), 1e-6, 0.5)
    assert not a30.lower((50, 100), (80, 100), 1e-3, 0.5)      # ratio 0.625 > 0.5
    assert not a30.lower((80, 100), (10, 100), 1e-6)           # wrong direction
    assert not a30.lower((60, 100), (70, 100), 0.2)            # not significant


def test_reading_precedence():
    r = a30.reading
    assert r(False, True, True, False, True, False).startswith("POSITIVE CONTROL FAILS")
    assert r(True, True, False, False, True, False).startswith("CAPABILITY FAILS")
    assert r(True, False, False, False, True, False).startswith("FULL MEDIATION")  # probe unidentified
    assert r(True, True, True, True, True, False).startswith("DAMAGE")
    assert r(True, True, True, False, True, False).startswith("FULL MEDIATION")
    assert r(True, True, True, False, True, True).startswith("PARTIAL MEDIATION")
    assert r(True, True, True, False, False, True).startswith("TWO ROUTES")


def _ro(d_share, r_share):
    return {"arms": {"full": {str(l): {"d": {"share": d_share}, "cp": {"share": 0.0},
                                       "r1": {"share": r_share}, "r2": {"share": r_share},
                                       "r3": {"share": r_share}} for l in range(16, 27)}}}


def test_conversion_rule():
    assert a30.conversion(_ro(0.03, 0.005)).startswith("CONVERSION")
    assert a30.conversion(_ro(0.01, 0.005)).startswith("NO CONVERSION")
