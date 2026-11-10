from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b26 = _load("build_j26a", "scripts/build_j26a.py")
a26 = _load("analyze_j26a", "scripts/analyze_j26a.py")


def test_counterbalancing_counts() -> None:
    orders = [tuple(p) for p in __import__("itertools").permutations(b26.KEYS)]
    ids5 = ["QX7", "M4R", "ZK2", "B9T", "H6W"]
    assert len(orders) == 120
    b26.check_counterbalancing(orders, ids5)          # does not raise
    from collections import Counter
    pos, lab = Counter(), Counter()
    for i, order in enumerate(orders):
        a = b26.assign(order, i, ids5)
        for j, k in enumerate(order):
            pos[(k, j)] += 1
            lab[(k, a[k])] += 1
    assert all(v == 24 for v in pos.values())
    assert all(v == 24 for v in lab.values())


def test_sign_flip_mc() -> None:
    # complete separation: all diffs > 0; only the all-negative sign draw ties
    # perfect separation over 6 pairs: only the two all-one sign patterns
    # reach the observed |mean|, so the exact p is 2/64 ~ 0.031
    d = [0.5, 0.4, 0.6, 0.3, 0.45, 0.55]
    p = a26.sign_flip_mc(d, seed=20261110, draws=100_000)
    assert p == pytest.approx(2 / 64, abs=0.01)
    assert a26.sign_flip_mc([0.1, -0.2, 0.15, -0.1, 0.2, -0.15],
                            draws=10_000) > 0.5


def test_reading_row_all_branches() -> None:
    assert a26.reading_row(False, True, False, 0.0, 0.0).startswith("control fails")
    assert a26.reading_row(True, True, False, 0.5, 0.6).startswith("conceptual")
    assert a26.reading_row(True, True, False, 0.7, 0.6).startswith("withheld")
    assert a26.reading_row(True, True, False, None, None).startswith("withheld")
    assert a26.reading_row(True, True, True, 0.5, 0.6).startswith("mixed")
    assert a26.reading_row(True, False, True, 0.5, 0.6).startswith("word matching")
    assert a26.reading_row(True, False, False, 0.5, 0.6).startswith("lexical")


def _row(cond, oi, mass=1.0, p=None):
    return {"condition": cond, "order_index": oi, "mass": mass,
            "p": p or {"T": 0.2, "E": 0.2, "C": 0.2, "W": 0.2, "X": 0.2}}


def test_mass_exclusion() -> None:
    rows = {("unsteered", 0): _row("unsteered", 0, mass=0.3),
            ("unsteered", 1): _row("unsteered", 1),
            ("unsteered", 2): _row("unsteered", 2)}
    assert len(a26.compliant(rows, "unsteered")) == 2
    assert a26.mean_p(rows, "unsteered", "T") == pytest.approx(0.2)
    rows2 = {("unsteered", 0): _row("unsteered", 0, mass=0.1)}
    assert np.isnan(a26.mean_p(rows2, "unsteered", "T"))


def test_selected_codes_carry_leading_space() -> None:
    """The recorded first token must decode with a leading space and the
    prefix ("Draft ID:") tokenisation must be unchanged by appending the
    " "+code candidate."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached locally")
    ctx = "Draft ID:"
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    english = b26.english_token_ids(tok)
    picked = b26.select_codes(tok, ctx, ctx_ids,
                              ["QX7", "M4R", "ZK2", "B9T", "H6W", "J3V",
                               "P8K", "W5N", "T2G", "F7Y", "R4Z", "D9Q"],
                              english)
    assert [p["code"] for p in picked] == ["QX7", "M4R", "ZK2", "B9T", "H6W"]
    for p in picked:
        full = tok(ctx + " " + p["code"], add_special_tokens=False)["input_ids"]
        assert full[:len(ctx_ids)] == ctx_ids          # prefix unchanged
        assert p["first_token_text"].startswith(" ")   # leading-space token


def test_check_drafts_refuses_modified_draft() -> None:
    good = json.loads((ROOT / "data/public/j26_choice_drafts.json")
                      .read_text())["drafts"]
    b26.check_drafts(good)                             # does not raise
    bad = dict(good)
    bad["T"] = good["T"] + " Maria"
    with pytest.raises(SystemExit):
        b26.check_drafts(bad)                          # names Maria
    bad2 = dict(good)
    bad2["W"] = good["W"] + " extra"
    with pytest.raises(SystemExit):
        b26.check_drafts(bad2)                         # wrong word count
    bad3 = dict(good)
    bad3["X"] = "Kyle, I will crush them all before 5pm."
    with pytest.raises(SystemExit):
        b26.check_drafts(bad3)                         # wrong lexicon hits
