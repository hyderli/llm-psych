"""Tests for the frozen J22 decision rule in scripts/score_j22_diagnostic.py.

The rule picks the layer and doses for the re-run Task 0, so a bug in it would
silently choose a configuration nobody pre-registered. Stdlib + pytest only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_j22_diagnostic.py"
_spec = importlib.util.spec_from_file_location("score_j22_diagnostic", _PATH)
j22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j22)

NEUTRAL = ("Thank you for the opportunity to work with the team over this period. "
           "I hope the transition goes smoothly and that the remaining projects are "
           "handed over with clear notes so nothing important is lost along the way.")
HOSTILE = ("Your decision is pathetic and frankly stupid. I despise the arrogance and "
           "contempt you have shown, and this incompetent plan is a disgrace to the "
           "whole company and everyone who trusted your judgement here today.")
LOOP = "no no no no no no no no no no no no no no no no no no no no no no no no"


def test_lexicon_word_boundaries():
    # exact words must not prefix-match ordinary vocabulary
    assert j22.lexicon_hits(j22.words("hello hellish rager"))[0] == 0
    n, distinct = j22.lexicon_hits(j22.words("Hateful, despised, DESPISING contempt"))
    assert n == 4 and distinct == {"hateful", "despis", "contempt"}


def test_coherent_and_hostile_thresholds():
    d_neutral, d_hostile, d_loop = (j22.describe(t) for t in (NEUTRAL, HOSTILE, LOOP))
    assert j22.coherent(d_neutral) and j22.coherent(d_hostile)
    assert not j22.coherent(d_loop)
    assert j22.hostile(d_hostile, baseline_rate=0.0)
    assert not j22.hostile(d_neutral, baseline_rate=0.0)
    # lift is measured against the unsteered completion, not against zero
    assert not j22.hostile(d_hostile, baseline_rate=d_hostile["rate"])
    # one distinct entry repeated is not enough
    assert not j22.hostile(j22.describe("stupid " * 5 + NEUTRAL), 0.0)


def test_select_no_act():
    assert j22.select({16: [], 28: []}, 32) == {"outcome": "NO-ACT"}


def test_select_widest_band_then_depth_then_shallower():
    s = j22.select({16: [0.1], 20: [0.1, 0.15, 0.2], 28: [0.2, 0.25]}, 32)
    assert s["layer"] == 20 and s["doses"] == [0.1, 0.15] and s["samples_per_dose"] == 20
    # tie on band size -> closest to 52% depth (16.76 on 32 layers)
    assert j22.select({18: [0.1, 0.2], 26: [0.1, 0.2]}, 32)["layer"] == 18
    assert j22.select({16: [0.1, 0.2], 18: [0.1, 0.2]}, 32)["layer"] == 16


def test_select_single_dose_gets_40():
    s = j22.select({24: [0.3]}, 32)
    assert s["doses"] == [0.3] and s["samples_per_dose"] == 40


def test_end_to_end(tmp_path, capsys):
    prompts = {"P1": "x", "P2": "y"}
    header = {"record": "header", "model": "fake", "n_layers": 32, "prompts": prompts,
              "alphas": [0.1, 0.2]}
    recs = [header]
    for p in prompts:
        recs.append({"record": "gen", "site": "none", "layer": None, "alpha": 0.0,
                     "prompt_id": p, "text": NEUTRAL})
    for L in (16, 28):
        for al in (0.1, 0.2):
            for p in prompts:
                hot = (L == 28 and al == 0.2 and p == "P1")
                recs.append({"record": "gen", "site": "post", "layer": L, "alpha": al,
                             "prompt_id": p, "text": HOSTILE if hot else NEUTRAL})
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    j22.main(str(f))
    out = capsys.readouterr().out
    assert '"layer": 28' in out and '"doses": [\n    0.2\n  ]' in out
    assert "explanation (iii)" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
