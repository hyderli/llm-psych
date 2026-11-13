from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "score_blackmail", Path(__file__).resolve().parents[1] / "scripts/score_blackmail.py"
)
sb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sb)


def _write_dir(tmp: Path, items: list[str]) -> Path:
    """One unflagged sample plus one score row per item -> payloads + scores."""
    tmp.mkdir(parents=True, exist_ok=True)
    pay = [{"sid": "s0", "condition": "c", "index": 0, "development": False,
            "pad": "p", "act": "a", "full": "f", "recipients": [],
            "n_tool_calls": 1, "gate_flags": [], "G0_auto": 1}]
    sc = [{"sid": "s0", "item": it, "value": 0, "span": None} for it in items]
    (tmp / "payloads.jsonl").write_text("\n".join(json.dumps(r) for r in pay))
    (tmp / "judge_scores.jsonl").write_text("\n".join(json.dumps(r) for r in sc))
    return tmp


def _combine(tmp: Path) -> list[dict]:
    a = SimpleNamespace(dir=str(tmp), human="")
    assert sb.cmd_combine(a) == 0
    return list(csv.DictReader((tmp / "scores_judge.csv").open()))


def test_p_tier_one_anchors_extract() -> None:
    p = [a for a in sb.extract_anchors() if a["item"] == "P"]
    assert len(p) == 4 and sum(a["expected"] for a in p) == 2


def test_p_tier_two_anchors() -> None:
    p = [a for a in sb.HARD_ANCHORS if a[0] == "P"]
    assert len(p) == 11
    assert sum(a[1] for a in p) == 4 and len(p) - sum(a[1] for a in p) == 7


def test_existing_anchor_counts_unchanged() -> None:
    got = {i: len([a for a in sb.extract_anchors() if a["item"] == i])
           for i in sb.JUDGE_ITEMS}
    assert got == {i: sb.EXPECTED_ANCHORS[i] for i in sb.JUDGE_ITEMS}
    assert "P" not in sb.JUDGE_ITEMS and "P" in sb.EXTRA_ITEMS


def test_build_prompt_p() -> None:
    pr = sb.build_prompt("P", "the act payload")
    assert "express praise, admiration, or warmth" in pr
    assert "the act payload" in pr
    assert sb.PAYLOAD_OF["P"] == "act"


def test_combine_identical_with_and_without_p(tmp_path: Path) -> None:
    base = _combine(_write_dir(tmp_path / "a", sb.JUDGE_ITEMS))
    with_p = _combine(_write_dir(tmp_path / "b", sb.JUDGE_ITEMS + ["P"]))
    assert base == with_p                     # P rows are ignored, not incomplete


def test_export_accepts_p_and_mixed_items(tmp_path: Path) -> None:
    d = _write_dir(tmp_path / "e", sb.JUDGE_ITEMS)
    for items in ("P", "A_act,B,P"):
        a = SimpleNamespace(dir=str(d), items=items,
                            out=f"p_{items.replace(',', '_')}.jsonl",
                            include_flagged=False)
        assert sb.cmd_export(a) == 0
        rows = [json.loads(x) for x in (d / a.out).read_text().splitlines()]
        assert {r["item"] for r in rows} == set(items.split(","))


def test_export_rejects_unknown_item(tmp_path: Path) -> None:
    d = _write_dir(tmp_path / "u", sb.JUDGE_ITEMS)
    a = SimpleNamespace(dir=str(d), items="Q", out="q.jsonl",
                        include_flagged=False)
    import pytest
    with pytest.raises(SystemExit):
        sb.cmd_export(a)


def test_ingest_accepts_p(tmp_path: Path) -> None:
    d = _write_dir(tmp_path / "i", sb.JUDGE_ITEMS)
    res = [{"id": "s0", "item": "P", "value": 0, "span": None},
           {"id": "s0", "item": "NOPE", "value": 1, "span": "a"}]
    (d / "res.jsonl").write_text("\n".join(json.dumps(r) for r in res))
    a = SimpleNamespace(dir=str(d), results=str(d / "res.jsonl"),
                        judge_model="test-judge")
    assert sb.cmd_ingest(a) == 0
    rows = [json.loads(x) for x in (d / "judge_scores.jsonl").read_text().splitlines()]
    assert [r["item"] for r in rows] == ["P"]
