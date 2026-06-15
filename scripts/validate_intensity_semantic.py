"""Numerical-intensity semantic validation (C2; Sofroniew et al. 2026 §2).

Each template family varies only a number; the number -> emotional-intensity
relation is **semantic**, and for the INVERSE families it runs opposite to the
raw digit (younger age at death = greater loss; sooner return = more joy). We
read the residual stream at the Assistant-colon, project it onto the story
emotion vector, and correlate (Spearman) with:
  * ``intensity_rank`` (the semantic order)  — should be strongly positive;
  * raw ``x`` (the digit)                     — diverges on inverse families.
If the projection tracks the *semantic* rank, including where it disagrees
with the raw number, the vector encodes meaning, not surface lexis. See
``plans/numerical-intensity-control.md``.

Usage:  uv run python scripts/validate_intensity_semantic.py --model qwen25_05b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from llm_psych.hooks import ResidualStreamRecorder  # noqa: E402
from llm_psych.models import load_model  # noqa: E402

EMOTIONS = ["admiration", "joy", "loathing", "sadness"]
STIMULI = _repo_root / "data" / "public" / "intensity_templates.jsonl"
RANK_PASS = 0.6  # |Spearman(proj, intensity_rank)| target (plans/numerical-intensity-control.md)


def _read_model_cfg(name: str) -> dict:
    path = _repo_root / "configs" / "model" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"model config not found: {path}")
    return yaml.safe_load(path.read_text())


def _check_hash(path: Path) -> None:
    reg = _repo_root / "configs" / "stimuli_hashes.yaml"
    if not reg.exists():
        return
    expected = (yaml.safe_load(reg.read_text()) or {}).get(path.name)
    if expected:
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"[stimuli] {path.name} MD5 {'OK' if actual == expected else 'MISMATCH (frozen set changed!)'}")


def _discover_vectors(story_dir: Path) -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = defaultdict(dict)
    for f in sorted(story_dir.glob("*_layer*.npy")):
        emotion, _, layer_s = f.stem.rpartition("_layer")
        if emotion and layer_s.isdigit():
            out[emotion][int(layer_s)] = f
    return dict(out)


def _pick_layer(vectors: dict[str, dict[int, Path]], requested: int | None, n_layers: int | None) -> int:
    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    if not shared:
        raise SystemExit("no layer shared across all emotions")
    if requested is not None:
        if requested not in shared:
            raise SystemExit(f"layer {requested} not in shared {shared}")
        return requested
    if n_layers:
        target = round(2 * n_layers / 3)
        return min(shared, key=lambda L: abs(L - target))
    return max(shared)


def _spear(a: list[float], b: list[float]) -> float:
    if len(set(b)) < 2 or len(set(a)) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_config", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 (default; model-faithful, avoids fp16 overflow, half the "
                         "memory of fp32). Use float32 for a slow CPU-only Mac run if bf16 is unsupported.")
    ap.add_argument("--vectors-dir", default="steering_vectors")
    ap.add_argument("--out-dir", default="results/vector_validation")
    ap.add_argument("--sweep", action="store_true",
                    help="report per-emotion inverse-family ρ(rank) across ALL shared layers "
                         "(layer-robustness curve) instead of the full table at ~2/3 depth")
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")
    _check_hash(STIMULI)
    rows = [json.loads(line) for line in STIMULI.read_text().splitlines() if line.strip()]

    cfg = _read_model_cfg(args.model_config)
    hf_model_id = cfg["hf_model_id"]
    model_key = hf_model_id.split("/")[-1]
    story_dir = _repo_root / args.vectors_dir / f"{model_key}-story"
    if not story_dir.exists():
        raise SystemExit(f"no story vectors at {story_dir} (pull them from HF first)")
    vectors = _discover_vectors(story_dir)
    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    if not shared:
        raise SystemExit("no layer shared across all emotions")
    if args.sweep:
        layers_to_run = shared
    else:
        layers_to_run = [_pick_layer(vectors, args.layer, cfg.get("n_layers"))]
    vec = {L: {e: np.load(vectors[e][L]).astype(np.float64) for e in EMOTIONS}
           for L in layers_to_run}

    lm = load_model(hf_model_id, revision=cfg.get("hf_revision"),
                    torch_dtype=getattr(torch, args.dtype), device_map=args.device)
    lm.model.eval()
    device = next(lm.model.parameters()).device
    rec = ResidualStreamRecorder(lm.model, layers_to_run, token_position="last")
    rec.attach()  # hooks are not registered until attach()/__enter__

    # readout activation per row, all layers captured per pass
    print(f"reading {len(rows)} templates at {len(layers_to_run)} layer(s) "
          f"({args.device}/{args.dtype}) ...")
    with torch.no_grad():
        try:
            for i, r in enumerate(rows, 1):
                prompt = lm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["text"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                inp = lm.tokenizer(prompt, return_tensors="pt").to(device)
                lm.model(**inp)
                r["_acts"] = {L: rec.activations[L][0].float().cpu().numpy().astype(np.float64)
                              for L in layers_to_run}
                if i % 25 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)}", flush=True)
        finally:
            rec.remove()

    fam: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        fam[(r["emotion"], r["family"])].append(r)

    def _mean(vals: list[float]) -> float:
        v = [x for x in vals if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    def analyze(L: int):
        """Per-family ρ's + inverse-family per-emotion means + neutral confound, at layer L."""
        vL = vec[L]
        per_inv: dict[str, list[float]] = defaultdict(list)
        per_all: dict[str, list[float]] = defaultdict(list)
        table: list[tuple] = []
        for (emotion, family), items in sorted(fam.items()):
            if emotion == "neutral":
                continue
            items.sort(key=lambda r: r["x"])
            proj = [float(r["_acts"][L] @ vL[emotion]) for r in items]
            rho_rank = _spear(proj, [r["intensity_rank"] for r in items])
            rho_x = _spear(proj, [r["x"] for r in items])
            direction = items[0]["direction"]
            per_all[emotion].append(rho_rank)
            if direction == "decreasing":
                per_inv[emotion].append(rho_rank)
            table.append((emotion, family, direction, len(items), rho_rank, rho_x))
        neutral_rho: list[float] = []
        for (emotion, family), items in fam.items():
            if emotion != "neutral":
                continue
            items.sort(key=lambda r: r["x"])
            xs = [r["x"] for r in items]
            for e in EMOTIONS:
                neutral_rho.append(abs(_spear([float(r["_acts"][L] @ vL[e]) for r in items], xs)))
        return per_inv, per_all, table, neutral_rho

    out_dir = _repo_root / args.out_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        lines = [
            f"# Numerical-intensity validation — {model_key} (layer sweep)",
            "",
            f"- Vectors: `{story_dir.relative_to(_repo_root)}` (story method)",
            "- Per-layer **inverse-family mean ρ(proj, rank)** per emotion (confound-controlled). "
            f"Robustness: is semantic tracking stable across the mid-late band? Target ≥ {RANK_PASS}.",
            "",
            "| layer | " + " | ".join(EMOTIONS) + " | neutral|ρ| |",
            "|" + "---|" * (len(EMOTIONS) + 2),
        ]
        for L in layers_to_run:
            per_inv, _, _, neutral_rho = analyze(L)
            cells = " | ".join(f"{_mean(per_inv[e]):+.2f}" for e in EMOTIONS)
            nz = np.nanmean(neutral_rho) if neutral_rho else float("nan")
            lines.append(f"| {L} | {cells} | {nz:.2f} |")
        report = out_dir / "intensity_semantic_sweep.md"
        report.write_text("\n".join(lines))
        print(f"swept {len(layers_to_run)} layers")
        print(f"Wrote {report.relative_to(_repo_root)}")
        return 0

    L = layers_to_run[0]
    per_emo_inv, per_emo_all, table, neutral_rho = analyze(L)
    lines = [
        f"# Numerical-intensity validation — {model_key}",
        "",
        f"- Vectors: `{story_dir.relative_to(_repo_root)}` (story method), layer **{L}** (~2/3 depth)",
        "- Read-out: Assistant-colon projection onto the emotion vector; Spearman vs rank and vs raw x",
        f"- Target: ρ(proj, intensity_rank) ≥ {RANK_PASS}; on INVERSE families ρ(rank) ≫ ρ(x)",
        "",
        "| emotion | family | dir | n | ρ(proj, rank) | ρ(proj, raw x) |",
        "|---|---|---|---|---|---|",
    ]
    for emotion, family, direction, n, rho_rank, rho_x in table:
        mark = " **(inverse)**" if direction == "decreasing" else ""
        lines.append(
            f"| {emotion} | {family}{mark} | {direction[:3]} | {n} "
            f"| {rho_rank:+.2f} | {rho_x:+.2f} |"
        )
    lines += [
        "",
        "## Semantic intensity — INVERSE families only (confound-controlled)",
        "",
        "On inverse families the number runs *against* the intended intensity, so "
        f"ρ(proj, rank) > 0 means the vector tracks meaning, not the digit. Target ≥ {RANK_PASS}. "
        "Run with --sweep to confirm this is not hostage to one layer.",
        "",
    ]
    for e in EMOTIONS:
        m = _mean(per_emo_inv[e])
        n = len([x for x in per_emo_inv[e] if not np.isnan(x)])
        flag = "✓" if (not np.isnan(m) and m >= RANK_PASS) else "⚠"
        plural = "y" if n == 1 else "ies"
        lines.append(f"- {flag} **{e}**: {m:+.2f}  ({n} inverse famil{plural})")
    lines += [
        "",
        "_secondary — number-confounded, do not over-read:_ mean ρ over ALL families — "
        + ", ".join(f"{e} {_mean(per_emo_all[e]):+.2f}" for e in EMOTIONS),
    ]
    if neutral_rho:
        lines += [
            "",
            "## Neutral control (number confound)",
            "",
            f"- mean |ρ(proj, x)| over neutral families × vectors: {np.nanmean(neutral_rho):.2f}  (want low)",
            f"- max |ρ(proj, x)|: {np.nanmax(neutral_rho):.2f}  (inflated by n=6; mean is the honest read)",
        ]
    lines += [
        "",
        "## How to read this",
        "",
        "ρ(proj, rank) strongly positive ⇒ the vector tracks semantic intensity. "
        "On the **inverse** families ρ(rank) is positive while ρ(raw x) is negative — "
        "a surface/digit account predicts ρ(x); only a semantic account predicts ρ(rank). "
        "That gap is the decisive concept-vs-surface evidence (`plans/numerical-intensity-control.md`).",
    ]
    report = out_dir / "intensity_semantic.md"
    report.write_text("\n".join(lines))
    for e in EMOTIONS:
        m = _mean(per_emo_inv[e])
        print(f"  {e}: inverse-family rho(rank) = {m:+.2f}"
              if not np.isnan(m) else f"  {e}: no inverse family")
    print(f"Wrote {report.relative_to(_repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
