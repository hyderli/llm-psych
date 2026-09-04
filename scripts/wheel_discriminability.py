#!/usr/bin/env python
"""Per-layer pairwise discriminability of the Plutchik-wheel cells.

For every candidate layer, build the C x C matrix of cross-validated pairwise
classification accuracy between story-level pooled activations, using a
**nearest-centroid** rule. For a single pair that rule is exactly the
difference-of-means linear classifier the steering vectors are built from, so
the matrix measures separability under the same linear read-out family used to
derive the vectors -- not some unrelated probe.

Cross-validation is **grouped by story topic** when the source corpora are
available: whole topics are held out, so the number reads as "generalises to
unseen topics" rather than "fits this topic set". The topic list is frozen and
shared across every cell, so topic cannot separate cells by construction;
grouping guards against per-topic idiosyncrasy instead.

Answers the two questions the logit lens could not:

1. **Which layer?** The lens has a depth bias pushing one way while ring
   distinctions collapse the other, so it cannot arbitrate. Ring separability
   (within-axis: serenity/joy/ecstasy) is the binding constraint, and this
   reports it per layer.
2. **Are the cells actually distinct?** Pairs that never separate at any layer
   are cells the wheel posits but the model does not represent apart.

Unlike derive, this is **pairwise**, so a partial cell set is not a footgun:
no grand mean is involved and every pair is independent of set membership. A
short set is still reported loudly.

Outputs to ``results/wheel_discriminability/<slug>/``:
    accuracy.npz       (n_layers, C, C) float32 + cell names + layer names
    layer_summary.csv  the aggregate curves
    report.md          recommended layer, per-axis detail, worst pairs

No GPU, no model weights, no authored stimuli.

Example::

    uv run python scripts/wheel_discriminability.py --model Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

_repo_root = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Wheel spec
# ---------------------------------------------------------------------------

def load_spec(spec_path: Path):
    spec = yaml.safe_load(spec_path.read_text())
    ring_order = spec.get("ring_order", ["high", "middle", "low"])
    cells: dict[str, dict] = {}
    for ax in spec["axes"]:
        for ring in ring_order:
            cells[ax["rings"][ring]] = {
                "kind": "ring",
                "axis": ax["name"],
                "ring": ring,
                "opposite_axis": ax["opposite"],
            }
    for d in spec["dyads"]:
        cells[d["name"]] = {
            "kind": "dyad",
            "axis": None,
            "ring": None,
            "components": list(d["components"]),
        }
    return spec, cells, spec["neutral"]["name"], ring_order


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def topic_folds(names, act_dir, story_dir, n_folds):
    """Fold id per row, grouped by story topic. None if unavailable.

    Story ids are ``<emotion>_<topic_idx>_<sample_idx>`` (see
    generate_emotion_stories.py), and ``<emotion>.meta.parquet`` lists them in
    npz row order. So the topic index comes straight off the id -- no need to
    read the corpora, and nothing to mismatch. The topic list is frozen and
    shared across cells, so topic_idx means the same thing everywhere.
    """
    try:
        import pandas as pd
    except ImportError:
        print("NOTE: pandas unavailable — ungrouped folds", file=sys.stderr)
        return None

    per_cell: dict[str, list[str]] = {}
    for nm in names:
        meta = act_dir / f"{nm}.meta.parquet"
        if not meta.exists():
            print(f"NOTE: no {meta.name} — ungrouped folds", file=sys.stderr)
            return None
        try:
            df = pd.read_parquet(meta)
        except Exception as exc:
            print(f"NOTE: could not read {meta.name} ({exc}) — ungrouped folds",
                  file=sys.stderr)
            return None
        col = "story_id" if "story_id" in df.columns else (
            "id" if "id" in df.columns else None)
        if col is None:
            print(f"NOTE: {meta.name} has no id column (has: "
                  f"{', '.join(df.columns)}) — ungrouped folds", file=sys.stderr)
            return None
        topics = []
        for sid in df[col].astype(str):
            parts = sid.rsplit("_", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                print(f"NOTE: unparseable story id {sid!r} in {meta.name} — "
                      "ungrouped folds", file=sys.stderr)
                return None
            topics.append(int(parts[1]))
        per_cell[nm] = topics

    all_topics = sorted({t for ts in per_cell.values() for t in ts})
    topic_fold = {t: i % n_folds for i, t in enumerate(all_topics)}
    folds = {nm: np.array([topic_fold[t] for t in ts], dtype=np.int32)
             for nm, ts in per_cell.items()}
    return folds, len(all_topics)


def index_folds(names, counts, n_folds):
    return {nm: np.arange(counts[nm], dtype=np.int32) % n_folds for nm in names}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def project_out_neutral_pcs(X: dict, neutral: str, k: int):
    """Remove the top-k neutral principal directions from every cell."""
    N = X[neutral]
    Nc = N - N.mean(0, keepdims=True)
    # right singular vectors of the centred neutral cloud
    _, _, Vt = np.linalg.svd(Nc, full_matrices=False)
    B = Vt[:k]                                    # (k, d)
    return {nm: A - (A @ B.T) @ B for nm, A in X.items()}


def layer_matrix(X: dict, names: list[str], folds: dict, n_folds: int):
    C = len(names)
    correct = np.zeros((C, C), np.float64)
    total = np.zeros((C, C), np.float64)

    for f in range(n_folds):
        M = np.empty((C, X[names[0]].shape[1]), np.float32)
        for ci, nm in enumerate(names):
            tr = folds[nm] != f
            if tr.sum() < 2:
                return None
            M[ci] = X[nm][tr].mean(0)
        Msq = (M ** 2).sum(1)

        for ci, nm in enumerate(names):
            te = folds[nm] == f
            n_te = int(te.sum())
            if n_te == 0:
                continue
            Xt = X[nm][te]
            # squared distance from each test row to each centroid
            D = (Xt ** 2).sum(1, keepdims=True) - 2.0 * (Xt @ M.T) + Msq[None, :]
            # for pair (ci, cj): correct iff closer to own centroid
            correct[ci] += (D[:, ci][:, None] < D).sum(0)
            total[ci] += n_te

    denom = total + total.T
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (correct + correct.T) / denom
    np.fill_diagonal(A, np.nan)
    return A.astype(np.float32)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def pair_groups(names, cells, neutral):
    idx = {nm: i for i, nm in enumerate(names)}
    emo = [nm for nm in names if nm != neutral]
    g = {k: [] for k in ("overall", "within_axis", "between_axis",
                         "opposite_axis", "dyad_vs_component", "vs_neutral")}
    for a in range(len(emo)):
        for b in range(a + 1, len(emo)):
            na, nb = emo[a], emo[b]
            ia, ib = idx[na], idx[nb]
            ca, cb = cells[na], cells[nb]
            g["overall"].append((ia, ib))
            if ca["kind"] == "ring" and cb["kind"] == "ring":
                if ca["axis"] == cb["axis"]:
                    g["within_axis"].append((ia, ib))
                else:
                    g["between_axis"].append((ia, ib))
                    if ca.get("opposite_axis") == cb["axis"]:
                        g["opposite_axis"].append((ia, ib))
            if ca["kind"] == "dyad" and cb["kind"] == "ring" and cb["ring"] == "middle" \
                    and cb["axis"] in ca["components"]:
                g["dyad_vs_component"].append((ia, ib))
            if cb["kind"] == "dyad" and ca["kind"] == "ring" and ca["ring"] == "middle" \
                    and ca["axis"] in cb["components"]:
                g["dyad_vs_component"].append((ia, ib))
    if neutral in idx:
        for nm in emo:
            g["vs_neutral"].append((idx[nm], idx[neutral]))
    return g


def group_mean(A, pairs):
    if not pairs:
        return float("nan")
    return float(np.nanmean([A[i, j] for i, j in pairs]))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", dest="model_key", required=True,
                    help="base model key, e.g. Llama-3.1-8B-Instruct")
    ap.add_argument("--track", default="story-wheel32")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--spec", default="configs/wheel.yaml")
    ap.add_argument("--activations-dir", default="activations")
    ap.add_argument("--out-dir", default="results/wheel_discriminability")
    ap.add_argument("--project-out-neutral-pcs", type=int, default=0, metavar="K",
                    help="remove top-K neutral PCs first, mirroring the vector "
                         "construction (default 0 = raw activations)")
    ap.add_argument("--worst", type=int, default=15, help="how many worst pairs to list")
    args = ap.parse_args()

    slug = f"{args.model_key}-{args.track}"
    act_dir = _repo_root / args.activations_dir / slug
    story_dir = _repo_root / "data" / "derived" / "stories" / args.model_key / args.track
    out_dir = _repo_root / args.out_dir / slug
    if not act_dir.is_dir():
        print(f"ERROR: no activations at {act_dir}", file=sys.stderr)
        return 1

    spec, cells, neutral, ring_order = load_spec(_repo_root / args.spec)
    expected = sorted(list(cells) + [neutral])
    present = sorted(p.stem for p in act_dir.glob("*.npz"))
    names = [n for n in expected if n in present]
    missing = [n for n in expected if n not in present]
    extra = [n for n in present if n not in expected]
    if missing:
        print(f"WARNING: {len(missing)} cell(s) absent from {act_dir}: {' '.join(missing)}",
              file=sys.stderr)
        print("         Pairwise accuracy is unaffected by set membership, but the "
              "aggregates below cover only what is present.", file=sys.stderr)
    if extra:
        print(f"WARNING: ignoring non-spec .npz: {' '.join(extra)}", file=sys.stderr)
    if len(names) < 2:
        print("ERROR: need at least two cells", file=sys.stderr)
        return 1

    handles = {nm: np.load(act_dir / f"{nm}.npz") for nm in names}
    layers = sorted(handles[names[0]].files, key=lambda s: int(s.split("_")[1]))
    for nm in names:
        if sorted(handles[nm].files, key=lambda s: int(s.split("_")[1])) != layers:
            print(f"ERROR: layer set differs in {nm}.npz", file=sys.stderr)
            return 1
    counts = {nm: handles[nm][layers[0]].shape[0] for nm in names}

    tf = topic_folds(names, act_dir, story_dir, args.folds)
    if tf is None:
        folds, n_topics, grouping = index_folds(names, counts, args.folds), None, "ungrouped"
    else:
        folds, n_topics = tf
        grouping = f"grouped by topic ({n_topics} topics)"

    print(f"{slug}: {len(names)} cells, {min(counts.values())}-{max(counts.values())} "
          f"stories/cell, {len(layers)} layers, {args.folds}-fold {grouping}")

    nonfinite: dict[str, dict[str, int]] = {}
    mats = np.empty((len(layers), len(names), len(names)), np.float32)
    for li, layer in enumerate(layers):
        X = {nm: handles[nm][layer].astype(np.float32) for nm in names}
        # Activations are stored float16 (extract_story_activations.py:208) while
        # pooling happens in float32, so any pooled value above 65504 lands as
        # inf. Such a story is unusable: its class mean, and after project_out
        # every coordinate of the vector, would be inf. Drop the row, loudly.
        lf = {}
        for nm in names:
            good = np.isfinite(X[nm]).all(axis=1)
            n_bad = int((~good).size - good.sum())
            if n_bad:
                nonfinite.setdefault(nm, {})[layer] = n_bad
                X[nm] = X[nm][good]
                lf[nm] = folds[nm][good]
            else:
                lf[nm] = folds[nm]
        if any(len(X[nm]) < args.folds for nm in names):
            print(f"ERROR: {layer} has a cell with fewer usable stories than folds",
                  file=sys.stderr)
            return 1
        if args.project_out_neutral_pcs and neutral in X:
            X = project_out_neutral_pcs(X, neutral, args.project_out_neutral_pcs)
        A = layer_matrix(X, names, lf, args.folds)
        if A is None:
            print(f"ERROR: too few stories for {args.folds} folds", file=sys.stderr)
            return 1
        mats[li] = A
        del X
        print(f"  {layer}: done", flush=True)

    groups = pair_groups(names, cells, neutral)
    keys = ["overall", "within_axis", "between_axis", "opposite_axis",
            "dyad_vs_component", "vs_neutral"]
    rows = [{"layer": l, **{k: group_mean(mats[i], groups[k]) for k in keys}}
            for i, l in enumerate(layers)]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "accuracy.npz", accuracy=mats,
                        cells=np.array(names), layers=np.array(layers))
    with (out_dir / "layer_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer"] + keys)
        w.writeheader()
        w.writerows(rows)

    # Recommended layer: ring separability is the binding constraint.
    wa = np.array([r["within_axis"] for r in rows])
    ov = np.array([r["overall"] for r in rows])
    best_overall = int(np.nanargmax(ov))
    have_rings = bool(groups["within_axis"]) and not np.all(np.isnan(wa))
    best_ring = int(np.nanargmax(wa)) if have_rings else best_overall

    L = []
    L.append(f"# Wheel discriminability — {slug}\n")
    L.append(f"- cells: {len(names)}"
             + (f" (**{len(missing)} missing**: {' '.join(missing)})" if missing else ""))
    L.append(f"- stories/cell: {min(counts.values())}–{max(counts.values())}")
    L.append(f"- CV: {args.folds}-fold, {grouping}")
    inp = ("raw activations" if not args.project_out_neutral_pcs
           else f"top-{args.project_out_neutral_pcs} neutral PCs projected out")
    L.append(f"- input: {inp}")
    L.append("\nCross-validated pairwise nearest-centroid accuracy; 0.50 = chance.\n")
    L.append("| layer | overall | within-axis (rings) | between-axis | opposite axes | dyad vs component | vs neutral |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        mark = " **<-- ring peak**" if i == best_ring else ""
        L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |{}".format(
            r["layer"], r["overall"], r["within_axis"], r["between_axis"],
            r["opposite_axis"], r["dyad_vs_component"], r["vs_neutral"], mark))

    if have_rings:
        L.append(f"\n**Ring separability peaks at `{layers[best_ring]}` "
                 f"({wa[best_ring]:.3f}); overall peaks at `{layers[best_overall]}` "
                 f"({ov[best_overall]:.3f}).**")
        L.append("\nWithin-axis is the binding constraint: between-axis separability "
                 "saturates early, so a layer chosen on overall accuracy will look fine "
                 "while the rings have already collapsed.\n")
    else:
        L.append(f"\n**No within-axis (ring) pairs in this cell set — recommending on "
                 f"overall accuracy instead: `{layers[best_overall]}` "
                 f"({ov[best_overall]:.3f}). Not a layer choice for the full wheel.**\n")

    if nonfinite:
        total_bad = sum(sum(d.values()) for d in nonfinite.values())
        L.append("## Non-finite activations (dropped)\n")
        L.append(f"{total_bad} (cell, layer, story) entries held `inf` and were "
                 "dropped. Activations are pooled in float32 but stored float16 "
                 "(`extract_story_activations.py:208`), so any pooled value above "
                 "65504 saturates. `derive_story_steering_vectors.py` has no finite "
                 "guard: an `inf` story poisons its class mean, and since the grand "
                 "mean is shared, every vector in the track.\n")
        L.append("| cell | layers affected | worst layer (rows dropped) |")
        L.append("|---|---|---|")
        for nm, d in sorted(nonfinite.items()):
            wl = max(d, key=lambda k: d[k])
            L.append(f"| {nm} | {len(d)} | {wl} ({d[wl]}) |")
        L.append("")

    A = mats[best_ring]
    L.append(f"## Per-axis ring separability at `{layers[best_ring]}`\n")
    L.append("| axis | high vs middle | middle vs low | high vs low |")
    L.append("|---|---|---|---|")
    idx = {nm: i for i, nm in enumerate(names)}
    for ax in spec["axes"]:
        r = ax["rings"]
        def acc(a, b):
            if r[a] in idx and r[b] in idx:
                return "{:.3f}".format(A[idx[r[a]], idx[r[b]]])
            return "—"
        L.append(f"| {ax['name']} | {acc('high','middle')} | {acc('middle','low')} "
                 f"| {acc('high','low')} |")

    flat = [(A[i, j], names[i], names[j]) for i, j in groups["overall"]]
    flat.sort()
    L.append(f"\n## Least separable pairs at `{layers[best_ring]}`\n")
    L.append("| pair | accuracy |")
    L.append("|---|---|")
    for v, a, b in flat[:args.worst]:
        L.append(f"| {a} / {b} | {v:.3f} |")

    never = [(float(np.nanmax(mats[:, i, j])), names[i], names[j])
             for i, j in groups["overall"]]
    never = [t for t in never if t[0] < 0.6]
    L.append(f"\n## Pairs below 0.60 at **every** layer ({len(never)})\n")
    if never:
        never.sort()
        for v, a, b in never:
            L.append(f"- {a} / {b} (best across layers: {v:.3f})")
        L.append("\nThese are cells the wheel posits but this model does not represent "
                 "separately. That is a finding, not a bug — but nothing downstream "
                 "should treat them as distinct conditions.")
    else:
        L.append("None — every pair clears 0.60 at some layer.")

    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_dir}/report.md")
    print(f"ring peak: {layers[best_ring]} ({wa[best_ring]:.3f})   "
          f"overall peak: {layers[best_overall]} ({ov[best_overall]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
