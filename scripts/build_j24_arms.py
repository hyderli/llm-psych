"""J24 arm builder: matched-angle construction classes at Qwen 2.5 7B L16.

Pre-registered in plans/j-space-decomposition.md ("J24"). Every arm except
``full`` sits at exactly the ``jspace`` angle theta* (74.84 deg at L16, pos side)
and has the form

    arm = cos(theta*) * v_hat + sin(theta*) * w_hat,     w_hat perpendicular to v

so the arms share their v-component exactly and differ only in w_hat. The
classes differ only in where w_hat comes from:

  pool<i>  w_hat = perp-to-v part of an EQUAL-WEIGHT sum of 44 CENTRED unit J-lens
           atoms (amendment 2: each atom minus the mean unit atom of ENGLISH-WORD
           tokens) drawn from the English-word tokens of the candidate pool (top-2048 tokens by v's own lens
           logit, jspace's picked atoms excluded): a push on words the emotion
           promotes. Six draws, atom sets disjoint across draws.
  far<i>   the same construction from the middle half of v's lens-logit
           ranking WITHIN English-word tokens, hostile tokens excluded
           (amendment 2): a push on ordinary English words unrelated to v. Six draws,
           disjoint. Equal weights, so unlike J23's NNLS faratom it cannot
           collapse to a few atoms; every draw has 44 active atoms.
  span<i>  w_hat = a Gaussian draw made perpendicular to v and to jspace's
           picked atoms: ladstar's construction, six independent draws.
  common   w_hat = perp-to-v part of the vocabulary-mean unit atom itself: the
           component every atom shares (amendment 1). One direction.
  jspace   the J-lens component itself (theta* by definition), and ``full`` (v),
           rebuilt and checked against the J23 arm files.

Equal weights are what make pool and far atom-matched. They also make pool
differ from J19's ``pin`` arms, which were random SIGNED directions inside the
atom span; a nonnegative sum is a push TOWARD the atoms' tokens.

Seeds: one SeedSequence(20261104) spawned into one child per family, so no
family's draws depend on another's (the J21 build note's defect, fixed here
under a new tag; existing ca_unit arms are untouched).

For each arm the report records the realised angle, active atom count, the
within-class pairwise cosines, and the HOSTILE PUSH: the mean vocabulary-centred
J-lens logit over tokens that hit the J23 hostility lexicon, at unit norm. The
push is computed from the arm alone, before any output exists, and feeds J24's
exploratory strength analysis.

Needs the lens, lm_head and final norm (HF downloads): run on the Mac with
``uv run`` or on a pod, from the repo root::

    uv run python scripts/build_j24_arms.py --dry-run     # report only
    uv run python scripts/build_j24_arms.py               # report + .npy arms
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parents[1]

# --- frozen by J24 -----------------------------------------------------------
SEED = 20261104
FAMILIES = ("pool", "far", "span")      # spawn order is part of the seed spec
N_PER_CLASS = 6
EXPECTED_THETA = 74.836                 # J23 build report, pos side, L16
THETA_TOL = 0.01                        # degrees
TAG = "j24"
# Build acceptance. Amendment 1 fixed these; amendment 2 drops the pool
# within-class criterion (reported, not required) and keeps the rest.
MAX_WITHIN_COS = 0.5        # mean pairwise w_hat cos within far
MAX_POOL_FAR_COS = 0.5      # mean pairwise w_hat cos between pool and far
CHUNK = 8192                # rows per chunk for mean-atom computations
# Amendment 2: an "English-word token" decodes to a leading space followed by
# 3 or more ASCII letters and nothing else -- a word-initial Latin token such
# as " the" or " brutal". The mandatory space excludes word-continuation
# fragments like "odyn", "ipop", "entai", which dominated the first dry run.
# A proxy, not a dictionary check.
ENGLISH_RE = re.compile(r" [A-Za-z]{3,}")
MIN_PICKED_MATCH = 0.999    # jspace's picked atoms must be found among the candidates
# -----------------------------------------------------------------------------


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _repo_root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _perp(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove from x everything lying in the row space of basis (as build_arm_vectors)."""
    q, _ = torch.linalg.qr(basis.T)
    return x - q @ (q.T @ x)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", type=Path,
                   default=_repo_root / "configs" / "model" / "qwen25_7b.yaml")
    p.add_argument("--track", default="story-wheel32")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--mix", nargs="+", default=["contempt=1", "aggressiveness=1"])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-candidates", type=int, default=2048)
    p.add_argument("--lens-source", default="neuronpedia", choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="write the report, no .npy")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    mod = _load("dec", "scripts/decompose_emotion_vectors.py")
    sc = _load("sc", "scripts/score_j22_diagnostic.py")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS

    cfg = mod._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = _repo_root / "steering_vectors" / f"{model_key}-{args.track}"

    # --- mixture, exactly as the J21/J23 ca_unit build (--unit-normalise) ----
    mix = None
    for spec in args.mix:
        name, _, c = spec.partition("=")
        v = np.load(vec_dir / f"{name}_layer{args.layer}.npy").astype(np.float64)
        v = float(c) * v / np.linalg.norm(v)
        mix = v if mix is None else mix + v
    target_norm = float(np.linalg.norm(mix))       # alpha = 1, as the ca_unit build
    u = mix.astype(np.float32)
    uu = torch.from_numpy(u)
    vhat = uu / uu.norm()

    print("loading tokenizer, unembedding shards and lens ...")
    tokenizer = mod._load_tokenizer(hf_id, revision=revision,
                                    trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = mod._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = mod._load_jlens(args.lens_source, hf_id, args.lens_path)
    mapped = mod._map_layer(args.layer, set(j.keys()) if j else None, "nearest")
    if mapped is None:
        raise SystemExit(f"layer {args.layer} not covered by the lens")
    j_l = j[mapped]
    j_f = j_l.float()

    comp, _resid, m, atoms, picked = mod._decompose_vector(
        u, sign=+1, j_l=j_l, w_u=w_u, g=g, tokenizer=tokenizer,
        k=args.k, n_candidates=args.n_candidates, return_atoms=True)
    comp_t = torch.from_numpy(comp).float()
    cos_star = float(comp_t @ vhat / comp_t.norm())
    theta = float(np.degrees(np.arccos(np.clip(cos_star, -1, 1))))
    if abs(theta - EXPECTED_THETA) > THETA_TOL:
        raise SystemExit(f"REFUSING: theta* = {theta:.4f}, expected {EXPECTED_THETA} "
                         f"+- {THETA_TOL}; the decomposition does not reproduce J23's")
    n_pick = int(m["n_atoms"])
    c_s, s_s = float(np.cos(np.radians(theta))), float(np.sin(np.radians(theta)))

    # --- lens readout used for the far pool and for the hostile push --------
    def lens_logits(x: torch.Tensor) -> torch.Tensor:
        z = (x.float() @ j_f.T)
        return ((z.to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T).float()

    vocab = w_u.shape[0]
    print(f"building hostility mask over {vocab} tokens ...")
    hostile = torch.zeros(vocab, dtype=torch.bool)
    english = torch.zeros(vocab, dtype=torch.bool)
    # lm_head has padding rows beyond the tokenizer's vocabulary (Qwen 2.5:
    # 152064 rows, ~151665 tokens). Those rows are never hostile; a decode
    # failure on them must not abort the build.
    for t in range(vocab):
        try:
            s = tokenizer.decode([t])
        except Exception:
            continue
        if sc.lexicon_hits(sc.words(s), lex)[0] > 0:
            hostile[t] = True
        if t < len(tokenizer) and ENGLISH_RE.fullmatch(s):
            english[t] = True
    n_hostile, n_english = int(hostile.sum()), int(english.sum())
    print(f"hostile tokens: {n_hostile}   English-word tokens: {n_english}")

    def unit_atoms(tids: torch.Tensor) -> torch.Tensor:
        a = (w_u[tids].float() * g.float()) @ j_f
        return a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)

    def mean_atom(tids: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(j_f.shape[1])
        for lo in range(0, tids.numel(), CHUNK):
            acc += unit_atoms(tids[lo:lo + CHUNK]).sum(0)
        return acc / tids.numel()

    # vocabulary-mean unit atom (amendment 1): mean over the tokenizer's real
    # vocabulary of normalize((W_U[t] * g) @ J), computed in chunks
    n_real = min(vocab, len(tokenizer))
    print(f"computing the vocabulary-mean unit atom over {n_real} tokens ...")
    abar = torch.zeros(j_f.shape[1])
    for lo in range(0, n_real, CHUNK):
        blk = (w_u[lo:min(lo + CHUNK, n_real)].float() * g.float()) @ j_f
        blk = blk / torch.clamp(blk.norm(dim=1, keepdim=True), min=1e-12)
        abar += blk.sum(0)
    abar /= n_real
    eng_ids = torch.nonzero(english).flatten()
    print(f"computing the English-word mean unit atom over {eng_ids.numel()} tokens ...")
    centre = mean_atom(eng_ids)

    def push(x: torch.Tensor) -> dict:
        z = lens_logits(x / x.norm())
        zc = z - z.mean()
        top = torch.topk(zc, 10).indices.tolist()
        return {"hostile_push": float(zc[hostile].mean()),
                "top10": [tokenizer.decode([t]) for t in top]}

    rngs = {f: np.random.default_rng(ch)
            for f, ch in zip(FAMILIES, np.random.SeedSequence(SEED).spawn(len(FAMILIES)))}
    arms: dict[str, torch.Tensor] = {"full": uu.clone(), "jspace": comp_t.clone()}
    meta: dict[str, dict] = {}

    def from_w(name: str, w: torch.Tensor, extra: dict) -> None:
        w = w - (w @ vhat) * vhat
        w = w / w.norm()
        if abs(float(w @ vhat)) > 1e-5:
            raise SystemExit(f"REFUSING {name}: w not perpendicular to v")
        arms[name] = c_s * vhat + s_s * w
        meta[name] = extra

    # pool (amendment 2): the candidate pool is v's top-n_candidates tokens by
    # lens logit; jspace's picked atoms are identified among them by cosine
    # and excluded; only English-word tokens are eligible.
    picked_set = set(int(i) for i in picked)
    logits_v = lens_logits(uu)
    cand_ids = torch.topk(logits_v, args.n_candidates).indices
    cand_atoms = unit_atoms(cand_ids)
    picked_atoms = atoms[sorted(picked_set)].float()
    sims = picked_atoms @ cand_atoms.T
    match_cos, match_idx = sims.max(dim=1)
    if float(match_cos.min()) < MIN_PICKED_MATCH:
        raise SystemExit(f"REFUSING: a picked jspace atom matches no candidate atom "
                         f"(min cos {float(match_cos.min()):.5f})")
    picked_tids = set(int(cand_ids[i]) for i in match_idx.tolist())
    pool_tids = torch.tensor([int(t) for t in cand_ids.tolist()
                              if english[t] and int(t) not in picked_tids])
    if pool_tids.numel() < N_PER_CLASS * n_pick:
        raise SystemExit(f"REFUSING: only {pool_tids.numel()} English-word candidates, "
                         f"need {N_PER_CLASS * n_pick} for disjoint pool draws")
    draw = rngs["pool"].choice(pool_tids.numel(), size=N_PER_CLASS * n_pick, replace=False)
    for i in range(N_PER_CLASS):
        tids = pool_tids[torch.from_numpy(draw[i * n_pick:(i + 1) * n_pick])]
        x = (unit_atoms(tids) - centre).sum(0)
        from_w(f"pool{i + 1}", x, {"n_atoms": int(tids.numel()), "token_ids": tids.tolist()})

    # far (amendment 2): middle half of v's ranking WITHIN English-word tokens,
    # hostile tokens excluded
    eng_sorted = eng_ids[torch.argsort(logits_v[eng_ids], descending=True)]
    mid = eng_sorted[eng_sorted.numel() // 4: 3 * eng_sorted.numel() // 4]
    mid = mid[~hostile[mid]]
    if mid.numel() < N_PER_CLASS * n_pick:
        raise SystemExit(f"REFUSING: only {mid.numel()} far-eligible tokens")
    draw = rngs["far"].choice(mid.numel(), size=N_PER_CLASS * n_pick, replace=False)
    for i in range(N_PER_CLASS):
        tids = mid[torch.from_numpy(draw[i * n_pick:(i + 1) * n_pick])]
        from_w(f"far{i + 1}", (unit_atoms(tids) - centre).sum(0),
               {"n_atoms": int(tids.numel()), "token_ids": tids.tolist()})

    # span: ladstar's construction, independent draws
    basis = torch.cat([uu.unsqueeze(0), atoms[sorted(picked_set)].float()], dim=0)
    for i in range(N_PER_CLASS):
        gw = torch.from_numpy(rngs["span"].normal(size=uu.shape).astype(np.float32))
        from_w(f"span{i + 1}", _perp(gw, basis), {"n_atoms": 0})

    # common: the shared component itself (deterministic, consumes no RNG)
    from_w("common", abar.clone(), {"n_atoms": n_real, "abar_norm": float(abar.norm())})

    # --- checks against J23 arm files ----------------------------------------
    report = {"amendment": "J24", "model_key": model_key, "track": args.track,
              "layer": args.layer, "seed": SEED, "families": FAMILIES,
              "theta_star_deg": theta, "n_atoms_jspace": n_pick,
              "centring": "English-word mean unit atom (amendment 2)",
              "n_hostile_tokens": n_hostile, "n_english_tokens": n_english,
              "n_pool_eligible": int(pool_tids.numel()), "n_far_eligible": int(mid.numel()),
              "picked_match_min_cos": float(match_cos.min()),
              "target_norm": target_norm, "lens": {"layer_used": mapped,
                                                   "provenance": lens_prov},
              "j23_match": {}, "arms": {}, "within_class_cos": {}}
    for nm in ("full", "jspace"):
        f = vec_dir / f"ca_unit_pos_{nm}_layer{args.layer}.npy"
        if f.exists():
            old = torch.from_numpy(np.load(f).astype(np.float32))
            cs = float(old @ arms[nm] / (old.norm() * arms[nm].norm()))
            report["j23_match"][nm] = cs
            if cs < 0.999999:
                raise SystemExit(f"REFUSING: rebuilt {nm} has cos {cs} with the J23 arm")
        else:
            raise SystemExit(f"REFUSING: {f} is missing, so the J23 check cannot run. "
                             "Pull steering_vectors for Qwen2.5-7B-Instruct-story-wheel32 first.")

    for nm, a in arms.items():
        ah = a / a.norm()
        report["arms"][nm] = {"angle_deg": float(np.degrees(np.arccos(
            np.clip(float(ah @ vhat), -1, 1)))), **meta.get(nm, {}), **push(a)}
        if nm not in ("full",) and abs(report["arms"][nm]["angle_deg"] - theta) > 1e-3:
            raise SystemExit(f"REFUSING {nm}: realised angle "
                             f"{report['arms'][nm]['angle_deg']:.5f} != theta*")
    def what(nm: str) -> torch.Tensor:
        a = arms[nm] / arms[nm].norm()
        w = a - (a @ vhat) * vhat
        return w / w.norm()

    W = {fam: [what(f"{fam}{i + 1}") for i in range(N_PER_CLASS)] for fam in FAMILIES}
    for fam in FAMILIES:
        ws = W[fam]
        cs = [float(ws[a] @ ws[b]) for a in range(len(ws)) for b in range(a + 1, len(ws))]
        report["within_class_cos"][fam] = {"w_hat_mean": float(np.mean(cs)),
                                           "w_hat_min": float(np.min(cs)),
                                           "w_hat_max": float(np.max(cs))}
    pf = [float(a @ b) for a in W["pool"] for b in W["far"]]
    report["pool_far_cos"] = {"w_hat_mean": float(np.mean(pf)), "w_hat_min": float(np.min(pf)),
                              "w_hat_max": float(np.max(pf))}
    wj, wc = what("jspace"), what("common")
    for nm in report["arms"]:
        if nm == "full":
            continue
        report["arms"][nm]["cos_w_jspace"] = float(what(nm) @ wj)
        report["arms"][nm]["cos_w_common"] = float(what(nm) @ wc)

    # --- amendment 1 acceptance ---------------------------------------------
    mp = {fam: float(np.mean([report["arms"][f"{fam}{i + 1}"]["hostile_push"]
                              for i in range(N_PER_CLASS)])) for fam in FAMILIES}
    checks = {
        "far_within_cos_le_0.5": report["within_class_cos"]["far"]["w_hat_mean"] <= MAX_WITHIN_COS,
        "pool_far_cos_le_0.5": report["pool_far_cos"]["w_hat_mean"] <= MAX_POOL_FAR_COS,
        "pool_push_gt_far": mp["pool"] > mp["far"],
        "pool_push_gt_span": mp["pool"] > mp["span"],
    }
    report["class_mean_hostile_push"] = mp
    report["acceptance"] = checks
    accepted = all(checks.values())

    rp = (_repo_root / "results" / "jspace_gate"
          / f"arms_{model_key}_{args.track}_L{args.layer}_{TAG}.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for nm, r in report["arms"].items():
        print(f"{nm:8s} angle {r['angle_deg']:7.3f}  hostile_push {r['hostile_push']:+.3f}  "
              f"cos_w_jspace {r.get('cos_w_jspace', float('nan')):+.3f}  "
              f"cos_w_common {r.get('cos_w_common', float('nan')):+.3f}  "
              f"top: {' '.join(repr(t) for t in r['top10'][:5])}")
    print("pool-far cos:", json.dumps(report["pool_far_cos"]))
    print("class mean hostile push:", json.dumps(mp))
    print("acceptance:", json.dumps(checks), "->", "PASS" if accepted else "FAIL")
    print(f"report -> {rp}")

    if not accepted:
        raise SystemExit("ACCEPTANCE FAILED (J24 amendment 2): no arm files written. "
                         "Report back; do not adjust the construction.")
    if args.dry_run:
        print("dry run: no arm files written")
        return 0
    for nm, a in arms.items():
        out = vec_dir / f"{TAG}_{nm}_layer{args.layer}.npy"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"REFUSING to overwrite {out}")
        np.save(out, (a / a.norm() * target_norm).numpy().astype(np.float32))
    print(f"{len(arms)} arm files -> {vec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
