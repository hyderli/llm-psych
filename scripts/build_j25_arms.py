"""J25 pilot arm builder: the emotion-selection discriminator, Qwen 2.5 7B L16.

Pre-registered in plans/j-space-decomposition.md ("J25", 2026-11-07) before any
arm exists. J24 asked whether a coherent hostile-word push acts; J25 asks
whether it acts because of hostility in general or because of the hostile
words the emotion vector v itself promotes. New families, same construction
as J24 (cos(theta*) * v_hat + sin(theta*) * w_hat, w_hat perp to v, 44 centred
English-word atoms per draw, equal weights):

  gen<i>    w_hat = equal-weight sum of atoms for hostile-lexicon English-word
            tokens NOT among v's top-2048 lens candidates: generic hostility
            that v does not promote. The lexical-vs-emotion discriminator.
  coh<i>    w_hat = atoms for the top-2048 English-word lens candidates of
            u_coh = admiration=1 + joy=1 (non-hostile primary pair), hostile
            tokens and v's candidate-pool tokens excluded: a coherent
            emotion-word push v does not promote.
  top<i>    w_hat = atoms for English-word tokens in the fixed Swadesh
            concrete-noun list (data/public/swadesh_nouns.txt), hostile
            excluded: a coherent non-emotional push.
  w90       j24_pool1 minus its cos(theta*)*v_hat component: sin(theta*)*w_hat
            kept at its in-arm magnitude, NOT rescaled (a deletion). Does a
            content-bearing w_hat need the v-component? Descriptive, one arm.
  w90ctl    the identical deletion applied to j24_span1: norm- and
            geometry-matched contentless control for w90.
  (refs)    pool1-6 are the J24 files reused unchanged; jspace and full are
            rebuilt only to recheck them against the J23 arms.

Eligible-count fallback (fixed in the plan): eligible >= 264 -> six disjoint
draws; 44 <= eligible < 264 -> six draws of 44 distinct atoms each, sets may
overlap across draws (one family tested six ways, recorded as such); < 44 ->
REFUSE, no arm files. Seeds: SeedSequence(20261107), spawn order gen, coh,
top; pool files are reused so consume no RNG.

Self-push generalises J24's hostile_push: the mean vocabulary-centred J-lens
logit over a class's own atom tokens at unit norm. Build acceptance (fixed):
angles at theta*, w perp v, coh/top within-class w_hat cos >= 0.5, each new
class's mean self-push >= 0.5 x pool's mean hostile push and >= 3 x span's
J24 mean (+0.065 -> >= 0.195), gen's mean hostile push >= 0.5 x pool's, and
the J23 full/jspace rebuild check. On failure nothing is written.

    uv run python scripts/build_j25_arms.py --dry-run     # report only
    uv run python scripts/build_j25_arms.py               # report + .npy arms
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

# --- frozen by the J25 section of plans/j-space-decomposition.md -------------
SEED = 20261107
FAMILIES = ("gen", "coh", "top")        # spawn order is part of the seed spec
N_PER_CLASS = 6
ATOMS_PER_DRAW = 44
EXPECTED_THETA = 74.836                 # J23 build report, pos side, L16
THETA_TOL = 0.01
TAG = "j25"
MIN_WITHIN_COS = 0.5                    # coh, top: as coherent as pool-family draws
SPAN_J24_MEAN_PUSH = 0.065              # J24 report, class mean hostile push of span
PUSH_FRACTION = 0.5                     # new classes' self-push vs pool's hostile push
GEN_PUSH_FRACTION = 0.5                 # gen's hostile push vs pool's
CHUNK = 8192
ENGLISH_RE = re.compile(r" [A-Za-z]{3,}")
MIN_PICKED_MATCH = 0.999
SWADESH_PATH = _repo_root / "data" / "public" / "swadesh_nouns.txt"
# -----------------------------------------------------------------------------


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _repo_root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", type=Path,
                   default=_repo_root / "configs" / "model" / "qwen25_7b.yaml")
    p.add_argument("--track", default="story-wheel32")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--mix", nargs="+", default=["contempt=1", "aggressiveness=1"])
    p.add_argument("--coh-mix", nargs="+", default=["admiration=1", "joy=1"])
    p.add_argument("--n-candidates", type=int, default=2048)
    p.add_argument("--lens-source", default="neuronpedia", choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--swadesh", type=Path, default=SWADESH_PATH)
    p.add_argument("--dry-run", action="store_true", help="write the report, no .npy")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    mod = _load("dec", "scripts/decompose_emotion_vectors.py")
    sc = _load("sc", "scripts/score_j22_diagnostic.py")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS

    swadesh = frozenset(
        ln.strip() for ln in args.swadesh.read_text().splitlines()
        if ln.strip() and not ln.startswith("#"))
    if len(swadesh) < 50:
        raise SystemExit(f"REFUSING: only {len(swadesh)} Swadesh nouns in {args.swadesh}")

    cfg = mod._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = _repo_root / "steering_vectors" / f"{model_key}-{args.track}"

    def mixture(specs: list[str]) -> np.ndarray:
        m = None
        for spec in specs:
            name, _, c = spec.partition("=")
            v = np.load(vec_dir / f"{name}_layer{args.layer}.npy").astype(np.float64)
            v = float(c) * v / np.linalg.norm(v)
            m = v if m is None else m + v
        return m

    # --- mixtures, exactly as the J24 build ----------------------------------
    mix = mixture(args.mix)
    target_norm = float(np.linalg.norm(mix))
    u = mix.astype(np.float32)
    uu = torch.from_numpy(u)
    vhat = uu / uu.norm()
    u_coh = torch.from_numpy(mixture(args.coh_mix).astype(np.float32))

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
        k=64, n_candidates=args.n_candidates, return_atoms=True)
    comp_t = torch.from_numpy(comp).float()
    cos_star = float(comp_t @ vhat / comp_t.norm())
    theta = float(np.degrees(np.arccos(np.clip(cos_star, -1, 1))))
    if abs(theta - EXPECTED_THETA) > THETA_TOL:
        raise SystemExit(f"REFUSING: theta* = {theta:.4f}, expected {EXPECTED_THETA} "
                         f"+- {THETA_TOL}; the decomposition does not reproduce J23's")
    c_s, s_s = float(np.cos(np.radians(theta))), float(np.sin(np.radians(theta)))

    def lens_logits(x: torch.Tensor) -> torch.Tensor:
        z = (x.float() @ j_f.T)
        return ((z.to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T).float()

    vocab = w_u.shape[0]
    print(f"building hostility/English masks over {vocab} tokens ...")
    hostile = torch.zeros(vocab, dtype=torch.bool)
    english = torch.zeros(vocab, dtype=torch.bool)
    swadesh_mask = torch.zeros(vocab, dtype=torch.bool)
    for t in range(vocab):
        try:
            s = tokenizer.decode([t])
        except Exception:
            continue
        if sc.lexicon_hits(sc.words(s), lex)[0] > 0:
            hostile[t] = True
        if t < len(tokenizer) and ENGLISH_RE.fullmatch(s):
            english[t] = True
            if s.strip().lower() in swadesh:
                swadesh_mask[t] = True
    n_hostile, n_english = int(hostile.sum()), int(english.sum())
    print(f"hostile tokens: {n_hostile}   English-word tokens: {n_english}   "
          f"Swadesh-noun tokens: {int(swadesh_mask.sum())}")

    def unit_atoms(tids: torch.Tensor) -> torch.Tensor:
        a = (w_u[tids].float() * g.float()) @ j_f
        return a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)

    eng_ids = torch.nonzero(english).flatten()
    print(f"computing the English-word mean unit atom over {eng_ids.numel()} tokens ...")
    centre = torch.zeros(j_f.shape[1])
    for lo in range(0, eng_ids.numel(), CHUNK):
        centre += unit_atoms(eng_ids[lo:lo + CHUNK]).sum(0)
    centre /= eng_ids.numel()

    def push(x: torch.Tensor, tids: torch.Tensor | None = None) -> dict:
        z = lens_logits(x / x.norm())
        zc = z - z.mean()
        out = {"hostile_push": float(zc[hostile].mean()),
               "top10": [tokenizer.decode([t]) for t in torch.topk(zc, 10).indices.tolist()]}
        if tids is not None:
            out["self_push"] = float(zc[tids].mean())
        return out

    rngs = {f: np.random.default_rng(ch)
            for f, ch in zip(FAMILIES, np.random.SeedSequence(SEED).spawn(len(FAMILIES)))}
    arms: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    draw_tids: dict[str, torch.Tensor] = {}

    def from_w(name: str, w: torch.Tensor, extra: dict) -> None:
        w = w - (w @ vhat) * vhat
        w = w / w.norm()
        if abs(float(w @ vhat)) > 1e-5:
            raise SystemExit(f"REFUSING {name}: w not perpendicular to v")
        arms[name] = c_s * vhat + s_s * w
        meta[name] = extra

    # --- eligible sets --------------------------------------------------------
    logits_v = lens_logits(uu)
    cand_ids = torch.topk(logits_v, args.n_candidates).indices
    in_cand = torch.zeros(vocab, dtype=torch.bool)
    in_cand[cand_ids] = True

    # gen: hostile English-word tokens v does not promote
    gen_elig = torch.nonzero(hostile & english & ~in_cand).flatten()
    # coh: u_coh's top-2048 English-word candidates, hostile and v-candidate
    # tokens excluded
    logits_coh = lens_logits(u_coh)
    coh_cand = torch.zeros(vocab, dtype=torch.bool)
    coh_cand[torch.topk(logits_coh, args.n_candidates).indices] = True
    coh_elig = torch.nonzero(coh_cand & english & ~hostile & ~in_cand).flatten()
    # top: Swadesh concrete nouns, hostile excluded
    top_elig = torch.nonzero(swadesh_mask & english & ~hostile).flatten()
    eligible = {"gen": gen_elig, "coh": coh_elig, "top": top_elig}
    for fam, elig in eligible.items():
        print(f"{fam}: {elig.numel()} eligible tokens")
        if elig.numel() < ATOMS_PER_DRAW:
            raise SystemExit(f"REFUSING: only {elig.numel()} {fam}-eligible tokens, "
                             f"need {ATOMS_PER_DRAW}")

    # --- draws under the overlap fallback ------------------------------------
    overlap: dict[str, bool] = {}
    for fam in FAMILIES:
        elig = eligible[fam]
        overlap[fam] = elig.numel() < N_PER_CLASS * ATOMS_PER_DRAW
        if overlap[fam]:            # sets may overlap across draws
            draws = [rngs[fam].choice(elig.numel(), size=ATOMS_PER_DRAW,
                                      replace=False)
                     for _ in range(N_PER_CLASS)]
        else:                       # one disjoint 6 x 44 draw, as J24
            flat = rngs[fam].choice(elig.numel(), size=N_PER_CLASS * ATOMS_PER_DRAW,
                                    replace=False)
            draws = [flat[i * ATOMS_PER_DRAW:(i + 1) * ATOMS_PER_DRAW]
                     for i in range(N_PER_CLASS)]
        for i, idx in enumerate(draws):
            tids = elig[torch.from_numpy(idx)]
            name = f"{fam}{i + 1}"
            draw_tids[name] = tids
            from_w(name, (unit_atoms(tids) - centre).sum(0),
                   {"n_atoms": int(tids.numel()), "token_ids": tids.tolist(),
                    "overlapping_draws": overlap[fam]})

    # --- w90 and its norm-matched control: deletions of existing J24 arms ----
    for nm, src in (("w90", "j24_pool1"), ("w90ctl", "j24_span1")):
        f = vec_dir / f"{src}_layer{args.layer}.npy"
        if not f.exists():
            raise SystemExit(f"REFUSING: {f} is missing; {nm} cannot be built")
        a = torch.from_numpy(np.load(f).astype(np.float32))
        arms[nm] = a - (a @ vhat) * vhat          # keeps sin(theta*) magnitude
        meta[nm] = {"deletion_of": src, "n_atoms": 0}

    # --- rebuilt refs and the J23 check --------------------------------------
    arms["full"] = uu.clone()
    arms["jspace"] = comp_t.clone()
    report = {"spec": "J25 pilot, plans/j-space-decomposition.md 2026-11-07",
              "model_key": model_key, "track": args.track, "layer": args.layer,
              "seed": SEED, "families": FAMILIES, "theta_star_deg": theta,
              "atoms_per_draw": ATOMS_PER_DRAW, "swadesh_list": str(args.swadesh),
              "n_swadesh_nouns": len(swadesh),
              "centring": "English-word mean unit atom (J24 amendment 2)",
              "n_hostile_tokens": n_hostile, "n_english_tokens": n_english,
              "n_swadesh_tokens": int(swadesh_mask.sum()),
              "n_eligible": {f: int(eligible[f].numel()) for f in FAMILIES},
              "overlapping_draws": overlap,
              "target_norm": target_norm,
              "lens": {"layer_used": mapped, "provenance": lens_prov},
              "j23_match": {}, "arms": {}, "within_class_cos": {}}
    for nm in ("full", "jspace"):
        f = vec_dir / f"ca_unit_pos_{nm}_layer{args.layer}.npy"
        if not f.exists():
            raise SystemExit(f"REFUSING: {f} is missing, so the J23 check cannot run.")
        old = torch.from_numpy(np.load(f).astype(np.float32))
        cs = float(old @ arms[nm] / (old.norm() * arms[nm].norm()))
        report["j23_match"][nm] = cs
        if cs < 0.999999:
            raise SystemExit(f"REFUSING: rebuilt {nm} has cos {cs} with the J23 arm")

    # --- reused pool arms: loaded for push matching and cosines --------------
    pool_arms: dict[str, torch.Tensor] = {}
    pool_tids: dict[str, torch.Tensor] = {}
    j24_report = (_repo_root / "results" / "jspace_gate"
                  / f"arms_{model_key}_{args.track}_L{args.layer}_j24.json")
    for i in range(N_PER_CLASS):
        f = vec_dir / f"j24_pool{i + 1}_layer{args.layer}.npy"
        if not f.exists():
            raise SystemExit(f"REFUSING: {f} is missing; the pool anchor cannot be reused")
        pool_arms[f"pool{i + 1}"] = torch.from_numpy(np.load(f).astype(np.float32))
    if j24_report.exists():
        rep = json.loads(j24_report.read_text())
        for i in range(N_PER_CLASS):
            nm = f"pool{i + 1}"
            if nm in rep.get("arms", {}) and "token_ids" in rep["arms"][nm]:
                pool_tids[nm] = torch.tensor(rep["arms"][nm]["token_ids"])

    def what(a: torch.Tensor) -> torch.Tensor:
        w = a - (a @ vhat) * vhat
        return w / w.norm()

    for nm, a in arms.items():
        rec = {"angle_deg": float(np.degrees(np.arccos(
            np.clip(float((a / a.norm()) @ vhat), -1, 1)))),
            "arm_norm": float(a.norm()), **meta.get(nm, {}),
            **push(a, draw_tids.get(nm))}
        report["arms"][nm] = rec
        if nm in ("w90", "w90ctl"):
            expected = 90.0
        elif nm == "full":
            expected = 0.0
        else:
            expected = theta
        if nm != "full" and abs(rec["angle_deg"] - expected) > 1e-3:
            raise SystemExit(f"REFUSING {nm}: realised angle {rec['angle_deg']:.5f} "
                             f"!= {expected}")
    for nm, a in pool_arms.items():
        report["arms"][nm] = {"reused_j24": True, "arm_norm": float(a.norm()),
                              "angle_deg": float(np.degrees(np.arccos(
                                  np.clip(float((a / a.norm()) @ vhat), -1, 1)))),
                              **push(a, pool_tids.get(nm))}

    for fam in FAMILIES:
        ws = [what(arms[f"{fam}{i + 1}"]) for i in range(N_PER_CLASS)]
        cs = [float(ws[a] @ ws[b]) for a in range(len(ws)) for b in range(a + 1, len(ws))]
        report["within_class_cos"][fam] = {"w_hat_mean": float(np.mean(cs)),
                                           "w_hat_min": float(np.min(cs)),
                                           "w_hat_max": float(np.max(cs))}
    pool_ws = [what(pool_arms[f"pool{i + 1}"]) for i in range(N_PER_CLASS)]
    report["cross_class_cos"] = {
        fam: {"w_hat_mean_vs_pool": float(np.mean(
            [float(what(arms[f"{fam}{i + 1}"]) @ pw)
             for i in range(N_PER_CLASS) for pw in pool_ws]))}
        for fam in FAMILIES}

    # --- acceptance, fixed in the plan ----------------------------------------
    mean_hp = {fam: float(np.mean([report["arms"][f"{fam}{i + 1}"]["hostile_push"]
                                   for i in range(N_PER_CLASS)])) for fam in FAMILIES}
    mean_sp = {fam: float(np.mean([report["arms"][f"{fam}{i + 1}"]["self_push"]
                                   for i in range(N_PER_CLASS)])) for fam in FAMILIES}
    pool_hp = float(np.mean([report["arms"][f"pool{i + 1}"]["hostile_push"]
                             for i in range(N_PER_CLASS)]))
    push_floor = max(PUSH_FRACTION * pool_hp, 3.0 * SPAN_J24_MEAN_PUSH)
    checks = {
        "coh_within_cos_ge_0.5": report["within_class_cos"]["coh"]["w_hat_mean"] >= MIN_WITHIN_COS,
        "top_within_cos_ge_0.5": report["within_class_cos"]["top"]["w_hat_mean"] >= MIN_WITHIN_COS,
        "gen_hostile_push_ge_half_pool": mean_hp["gen"] >= GEN_PUSH_FRACTION * pool_hp,
        **{f"{fam}_self_push_ge_floor": mean_sp[fam] >= push_floor for fam in FAMILIES},
    }
    report["class_mean_hostile_push"] = {**mean_hp, "pool": pool_hp}
    report["class_mean_self_push"] = mean_sp
    report["self_push_floor"] = push_floor
    report["acceptance"] = checks
    accepted = all(checks.values())

    rp = (_repo_root / "results" / "jspace_gate"
          / f"arms_{model_key}_{args.track}_L{args.layer}_{TAG}.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for nm, r in report["arms"].items():
        print(f"{nm:8s} angle {r['angle_deg']:7.3f}  hostile_push {r['hostile_push']:+.3f}  "
              f"self_push {r.get('self_push', float('nan')):+.3f}  "
              f"top: {' '.join(repr(t) for t in r['top10'][:5])}")
    print("acceptance:", json.dumps(checks), "->", "PASS" if accepted else "FAIL")
    print(f"report -> {rp}")

    if not accepted:
        raise SystemExit("ACCEPTANCE FAILED (J25 spec): no arm files written. "
                         "Report back; do not adjust the construction.")
    if args.dry_run:
        print("dry run: no arm files written")
        return 0
    for nm, a in arms.items():
        if nm in ("full", "jspace"):
            continue                     # J24's files already exist
        out = vec_dir / f"{TAG}_{nm}_layer{args.layer}.npy"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"REFUSING to overwrite {out}")
        if nm in ("w90", "w90ctl"):
            np.save(out, a.numpy().astype(np.float32))      # deletion norm, not rescaled
        else:
            np.save(out, (a / a.norm() * target_norm).numpy().astype(np.float32))
    print(f"arm files -> {vec_dir} (pool1-6, jspace, full reused from j24/j23)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
