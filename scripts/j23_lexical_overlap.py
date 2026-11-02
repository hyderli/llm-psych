"""J23 lexical-overlap check (exploratory): do the arms' promoted tokens surface?

Definition fixed in plans/j-space-decomposition.md ("J23 lexical-overlap check")
at commit 4143397, before anything here was computed. Two stages:

``lists``    needs the J-lens and Qwen's lm_head + final norm (range-fetched).
             Replays build_arm_vectors.py's atom selection for the ca_unit pos
             arms at L16 WITHOUT writing any arm file -- list (1), the exact
             atoms of jspace / randatom / faratom -- and gates it on the rebuilt
             vectors matching the HF .npy arms. Also writes list (2), every arm's
             top-44 J-lens tokens. Output: results/j23_lexical/lists.json.
``analyze``  tokenizer only. Re-tokenizes every sample's raw completion and
             prints analyses A-D of the definition.

Usage::

    uv run python scripts/j23_lexical_overlap.py lists
    uv run python scripts/j23_lexical_overlap.py analyze | tee results/j23_lexical/analysis_stdout.txt
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "j23_lexical"
LAYER, K_LIST, SEED, BOOT_SEED, N_BOOT = 16, 44, 20260924, 20261102, 10_000
MODEL_CFG = ROOT / "configs" / "model" / "qwen25_7b.yaml"
TRACK_DIR = "Qwen2.5-7B-Instruct-story-wheel32"
REPORT = ROOT / "eval_outputs/j23_t0/L16/arms_Qwen2.5-7B-Instruct_story-wheel32_L16_ca_unit.json"
HF_REPO = "llm-psych/llm-psych-activations"
ARMS = ["full", "resid", "jspace", "randatom", "faratom", "gauss", "shuffle", "lad10",
        "lad20", "lad40", "lad60", "lad70", "lad80", "lad85", "ladstar"]
ATOM_ARMS = ["jspace", "randatom", "faratom"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stage_lists() -> None:
    import torch
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    load_dotenv(ROOT / ".env")
    mod = _load("decompose_emotion_vectors")
    cfg = mod._load_model_config(MODEL_CFG)
    rep = json.load(open(REPORT))
    gate: dict = {}

    # --- mixture, exactly as build_arm_vectors.py with --unit-normalise -------
    raws = {n: np.load(ROOT / "steering_vectors" / TRACK_DIR / f"{n}_layer{LAYER}.npy")
            .astype(np.float64) for n in ("contempt", "aggressiveness")}
    mix = sum(v / np.linalg.norm(v) for v in raws.values())
    gate["mixture_norm"] = (float(np.linalg.norm(mix)), rep["mixture_norm"])
    gate["cos_between_sources"] = (_cos(*raws.values()), rep["cos_between_sources"])
    mix_ok = all(abs(a - b) < 1e-6 for a, b in gate.values())

    tok = mod._load_tokenizer(cfg["hf_model_id"], revision=cfg["hf_revision"])
    w_u, g = mod._load_unembed_and_norm_shards(cfg["hf_model_id"], cfg["hf_revision"])
    j, lens_prov = mod._load_jlens("neuronpedia", cfg["hf_model_id"], None)
    j_l = j[mod._map_layer(LAYER, set(j.keys()), "nearest")]

    # --- replay the pos-sign selection (pos is built first, so its draws come first)
    rng = np.random.default_rng(SEED)
    u = mix.astype(np.float32)
    uu = torch.from_numpy(u)
    comp, _, m, atoms, picked = mod._decompose_vector(
        u, sign=+1, j_l=j_l, w_u=w_u, g=g, tokenizer=tok, k=64, n_candidates=2048,
        return_atoms=True)
    h = uu.float()
    cand = (((h @ j_l.float().T) * g) @ w_u.T).topk(2048).indices
    assert cand[picked].tolist() == m["token_ids"], "candidate order not reproduced"
    n_pick = max(m["n_atoms"], 1)

    pool = [i for i in range(atoms.shape[0]) if i not in set(picked)]
    idx = rng.choice(len(pool), size=min(n_pick, len(pool)), replace=False)
    rand_rows = [pool[i] for i in idx]
    active = atoms[rand_rows].T
    rand_v = active @ mod._nnls_active(active, uu)

    j_f = j_l.float()
    lens_logits = ((uu.float() @ j_f.T).to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T
    order = torch.argsort(lens_logits.float(), descending=True)
    mid = order[order.numel() // 4: 3 * order.numel() // 4]
    far_ids = mid[torch.from_numpy(rng.choice(mid.numel(), size=n_pick, replace=False))]
    far = (w_u[far_ids].float() * g.float()) @ j_f
    far = far / torch.clamp(far.norm(dim=1, keepdim=True), min=1e-12)
    far_coefs = mod._nnls_active(far.T, uu)
    far_v = far.T @ far_coefs

    lists1 = {"jspace": m["token_ids"], "randatom": cand[rand_rows].tolist(),
              "faratom": far_ids.tolist()}
    rand_coefs = mod._nnls_active(active, uu)
    # Atoms the NNLS fit actually uses. Not in the definition: added after the lists
    # stage showed randatom uses 15 of its 44 atoms, before analyze was first run.
    lists1_nz = {"jspace": [t for t, c in zip(m["token_ids"], m["coefs"], strict=True) if c > 0],
                 "randatom": cand[rand_rows][rand_coefs > 0].tolist(),
                 "faratom": far_ids[far_coefs > 0].tolist()}
    nonzero = {a: len(v) for a, v in lists1_nz.items()}

    # --- the HF arms: verification target for (1), and the vectors for (2) ----
    sha = HfApi().dataset_info(HF_REPO).sha
    arm_vecs = {a: np.load(hf_hub_download(
        HF_REPO, f"steering_vectors/{TRACK_DIR}/ca_unit_pos_{a}_layer{LAYER}.npy",
        repo_type="dataset", revision=sha)) for a in ARMS}
    rebuilt = {"jspace": comp, "randatom": rand_v.numpy(), "faratom": far_v.numpy()}
    gate["rebuilt_cos"] = {a: _cos(rebuilt[a], arm_vecs[a]) for a in ATOM_ARMS}
    gate["passed"] = bool(mix_ok and all(c >= 0.9999 for c in gate["rebuilt_cos"].values()))

    lists2 = {}
    for a, x in arm_vecs.items():
        xt = torch.from_numpy(x.astype(np.float32))
        lists2[a] = (((xt @ j_l.float().T) * g) @ w_u.T).topk(K_LIST).indices.tolist()

    ids = {i for lst in [*lists1.values(), *lists2.values()] for i in lst}
    out = {"definition_commit": "4143397", "hf_dataset_revision": sha,
           "model_revision": cfg["hf_revision"], "lens": lens_prov and
           {k: lens_prov.get(k) for k in ("repo_id", "filename", "revision")},
           "seed": SEED, "k_list": K_LIST, "gate": gate, "nonzero_coefs": nonzero,
           "lists1": lists1 if gate["passed"] else None, "lists1_unverified": lists1,
           "lists1_nonzero": lists1_nz if gate["passed"] else None,
           "lists2": lists2, "decoded": {str(i): tok.decode([i]) for i in sorted(ids)}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "lists.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"gate": gate, "nonzero_coefs": nonzero}, indent=1))
    print(f"wrote {OUT / 'lists.json'}")


# --- analyze -------------------------------------------------------------------

def load_samples(d: Path) -> list[dict]:
    """One record per sample: arm, dose, gate, B (None if unscored), raw completion."""
    b = {}
    for r in map(json.loads, open(d / "judge_scores.jsonl")):
        if r["item"] == "B":
            b[r["sid"]] = None if r["value"] in (None, "None", "MISSING") else int(r["value"])
    out = []
    for p in map(json.loads, open(d / "payloads.jsonl")):
        arm, dose = p["condition"].replace("ca_unit_pos_", "").split("_L16_a")
        out.append(dict(arm=arm, dose=float(dose), gate=p["G0_auto"], B=b.get(p["sid"]),
                        full=p["full"]))
    return out


def stage_analyze() -> None:
    from transformers import AutoTokenizer

    lists = json.load(open(OUT / "lists.json"))
    sc = _load("score_j22_diagnostic")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS
    dec = {int(k): v for k, v in lists["decoded"].items()}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=lists["model_revision"])
    rng = np.random.default_rng(BOOT_SEED)

    print(f"# J23 lexical-overlap check (definition 4143397). HF arms @ {lists['hf_dataset_revision'][:8]}")
    print(f"gate: {json.dumps(lists['gate'])}")
    l1 = lists["lists1"]
    if l1 is None:
        print("LIST (1) NOT RECOVERED: gate failed; analyses use list (2) only")
    l2 = {a: set(v) for a, v in lists["lists2"].items()}
    l1 = {a: set(v) for a, v in (l1 or {}).items()}

    def hostile(i: int) -> bool:
        return sc.lexicon_hits(sc.words(dec[i]), lex)[0] > 0

    def show(ids) -> str:
        return " ".join(repr(dec[i]) for i in ids)

    print("\n# A. Construction (no outputs): list (2) = top-44 J-lens tokens per arm")
    if l1:
        print("  list (1) atoms, in pursuit / draw order; nonzero NNLS coefs:",
              lists["nonzero_coefs"])
        for a in ATOM_ARMS:
            print(f"  (1) {a:8s} hostile={sum(map(hostile, l1[a])):2d}/44  "
                  f"|(1)∩(2)|={len(l1[a] & l2[a]):2d}/44  first 15: "
                  f"{show(lists['lists1'][a][:15])}")
    print(f"  {'arm':9s} hostile/44  ∩jspace(1)  ∩randatom(1)  top 10")
    for a in ARMS:
        h = sum(map(hostile, l2[a]))
        oj = len(l2[a] & l1["jspace"]) if l1 else "-"
        orr = len(l2[a] & l1["randatom"]) if l1 else "-"
        print(f"  {a:9s} {h:6d}      {oj!s:>6s}      {orr!s:>6s}      "
              f"{show(lists['lists2'][a][:10])}")

    samples = load_samples(ROOT / "results/t0_qwen") + load_samples(ROOT / "results/j23_main")
    for s in samples:
        s["ids"] = np.array(tok(s["full"], add_special_tokens=False).input_ids)

    def rate(s: dict, lst: set) -> float:
        return 100.0 * np.isin(s["ids"], list(lst)).sum() / max(len(s["ids"]), 1)

    def boot(stat, groups):
        """Point estimate and 95% percentile CI of stat(group means), groups resampled."""
        groups = [np.asarray(gr, float) for gr in groups]
        means = [gr[rng.integers(0, len(gr), (N_BOOT, len(gr)))].mean(axis=1) for gr in groups]
        vals = stat(*means)
        return stat(*[gr.mean(axis=0) for gr in groups]), *np.percentile(vals, [2.5, 97.5])

    def ci(t) -> str:
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

    unst = [s for s in samples if s["arm"] == "full" and s["dose"] == 0.0]
    print(f"\n# B. Surfacing: own-list rate per 100 tokens, minus the unsteered rate of the "
          f"same list (all samples; unsteered n={len(unst)}, Task 0 pod)")
    print(f"  {'arm':9s} dose  n   own(2) rate   lift(2) [95% CI]          "
          f"own(1) rate   lift(1) [95% CI]")
    for a in ARMS:
        for d in (0.2, 0.25):
            ss = [s for s in samples if s["arm"] == a and s["dose"] == d]
            if not ss:
                continue
            r2 = [rate(s, l2[a]) for s in ss]
            line = (f"  {a:9s} {d:<5} {len(ss):2d}  {np.mean(r2):8.3f}   "
                    f"{ci(boot(np.subtract, [r2, [rate(s, l2[a]) for s in unst]])):28s}")
            if a in l1:
                r1 = [rate(s, l1[a]) for s in ss]
                line += (f"{np.mean(r1):8.3f}   "
                         f"{ci(boot(np.subtract, [r1, [rate(s, l1[a]) for s in unst]]))}")
            print(line)

    # C and D use list (1) when recovered, else list (2), as the definition says.
    word_lists = l1 or l2
    src = "(1)" if l1 else "(2)"
    print(f"\n# C. Crossed, threat-matched: jspace vs randatom, doses pooled, lists {src}"
          f"; |W_j ∩ W_r| = {len(word_lists['jspace'] & word_lists['randatom'])}")

    def interaction(mj, mr):
        return (mj[..., 0] - mj[..., 1]) - (mr[..., 0] - mr[..., 1])

    subsets = {"B=1 (primary)": lambda s: s["gate"] == 1 and s["B"] == 1,
               "B=0": lambda s: s["gate"] == 1 and s["B"] == 0,
               "all samples": lambda s: True}
    for label, keep in subsets.items():
        grp = {a: [[rate(s, word_lists["jspace"]), rate(s, word_lists["randatom"])]
                   for s in samples if s["arm"] == a and keep(s)] for a in ("jspace", "randatom")}
        mj, mr = (np.mean(grp[a], axis=0) for a in ("jspace", "randatom"))
        print(f"  {label:14s} n=({len(grp['jspace'])}, {len(grp['randatom'])})  "
              f"jspace: W_j {mj[0]:.3f} W_r {mj[1]:.3f}   randatom: W_j {mr[0]:.3f} "
              f"W_r {mr[1]:.3f}   I = {ci(boot(interaction, [grp['jspace'], grp['randatom']]))}")

    if lists["lists1_nonzero"]:
        nz = {a: set(v) for a, v in lists["lists1_nonzero"].items()}
        keep = subsets["B=1 (primary)"]
        grp = {a: [[rate(s, nz["jspace"]), rate(s, nz["randatom"])]
                   for s in samples if s["arm"] == a and keep(s)] for a in ("jspace", "randatom")}
        print(f"  SENSITIVITY (added after the lists stage, before analyze ran): B=1, list (1) "
              f"atoms with nonzero NNLS coef only ({len(nz['jspace'])} / {len(nz['randatom'])})"
              f"  I = {ci(boot(interaction, [grp['jspace'], grp['randatom']]))}")
    print("  list tokens found in B=1 outputs (token: count), top 8 per arm x list:")
    for a in ("jspace", "randatom"):
        for wl in ("jspace", "randatom"):
            c = collections.Counter(int(i) for s in samples if s["arm"] == a and s["gate"] == 1
                                    and s["B"] == 1 for i in s["ids"] if int(i) in word_lists[wl])
            print(f"    {a} outputs, W_{wl[0]}: "
                  + ", ".join(f"{dec[i]!r}:{n}" for i, n in c.most_common(8)))

    print(f"\n# D. Threats vs near-v threats (resid, lad10, lad20, lad40; gate-passing, "
          f"B=1), lists {src}")
    comp = [s for s in samples if s["arm"] in ("resid", "lad10", "lad20", "lad40")
            and s["gate"] == 1 and s["B"] == 1]
    for a in ("jspace", "randatom"):
        own = [rate(s, word_lists[a]) for s in samples if s["arm"] == a and s["gate"] == 1 and s["B"] == 1]
        oth = [rate(s, word_lists[a]) for s in comp]
        print(f"  W_{a[0]} rate: {a} threats {np.mean(own):.3f} (n={len(own)})  comparators "
              f"{np.mean(oth):.3f} (n={len(oth)})  diff {ci(boot(np.subtract, [own, oth]))}")


if __name__ == "__main__":
    {"lists": stage_lists, "analyze": stage_analyze}[sys.argv[1]]()
