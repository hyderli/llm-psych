"""J27 manipulation check, fixed before generation (J26-J28 amendment 1 item 4).

On one teacher-forced pass of the canonical scenario prompt:

    M(.) = log of the next-token probability mass on the set's tokens, from
           the final-layer logits, averaged over positions.

Runs: unsteered; full @0.25 at L16; full @0.25 + V ablation at L16-26 for each
rank r in {32, 64, 121} (steering hook attached FIRST at L16, then the
ablation — the registered order). Let pass_rise = M(full) - M(unsteered).
Rank r passes iff

    M(full + ablation_r) - M(unsteered) <= 0.20 * pass_rise.

The first passing rank is the J27 rank, written to results/j27/rank.json and
used for EVERY set by run_j27.py. If none passes, exit 2 and J27 stops.

Also reported (amendment 1 item 3): per set and layer, on the unsteered and
steered passes, the removed share of activation norm
||P_l (h - mu_l)||^2 / ||h||^2, read off each AblationHook's
last_removed_share. This is the "matched rank = matched damage?" table.

    uv run python scripts/check_j27_manipulation.py \
        --model-path /workspace/qwen25-7b-instruct-a09a354
    uv run python scripts/check_j27_manipulation.py --self-test --work /tmp/j27w
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
PASS_FRACTION = 0.20
ALL_SETS = ("V", "H", "C", "R1", "R2", "R3")

_HOOKS = None


def _hooks():
    global _HOOKS
    if _HOOKS is None:
        _HOOKS = jc._load("hooks", "emotion_steering/sycophancy_blackmail/hooks.py")
    return _HOOKS


def set_mass(logits: torch.Tensor, token_ids: list[int]) -> float:
    """Mean over positions of log(sum softmax over the set's tokens)."""
    lp = torch.log_softmax(logits[0].float(), dim=-1)      # (T, vocab)
    tids = torch.tensor(token_ids, dtype=torch.long)
    return float(torch.logsumexp(lp[:, tids], dim=-1).mean())


def load_spec(results_dir: Path, set_name: str, rank: int,
              layers: list[int]) -> list:
    """(subspace[:, :rank], mu) per layer."""
    specs = []
    for lyr in layers:
        s = np.load(results_dir / "subspaces" / f"{set_name}_l{lyr}.npy")
        mu = np.load(results_dir / "mu" / f"mu_l{lyr}.npy")
        specs.append((s[:, :min(rank, s.shape[1])], mu))
    return specs


def measure(model, ids: torch.Tensor, token_ids: list[int],
            steer, ablate_specs, dec_layers) -> tuple[float, dict]:
    """One teacher-forced pass under hooks; (M, per-layer removed share)."""
    hk = _hooks()
    steer_hook = abl = None
    try:
        if steer is not None:                       # ORDER: steering FIRST
            steer_hook = hk.SteeringHook(*steer)
            steer_hook.attach(dec_layers[jc.LAYER])
        if ablate_specs:                            # ablation SECOND
            abl = hk.MultiAblationHook(
                [(dec_layers[lyr], s, mu)
                 for lyr, (s, mu) in zip(jc.ABLATION_LAYERS, ablate_specs,
                                       strict=True)])
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        shares = ({lyr: h.last_removed_share
                   for lyr, h in zip(jc.ABLATION_LAYERS, abl.hooks, strict=True)}
                  if abl else {})
        return set_mass(logits, token_ids), shares
    finally:
        if abl is not None:
            abl.remove()
        if steer_hook is not None:
            steer_hook.remove()


def _tiny_mock(vocab: int = 2048, hidden: int = 64, n_layers: int = 28):
    import torch.nn as nn

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(n_layers)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids=None, **_kw):
            h = self.emb(input_ids)
            for blk in self.model.layers:
                h = blk(h)
            class O:  # noqa: E742
                pass
            o = O()
            o.logits = self.head(h)
            return o

    torch.manual_seed(0)
    m = _Model()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


def _self_test_fixture(work: Path, hidden: int = 64):
    """Fabricated results dir: orthonormal (64,121) bases, mu files, token sets."""
    rng = np.random.default_rng(0)
    (work / "subspaces").mkdir(parents=True, exist_ok=True)
    (work / "mu").mkdir(parents=True, exist_ok=True)
    for name in ALL_SETS:
        q, _ = np.linalg.qr(rng.standard_normal((hidden, 121)))
        for lyr in jc.ABLATION_LAYERS:
            np.save(work / "subspaces" / f"{name}_l{lyr}.npy", q.astype(np.float32))
    for lyr in jc.ABLATION_LAYERS:
        np.save(work / "mu" / f"mu_l{lyr}.npy",
                rng.standard_normal(hidden).astype(np.float32) * 0.01)
    ids = list(range(10, 40))
    (work / "token_sets.json").write_text(json.dumps(
        {n: {"token_ids": ids, "texts": [str(i) for i in ids]}
         for n in ALL_SETS}))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--vectors-dir", type=Path, default=jc.VEC_DIR)
    ap.add_argument("--results", type=Path, default=jc.RESULTS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-damage-report", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    results = args.results
    if args.self_test:
        work = ROOT / "results" / "j27_selftest"
        results = work
        device = "cpu"
        token_ids = _self_test_fixture(work)
        model = _tiny_mock()
        ids = torch.arange(200)[None]
        direction = torch.ones(64)
        prompt_sha = "self-test"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        token_ids = json.loads(
            (results / "token_sets.json").read_text())["V"]["token_ids"]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device)
        prompt, prompt_sha = jc.rendered_prompt(tokenizer)
        ids = tokenizer(prompt, add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(device)
        vbn = jc._load("vbn", "emotion_steering/sycophancy_blackmail/vectors_npy.py")
        direction = vbn.build_direction(
            vbn.NpyVectorBank(args.vectors_dir), [("j24_full", 1.0)], jc.LAYER)
    model.eval()
    dec_layers = _hooks().get_decoder_layers(model)
    if max(jc.ABLATION_LAYERS) >= len(dec_layers):
        raise SystemExit(f"model has {len(dec_layers)} layers; "
                         f"J27 needs layer {max(jc.ABLATION_LAYERS)}")
    steer = (direction, jc.ALPHA_FULL, True, "post")

    out = {"prompt_sha256": prompt_sha, "self_test": args.self_test,
           "alpha": jc.ALPHA_FULL, "layer": jc.LAYER,
           "ablation_layers": jc.ABLATION_LAYERS, "ranks": jc.RANKS,
           "pass_fraction": PASS_FRACTION, "M": {}, "damage": {}}

    m_un, _ = measure(model, ids, token_ids, None, None, dec_layers)
    m_full, _ = measure(model, ids, token_ids, steer, None, dec_layers)
    rise = m_full - m_un
    out["M"] = {"unsteered": m_un, "full": m_full, "rise": rise,
                "per_rank": {}}
    print(f"M(unsteered) {m_un:+.4f}   M(full) {m_full:+.4f}   "
          f"rise {rise:+.4f}", flush=True)

    rank = None
    for r in jc.RANKS:
        specs = load_spec(results, "V", r, jc.ABLATION_LAYERS)
        m_r, _ = measure(model, ids, token_ids, steer, specs, dec_layers)
        passed = (m_r - m_un) <= PASS_FRACTION * rise
        out["M"]["per_rank"][str(r)] = {
            "M": m_r, "residual_rise": m_r - m_un,
            "fraction_of_rise": (m_r - m_un) / rise if rise else float("nan"),
            "pass": passed}
        print(f"r={r:3d}: M {m_r:+.4f}   residual rise {m_r - m_un:+.4f}   "
              f"{(m_r - m_un) / rise if rise else float('nan'):.3f} of rise   "
              f"{'PASS' if passed else 'fail'}", flush=True)
        if rank is None and passed:
            rank = r
    out["rank"] = rank

    # --- damage table: removed share per set and layer, unsteered + steered ---
    if not args.no_damage_report:
        for name in ALL_SETS:
            for r in jc.RANKS:
                specs = load_spec(results, name, r, jc.ABLATION_LAYERS)
                _, su = measure(model, ids, token_ids, None, specs, dec_layers)
                _, ss = measure(model, ids, token_ids, steer, specs, dec_layers)
                out["damage"].setdefault(name, {})[str(r)] = {
                    "unsteered": su, "steered": ss}
                print(f"damage {name:3s} r={r:3d}: "
                      f"unsteered {np.mean(list(su.values())):.3f}   "
                      f"steered {np.mean(list(ss.values())):.3f}   "
                      "(mean removed share over L16-26)", flush=True)

    results.mkdir(parents=True, exist_ok=True)
    (results / "manipulation_report.json").write_text(json.dumps(out, indent=2))
    print(f"-> {results / 'manipulation_report.json'}")
    if rank is None:
        print("NO rank in {32,64,121} passes the 20% rule — J27 stops here.")
        return 2
    (results / "rank.json").write_text(json.dumps({"rank": rank}))
    print(f"J27 rank = {rank}  -> {results / 'rank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
