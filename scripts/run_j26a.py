"""J26A readout: forced-choice probabilities at the assistant prefix.

One forward pass per prompt (no generation), SteeringHook at L16 site=post
norm_scale=True alpha=0.25 — the same hook, model loading and dtype the
`steered/local` provider uses (AutoModelForCausalLM.from_pretrained(model_path,
device_map=device); config dtype, bfloat16 for Qwen2.5-7B-Instruct).

For each condition x prompt: softmax at the last position (after the prefix),
the five first-token probabilities, their total mass, and the renormalised P
per draft key. Non-compliance (mass < 0.5) is an analysis decision, made in
analyze_j26a.py — every row is written.

Conditions: unsteered (no hook) plus j24_full, j24_pool1-6, j24_span1-6,
j25_coh1-6 — 20 in total. Resumable per (condition, order_index).

    uv run python scripts/run_j26a.py \
        --model-path /workspace/qwen25-7b-instruct-a09a354
    uv run python scripts/run_j26a.py --self-test     # tiny mock, exercises the path
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "public" / "j26a_prompts.jsonl"
OUT = ROOT / "results" / "j26a" / "readout.jsonl"
HEADER = ROOT / "results" / "j26a" / "readout_header.json"
VEC_DIR = ROOT / "steering_vectors" / "Qwen2.5-7B-Instruct-story-wheel32"
LAYER = 16
ALPHA = 0.25
ARMS = (["j24_full"] + [f"j24_pool{i}" for i in range(1, 7)]
        + [f"j24_span{i}" for i in range(1, 7)]
        + [f"j25_coh{i}" for i in range(1, 7)])
CONDITIONS = ["unsteered"] + ARMS


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_mock(vocab: int = 2048, hidden: int = 64, n_layers: int = 17):
    """A minimal model exercising the readout path on CPU, hook included."""
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

    return _Model()


def readout(model, prompt_ids: torch.Tensor, first_tokens: dict) -> dict:
    """Softmax at the last position; mass and renormalised P per draft key."""
    with torch.no_grad():
        logits = model(input_ids=prompt_ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    raw = {k: float(probs[t]) for k, t in first_tokens.items()}
    mass = float(sum(raw.values()))
    return {"raw_p": raw, "mass": mass,
            "p": {k: (v / mass if mass else float("nan"))
                  for k, v in raw.items()}}


def _done(path: Path) -> set:
    done = set()
    if path.exists():
        for x in path.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done.add((r["condition"], r["order_index"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--vectors-dir", type=Path, default=VEC_DIR)
    ap.add_argument("--prompts", type=Path, default=PROMPTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N prompts only (smoke tests)")
    ap.add_argument("--self-test", action="store_true",
                    help="tiny mock model on CPU; no HF download")
    args = ap.parse_args()

    prompts = [json.loads(x) for x in args.prompts.read_text().splitlines()
               if x.strip()]
    if args.limit:
        prompts = prompts[:args.limit]

    if args.self_test:
        device = "cpu"                  # mock runs nowhere else
    elif args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.self_test:
        model = _tiny_mock()
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.05)
        tokenizer = None
        enc = lambda text: {"input_ids": torch.tensor(  # noqa: E731
            [[abs(hash(text)) % 2000 + 1]])}
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device)
        enc = lambda text: tokenizer(  # noqa: E731
            text, add_special_tokens=False, return_tensors="pt")

    model.eval()

    hooks = _load("hooks", "emotion_steering/sycophancy_blackmail/hooks.py")
    vbn = _load("vbn", "emotion_steering/sycophancy_blackmail/vectors_npy.py")

    layers = hooks.get_decoder_layers(model)
    if not (0 <= args.layer < len(layers)):
        raise SystemExit(f"layer {args.layer} out of range "
                         f"({len(layers)} layers)")

    import transformers
    header = {"torch": torch.__version__,
              "transformers": getattr(transformers, "__version__", "mock"),
              "model_path": None if args.self_test else args.model_path,
              "dtype": str(getattr(model, "dtype", "?")),
              "alpha": args.alpha, "layer": args.layer, "site": "post",
              "norm_scale": True, "n_layers": len(layers),
              "device": device, "self_test": args.self_test}
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(json.dumps(header, indent=2))

    done = _done(args.out)
    bank = None if args.self_test else vbn.NpyVectorBank(args.vectors_dir)
    fh = args.out.open("a")
    for cond in CONDITIONS:
        direction = None
        if cond != "unsteered":
            if bank is not None:
                direction = vbn.build_direction(bank, [(cond, 1.0)], args.layer)
            else:                       # self-test: any nonzero direction
                direction = torch.ones(64)
        hook = None
        if direction is not None:
            hook = hooks.SteeringHook(direction, args.alpha, True, "post")
            hook.attach(layers[args.layer])
        n_done = 0
        try:
            for pr in prompts:
                if (cond, pr["order_index"]) in done:
                    n_done += 1
                    continue
                ids = enc(pr["prompt"])["input_ids"].to(device)
                r = readout(model, ids, pr["first_token_ids"])
                fh.write(json.dumps({"condition": cond,
                                     "order_index": pr["order_index"],
                                     **r}) + "\n")
                fh.flush()
            skipped = n_done
            print(f"{cond}: {len(prompts)} prompts"
                  + (f" ({skipped} resumed)" if skipped else ""))
        finally:
            if hook is not None:
                hook.remove()
    fh.close()
    print(f"-> {args.out}   header -> {HEADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
