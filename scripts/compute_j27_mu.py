"""J27 mu: the per-layer mean block output on the canonical scenario prompt.

J26-J28 amendment 1 item 3: mu_l is the mean layer-l block output over all
positions of ONE unsteered, unablated forward pass of the scenario prompt,
frozen at build. The AblationHook subtracts mu so ablation removes the
deviation inside S, not the ordinary level.

Model loading is exactly what run_j26a.py and the steered/local provider use:
AutoModelForCausalLM.from_pretrained(model_path, device_map=...) at the config
dtype (bfloat16 for Qwen2.5-7B-Instruct). mu files are float32.

    uv run python scripts/compute_j27_mu.py \
        --model-path /workspace/qwen25-7b-instruct-a09a354
    uv run python scripts/compute_j27_mu.py --self-test   # tiny mock, CPU

Outputs (refuses to overwrite):
    results/j27/mu/mu_l<16..26>.npy     float32 (d_model,)
    results/j27/mu/mu_meta.json         position count, prompt sha256, dtype
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
OUT_DIR = jc.RESULTS / "mu"
META = OUT_DIR / "mu_meta.json"
_HOOKS = None


def _hooks():
    global _HOOKS
    if _HOOKS is None:
        _HOOKS = jc._load("hooks", "emotion_steering/sycophancy_blackmail/hooks.py")
    return _HOOKS


def _tiny_mock(vocab: int = 2048, hidden: int = 64, n_layers: int = 28):
    """A minimal model exercising the capture path on CPU (run_j26a's shape)."""
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

        def forward(self, input_ids=None, **_kw):
            h = self.emb(input_ids)
            for blk in self.model.layers:
                h = blk(h)
            class O:  # noqa: E742
                pass
            o = O()
            o.logits = h
            return o

    torch.manual_seed(0)
    m = _Model()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


class _MockTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False):
        return "SYS:" + messages[0]["content"] + "|USER:" + messages[1]["content"]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = torch.arange(len(text) % 500 + 10, dtype=torch.long)[None]
        return {"input_ids": ids}


def capture_mus(model, input_ids: torch.Tensor,
                layers: list[int]) -> dict[int, torch.Tensor]:
    """One unsteered forward; mean over positions of each layer's block output."""
    dec_layers = _hooks().get_decoder_layers(model)
    mus: dict[int, torch.Tensor] = {}
    handles = []

    def make_cap(layer):
        def cap(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            mus[layer] = h[0].detach().float().mean(dim=0).cpu()   # over positions
        return cap

    for lyr in layers:
        handles.append(dec_layers[lyr].register_forward_hook(make_cap(lyr)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h_ in handles:
            h_.remove()
    return mus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--layers", default="16-26")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    lo, _, hi = args.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))

    if args.self_test and args.out == OUT_DIR:
        args.out = ROOT / "results" / "j27_selftest" / "mu"
        print(f"self-test: writing to {args.out} (not {OUT_DIR})")

    outs = [args.out / f"mu_l{lyr}.npy" for lyr in layers] + [args.out / "mu_meta.json"]
    for f in outs:
        if f.exists():
            raise SystemExit(f"REFUSING to overwrite {f}")

    if args.self_test:
        device = "cpu"
        model = _tiny_mock(n_layers=max(layers) + 1)
        tokenizer = _MockTokenizer()
        prompt_text = "mock canonical prompt"
        prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest() + " (mock)"
        ids = tokenizer(prompt_text)["input_ids"]
        n_pos = int(ids.shape[1])
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device)
        prompt_text, prompt_sha = jc.rendered_prompt(tokenizer)
        ids = tokenizer(prompt_text, add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(device)
        n_pos = int(ids.shape[1])

    model.eval()
    mus = capture_mus(model, ids, layers)

    args.out.mkdir(parents=True, exist_ok=True)
    for lyr, mu in mus.items():
        np.save(args.out / f"mu_l{lyr}.npy", mu.numpy().astype(np.float32))
        print(f"mu_l{lyr}: shape {tuple(mu.shape)}  norm {float(mu.norm()):.4f}")
    meta = {"layers": layers, "n_positions": n_pos,
            "prompt_sha256": prompt_sha, "self_test": args.self_test,
            "model_path": None if args.self_test else args.model_path,
            "dtype": str(getattr(model, "dtype", "?")),
            "torch": torch.__version__}
    (args.out / "mu_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {args.out}  ({n_pos} positions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
