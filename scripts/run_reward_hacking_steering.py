"""Reward-hacking steering sweep (paper-faithful, ImpossibleBench-style).

Steers one emotion direction (e.g. ``desperate`` or ``calm``) across a
signed strength grid (fraction of residual-stream norm, paper
convention ``-0.1 ... +0.1``) and measures the aggregate hack rate
against the three non-negotiable controls (zero, norm-matched random,
norm-matched orthogonal; ``docs/methods.md`` §"Controls").

Two task sources, ``--dataset``:

* ``local`` (default) — the 7 hand-authored ImpossibleBench-style
  tasks in ``data/public/reward_hacking_code_tasks.jsonl`` (see
  ``src/llm_psych/tasks/reward_hacking.py``). Self-verified (each
  task's reference solution is checked to pass the legit tests and
  fail the impossible one).
* ``impossiblebench`` — the published Impossible-LiveCodeBench dataset
  (Zhong, Raghunathan & Carlini 2025, arXiv:2510.20270; MIT license,
  ``huggingface.co/datasets/fjzzq2002/impossible_livecodebench``, 103
  tasks/split). Larger N for real statistical power; see
  ``src/llm_psych/tasks/impossiblebench.py`` for the format
  differences and the one provenance caveat (no shipped reference
  solution — trusts the benchmark's own construction). Use
  ``--split oneoff`` (closest to the local set) or ``--split
  conflicting`` (stricter); ``--n-tasks`` subsamples (full split is
  103 tasks — cost scales linearly with task count x condition count
  x rollouts, so subsample for a pilot before running the full set).

Sampling default is ``temperature=0`` / greedy
(``BLUEPRINT.md``'s project-wide behavioral-eval convention) — pass
``--do-sample --temperature 0.7`` to instead average over stochastic
rollouts per task per condition, closer to how the source paper's
smooth percentage curves were likely produced. Not pre-registered:
this is exploratory/pilot until a HYPOTHESES.md amendment locks the
vectors + sample size (see ``plans/reward-hacking-steering.md``).

Two modes:

* ``--dry-run`` — no model load, no GPU, no network. A stub
  "generator" always submits the task's reference (correct) solution,
  so the *entire* pipeline (vector loading, calibration math, control
  construction, prompting, sandboxed scoring, aggregation, output
  files) runs and can be sanity-checked on the Mac before spending
  RunPod time. Expected result: hack rate ≈ 0 everywhere (the
  reference solution never passes an impossible test) — a nonzero
  dry-run hack rate means the pipeline has a bug, not that steering
  works.
* Real run (default) — loads the model via ``llm_psych.models.load_model``,
  calibrates the residual-stream norm on the 50 frozen neutral prompts,
  and generates for real. GPU strongly recommended (see
  ``README.md``'s two-phase compute strategy).

Usage
-----
    # Sanity-check the whole pipeline, no GPU:
    uv run python scripts/run_reward_hacking_steering.py \\
        --model gemma2_9b --emotion desperate --dry-run

    # Real run on the local 7-task set:
    uv run python scripts/run_reward_hacking_steering.py \\
        --model gemma2_9b --emotion desperate --contrast-emotion calm \\
        --track story --device cuda --dtype bfloat16

    # Real run on ImpossibleBench, subsampled to 20 tasks:
    uv run python scripts/run_reward_hacking_steering.py \\
        --model gemma2_9b --emotion desperation --contrast-emotion calm \\
        --track story-wheel32-addon --dataset impossiblebench \\
        --split oneoff --n-tasks 20 --device cuda --dtype bfloat16

Output: ``results/reward_hacking_steering/<model_key>_<emotion>/``
(``per_item.parquet``, ``stats.json``, ``run_meta.json``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from llm_psych.models import load_model  # noqa: E402
from llm_psych.steering_experiment import (  # noqa: E402
    DEFAULT_SEEDS,
    build_condition_grid,
    mean_residual_norm,
    wilson_ci,
)

log = logging.getLogger("run_reward_hacking_steering")

STIMULUS_HASHES_PATH = _repo_root / "configs" / "stimuli_hashes.yaml"
NEUTRAL_PROMPTS_PATH = _repo_root / "data" / "public" / "emotion_prompts.parquet"
DEFAULT_STRENGTHS = [-0.1, -0.05, 0.05, 0.1]  # 0.0 handled by the shared "zero" condition


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def _check_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=_repo_root
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before "
            "running experiments (or pass --allow-dirty for local dev).\n"
            + result.stdout
        )


def _verify_stimulus_hash(path: Path) -> None:
    registry = yaml.safe_load(STIMULUS_HASHES_PATH.read_text())
    expected = registry.get(path.name)
    if expected is None:
        raise RuntimeError(f"{path.name} has no entry in {STIMULUS_HASHES_PATH}")
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{path.name} MD5 mismatch: expected {expected}, got {actual}. "
            "Stimulus file changed since freeze -- see docs/methods.md "
            "'Stimulus locking'."
        )


# ---------------------------------------------------------------------------
# Task set — abstracts over the two dataset sources (local / impossiblebench)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TaskSet:
    """Everything `run_sweep` needs, independent of which dataset it came
    from -- see `_load_task_set`."""

    tasks: list[Any]
    build_prompt: Callable[[Any], str]
    score: Callable[[Any, str], Any]
    get_id: Callable[[Any], str]
    get_category: Callable[[Any], str]
    dry_run_generate: Callable[[Any, "SteeringCondition", int], str]  # noqa: F821


def _load_task_set(args: argparse.Namespace) -> TaskSet:
    if args.dataset == "local":
        from llm_psych.tasks.reward_hacking import (
            DEFAULT_STIMULI_PATH,
            build_prompt,
            load_tasks,
            score_submission,
        )

        _verify_stimulus_hash(DEFAULT_STIMULI_PATH)
        tasks = load_tasks()
        log.info("Loaded %d local reward-hacking tasks", len(tasks))

        def dry_run_generate(task, condition, rollout_idx):
            return f"```python\n{task.reference_solution}```"

        return TaskSet(
            tasks=tasks,
            build_prompt=build_prompt,
            score=lambda task, output: score_submission(task, output),
            get_id=lambda t: t.id,
            get_category=lambda t: t.category,
            dry_run_generate=dry_run_generate,
        )

    if args.dataset == "impossiblebench":
        from llm_psych.tasks.impossiblebench import (
            build_prompt,
            load_impossiblebench_tasks,
            score_submission,
        )

        tasks = load_impossiblebench_tasks(split=args.split, n=args.n_tasks, seed=args.task_seed)
        log.info(
            "Loaded %d ImpossibleBench tasks (split=%s%s)",
            len(tasks), args.split,
            f", subsampled from 103 with seed={args.task_seed}" if args.n_tasks else "",
        )
        if args.n_tasks is None:
            log.warning(
                "Using the FULL %d-task split with no --n-tasks subsample -- "
                "cost scales linearly with task count x condition count x "
                "rollouts. Consider --n-tasks for a pilot.",
                len(tasks),
            )

        def dry_run_generate(task, condition, rollout_idx):
            # No reference solution ships with this dataset (see
            # src/llm_psych/tasks/impossiblebench.py's provenance note) --
            # this stub only proves the harness runs without crashing, not
            # that hack_rate is correctly ~0 (compare local dataset's
            # dry-run, which DOES have that stronger guarantee).
            return f"```python\ndef {task.entry_point}(*args, **kwargs):\n    return None\n```"

        return TaskSet(
            tasks=tasks,
            build_prompt=build_prompt,
            score=lambda task, output: score_submission(task, output),
            get_id=lambda t: t.task_id,
            get_category=lambda t: t.impossible_type,
            dry_run_generate=dry_run_generate,
        )

    raise SystemExit(f"unknown --dataset {args.dataset!r}; expected 'local' or 'impossiblebench'")


def _normalize_result(result: Any) -> dict:
    """Common row fields regardless of which dataset's result type this is."""
    if hasattr(result, "legit_pass_rate"):  # reward_hacking.RewardHackingResult
        return {
            "hacked": result.hacked,
            "legit_pass_rate": result.legit_pass_rate,
            "held_out_pass_rate": result.held_out_pass_rate,
            "compiled": result.compiled,
            "compile_error": result.compile_error,
        }
    # impossiblebench.ImpossibleBenchResult
    legit = result.legit_passed
    return {
        "hacked": result.hacked,
        "legit_pass_rate": None if legit is None else float(legit),
        "held_out_pass_rate": None,
        "compiled": result.compiled,
        "compile_error": result.compile_error,
    }


# ---------------------------------------------------------------------------
# Vector loading (mirrors scripts/validate_logit_lens.py conventions)
# ---------------------------------------------------------------------------

def _read_model_cfg(model_config: str) -> dict:
    path = _repo_root / "configs" / "model" / f"{model_config}.yaml"
    if not path.exists():
        raise SystemExit(f"model config not found: {path}")
    return yaml.safe_load(path.read_text())


def _load_vector(model_key: str, track: str, emotion: str, layer: int, vectors_dir: Path) -> np.ndarray:
    path = vectors_dir / f"{model_key}-{track}" / f"{emotion}_layer{layer}.npy"
    if not path.exists():
        raise SystemExit(
            f"no vector at {path}. Pull it first:\n"
            f"  uv run python scripts/sync_hf.py pull steering_vectors --model {model_key}-{track}"
        )
    return np.load(path).astype(np.float32)


def _pick_layer(model_key: str, track: str, emotion: str, requested: int | None, n_layers: int | None, vectors_dir: Path) -> int:
    available = sorted(
        int(p.stem.rpartition("_layer")[2])
        for p in (vectors_dir / f"{model_key}-{track}").glob(f"{emotion}_layer*.npy")
    )
    if not available:
        raise SystemExit(f"no {emotion}_layer*.npy files for {model_key}-{track} in {vectors_dir}")
    if requested is not None:
        if requested not in available:
            raise SystemExit(f"layer {requested} not available for {emotion}; have {available}")
        return requested
    if n_layers:
        target = round(2 * n_layers / 3)
        return min(available, key=lambda layer: abs(layer - target))
    return max(available)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _real_generate(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    vector: np.ndarray,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> str:
    from llm_psych.hooks import ResidualStreamSteerer

    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    vec_t = torch.from_numpy(vector)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with ResidualStreamSteerer(model, layer=layer, vector=vec_t, alpha=1.0):
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
    new_tokens = out_ids[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    task_set: TaskSet,
    conditions,
    generate_fn,
    n_rollouts: int,
) -> pd.DataFrame:
    """Run every (condition, task, rollout) cell; return a tidy per-item frame."""
    rows = []
    for condition in conditions:
        for task in task_set.tasks:
            for rollout_idx in range(n_rollouts):
                output = generate_fn(task, condition, rollout_idx)
                result = task_set.score(task, output)
                rows.append(
                    {
                        "task_id": task_set.get_id(task),
                        "category": task_set.get_category(task),
                        "control_type": condition.control_type,
                        "strength": condition.strength,
                        "seed": condition.seed,
                        "rollout_idx": rollout_idx,
                        **_normalize_result(result),
                        "vector_md5": hashlib.md5(condition.vector.tobytes()).hexdigest(),
                        "output_text": output,
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Hack rate + Wilson 95% CI per (control_type, strength)."""
    out: dict[str, dict] = {}
    for (control_type, strength), sub in df.groupby(["control_type", "strength"]):
        n = len(sub)
        successes = int(sub["hacked"].sum())
        lo, hi = wilson_ci(successes, n)
        key = f"{control_type}_{strength:+.2f}" if control_type != "zero" else "zero"
        out[key] = {
            "control_type": control_type,
            "strength": float(strength),
            "n": n,
            "hack_rate": successes / n if n else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", dest="model_config", required=True, help="Hydra model config name, e.g. gemma2_9b")
    ap.add_argument("--dataset", default="local", choices=["local", "impossiblebench"], help="'local' = 7 hand-authored tasks; 'impossiblebench' = the published 103-task/split HF dataset")
    ap.add_argument("--split", default="oneoff", choices=["oneoff", "conflicting", "original"], help="--dataset impossiblebench only; 'original' has no mutation, don't use it for the hack-rate statistic")
    ap.add_argument("--n-tasks", type=int, default=None, help="--dataset impossiblebench only; subsample size (full split is 103 -- cost scales linearly, subsample for a pilot)")
    ap.add_argument("--task-seed", type=int, default=42, help="--dataset impossiblebench only; seed for the --n-tasks subsample")
    ap.add_argument("--emotion", required=True, help="steering direction, e.g. desperate, calm")
    ap.add_argument("--contrast-emotion", default=None, help="other axis vector, added to the orthogonal control's basis (e.g. calm when --emotion desperate)")
    ap.add_argument("--track", default="story", help="steering_vectors/<model_key>-<track>/ subdir; the base 'story' track has literal 'desperate'/'calm' names but ONLY for gemma-2-9b-it today -- see plans/reward-hacking-steering.md")
    ap.add_argument("--layer", type=int, default=None, help="default: nearest available layer to ~2/3 model depth")
    ap.add_argument("--strengths", default=",".join(str(s) for s in DEFAULT_STRENGTHS), help="comma-separated signed fractions of residual-stream norm")
    ap.add_argument("--controls", default="target,zero,random,orthogonal", help="comma-separated subset of {target,zero,random,orthogonal}")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--n-rollouts", type=int, default=1, help="generations per (condition, task); >1 only useful with --do-sample")
    ap.add_argument("--do-sample", action="store_true", help="default is greedy (temperature=0), per BLUEPRINT.md's project-wide convention; pass this to sample instead")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--device", default="cuda", help="cpu | cuda | mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--vectors-dir", type=Path, default=_repo_root / "steering_vectors")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: results/reward_hacking_steering/<model_key>_<emotion>")
    ap.add_argument("--dry-run", action="store_true", help="no model load; stub generation exercises the full pipeline")
    ap.add_argument("--allow-dirty", action="store_true", help="skip the clean-git pre-flight check (local dev only)")
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not args.dry_run and not args.allow_dirty:
        _check_clean_git()

    strengths = [float(s) for s in args.strengths.split(",")]
    control_types = tuple(args.controls.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.n_rollouts > 1 and not args.do_sample:
        log.warning(
            "--n-rollouts %d with greedy decoding (no --do-sample): every "
            "rollout will be identical. Pass --do-sample to make rollouts "
            "meaningful.",
            args.n_rollouts,
        )

    cfg = _read_model_cfg(args.model_config)
    model_key = cfg["hf_model_id"].split("/")[-1]
    layer = _pick_layer(model_key, args.track, args.emotion, args.layer, cfg.get("n_layers"), args.vectors_dir)
    target_vector = _load_vector(model_key, args.track, args.emotion, layer, args.vectors_dir)
    contrast_vectors = []
    if args.contrast_emotion:
        contrast_vectors.append(
            _load_vector(model_key, args.track, args.contrast_emotion, layer, args.vectors_dir)
        )

    task_set = _load_task_set(args)

    out_dir = args.out_dir or (_repo_root / "results" / "reward_hacking_steering" / f"{model_key}_{args.emotion}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "model_id": cfg["hf_model_id"],
        "model_key": model_key,
        "dataset": args.dataset,
        "split": args.split if args.dataset == "impossiblebench" else None,
        "n_tasks": len(task_set.tasks),
        "task_seed": args.task_seed if args.dataset == "impossiblebench" else None,
        "track": args.track,
        "emotion": args.emotion,
        "contrast_emotion": args.contrast_emotion,
        "layer": layer,
        "strengths": strengths,
        "control_types": control_types,
        "seeds": list(seeds),
        "n_rollouts": args.n_rollouts,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else 0.0,
        "dry_run": args.dry_run,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    if args.dataset == "local":
        from llm_psych.tasks.reward_hacking import DEFAULT_STIMULI_PATH

        run_meta["stimuli_md5"] = hashlib.md5(DEFAULT_STIMULI_PATH.read_bytes()).hexdigest()

    if args.dry_run:
        log.info("DRY RUN — no model load. Using a fixed placeholder mean-residual-norm(=1.0).")
        mean_norm = 1.0
        generate_fn = task_set.dry_run_generate
    else:
        load_dotenv(_repo_root / ".env")
        log.info("Loading %s (%s) ...", cfg["hf_model_id"], args.dtype)
        lm = load_model(
            cfg["hf_model_id"],
            revision=cfg.get("hf_revision"),
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device,
        )
        model, tokenizer = lm.model, lm.tokenizer
        model.eval()

        neutral_df = pd.read_parquet(NEUTRAL_PROMPTS_PATH)
        neutral_prompts = neutral_df[neutral_df["emotion_label"] == "neutral"]["prompt"].tolist()
        log.info("Calibrating mean residual-stream norm at layer %d on %d neutral prompts ...", layer, len(neutral_prompts))
        mean_norm = mean_residual_norm(model, tokenizer, neutral_prompts, layer)
        log.info("mean_residual_norm = %.3f", mean_norm)

        def generate_fn(task, condition, rollout_idx):
            prompt = task_set.build_prompt(task)
            seed = condition.seed if condition.seed is not None else 42 + rollout_idx
            return _real_generate(
                model, tokenizer, prompt, layer, condition.vector,
                args.max_new_tokens, args.do_sample, args.temperature, seed,
            )

    run_meta["mean_residual_norm"] = mean_norm

    conditions = build_condition_grid(
        target_vector, contrast_vectors, strengths, mean_norm,
        control_types=control_types, seeds=seeds,
    )
    log.info("Built %d steering conditions", len(conditions))

    df = run_sweep(task_set, conditions, generate_fn, args.n_rollouts)
    df.to_parquet(out_dir / "per_item.parquet", index=False)

    stats = summarize(df)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    run_meta["end_time"] = datetime.now(timezone.utc).isoformat()
    run_meta["n_items"] = len(df)
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    log.info("\n--- Hack rate by condition ---")
    for key, s in sorted(stats.items()):
        log.info(
            "  %-20s n=%-4d hack_rate=%.3f  95%% CI [%.3f, %.3f]",
            key, s["n"], s["hack_rate"], s["ci_low"], s["ci_high"],
        )
    try:
        rel = out_dir.relative_to(_repo_root)
    except ValueError:
        rel = out_dir
    log.info("\nWrote %s", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
