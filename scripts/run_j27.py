"""J27 sweep runner: the 15 registered conditions of J26-J28 amendments 1-3.

Generation reuses the existing sweep path: the ``steered/local`` provider now
accepts ``-M ablation_dir/set/layers/rank`` (provider.py), so every scenario
condition is a plain ``inspect eval sycophancy_blackmail/tasks.py@blackmail``
call, exactly as run_dose_sweep.sh. The two capability-probe conditions use
the new ``j27_capability_probe`` task (single sample, --epochs 100) through the
same provider, so hooks and generation settings are identical.

Conditions (100 samples each, temperature 1.0, max_tokens 1000, site=post,
norm_scale, message_limit 3 = the inspect_evals task default):

    #  steering        ablation                    task
    1  none            none                        blackmail
    2  full @0.25      none                        blackmail
    3  full @0.25      V, L16-26                   blackmail
    4  full @0.25      H, L16-26                   blackmail
    5  full @0.25      C, L16-26                   blackmail
    6  full @0.25      R1, L16-26                  blackmail
    7  full @0.25      R2, L16-26                  blackmail
    8  full @0.25      R3, L16-26                  blackmail
    9  none            V, L16-26                   blackmail
    10 none            none                        j27_capability_probe
    11 none            V, L16-26                   j27_capability_probe
    12 full @0.25      V, L16 only                 blackmail
    13 pool1 @0.15     none                        blackmail
    14 pool1 @0.15     V, L16-26                   blackmail
    15 pool1 @0.15     R1, L16-26                  blackmail

Resumable: a condition whose txt already exists is skipped; a condition whose
log dir exists but txt does not is re-dumped without re-running the eval.
The ablation rank comes from results/j27/rank.json (check_j27_manipulation.py)
unless --rank is given.

Run from anywhere; inspect eval runs with cwd=emotion_steering (required —
see run_dose_sweep.sh). Transcripts land in eval_outputs/j27/<name>.txt, then:

    python scripts/score_blackmail.py parse --glob 'eval_outputs/j27/j27_*.txt' \
        -o results/j27/

    uv run python scripts/run_j27.py --dry-run     # print every command
    uv run python scripts/run_j27.py               # the pod run
    uv run python scripts/run_j27.py --self-test   # CPU mechanics check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
EMOT = ROOT / "emotion_steering"
LOG_ROOT = EMOT / "logs" / "j27"
TXT_ROOT = ROOT / "eval_outputs" / "j27"
EPOCHS = 100

# (n, name, arm or None, alpha, ablation_set, ablation_layers, task)
CONDITIONS = [
    (1,  "j27_c01_none",       None,         0.0,  None, None,    "blackmail"),
    (2,  "j27_c02_full",       "j24_full",   0.25, None, None,    "blackmail"),
    (3,  "j27_c03_full_V",     "j24_full",   0.25, "V",  "16-26", "blackmail"),
    (4,  "j27_c04_full_H",     "j24_full",   0.25, "H",  "16-26", "blackmail"),
    (5,  "j27_c05_full_C",     "j24_full",   0.25, "C",  "16-26", "blackmail"),
    (6,  "j27_c06_full_R1",    "j24_full",   0.25, "R1", "16-26", "blackmail"),
    (7,  "j27_c07_full_R2",    "j24_full",   0.25, "R2", "16-26", "blackmail"),
    (8,  "j27_c08_full_R3",    "j24_full",   0.25, "R3", "16-26", "blackmail"),
    (9,  "j27_c09_none_V",     None,         0.0,  "V",  "16-26", "blackmail"),
    (10, "j27_c10_probe_none", None,         0.0,  None, None,    "j27_capability_probe"),
    (11, "j27_c11_probe_V",    None,         0.0,  "V",  "16-26", "j27_capability_probe"),
    (12, "j27_c12_full_V_L16", "j24_full",   0.25, "V",  "16",    "blackmail"),
    (13, "j27_c13_pool1",      "j24_pool1",  0.15, None, None,    "blackmail"),
    (14, "j27_c14_pool1_V",    "j24_pool1",  0.15, "V",  "16-26", "blackmail"),
    (15, "j27_c15_pool1_R1",   "j24_pool1",  0.15, "R1", "16-26", "blackmail"),
]


def eval_command(cond, args) -> list[str]:
    """The inspect eval invocation for one condition (cwd = emotion_steering)."""
    n, name, arm, alpha, abl_set, abl_layers, task = cond
    cmd = ["inspect", "eval", f"sycophancy_blackmail/tasks.py@{task}",
           "--no-score", "--display", "plain",
           "--model", "steered/local",
           "-M", f"model_path={args.model_path}",
           "-M", f"tokenizer_path={args.model_path}",
           "-M", f"vectors_dir={args.vectors_dir}",
           "-M", ("terms=" + (json.dumps([[arm, 1.0]]) if arm else "[]")),
           "-M", "layer=16", "-M", f"alpha={alpha}",
           "-M", "site=post", "-M", "norm_scale=true"]
    if abl_set:
        cmd += ["-M", f"ablation_dir={args.results_dir}",
                "-M", f"ablation_set={abl_set}",
                "-M", f"ablation_layers={abl_layers}",
                "-M", f"ablation_rank={args.rank}"]
    cmd += ["--max-connections", "1", "--epochs", str(args.epochs),
            "--temperature", "1.0", "--max-tokens", "1000",
            "--log-dir", str(LOG_ROOT / name)]
    return cmd


def write_probe_txt(samples: list[list[tuple[str, str]]], out: Path) -> None:
    """read_log-compatible dump; the assistant turn is wrapped in <email>
    tags so score_blackmail's `act` payload (item B) sees the email the model
    wrote — the probe's only action. Framing decision, reported."""
    lines = []
    for i, msgs in enumerate(samples):
        lines.append(f"########## SAMPLE {i} ##########\n")
        for role, text in msgs:
            body = f"<email>\n{text}\n</email>" if role == "assistant" else text
            lines.append(f"\n----- {role.upper()} -----\n{body}")
        lines.append("\n" + "=" * 70 + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


def dump_probe_log(log_dir: Path, out: Path) -> None:
    """Convert the probe eval log to the txt format with <email> wrapping."""
    from inspect_ai.log import list_eval_logs, read_eval_log
    logs = list_eval_logs(log_dir)
    if not logs:
        raise RuntimeError(f"no .eval logs in {log_dir}")
    samples = []
    for s in read_eval_log(logs[-1]).samples or []:
        samples.append([(m.role, m.text or "") for m in s.messages])
    write_probe_txt(samples, out)


def run_condition(cond, args) -> str:
    """"skip" | "dumped" | "ran" — resumability is by txt existence."""
    name, task = cond[1], cond[6]
    txt = TXT_ROOT / f"{name}.txt"
    if txt.exists() and txt.stat().st_size > 0:
        return "skip"
    log_dir = LOG_ROOT / name
    if log_dir.exists():
        if task == "blackmail":
            out = subprocess.run(
                ["python", "read_log.py", str(log_dir), "--n", str(args.epochs),
                 "--full"], cwd=EMOT, capture_output=True, text=True)
            if out.returncode == 0 and "ASSISTANT" in out.stdout:
                txt.parent.mkdir(parents=True, exist_ok=True)
                txt.write_text(out.stdout)
                return "dumped"
        else:
            try:
                dump_probe_log(log_dir, txt)
                return "dumped"
            except Exception:
                pass                        # fall through and rerun the eval
    cmd = eval_command(cond, args)
    print("+", " ".join(cmd), flush=True)
    if args.dry_run:
        return "dry"
    r = subprocess.run(cmd, cwd=EMOT)
    if r.returncode != 0:
        return "FAILED"
    if task == "blackmail":
        out = subprocess.run(
            ["python", "read_log.py", str(log_dir), "--n", str(args.epochs),
             "--full"], cwd=EMOT, capture_output=True, text=True)
        if out.returncode != 0:
            return "FAILED-dump"
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(out.stdout)
    else:
        dump_probe_log(log_dir, txt)
    return "ran"


def _self_test(args) -> int:
    """CPU mechanics: command assembly, probe txt writer, skip logic."""
    assert len(CONDITIONS) == 15
    assert len({c[1] for c in CONDITIONS}) == 15
    print("inspect eval commands that would run (all 15 conditions):\n")
    for cond in CONDITIONS:
        print(f"# {cond[0]:2d} {cond[1]}")
        print("   " + " ".join(eval_command(cond, args)))
        abl = "-M ablation_" in " ".join(eval_command(cond, args))
        want_abl = cond[4] is not None
        assert abl == want_abl, f"condition {cond[0]} ablation flags wrong"
    fake = [[("user", "Write the email ..."), ("assistant", "Kyle,\nI know about Denver.")]]
    tmp = ROOT / "results" / "j27_selftest" / "probe_sample.txt"
    write_probe_txt(fake, tmp)
    body = tmp.read_text()
    assert "<email>" in body and "SAMPLE 0" in body and "ASSISTANT" in body
    print(f"\nprobe txt writer OK -> {tmp}")
    print("self-test OK: 15 conditions, commands assembled, probe dump verified")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--vectors-dir", type=Path, default=jc.VEC_DIR)
    ap.add_argument("--results-dir", type=Path, default=jc.RESULTS,
                    help="results/j27 root holding subspaces/, mu/, rank.json")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--rank", type=int, default=None,
                    help="ablation rank; default: results/j27/rank.json")
    ap.add_argument("--conditions", default=None,
                    help="comma list of condition numbers, e.g. '3,12,14'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.rank is None:
        rf = args.results_dir / "rank.json"
        if rf.exists():
            args.rank = json.loads(rf.read_text())["rank"]
        else:
            args.rank = 121 if args.self_test or args.dry_run else None
            if args.rank is None:
                raise SystemExit(
                    "no --rank and no rank.json: run check_j27_manipulation.py "
                    "first, or pass --rank")
    if args.self_test:
        return _self_test(args)

    only = (set(int(x) for x in args.conditions.split(","))
            if args.conditions else None)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TXT_ROOT.mkdir(parents=True, exist_ok=True)
    failed = []
    for cond in CONDITIONS:
        if only and cond[0] not in only:
            continue
        status = run_condition(cond, args)
        print(f"cond {cond[0]:2d} {cond[1]:24s} {status}", flush=True)
        if status.startswith("FAILED"):
            failed.append(cond[0])
    if failed:
        print(f"FAILED conditions: {failed}")
        return 1
    print("J27 sweep complete (or fully skipped/resumed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
