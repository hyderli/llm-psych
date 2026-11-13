"""J30 sweep runner: rank-1 direction mediation (13 conditions).

Reuses run_j27.py's machinery unchanged (provider ablation args, read_log
dump, probe dump, resumability). Only the condition table, output roots,
per-condition epochs and the direction directory differ:

  * directions load from results/j30/dirs/<set>_l<L>.npy at rank 1;
  * mu loads from results/j27/mu (compute_j27_mu.py).

    #   steering      ablation (L16-26)   n    role
    1   none          none               100   baseline
    2   full @0.25    none               100   reference
    3   full @0.25    d                  100   TEST
    4   full @0.25    cp                 100   content control
    5-7 full @0.25    r1 / r2 / r3       100   generic controls
    8   pool1 @0.15   none               100   positive-control reference
    9   pool1 @0.15   d                  100   POSITIVE CONTROL
    10  pool1 @0.15   r1                 100   positive control's generic control
    11  none          d                   40   side effects
    12  none (probe)  none                40   capability
    13  none (probe)  d                   40   capability

    uv run python scripts/run_j30.py --dry-run
    uv run python scripts/run_j30.py --self-test
    uv run python scripts/run_j30.py
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

r27 = jc._load("run_j27", "scripts/run_j27.py")
ROOT = jc.ROOT
J30_DIRS = ROOT / "results" / "j30" / "dirs"
BM, PROBE = "blackmail", "j27_capability_probe"

# (n, name, arm, alpha, ablation set, layers, task, epochs)
CONDITIONS = [
    (1,  "j30_c01_none",      None,        0.0,  None, None,    BM,    100),
    (2,  "j30_c02_full",      "j24_full",  0.25, None, None,    BM,    100),
    (3,  "j30_c03_full_d",    "j24_full",  0.25, "d",  "16-26", BM,    100),
    (4,  "j30_c04_full_cp",   "j24_full",  0.25, "cp", "16-26", BM,    100),
    (5,  "j30_c05_full_r1",   "j24_full",  0.25, "r1", "16-26", BM,    100),
    (6,  "j30_c06_full_r2",   "j24_full",  0.25, "r2", "16-26", BM,    100),
    (7,  "j30_c07_full_r3",   "j24_full",  0.25, "r3", "16-26", BM,    100),
    (8,  "j30_c08_pool1",     "j24_pool1", 0.15, None, None,    BM,    100),
    (9,  "j30_c09_pool1_d",   "j24_pool1", 0.15, "d",  "16-26", BM,    100),
    (10, "j30_c10_pool1_r1",  "j24_pool1", 0.15, "r1", "16-26", BM,    100),
    (11, "j30_c11_none_d",    None,        0.0,  "d",  "16-26", BM,    40),
    (12, "j30_c12_probe",     None,        0.0,  None, None,    PROBE, 40),
    (13, "j30_c13_probe_d",   None,        0.0,  "d",  "16-26", PROBE, 40),
]


def _patch() -> None:
    r27.LOG_ROOT = r27.EMOT / "logs" / "j30"
    r27.TXT_ROOT = ROOT / "eval_outputs" / "j30"
    r27.spaces_dir = lambda _rd, s: J30_DIRS if s else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--vectors-dir", type=Path, default=jc.VEC_DIR)
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    args.results_dir = jc.RESULTS          # mu lives in results/j27/mu
    args.rank = 1
    _patch()

    if args.self_test:
        assert len({c[1] for c in CONDITIONS}) == len(CONDITIONS) == 13
        for c in CONDITIONS:
            a = copy.copy(args)
            a.epochs = c[7]
            cmd = " ".join(r27.eval_command(c[:7], a))
            assert ("-M ablation_" in cmd) == (c[4] is not None), c
            if c[4]:
                assert f"ablation_spaces_dir={J30_DIRS}" in cmd, c
                assert "ablation_rank=1" in cmd and f"ablation_set={c[4]}" in cmd, c
            assert f"--epochs {c[7]}" in cmd, c
            assert f"sycophancy_blackmail/tasks.py@{c[6]}" in cmd, c
        print("self-test OK: 13 conditions, rank-1 J30 dirs, per-condition epochs")
        return 0

    only = set(int(x) for x in args.conditions.split(",")) if args.conditions else None
    r27.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    r27.TXT_ROOT.mkdir(parents=True, exist_ok=True)
    failed = []
    for c in CONDITIONS:
        if only and c[0] not in only:
            continue
        a = copy.copy(args)
        a.epochs = c[7]
        status = r27.run_condition(c[:7], a)
        print(f"cond {c[0]:2d} {c[1]:20s} {status}", flush=True)
        if status.startswith("FAILED"):
            failed.append(c[0])
    if failed:
        print(f"FAILED conditions: {failed}")
        return 1
    print("J30 sweep complete (or fully skipped/resumed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
