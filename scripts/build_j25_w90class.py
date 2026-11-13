"""J25 amendment 3 (2026-11-10): the w90 / w90ctl v-deletion as a class, i = 2..6.

`w90_i`   = j24_pool<i> minus its v-hat component, v-hat = unit(j24_full).
`w90ctl_i`= j24_span<i> minus its v-hat component. Same deletion as amendment 1's
single w90/w90ctl pair, which are the i = 1 members and are NOT rebuilt. No RNG.

Acceptance (fixed in the amendment; on any failure nothing is written):
  * every arm at 90 deg +/- 1e-3 to v-hat;
  * norm ratio to its source = sin(74.836 deg) = 0.96518 +/- 1e-4;
  * recomputing i = 1 gives cos >= 0.999999 and matching norm against the
    existing j25_w90_layer16.npy / j25_w90ctl_layer16.npy files.

Writes j25_w90_<i>_layer16.npy and j25_w90ctl_<i>_layer16.npy for i = 2..6
(float32, deletion norm kept, like build_j25_arms.py) and refuses to overwrite.
Report: results/j25/w90class_build_report.json.

    uv run python scripts/build_j25_w90class.py --dry-run   # report only
    uv run python scripts/build_j25_w90class.py           # report + .npy arms
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VEC_DIR = ROOT / "steering_vectors" / "Qwen2.5-7B-Instruct-story-wheel32"
REPORT = ROOT / "results" / "j25" / "w90class_build_report.json"
DRY_REPORT = ROOT / "results" / "j25" / "w90class_dryrun_report.json"
LAYER = 16
N = 6
ANGLE_TOL = 1e-3                       # degrees
EXPECTED_RATIO = 0.96518               # sin(74.836 deg)
RATIO_TOL = 1e-4
I1_COS_MIN = 0.999999
I1_NORM_TOL = 1e-5                     # relative, "matching norm"


def _load(name: str) -> np.ndarray:
    f = VEC_DIR / f"{name}_layer{LAYER}.npy"
    if not f.exists():
        raise SystemExit(f"REFUSING: {f} is missing")
    return np.load(f).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="run all checks and write the report, but no .npy files")
    args = ap.parse_args()
    report_path = DRY_REPORT if args.dry_run else REPORT

    v = _load("j24_full")
    vhat = v / np.linalg.norm(v)

    arms: dict[str, np.ndarray] = {}
    report = {"spec": "J25 amendment 3, plans/j-space-decomposition.md 2026-11-10",
              "layer": LAYER, "v_source": "j24_full", "expected_ratio": EXPECTED_RATIO,
              "angles_deg": {}, "norm_ratios": {}, "i1_recompute": {},
              "files_written": {}, "dry_run": bool(args.dry_run)}
    fails: list[str] = []

    for fam, src in (("w90", "j24_pool"), ("w90ctl", "j24_span")):
        for i in range(1, N + 1):
            a = _load(f"{src}{i}")
            w = a - (a @ vhat) * vhat
            name = f"{fam}_layer{LAYER}" if i == 1 else f"{fam}_{i}_layer{LAYER}"
            arms[name] = w
            ang = float(np.degrees(np.arccos(
                np.clip(float((w / np.linalg.norm(w)) @ vhat), -1, 1))))
            ratio = float(np.linalg.norm(w) / np.linalg.norm(a))
            report["angles_deg"][name] = ang
            report["norm_ratios"][name] = ratio
            if abs(ang - 90.0) > ANGLE_TOL:
                fails.append(f"{name}: angle {ang:.6f} != 90 +/- {ANGLE_TOL}")
            if abs(ratio - EXPECTED_RATIO) > RATIO_TOL:
                fails.append(f"{name}: norm ratio {ratio:.6f} != "
                             f"{EXPECTED_RATIO} +/- {RATIO_TOL}")

    # i = 1 must reproduce the files build_j25_arms.py already wrote.
    for fam in ("w90", "w90ctl"):
        new = arms[f"{fam}_layer{LAYER}"]
        old = _load(f"j25_{fam}")
        cos = float(new @ old / (np.linalg.norm(new) * np.linalg.norm(old)))
        rel = abs(np.linalg.norm(new) - np.linalg.norm(old)) / np.linalg.norm(old)
        report["i1_recompute"][fam] = {"cos": cos, "rel_norm_diff": rel}
        if cos < I1_COS_MIN:
            fails.append(f"j25_{fam}: i=1 recompute cos {cos:.9f} < {I1_COS_MIN}")
        if rel > I1_NORM_TOL:
            fails.append(f"j25_{fam}: i=1 norm differs by {rel:.3g} relative")

    report["acceptance"] = {"pass": not fails, "failures": fails}
    print(json.dumps(report, indent=2))
    if fails:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        raise SystemExit("ACCEPTANCE FAILED (J25 amendment 3): no arm files "
                         f"written. Report -> {report_path}")

    # Only i = 2..6 are new; refuse to overwrite anything that exists.
    todo = {nm: a for nm, a in arms.items()
            if nm not in (f"w90_layer{LAYER}", f"w90ctl_layer{LAYER}")}
    for nm in todo:
        out = VEC_DIR / f"j25_{nm}.npy"
        if out.exists():
            raise SystemExit(f"REFUSING to overwrite {out}")

    if args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"dry run: no arm files written. Report -> {report_path}")
        return 0

    for nm, a in sorted(todo.items()):
        out = VEC_DIR / f"j25_{nm}.npy"
        np.save(out, a.astype(np.float32))
        report["files_written"][out.name] = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"wrote {out.name}  sha256 {report['files_written'][out.name][:16]}...")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
