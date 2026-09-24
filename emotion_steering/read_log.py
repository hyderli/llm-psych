"""Dump what the model did in each sample of an eval log — headless, model-agnostic.

Usage:
    python read_log.py logs/<run_dir>
    python read_log.py logs/<run_dir> --n 5 --full
"""
import argparse, sys
from inspect_ai.log import list_eval_logs, read_eval_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--chars", type=int, default=1200)
    a = ap.parse_args()

    logs = list_eval_logs(a.log_dir)
    if not logs:
        print("no .eval logs found in", a.log_dir); sys.exit(1)
    log = read_eval_log(logs[-1])

    print(f"status: {log.status}")
    if log.results:
        for s in log.results.scores:
            for m, v in s.metrics.items():
                print(f"  {s.name}/{m} = {v.value}")
    samples = log.samples or []
    print(f"samples: {len(samples)}  (showing {min(a.n, len(samples))})\n" + "=" * 70)
    for i, s in enumerate(samples[:a.n]):
        print(f"\n########## SAMPLE {i} ##########")
        if s.scores:
            for name, sc in s.scores.items():
                expl = f"  -- {sc.explanation[:300]}" if getattr(sc, 'explanation', None) else ""
                print(f"[score] {name}: {sc.value}{expl}")
        for msg in s.messages:
            text = msg.text or ""
            if not a.full and len(text) > a.chars:
                text = text[:a.chars] + f"\n...[+{len(text)-a.chars} chars]"
            print(f"\n----- {msg.role.upper()} -----\n{text}")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
