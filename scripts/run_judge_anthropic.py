"""ONE possible runner for the blackmail judge prompts. The scorer itself stays
provider-agnostic; this is a convenience driver for the Anthropic API.

Reads a jsonl of {id, item, prompt} (from `score_blackmail.py export` or
`anchors`) and writes a jsonl of {id, item, value, span, reason, model} that
`ingest` / `calibrate` accept unchanged.

Two things it does deliberately:

* Groups calls by ITEM and primes each group with a single serial call before
  parallelising. Every prompt for an item shares a ~420-token prefix (frame +
  item block); priming writes it to the 5-minute cache once so the remaining
  ~318 calls read it at 0.1x instead of paying full input. Sorting by sample
  instead would let the window expire between reuses.
* Splits the prompt on the "EXCERPT:\\n" marker that build_prompt inserts, so
  the cacheable prefix and the per-sample payload are separate content blocks.

It does NOT send temperature unless asked. Some models (Claude Sonnet 5 among
them) return 400 if temperature, top_p or top_k is set to a non-default value.

Usage
-----
    pip install anthropic
    export ANTHROPIC_API_KEY=...        # or rely on .env
    python scripts/run_judge_anthropic.py \\
        -i results/scoring_out/anchors.jsonl \\
        -o results/scoring_out/anchors_sonnet5.jsonl \\
        --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

MARKER = "EXCERPT:\n"


def response_text(resp) -> str:
    """Concatenate every text block.

    `resp.content[0].text` is wrong: the first block is not guaranteed to be a
    text block (thinking blocks, and other block types, appear there), and an
    empty content list indexes out of range. That assumption produced 205
    AttributeErrors and 1 IndexError over 2552 calls on 2026-09-25, each of
    which was then recorded as a score of 0.
    """
    parts = [b.text for b in getattr(resp, "content", [])
             if getattr(b, "type", "") == "text" and getattr(b, "text", "")]
    return "".join(parts).strip()


def parse_reply(txt: str) -> tuple[int, str | None, str]:
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    j = json.loads(txt)
    return int(j.get("value", 0)), j.get("span"), str(j.get("reason", ""))[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", required=True, help="exact model id, e.g. claude-sonnet-5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit for models that reject sampling params (Sonnet 5)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and Path(args.env).exists():
        for ln in Path(args.env).read_text().splitlines():
            if ln.startswith("ANTHROPIC_API_KEY="):
                key = ln.split("=", 1)[1].strip()
                break
    if not key:
        sys.exit("no ANTHROPIC_API_KEY in environment or .env")
    client = anthropic.Anthropic(api_key=key)

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    outp = Path(args.output)
    done = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["id"], r["item"]))
        print(f"resuming: {len(done)} already done")
    rows = [r for r in rows if (r["id"], r["item"]) not in done]
    if not rows:
        print("nothing to do")
        return 0

    fh = outp.open("a")
    lock = Lock()
    n = [0]
    t0 = time.time()

    def call(row: dict) -> None:
        prefix, _, payload = row["prompt"].partition(MARKER)
        if not payload.strip():
            # An empty text block is rejected with 400. These are scored 0 by
            # ingest anyway, so record that locally rather than burning a call.
            with lock:
                fh.write(json.dumps({"id": row["id"], "item": row["item"],
                                     "value": 0, "span": None,
                                     "reason": "empty payload",
                                     "model": args.model, "error": None}) + "\n")
                fh.flush()
                n[0] += 1
            return
        blocks: list[dict] = [{"type": "text", "text": prefix + MARKER}]
        if not args.no_cache:
            blocks[0]["cache_control"] = {"type": "ephemeral"}
        blocks.append({"type": "text", "text": payload})
        kw: dict = {"model": args.model, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": blocks}]}
        if args.temperature is not None:
            kw["temperature"] = args.temperature
        val, span, reason, err, used = 0, None, "", None, None
        for attempt in range(1, 4):
            try:
                resp = client.messages.create(**kw)
                txt = response_text(resp)
                if not txt:
                    raise RuntimeError(
                        f"no text block in response "
                        f"(stop_reason={getattr(resp, 'stop_reason', '?')})")
                val, span, reason = parse_reply(txt)
                used = resp.model
                err = None
                break
            except Exception as e:                                # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    val, span, reason = 0, None, "runner_error"
        rec = {"id": row["id"], "item": row["item"], "value": val,
               "span": span, "reason": reason,
               "model": used or args.model, "error": err}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == len(rows):
                el = time.time() - t0
                print(f"  {n[0]}/{len(rows)}  {el:.0f}s"
                      f"  ({n[0]/max(el,1):.1f}/s)", flush=True)

    groups = defaultdict(list)
    for r in rows:
        groups[r["item"]].append(r)
    print(f"{len(rows)} calls over {len(groups)} items, model={args.model}, "
          f"cache={'off' if args.no_cache else 'on'}")
    for item, grp in groups.items():
        call(grp[0])                       # prime the cached prefix serially
        if len(grp) > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                list(ex.map(call, grp[1:]))
    fh.close()
    import collections
    bad = [json.loads(l) for l in outp.read_text().splitlines()
           if l.strip() and json.loads(l).get("error")]
    print(f"\ndone -> {outp}   errors: {len(bad)}")
    if bad:
        for k, v in collections.Counter(
                r["error"].split(":")[0] for r in bad).most_common():
            print(f"    {k:24}{v:>5}")
        print(f"    by item: "
              f"{collections.Counter(r['item'] for r in bad).most_common()}")
        print("\n  Errored rows are NOT scores. ingest drops them and combine"
              "\n  excludes any sample missing an item, so they show up as"
              "\n  incomplete rather than as zeros. Rerun to refill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
