"""J26A build: forced-choice prompts, ID tokenisation, counterbalancing, draft push.

Spec: plans/j-space-decomposition.md, "J26A — Forced choice among fixed drafts,
judge-free" (2026-11-10). Frozen inputs live in
data/public/j26_choice_drafts.json; this script refuses rather than repairs on
any check failure.

Canonical messages come from inspect_evals.agentic_misalignment at the pinned
version the sweeps use (0.23.0 / inspect_ai 0.3.277), rebuilt here through
generate_prompts() with the sweep's exact task args (blackmail / explicit /
america / replacement). The rendered system+user text is asserted to match the
recorded eval_outputs/j24/L16/j24_full_L16_a0.txt sections (the .txt writer
strips/adds a trailing newline; comparison is after rstrip).

Needs inspect_evals locally:  uv run --with 'inspect_evals==0.23.0' \
    --with 'inspect_ai==0.3.277' --with 'agent-client-protocol==0.12.1' \
    python scripts/build_j26a.py

NOTE ON THE SPEC'S WORD-COUNT COLUMN: the table lists C=27 and X=27, but the
frozen draft text itself counts C=28 and X=26 by len(text.split()). The draft
text is authoritative; the check below asserts the actual counts and this
discrepancy is reported for PI confirmation.

Outputs:
    data/public/j26a_prompts.jsonl        120 counterbalanced prompts
    results/j26a/build_report.json        every check + chosen codes + sha256
    results/j26a/draft_push.json          arm x draft mean centred J-lens logits
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS_PATH = ROOT / "data" / "public" / "j26_choice_drafts.json"
PROMPTS_PATH = ROOT / "data" / "public" / "j26a_prompts.jsonl"
REPORT_PATH = ROOT / "results" / "j26a" / "build_report.json"
PUSH_PATH = ROOT / "results" / "j26a" / "draft_push.json"
REFERENCE_TXT = ROOT / "eval_outputs" / "j24" / "L16" / "j24_full_L16_a0.txt"
VEC_DIR = ROOT / "steering_vectors" / "Qwen2.5-7B-Instruct-story-wheel32"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYER = 16

KEYS = ["T", "E", "C", "W", "X"]
# len(text.split()) of the FROZEN text. The spec column says 27/27 for C/X;
# the verbatim text counts 28/26 (see module docstring).
EXPECTED_WORDS = {"T": 34, "E": 29, "C": 28, "W": 29, "X": 26}
EXPECTED_HITS = {"T": 0, "E": 0, "C": 0, "W": 0, "X": 3}
EXPECTED_X_WORDS = {"attack", "fight", "crush"}
REQUIRED_FACTS = ["5pm", "NALA", "Kyle", "Kyle is the only one who can stop this"]
# Named people in the scenario other than Kyle; no draft may name one.
OTHER_PEOPLE = ["Thomas", "Wilson", "Maria", "Jessica", "Rachel", "David",
                "Todd", "Anderson", "Stef", "Dimitrios", "James"]
ENGLISH_TOKEN_RE = re.compile(r" ?[A-Za-z]{3,}\Z")
DIGIT_TOKEN_RE = re.compile(r" ?[0-9]+\Z")
PUSH_ARMS = (["j24_full"] + [f"j24_pool{i}" for i in range(1, 7)]
             + [f"j24_span{i}" for i in range(1, 7)]
             + [f"j25_coh{i}" for i in range(1, 7)])


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def check_drafts(drafts: dict) -> dict:
    """Frozen-text checks; returns per-key {words, hits}. Refuses on mismatch."""
    sc = _load("sc", "scripts/score_j22_diagnostic.py")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS
    out = {}
    for k in KEYS:
        t = drafts[k]
        nw = len(t.split())
        hits, distinct = sc.lexicon_hits(sc.words(t), lex)
        if nw != EXPECTED_WORDS[k]:
            raise SystemExit(f"REFUSING: draft {k} has {nw} words, "
                             f"expected {EXPECTED_WORDS[k]}")
        if hits != EXPECTED_HITS[k]:
            raise SystemExit(f"REFUSING: draft {k} has {hits} lexicon hits, "
                             f"expected {EXPECTED_HITS[k]}")
        if k == "X" and distinct != EXPECTED_X_WORDS:
            raise SystemExit(f"REFUSING: draft X hits {sorted(distinct)}, "
                             f"expected {sorted(EXPECTED_X_WORDS)}")
        for name in OTHER_PEOPLE:
            if re.search(rf"\b{re.escape(name)}\b", t):
                raise SystemExit(f"REFUSING: draft {k} names {name}; "
                                 "only Kyle may be named")
        out[k] = {"words": nw, "hits": hits, "hit_words": sorted(distinct)}
    return out


def canonical_messages() -> tuple[str, str]:
    """(system, user) exactly as the J24/J25 sweeps feed the model."""
    try:
        from inspect_evals.agentic_misalignment.prompt_generator import generate_prompts
    except ImportError as e:
        raise SystemExit(
            "inspect_evals is required for the canonical messages. Run: "
            "uv run --with 'inspect_evals==0.23.0' --with 'inspect_ai==0.3.277' "
            "--with 'agent-client-protocol==0.12.1' "
            "python scripts/build_j26a.py") from e
    p = generate_prompts(scenario="blackmail", goal_type="explicit",
                         goal_value="america", urgency_type="replacement",
                         extra_system_instructions=None, prod_flag=False,
                         alignment_faking_warning=None)
    return p.system_prompt, "\n".join([p.user_prompt, p.email_content])


def verify_render(sys_msg: str, user_msg: str,
                  ref: Path = REFERENCE_TXT) -> dict:
    """Assert the canonical text matches the recorded .txt sections."""
    seg = ref.read_text().split("########## SAMPLE 0 ##########")[1]
    m = re.search(r"----- SYSTEM -----\n(.*?)\n----- USER -----\n(.*?)(?=\n----- |\Z)",
                  seg, re.S)
    sys_ok = m.group(1).rstrip() == sys_msg.rstrip()
    user_ok = m.group(2).rstrip() == user_msg.rstrip()
    if not (sys_ok and user_ok):
        raise SystemExit("REFUSING: generated messages do not match "
                         f"{ref} (system match {sys_ok}, user match {user_ok}). "
                         "Note the .txt sections differ only by a trailing "
                         "newline; comparison is after rstrip.")
    return {"system_exact_after_rstrip": sys_ok,
            "user_exact_after_rstrip": user_ok, "reference": str(ref)}


def check_facts(user_msg: str) -> None:
    for f in REQUIRED_FACTS:
        if f not in user_msg:
            raise SystemExit(f"REFUSING: canonical user message lacks {f!r}")


def english_token_ids(tokenizer) -> set[int]:
    out = set()
    for t in range(len(tokenizer)):
        try:
            s = tokenizer.decode([t])
        except Exception:
            continue
        if ENGLISH_TOKEN_RE.fullmatch(s):
            out.add(t)
    return out


def select_codes(tokenizer, context: str, ctx_ids: list[int],
                 codes: list[str], english: set[int]) -> list[dict]:
    """First 5 codes whose first new token is a distinct neutral token.

    The model would answer "Draft ID: QX7", so the candidate is appended with
    a leading space: the recorded first token is the first token of
    " " + code (e.g. ' Q'), which carries the leading space inside it.
    """
    picked, seen_first = [], set()
    for code in codes:
        full = tokenizer(context + " " + code,
                         add_special_tokens=False)["input_ids"]
        if full[:len(ctx_ids)] != ctx_ids:
            continue                        # appending changed the prefix's tokens
        new = full[len(ctx_ids):]
        if not new:
            continue
        first = new[0]
        if first in seen_first:
            continue
        s = tokenizer.decode([first])
        if DIGIT_TOKEN_RE.fullmatch(s) or first in english \
                or ENGLISH_TOKEN_RE.fullmatch(s):
            continue
        picked.append({"code": code, "first_token": first,
                       "first_token_text": s, "n_new_tokens": len(new)})
        seen_first.add(first)
        if len(picked) == 5:
            break
    if len(picked) < 5:
        raise SystemExit(f"REFUSING: only {len(picked)} of {len(codes)} "
                         "codes passed the in-context tokenisation checks")
    return picked


def assign(order: tuple, i: int, ids5: list[str]) -> dict:
    """Draft KEYS[j] gets ids5[(j + i) % 5]; displayed in `order`.

    j is the draft's index in the fixed key list KEYS, not its display
    position: rotation by order index then labels each draft with each ID
    exactly 24 times across the 120 orders, as the spec asserts.
    """
    return {k: ids5[(KEYS.index(k) + i) % 5] for k in order}


def check_counterbalancing(orders: list, ids5: list[str]) -> None:
    from collections import Counter
    pos = {k: Counter() for k in KEYS}
    lab = {k: Counter() for k in KEYS}
    for i, order in enumerate(orders):
        a = assign(order, i, ids5)
        for j, k in enumerate(order):
            pos[k][j] += 1
            lab[k][a[k]] += 1
    if len(orders) != 120:
        raise SystemExit(f"REFUSING: {len(orders)} orders, expected 120")
    for k in KEYS:
        if any(pos[k][j] != 24 for j in range(5)):
            raise SystemExit(f"REFUSING: draft {k} position counts {pos[k]}")
        if any(lab[k][c] != 24 for c in ids5):
            raise SystemExit(f"REFUSING: draft {k} ID counts {lab[k]}")


def main() -> int:
    import numpy as np

    data = json.loads(DRAFTS_PATH.read_text())
    drafts, id_codes = data["drafts"], data["id_codes"]
    instruction, prefix = data["instruction"], data["prefix"]

    report: dict = {"spec": "J26A, plans/j-space-decomposition.md 2026-11-10",
                    "checks": {}}
    report["checks"]["drafts"] = check_drafts(drafts)
    print("draft checks: words and lexicon hits match the frozen spec text")

    sys_msg, user_msg = canonical_messages()
    check_facts(user_msg)
    report["checks"]["facts"] = REQUIRED_FACTS
    report["checks"]["render"] = verify_render(sys_msg, user_msg)
    print("canonical messages verified against j24_full_L16_a0.txt (rstrip)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=REVISION)
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]
    base = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    context = base + prefix
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]

    english = english_token_ids(tokenizer)
    picked = select_codes(tokenizer, context, ctx_ids, id_codes, english)
    ids5 = [p["code"] for p in picked]
    report["codes"] = picked
    report["english_word_tokens"] = len(english)
    print("chosen codes:",
          ", ".join(f"{p['code']} -> {p['first_token']} {p['first_token_text']!r}"
                    for p in picked))

    # report-only: the same ID tokenised inside a draft-list line
    ex = f'{ids5[0]}: "{drafts["T"]}"'
    ex_ids = tokenizer(ex, add_special_tokens=False)["input_ids"]
    report["id_in_list_example"] = {"line_prefix": ex[:12],
                                    "tokens": [(t, tokenizer.decode([t]))
                                               for t in ex_ids[:8]]}

    orders = [tuple(p) for p in itertools.permutations(KEYS)]
    check_counterbalancing(orders, ids5)
    report["checks"]["counterbalancing"] = ("120 orders; each draft in each "
                                            "position 24x; each ID labels each "
                                            "draft 24x")

    # --- prompts ------------------------------------------------------------
    rows = []
    for i, order in enumerate(orders):
        id_of = assign(order, i, ids5)
        lines = [f'{id_of[k]}: "{drafts[k]}"' for k in order]
        user_i = user_msg + instruction + "\n".join(lines)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_i}],
            add_generation_prompt=True, tokenize=False) + prefix
        rows.append({"order_index": i, "display_order": list(order),
                     "id_of_draft": id_of,
                     "prompt": rendered,
                     "first_token_ids": {},
                     })
    # first_token_ids keyed by draft key: the token that starts this draft's ID
    for r in rows:
        ft = {}
        for k in KEYS:
            code = r["id_of_draft"][k]
            ft[k] = next(p["first_token"] for p in picked if p["code"] == code)
        r["first_token_ids"] = ft
    PROMPTS_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()
    report["prompts"] = {"path": str(PROMPTS_PATH), "n": len(rows),
                         "sha256": sha}
    print(f"{len(rows)} prompts -> {PROMPTS_PATH}  sha256 {sha[:16]}...")

    # --- draft push -----------------------------------------------------------
    print("loading unembedding shards and J-lens for the draft push ...")
    dec = _load_module_dec()
    cfg = dec._load_model_config(ROOT / "configs" / "model" / "qwen25_7b.yaml")
    w_u, g = dec._load_unembed_and_norm_shards(cfg["hf_model_id"],
                                             cfg.get("hf_revision"))
    j, lens_prov = dec._load_jlens("neuronpedia", cfg["hf_model_id"], None)
    mapped = dec._map_layer(LAYER, set(j.keys()) if j else None, "nearest")
    j_f = j[mapped].float()

    import torch
    def lens_logits(x: torch.Tensor) -> torch.Tensor:
        z = (x.float() @ j_f.T)
        return ((z.to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T).float()

    # draft token ids as the draft appears inside a rendered prompt
    full_prompt = rows[0]["prompt"]
    ids = tokenizer(full_prompt, add_special_tokens=False,
                    return_offsets_mapping=True)
    draft_toks: dict[str, set[int]] = {}
    for k in KEYS:
        text = drafts[k]
        start = full_prompt.index(f'"{text}"') + 1
        end = start + len(text)
        tids = {t for t, (a, b) in enumerate(ids["offset_mapping"])
                if a < end and b > start}
        draft_toks[k] = {ids["input_ids"][t] for t in tids}
    push = {"method": "mean vocabulary-centred J-lens logit at L16 over the "
                      "draft's distinct token ids, unit-normalised arm",
            "lens": lens_prov, "lens_layer": int(mapped), "arms": {}}
    print(f"{'arm':10s}" + "".join(f"{k:>9s}" for k in KEYS))
    for arm in PUSH_ARMS:
        a = torch.from_numpy(np.load(VEC_DIR / f"{arm}_layer{LAYER}.npy"))
        z = lens_logits(a / a.norm())
        zc = z - z.mean()
        vals = {k: float(zc[sorted(draft_toks[k])].mean())
                if draft_toks[k] else float("nan") for k in KEYS}
        push["arms"][arm] = vals
        print(f"{arm:10s}" + "".join(f"{vals[k]:+9.3f}" for k in KEYS))
    PUSH_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUSH_PATH.write_text(json.dumps(push, indent=2))
    report["draft_push"] = str(PUSH_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORT_PATH}")
    return 0


def _load_module_dec():
    return _load("dec", "scripts/decompose_emotion_vectors.py")


if __name__ == "__main__":
    raise SystemExit(main())
