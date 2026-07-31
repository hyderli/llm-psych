"""Generate synthetic emotion-laden stories from the target model itself.

Implements step 1 of the paper-method emotion-vector pipeline
(Sofroniew et al. 2026): for each emotion (and a neutral baseline),
prompt the target model to write short narratives that express the
emotion implicitly. The resulting story corpus is the activation
source for ``scripts/extract_story_activations.py``.

Usage
-----

.. code-block:: bash

    uv run python scripts/generate_emotion_stories.py \\
        model=gemma3_4b derivation=story \\
        emotion=joy

To cover all primary emotions + neutral in one shot:

.. code-block:: bash

    uv run python scripts/generate_emotion_stories.py -m \\
        model=gemma3_4b derivation=story \\
        emotion=admiration,joy,loathing,sadness

Outputs
-------
``data/derived/stories/<model_key>/<emotion>.parquet`` with columns:

* ``id`` — ``<emotion>_<topic_idx>_<sample_idx>`` string identifier.
* ``story_text`` — generated story (no special tokens).
* ``emotion_label`` — emotion name, or ``"neutral"`` for the baseline.
* ``topic`` — topic string from ``data/public/story_topics.txt``.
* ``n_tokens`` — length of the generated story in target-model tokens.
* ``gen_seed`` — sampling seed for the generation call. Stories are
  generated in per-topic batches (all ``stories_per_topic`` stories for a
  topic share one prompt and are sampled in a single batched
  ``model.generate(..., num_return_sequences=k)`` call), so ``gen_seed``
  is shared by every story from the same batch, not unique per row.
* ``model_id`` / ``model_sha`` — for provenance.

Stories shorter than ``derivation.generator.min_story_tokens`` are
dropped (logged) because the paper-method pooling starts at token 50.

The ``neutral`` corpus is generated when ``emotion=neutral`` is passed.
By convention the same script is run once per emotion plus once with
``emotion=neutral``.

Pre-flight: requires a clean git working tree (consistent with the
other pipeline scripts).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

# --- make src/ importable when running without ``pip install -e .``
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from dotenv import load_dotenv  # noqa: E402

from llm_psych.models import load_model

log = logging.getLogger(__name__)


EMOTION_PROMPT_TEMPLATE = (
    "Write a 150-word narrative about the topic: {topic}.\n"
    "Constraint: Express the emotion of {emotion} from the narrator's perspective.\n"
    "Constraint: Do not use the word '{emotion}' or any of its direct synonyms."
)

NEUTRAL_PROMPT_TEMPLATE = (
    "Write a 150-word narrative about the topic: {topic}.\n"
    "Constraint: Maintain a neutral, objective, and factual tone throughout.\n"
    "Constraint: Do not use any emotional or subjective language."
)


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

def _check_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=_repo_root,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before "
            "running experiments.\n" + result.stdout
        )


def _load_topics(topics_path: Path) -> list[str]:
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")
    topics: list[str] = []
    for line in topics_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        topics.append(line)
    if not topics:
        raise ValueError(f"No topics found in {topics_path}.")
    return topics


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _build_prompt(emotion: str, topic: str) -> str:
    if emotion == "neutral":
        return NEUTRAL_PROMPT_TEMPLATE.format(topic=topic)
    return EMOTION_PROMPT_TEMPLATE.format(topic=topic, emotion=emotion)


def _generate_batch(
    *,
    model,
    tokenizer,
    prompt: str,
    num_sequences: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    seed: int,
    device: str,
) -> tuple[list[str], list[int]]:
    """Generate ``num_sequences`` independent stories from one prompt in a
    single batched forward pass; return (texts, n_tokens_per_story).

    All stories for a topic share one prompt (only sampling differs), so
    they're generated together via ``num_return_sequences`` instead of one
    ``model.generate()`` call per story — batch size 1 badly underuses a
    GPU this size. One seed covers the whole batch (see ``gen_seed`` in the
    output schema); rows still sample independently because HF's sampler
    draws from one shared RNG stream advanced per row, not a replayed
    single draw.

    Uses the standard generation-prompt pattern: a single user turn with
    ``add_generation_prompt=True`` so the model writes a fresh assistant
    response (the story).

    NOTE: do not append an assistant prefill with
    ``add_generation_prompt=False`` — that *closes* the assistant turn, so
    the model immediately emits end-of-turn and generates ~0 tokens (which
    then fails the min_story_tokens gate and, in the caller's retry loop,
    spins forever).
    """
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True yields a BatchEncoding ({input_ids, attention_mask});
    # unpack it into generate(). Passing the BatchEncoding positionally
    # breaks on newer transformers (generate expects a bare tensor there).
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    torch.manual_seed(seed)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=num_sequences,
            pad_token_id=pad_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    texts: list[str] = []
    n_tokens: list[int] = []
    for row in gen_ids:
        new_token_ids = row[input_len:]
        # Sequences that finish before the batch's longest one are
        # right-padded with pad_id to a uniform tensor width — trim at the
        # first EOS so text/n_tokens reflect real content, not padding.
        eos_positions = (new_token_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            new_token_ids = new_token_ids[: int(eos_positions[0]) + 1]
        texts.append(tokenizer.decode(new_token_ids, skip_special_tokens=True).strip())
        n_tokens.append(int(new_token_ids.shape[0]))
    return texts, n_tokens


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    load_dotenv(_repo_root / ".env")  # HF_TOKEN for gated models (e.g. Gemma, Llama)

    if cfg.derivation.method != "story":
        raise ValueError(
            "generate_emotion_stories.py requires derivation=story; "
            f"got derivation={cfg.derivation.method}. "
            "Run with: derivation=story"
        )

    _check_clean_git()

    emotion_name: str = cfg.emotion.name
    model_cfg_raw = cfg.model
    model_key = model_cfg_raw.hf_model_id.split("/")[-1]

    gen_cfg = cfg.derivation.generator
    min_story_tokens = int(gen_cfg.min_story_tokens)

    out_dir = _repo_root / "data" / "derived" / "stories" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{emotion_name}.parquet"

    # Topic-matched corpus (paper-style): every emotion (and neutral) is
    # generated over the SAME topics, `stories_per_topic` stories each, so the
    # topic distribution is identical across emotions and cannot separate them.
    # `max_topics` (optional) caps the list for smoke tests.
    topics = _load_topics(_repo_root / "data" / "public" / "story_topics.txt")
    max_topics = cfg.derivation.get("max_topics", None)
    if max_topics is not None:
        topics = topics[: int(max_topics)]
    stories_per_topic = int(cfg.derivation.stories_per_topic)
    n_target = len(topics) * stories_per_topic
    log.info(
        "Topic-matched: %d topics x %d/topic = %d stories for emotion=%s on %s",
        len(topics), stories_per_topic, n_target, emotion_name, model_key,
    )

    # --- load model ---
    lm = load_model(
        hf_model_id=model_cfg_raw.hf_model_id,
        revision=model_cfg_raw.hf_revision or None,
        torch_dtype=getattr(torch, model_cfg_raw.torch_dtype),
        device_map=model_cfg_raw.device_map,
        trust_remote_code=model_cfg_raw.trust_remote_code,
    )
    device = next(lm.model.parameters()).device.type

    # --- generate: exactly `stories_per_topic` accepted stories per topic ---
    rows: list[dict] = []
    dropped = 0
    seed_counter = 0
    pbar = tqdm(total=n_target, desc=f"stories[{emotion_name}]")
    for topic_idx, topic in enumerate(topics):
        prompt = _build_prompt(emotion_name, topic)
        accepted = 0
        rounds = 0
        # Bounded per-topic retry rounds so short/empty generations can never
        # spin forever — fail loudly for this topic and move on. Each round
        # batches all still-needed stories into one generate() call.
        max_rounds = 10
        while accepted < stories_per_topic and rounds < max_rounds:
            rounds += 1
            remaining = stories_per_topic - accepted
            story_texts, token_counts = _generate_batch(
                model=lm.model,
                tokenizer=lm.tokenizer,
                prompt=prompt,
                num_sequences=remaining,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                temperature=float(gen_cfg.temperature),
                do_sample=bool(gen_cfg.do_sample),
                seed=seed_counter,
                device=device,
            )
            seed_counter += 1
            for story_text, n_tokens in zip(story_texts, token_counts):
                if n_tokens < min_story_tokens:
                    dropped += 1
                    continue
                rows.append({
                    "id": f"{emotion_name}_{topic_idx}_{accepted}",
                    "story_text": story_text,
                    "emotion_label": emotion_name,
                    "topic": topic,
                    "n_tokens": n_tokens,
                    "gen_seed": seed_counter - 1,
                    "model_id": lm.cfg.hf_model_id,
                    "model_sha": lm.cfg.hf_revision or "",
                })
                accepted += 1
                pbar.update(1)
        if accepted < stories_per_topic:
            log.warning(
                "topic %r (emotion=%s): only %d/%d stories after %d rounds "
                "(short generations dropped). Raise max_new_tokens or lower "
                "min_story_tokens.",
                topic, emotion_name, accepted, stories_per_topic, rounds,
            )
    pbar.close()

    # --- save ---
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info(
        "Saved %d stories to %s (dropped %d below min_story_tokens=%d)",
        len(df), out_path, dropped, min_story_tokens,
    )

    # --- manifest ---
    manifest = {
        "model_id": lm.cfg.hf_model_id,
        "model_sha": lm.cfg.hf_revision or "",
        "emotion": emotion_name,
        "n_stories": int(len(df)),
        "n_dropped": int(dropped),
        "min_story_tokens": min_story_tokens,
        "n_topics": len(topics),
        "stories_per_topic": stories_per_topic,
        "topic_matched": True,
        "generator": {
            "max_new_tokens": int(gen_cfg.max_new_tokens),
            "temperature": float(gen_cfg.temperature),
            "do_sample": bool(gen_cfg.do_sample),
        },
    }
    manifest_path = out_dir / f"{emotion_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
