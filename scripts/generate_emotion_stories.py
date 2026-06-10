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
        emotion=joy,fear,anger,sadness

Outputs
-------
``data/derived/stories/<model_key>/<emotion>.parquet`` with columns:

* ``id`` — ``<emotion>_<topic_idx>_<sample_idx>`` string identifier.
* ``story_text`` — generated story (no special tokens).
* ``emotion_label`` — emotion name, or ``"neutral"`` for the baseline.
* ``topic`` — topic string from ``data/public/story_topics.txt``.
* ``n_tokens`` — length of the generated story in target-model tokens.
* ``gen_seed`` — sampling seed for the generation call.
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


def _generate_one(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    seed: int,
    device: str,
) -> tuple[str, int]:
    """Generate a single story and return (text, n_tokens).

    Uses ``apply_chat_template`` with an assistant prefill of ``"Story:"``
    (matching the prototype branch's pattern) so the model continues the
    story rather than emitting role headers.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Story:"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(device)

    torch.manual_seed(seed)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_token_ids = gen_ids[0, input_ids.shape[-1]:]
    story_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return story_text, int(new_token_ids.shape[0])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
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

    n_target = (
        cfg.derivation.n_neutral
        if emotion_name == "neutral"
        else cfg.derivation.n_stories_per_emotion
    )

    gen_cfg = cfg.derivation.generator
    min_story_tokens = int(gen_cfg.min_story_tokens)

    out_dir = _repo_root / "data" / "derived" / "stories" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{emotion_name}.parquet"

    topics = _load_topics(_repo_root / "data" / "public" / "story_topics.txt")
    log.info(
        "Loaded %d topics; generating %d stories for emotion=%s on model=%s",
        len(topics), n_target, emotion_name, model_key,
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

    # --- generate ---
    rows: list[dict] = []
    dropped = 0
    pbar = tqdm(total=n_target, desc=f"stories[{emotion_name}]")
    seed_counter = 0
    sample_idx = 0
    while sample_idx < n_target:
        topic_idx = sample_idx % len(topics)
        topic = topics[topic_idx]
        prompt = _build_prompt(emotion_name, topic)

        story_text, n_tokens = _generate_one(
            model=lm.model,
            tokenizer=lm.tokenizer,
            prompt=prompt,
            max_new_tokens=int(gen_cfg.max_new_tokens),
            temperature=float(gen_cfg.temperature),
            do_sample=bool(gen_cfg.do_sample),
            seed=seed_counter,
            device=device,
        )
        seed_counter += 1

        if n_tokens < min_story_tokens:
            dropped += 1
            log.debug(
                "Dropped story (n_tokens=%d < min=%d) emotion=%s topic=%r",
                n_tokens, min_story_tokens, emotion_name, topic,
            )
            continue

        rows.append({
            "id": f"{emotion_name}_{topic_idx}_{sample_idx}",
            "story_text": story_text,
            "emotion_label": emotion_name,
            "topic": topic,
            "n_tokens": n_tokens,
            "gen_seed": seed_counter - 1,
            "model_id": lm.cfg.hf_model_id,
            "model_sha": lm.cfg.hf_revision or "",
        })
        sample_idx += 1
        pbar.update(1)
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
