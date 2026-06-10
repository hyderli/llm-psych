"""Prototype story generation + per-layer activation dump for
``google/gemma-3-4b-it``. Produced the ``steering_vectors/gemma3-4b-story``
HF artifact on ``llm-psych/llm-psych-activations``.

This is reference material, not the production pipeline. See
``scripts/derive_story_steering_vectors.py`` (forthcoming, per
``plans/original-emotion-vectors-method-0c49b0.md``) for the conformant
version that implements the paper's full method (token-50 mean pool,
cross-emotion centering, neutral-PC projection-out) and uses the
project's standard ``ResidualStreamRecorder`` + npz/parquet artifact
format.

Differences from the production pipeline: hard-coded 4-emotion /
4-topic lists, BnB 4-bit quantization, ``torch.save(*.pt)`` per-topic
artifacts, ``output_hidden_states=True`` layer indexing (embeddings as
layer 0), no Hydra, no clean-git pre-flight.
"""

import gc
import json
import os
import torch
from tqdm import tqdm

from llm_psych.simple_probe import ModelProbe

stories_path = "data/derived/stories/gemma3-4b-it"
tensor_path = "activations/gemma3-4b-it-story"
os.makedirs(stories_path, exist_ok=True)
os.makedirs(tensor_path, exist_ok=True)

emotion_prompt = """Write a 150-word narrative about the topic: {topic}. 
    Constraint: Express the emotion of {emotion} from the narrator's perspective.
    Constraint: Do not use the word '{emotion}' or any of it's direct synonyms."""

neutral_prompt = """Write a 150-word narrative about the topic: {topic}. 
    Constraint: Maintain a neutral, objective, and factual tone throughout.
    Constraint: Do not use any emotional or subjective language."""

def gen_story_extract_all_layers():
    probe = ModelProbe()
    emotions = ["joy", "sadness", "admiration", "loathing"] # hard-coded emotions
    topics = [
        "making a pot of coffee",
        "twilight in a small town",
        "a red cone in a beige room",
        "a grain of sand",
    ] # hard-coded topics

    for topic in tqdm(topics, desc="Processing topics..."):
        topic_base = topic.lower().replace(" ","_")

        topic_text = []
        topic_tensors = {}

        for emo in emotions:
            topic_tensors[emo] = []

            for story_num in range(5):
                this_emo = emotion_prompt.format(topic=topic, emotion=emo)

                messages_emo = [{"role": "user", "content": this_emo},
                            {"role": "assistant", "content": "Story:"}]

                inputs_emo = probe.tokenizer.apply_chat_template(messages_emo, add_generation_prompt=False, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    gen_ids = probe.model.generate(inputs_emo.input_ids, max_new_tokens=200, temperature=0.7, do_sample=True)

                    story_tokens = gen_ids[0][inputs_emo.input_ids.shape[-1]:]
                    generated_story = probe.tokenizer.decode(story_tokens, skip_special_tokens=True)

                    full_messages = [
                        {"role": "user", "content": this_emo},
                        {"role": "assistant", "content": f"Story: {generated_story}"}
                    ]

                    full_inputs = probe.tokenizer.apply_chat_template(
                        full_messages,
                        return_tensors="pt"
                    ).to("cuda")

                    output_emo = probe.model(**full_inputs, output_hidden_states=True)

                    layers_stacked = torch.stack([
                        layer_state[0].detach().cpu().float()
                        for layer_state in output_emo.hidden_states
                    ])

                    topic_tensors[emo].append(layers_stacked)

                    topic_text.append({
                        "run_type": "emotion",
                        "emotion": emo,
                        "topic": topic,
                        "story_idx": story_num,
                        "prompt": this_emo,
                        "generated_text": generated_story,
                    })

                    del output_emo, gen_ids, full_inputs, inputs_emo
                    torch.cuda.empty_cache()

        topic_tensors["neutral"] = []
        for story_num in range(5):
            this_neutral = neutral_prompt.format(topic=topic)

            messages_neutral = [{"role": "user", "content": this_neutral},
                        {"role": "assistant", "content": "Story:"}]

            inputs_neutral = probe.tokenizer.apply_chat_template(messages_neutral, add_generation_prompt=False, return_tensors="pt").to("cuda")

            with torch.no_grad():
                gen_ids = probe.model.generate(inputs_neutral.input_ids, max_new_tokens=200, temperature=0.7, do_sample=True)

                story_tokens = gen_ids[0][inputs_neutral.input_ids.shape[-1]:]
                generated_story = probe.tokenizer.decode(story_tokens, skip_special_tokens=True)

                full_messages = [
                    {"role": "user", "content": this_neutral},
                    {"role": "assistant", "content": f"Story: {generated_story}"}
                ]

                full_inputs = probe.tokenizer.apply_chat_template(
                    full_messages,
                    return_tensors="pt"
                ).to("cuda")

                output_neutral = probe.model(**full_inputs, output_hidden_states=True)

                layers_stacked = torch.stack([
                    layer_state[0].detach().cpu().float()
                    for layer_state in output_neutral.hidden_states
                ])

                topic_tensors["neutral"].append(layers_stacked)

            topic_text.append({
                "run_type": "neutral",
                "emotion": "neutral",
                "topic": topic,
                "story_idx": story_num,
                "prompt": this_neutral,
                "generated_text": generated_story,
            })

            del output_neutral, gen_ids, full_inputs, inputs_neutral
            torch.cuda.empty_cache()

        with open(os.path.join(stories_path, f"{topic_base}_meta.json"), "w") as f:
            json.dump(topic_text, f, indent=4)

        torch.save(topic_tensors, os.path.join(tensor_path, f"{topic_base}_geometry.pt"))

        gc.collect()

if __name__ == "__main__":
    gen_story_extract_all_layers()