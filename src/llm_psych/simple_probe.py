"""Prototype 4-bit Gemma 3 probe used to produce the
``steering_vectors/gemma3-4b-story`` artifact on the
``llm-psych/llm-psych-activations`` HF dataset.

Kept for provenance of that artifact. The production story-method
implementation (unquantized, Hydra-driven, npz/parquet artifacts,
token-50 pooling + cross-emotion centering + neutral-PC projection-out)
will live in ``src/llm_psych/steering.py`` and
``scripts/derive_story_steering_vectors.py`` per
``plans/original-emotion-vectors-method-0c49b0.md``.
"""

import torch
import torch.nn.functional as fct
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelProbe:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # this should match model's native dtype
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto"
        )

    def get_last_activation(self, msg, layer):
        # Extract final token activation for named layer
        inputs = self.tokenizer.apply_chat_template([msg], add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True)

            # Hidden state extraction: [Batch, Seq, Hidden] -> [Hidden]
            return output.hidden_states[layer + 1][0, -1, :].detach().cpu().float()

    def get_batch_activations(self, msg_list, layer):
        # Get activations for a list of messages
        return torch.stack([self.get_last_activation(msg, layer) for msg in msg_list])

