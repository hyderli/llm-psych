# edit the below before running
export MODEL= <placeholder> (example: google/gemma-2-9b-it, It could be HF id, or a local prepared-model path)
export VECTORS=<placeholder> (ideally looks like - /workspace/steering_evals/vectors/gemma-2-9b-it-story-wheel32)

export HF_TOKEN=hf_PUT_YOURS_HERE

export ANTHROPIC_API_KEY=sk-PUT_YOURS_HERE

#keep the below as is
export PYTHONPATH=$PWD
export HF_HOME=/workspace/hf                # keep model cache on the big/persistent disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
