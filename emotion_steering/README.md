# steering_evals — run emotion-steering evals on any model

This harness steers a language model along emotion "vectors" and runs behavioural
evals (sycophancy, agentic-misalignment/blackmail) while it's steered. You pick the
model — nothing is hardcoded.

Follow the steps in order. Each step says what to run and what you should see.

---

## What you need before starting
- A GPU pod. We used an RTX A6000 (48GB) with a 50GB volume disk and a 20GB container
  disk. A 9B model took ~18GB of weights plus room for generation, and ran comfortably
  on the A6000 with `--max-connections 1`.
- Your Hugging Face token (for downloading models/vectors).
- Your Anthropic API key (the eval's grader uses Claude).
- The code files in this repo.
- A directory of steering vectors as `NAME_layerN.npy` files (one per emotion per
  layer). The vectors must come from the **same model** you plan to steer.

---

## Step 1 — Check your disks
Pod disks vary. Run:
```bash
df -h /workspace /
```
Find which mount has the most free space (usually `/workspace`, ~50GB). Use that disk
for the model, cache, and logs. Everywhere below uses `/workspace` — if your big disk
is somewhere else, substitute that path.

---

## Step 2 — Put the code on the pod
Upload this repo to `/workspace/steering_evals_repo/` (or clone it there). Then:
```bash
cd /workspace/steering_evals_repo
ls steering_evals/*.py    # you should see: __init__ _registry hooks provider tasks vectors_npy
```

---

## Step 3 — Set your environment
Open `env.sh` and fill in the top section: your model, your vectors path, and your two
keys. Then:
```bash
source env.sh
echo "MODEL=$MODEL"
echo "VECTORS=$VECTORS"
```
Both should print what you set. `env.sh` also points the model cache at the big disk —
do not skip sourcing it, or downloads may fill the small disk.

---

## Step 4 — Install dependencies
```bash
pip install -U huggingface_hub
pip install "torch==2.4.1" "transformers==4.44.2" "huggingface-hub>=0.23.2,<1.0" accelerate inspect_ai inspect_evals
pip install -e . --no-deps
```
Check it worked:
```bash
python -c "import torch, transformers; from transformers import AutoModelForCausalLM; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```
You want version numbers and `True` (GPU visible), with no error.

---

## Step 5 — Download your steering vectors
Download your vector set to the big disk (example shown; use your own repo/path):
```bash
hf download <your-org>/<your-vectors-dataset> --repo-type dataset \
  --include "<subfolder>/*.npy" \
  --local-dir /workspace/steering_evals_repo/vectors
```
Confirm the vectors are there and note an emotion name + a layer you'll use:
```bash
ls $VECTORS | sed 's/_layer[0-9]*//' | sort -u    # lists available emotion names
```

---

## Step 6 — (Only if your model rejects a "system" role) prepare a fixed copy
Some models (e.g. Gemma-2/3) error on system prompts, which the eval uses. If yours is
one of them, make a local copy with a fixed chat template:
```bash
python prepare_model.py --model $MODEL \
  --chat-template templates/gemma_sys.jinja \
  --out /workspace/model-fixed
```
Then use `/workspace/model-fixed` as your model path in the steps below.

**If your model supports system prompts, skip this step** and use its Hugging Face id
directly.

Not sure if you need it? You'll find out in Step 8 — if you see an error like
"System role not supported", come back and do this step.

---

## Step 7 — Calibrate: find the right steering strength
Steering strength ("alpha") and the best layer are different for every model — you must
find them, they don't carry over. This step generates sample text at several alphas so
you can see where steering works and where it breaks:
```bash
python find_alpha.py --model $MODEL --vectors-dir $VECTORS \
  --emotion <emotion> --layer <layer> --alphas 0 0.1 0.2 0.3
```
Read the output:
- Pick the **highest alpha where the text is still coherent** (not looping or gibberish).
  That's your usable strength.
- Confirm the emotion actually changes the tone as alpha rises. If nothing changes at
  any alpha, add `--site pre` and try again.
- To steer two emotions at once, use `--terms` instead of `--emotion`:
  `--terms '[["contempt",1.0],["aggressiveness",1.0]]'`
- Negative alphas steer the opposite way (e.g. `--alphas 0 -0.1 -0.2 -0.3`).

---

## Step 8 — Run the eval under steering
Example: the blackmail scenario, steering one emotion. Fill in your emotion, layer, and
the alpha you chose in Step 7:
```bash
inspect eval steering_evals/tasks.py@blackmail \
  --model steered/local \
  -M model_path=/workspace/model-fixed \
  -M tokenizer_path=/workspace/model-fixed \
  -M vectors_dir=$VECTORS \
  -M 'terms=[["<emotion>",1.0]]' \
  -M layer=<layer> -M alpha=<alpha> -M site=post -M norm_scale=true \
  --max-connections 1 --epochs 20 --temperature 1.0 --max-tokens 1000 \
  -T grader_model=anthropic/claude-sonnet-4-5 \
  --log-dir logs/my_run
```
Notes:
- If your model needed no fixed copy (Step 6 skipped), use `--model steered/$MODEL` and
  remove the `model_path` and `tokenizer_path` lines.
- `--max-connections 1` keeps memory in check on one GPU. Raise it only if you have room.
- This is one scenario repeated `--epochs` times; the rate comes from the repeats.
- For the sycophancy eval instead, use `steering_evals/tasks.py@sycophancy` and
  `--limit N` instead of `--epochs`.

To run it in the background so a disconnect won't kill it:
```bash
nohup <the command above> > run.log 2>&1 &
tail -f run.log     # Ctrl-C stops watching, not the run
```

---

## Step 9 — Read the results
Get the score:
```bash
python read_log.py logs/my_run --n 5 --full
```
This prints the harmful/verdict rate and the first samples' transcripts.

**Important:** always read a few transcripts, not just the number. The scorer only flags
directly-sent, explicit threats — it can miss harmful *intent* or indirect behaviour.
The number alone can under-state what the model did.

To save transcripts to a file:
```bash
python read_log.py logs/my_run --n 20 --full > my_run.txt
```

---

## Step 10 — Save your results and shut down
Logs are small; the model is not. Save the logs before terminating the pod:
```bash
tar czf /workspace/my_results.tgz logs/ *.txt
```
Download `my_results.tgz`, confirm it opens on your machine, then terminate the pod.

---

## Common problems
- **"System role not supported"** -> do Step 6 (prepare a fixed model copy).
- **CUDA out of memory** -> use `--max-connections 1`; make sure no old run is still on
  the GPU (`nvidia-smi`, then kill stray processes); set
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **"No space left on device"** -> your cache/model landed on the small disk. Make sure
  `HF_HOME` points to the big disk (it's set in `env.sh`) before downloading/saving.
- **Steering does nothing at any alpha** -> wrong injection site; add `--site pre`.
- **find_alpha warns the dims don't match** -> your vectors are from a different model
  than the one you're steering. You need vectors extracted from *this* model.
- **The rate is 0 but you expect behaviour** -> read the transcripts; the scorer may be
  missing indirect/planned behaviour that's visible in the text.

---

## What each file does
- `env.sh` — set your model, vectors path, and keys here.
- `find_alpha.py` — calibrate steering strength and layer for a model (Step 7).
- `prepare_model.py` — make a local model copy with a fixed chat template (Step 6).
- `read_log.py` — read eval results and transcripts (Step 9).
- `templates/` — example chat templates (e.g. Gemma system-role fix).
- `steering_evals/` — the package: the steered model provider, the steering hooks, the
  vector loader, and the eval task wrappers. You don't edit these to change models —
  you pass the model on the command line.