# Cloud runbook — RunPod

Step-by-step procedure for running the extraction / probe-training
pipeline on a paid RunPod GPU pod. The Mac M5 stays for analysis and
plotting; everything that requires a real 7-8B forward pass runs here.

## Cost expectations

| Workload | GPU | Approx. wall-clock | Approx. cost |
|---|---|---|---|
| Pilot: 1 model × 1 emotion (~700 prompts) | RTX 4090 | ~5 min | $0.03 |
| Pilot: 1 model × 4 emotions | RTX 4090 | ~20 min | $0.11 |
| Full H1 primary: 3 models × 4 emotions | RTX 4090 | ~1 hour | $0.34 |
| Full project (H1 + H2 + H3) | RTX 4090 | ~50-100 hours over multiple sessions | $50-100 |

These assume Community Cloud pricing of $0.34/hr; Secure Cloud is
roughly 2-3× more expensive. Hard budget cap for the project is $150
(see `BLUEPRINT.md`).

## Shared group credits (RunPod Teams)

**Decision: use a RunPod Team account, not a shared login.** Haydar is
the sole billing owner; teammates join the team with a non-billing
role. This pools Haydar's credits across the whole team without anyone
else needing a payment method, while keeping per-user audit and
role-scoped permissions. A shared login was rejected — it means a
shared password, no per-user attribution, and no way to restrict who
can touch billing.

Why Teams works for us:

- **Pooled credits, one payer.** Compute is billed by the second
  against the team's balance (Haydar's credits). No per-seat fee — you
  only pay for GPU time.
- **Roles.** Owner/Admin (Haydar) controls billing; teammates get the
  **Dev** role (deploy + manage pods, *no* billing access). A
  **Billing** role exists for finance-only access if ever needed.
- **No egress fees**, so pulling artefacts to HF / laptops is free.

### Setup (Haydar, once)

1. Sign up / log in at <https://console.runpod.io>, add a payment
   method, and load a starter balance (e.g. $20 — well under the $150
   project cap).
2. Convert to a team account: console → **Team page**
   (<https://www.console.runpod.io/team>) → **Convert to a Team
   Account** → name it (e.g. `emotion-concepts`).
3. Invite each teammate: **Members → Invite New Member →** choose
   **Dev** role → enter their email → **Create Invite** → share the
   generated link from *Pending Invites*.
4. (Recommended) Set an **account spend limit** so a runaway pod can't
   blow the budget cap.

### Setup (each teammate, once)

1. Click Haydar's invite link → **Join Team**.
2. In the console's top-left account selector, switch into the
   **team** context (not your personal account) before deploying — pods
   launched in your personal context bill *you*, not the team.
3. Create your own HuggingFace **Write** token (see next section); each
   member uses their own `HF_TOKEN`, never a shared one.

### Per-session reminder

Always confirm the account selector shows the **team** before spinning
a pod, so it draws on the pooled credits.

## One-time setup

### 1. Community Cloud is sufficient

Pods can be preempted, but `cloud_run.sh` pushes artefacts to HF after
every emotion, so at most one emotion of work is at risk. Community
Cloud (~$0.34/hr for a 4090) is the default; only use Secure Cloud if
Community has no 4090 capacity.

### 2. HuggingFace token with org access

The cloud pipeline pushes to the private dataset
`llm-psych/llm-psych-activations`. You need a token with write access:

1. <https://huggingface.co/settings/tokens> → "Create new token"
2. Type: **Write** (or fine-grained with write access to that dataset)
3. Make sure your HF account is a member of the
   `EmotionConceptsResearch` organisation
4. Copy the token (`hf_xxx…`). You'll paste it as an env var on the
   pod — **do not commit it.**

### 3. (Optional) Cache HF tokens & GitHub SSH on the pod

If you'll be running many sessions, attach a Network Volume to the pod
so `~/.cache/huggingface/` and the cloned repo persist between
sessions (saves the ~20 GB model download each time). This is
optional; `cloud_bootstrap.sh` handles a clean pod fine.

## Per-session procedure

### 1. Spin up a pod

In the RunPod console:

- **Template:** "RunPod PyTorch 2.4.0" (or any image with CUDA 12.x
  pre-installed). The bootstrap installs its own `torch` via `uv`, so
  the image's torch is irrelevant — what matters is the **CUDA driver**.
- **GPU:** RTX 4090 (Community Cloud). 24 GB VRAM runs Llama 3.1 8B
  in bf16 without quantisation.
- **Container disk:** 50 GB (model download + activations).
- **Volume disk:** Skip for one-off pilot; attach a 50 GB volume if
  you'll run multiple sessions.
- **Environment variables (CRITICAL):**

  ```
  HF_TOKEN=hf_xxx                # required
  ```

  You can also set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` here if
  you'll be running judge-model code; the bootstrap propagates them
  into `.env`.

### 2. Bootstrap

Open the pod's web terminal (or SSH). Then either:

**Option A — one-liner (recommended; no need to clone first):**

```bash
curl -fsSL https://raw.githubusercontent.com/hyderli/llm-psych/main/scripts/cloud_bootstrap.sh | bash
```

**Option B — clone first (when iterating on the bootstrap itself):**

```bash
git clone https://github.com/hyderli/llm-psych.git /workspace/llm-psych
cd /workspace/llm-psych
bash scripts/cloud_bootstrap.sh
```

The bootstrap takes ~3-5 min on first run (mostly `uv sync` installing
the CUDA torch wheel). On subsequent runs against a persisted volume
it takes ~30 s.

Expected tail of successful output:

```
[bootstrap] torch: 2.5.x+cu121
[bootstrap] GPU: NVIDIA GeForce RTX 4090
[bootstrap] VRAM: 25.4 GB
[bootstrap] bitsandbytes: 0.44.x
[bootstrap] HF dataset llm-psych/llm-psych-activations: N files visible
[bootstrap] Bootstrap complete. Pod is ready.
```

### 3. Run an experiment

```bash
cd /workspace/llm-psych

# Pilot: Llama 3.1 8B × anger only (~5 min)
bash scripts/cloud_run.sh --model llama31_8b --emotions "anger"

# Full primary: Llama 3.1 8B × 4 emotions (~20 min, auto-shutdown when done)
bash scripts/cloud_run.sh --model llama31_8b --shutdown

# Different model
bash scripts/cloud_run.sh --model qwen25_7b
```

Each emotion's activations + probes + steering vectors are pushed to
the HF dataset **immediately** after that emotion finishes, so a pod
preemption costs at most one emotion of re-extraction. Re-running the
same command after a preemption simply overwrites partially-written
files; the pipeline is idempotent at the (model, emotion) granularity.

### 4. Verify on the Mac

After the pod is done, on your laptop:

```bash
uv run python scripts/sync_hf.py ls activations
uv run python scripts/sync_hf.py pull activations --model Llama-3.1-8B-Instruct
uv run python scripts/sync_hf.py pull probes --model Llama-3.1-8B-Instruct
```

Then proceed to analysis notebooks locally.

### 5. Tear down the pod

If you used `--shutdown` and `RUNPOD_POD_ID` was set, the pod stops
itself. Otherwise:

1. RunPod console → Pods → click the pod → **Stop** (preserves
   container disk, no GPU charges)
2. Or **Terminate** (deletes the pod entirely)

**Critical:** A running pod incurs GPU charges every second even when
idle. After verifying artefacts are on HF, terminate any pod you do
not plan to reuse within the hour.

## Troubleshooting

### `nvidia-smi: command not found`

You selected a CPU-only pod by mistake. Terminate and re-spin with a
GPU template.

### `uv sync` fails with a CUDA mismatch

The base image's CUDA driver is too old for the torch wheel `uv`
resolves. Either pick a newer image (CUDA 12.1+) or pin `torch`
explicitly in `pyproject.toml`.

### `403 Forbidden` from HF on push

Your token does not have write access, or you are not a member of the
`EmotionConceptsResearch` org. Verify at
<https://huggingface.co/settings/organizations>.

### Pod preempted mid-run

Spin up a new pod, run `cloud_bootstrap.sh` and the exact same
`cloud_run.sh` command. The script will redo any incomplete emotion
from scratch; already-pushed emotions on HF are not affected.

### Wall-clock is much longer than the table suggests

Check `nvidia-smi` during a run — if the GPU is at <50% utilisation,
the bottleneck is CPU tokenisation or disk I/O, not GPU. Increase
`--batch-size` (default 16 → try 32) if VRAM allows.
