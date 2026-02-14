---
name: mimickit-deepmimic-training
description: Run and stabilize MimicKit DeepMimic humanoid PPO on Newton backend across the full path from training to inference to visualization, with tmux keep-alive and fallback for NCCL/CUDA OOM or visualization stack failures. Use when a user asks to start, evaluate, visualize, resume, or debug DeepMimic runs.
---

# MimicKit DeepMimic Training (Newton + tmux)

## Goal

Apply a reproducible workflow to:
- launch DeepMimic humanoid PPO training quickly
- validate trained model with explicit inference output
- validate visual playback for one episode
- keep long training alive in tmux across SSH disconnects
- promote from bounded warmup to longrun safely
- handle multi-device startup failures with deterministic fallback

## Fast E2E Path (Train -> Inference -> Visualization)

Use this when the user wants one complete runnable case quickly.

### Stage 1: short training (produce model)

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/train/deepmimic_humanoid_e2e_${TS}"

python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 \
  --num_envs 512 \
  --max_samples 131072 \
  --out_dir "${OUT_DIR}"
```

Pass conditions:
- exit code `0`
- `${OUT_DIR}/model.pt` exists
- `${OUT_DIR}/log.txt` exists

### Stage 2: inference check (headless test)

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize false \
  --devices cuda:0 \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file "${OUT_DIR}/model.pt"
```

Pass condition:
- log prints `Mean Return` and `Episodes`

### Stage 3: visual playback check (1 episode)

On some WSL/Xwayland setups, shader compilation fails (`GLSL 1.50 is not supported`) unless GL/GLSL versions are overridden.

```bash
MESA_GL_VERSION_OVERRIDE=3.3 \
MESA_GLSL_VERSION_OVERRIDE=330 \
timeout --signal=TERM --kill-after=10 180 \
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize true \
  --devices cuda:0 \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file "${OUT_DIR}/model.pt"
```

Pass condition:
- exit code `0` and log prints `Mean Return`

Notes:
- `Warp CUDA error 304` warnings may appear in visualization mode when CUDA/GL interop is unavailable; this is acceptable if playback and metrics complete.
- keep `test/visualize` single-GPU for reliability.

## Standard Workflow

### 1. Confirm run target

Collect:
- method: `deepmimic`
- task: humanoid PPO
- backend: `newton`
- mode: warmup then longrun
- device plan: dual-device preferred, fallback to single-device if needed

### 2. Precheck environment

Run:
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit
python -c "import torch,newton,warp as wp,mimickit; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), newton.__version__, wp.__version__)"
tmux -V
nvidia-smi
```

Require:
- `torch.cuda.is_available()` is `True`
- `newton` and `warp` import successfully
- required motion files exist under `data/motions/`

For dual-device runs, also check free memory per GPU:
```bash
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
```

Practical rule:
- if one GPU has large persistent memory occupancy, dual-device NCCL init may fail before stable training starts
- prefer probing with small `num_envs` ladder instead of directly starting at high scale

### 2.1 Dual-device NCCL profile (validated on this host)

On this machine, dual-device NCCL failed with default settings but passed with:
- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `NCCL_CUMEM_ENABLE=0`

Export before any dual-device run:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

If needed for debugging:
```bash
export NCCL_DEBUG=INFO
```

### 3. Start tmux session

Use a dedicated session:
```bash
tmux new -s mk_deepmimic_newton_train
```

Recommended monitor split:
- `Ctrl+b` then `%`
- run `watch -n 1 nvidia-smi` in the second pane

### 4. Run bounded warmup first

Use `--max_samples` to cap warmup:
```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 \
  --num_envs 1024 \
  --max_samples 1638400 \
  --out_dir output/train/deepmimic_newton_warmup_<timestamp>
```

Warmup pass condition:
- process exits cleanly
- output dir contains `model.pt` and `log.txt`

### 5. Dual-device warmup probe (recommended)

Use dual-device only after warmup probe passes.

Probe pattern:
1. keep configs fixed (`arg_file`, `engine_config`)
2. run short bounded warmup with timeout
3. lower `num_envs` until one probe passes

Example (single probe):
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

timeout --signal=TERM --kill-after=30 480 \
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 128 \
  --max_samples 131072 \
  --out_dir output/train/deepmimic_newton_warmup_<timestamp>_dual_e128
```

Probe ladder:
- `num_envs`: `320 -> 256 -> 192 -> 160 -> 128 -> 96 -> 64`
- stop at first passing scale

Validated passing scale on this host:
- warmup passed at `num_envs=256` with the NCCL profile above

### 6. Promote to longrun

After warmup pass, start longrun without `--max_samples`:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs <best_probe_envs> \
  --out_dir output/train/deepmimic_newton_longrun_<timestamp>
```

Longrun pass condition:
- training process stays alive in tmux
- `log.txt` continues to update
- console table keeps printing `Iteration`, `Samples`, `Train_Return`, `Loss`

## Fallback Ladder

### Failure class A: NCCL or CUDA OOM at startup (multi-device)

Do this:
1. Keep `arg_file` and `engine_config` unchanged.
2. Apply the validated NCCL profile (`P2P_DISABLE`, `IB_DISABLE`, `CUMEM_ENABLE=0`).
3. Run dual-device probe ladder with timeout (`320 -> ... -> 64`).
4. If all fail, switch to single device: `--devices cuda:0`.
5. Record failure signature and free-memory snapshot for later triage.

### Failure class B: Warmup failed (missing outputs)

Do this:
1. Check tmux pane tail and stderr traceback.
2. Confirm `output/train/.../` is writable.
3. Re-run warmup with lower `num_envs`.

### Failure class C: Longrun exits unexpectedly

Do this:
1. Re-run longrun from same configs with lower `num_envs`.
2. Keep one change per retry (only scale or only devices).
3. Verify process health via `ps` and log growth.

### Failure class D: Run stuck after NCCL abort

Symptoms:
- pane shows `Abort COMPLETE` and `Cuda failure 2 'out of memory'`
- parent process still appears alive and no new training iterations

Do this:
1. terminate current `run.py` and timeout wrapper
2. mark this scale as failed
3. continue next probe scale
4. do not wait indefinitely for automatic exit

## GPU Utilization Reporting

During training, report GPU status at fixed intervals.

Per-minute snapshot:
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
```

Suggested reporting cadence:
- every 1 minute during startup/probe
- every 3-5 minutes during stable longrun

Report template:
- current phase (`warmup_probe` / `longrun`)
- GPU0/GPU1 util%
- GPU0/GPU1 memory used
- power draw
- whether training iterations are advancing
- latest failure marker if any

## Command Patterns

Minimal smoke test:
```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --num_envs 1 \
  --visualize false \
  --test_episodes 1 \
  --devices cuda:0
```

Dual-device warmup probe:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

timeout --signal=TERM --kill-after=30 480 \
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 128 \
  --max_samples 131072 \
  --out_dir output/train/deepmimic_newton_warmup_<timestamp>_dual_e128
```

Dual-device longrun (after probe success):
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs <best_probe_envs> \
  --out_dir output/train/deepmimic_newton_longrun_<timestamp>_dual
```

Single-device fallback:
```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 \
  --num_envs 1024 \
  --out_dir output/train/deepmimic_newton_longrun_<timestamp>
```

## Session Ops

Detach:
- `Ctrl+b`, then `d`

Re-attach:
```bash
tmux ls
tmux attach -t mk_deepmimic_newton_train
```

Kill stuck session:
```bash
tmux kill-session -t mk_deepmimic_newton_train
```

## Output Checklist

When completing a DeepMimic training task, report:
- exact warmup and longrun commands
- selected devices and `num_envs`
- warmup output path and whether `model.pt`/`log.txt` exist
- longrun output path and whether training process is alive
- inference command and the printed `Mean Return`
- visualization command, exit code, and whether `Mean Return` is printed
- if fallback used, which step solved the issue
- whether NCCL profile env vars were applied

## Latest Verified Run (2026-02-12, E2E)

Train -> inference -> visualization verified for:
- case: `args/deepmimic_humanoid_ppo_args.txt`
- train output: `output/train/deepmimic_humanoid_e2e_20260212_131701`
- train status: pass (`model.pt` and `log.txt` generated)
- inference status: pass (`Mean Return: 4.6156`, `Episodes: 10`)
- visualization status: pass with Mesa override
  - required env: `MESA_GL_VERSION_OVERRIDE=3.3`, `MESA_GLSL_VERSION_OVERRIDE=330`
  - result: `Mean Return: 4.1834`, `Episodes: 1`

## Latest Verified Run (2026-02-11)

Dual-device successful run:
- warmup: `output/train/deepmimic_newton_warmup_20260211_214747_dualfix_e256`
- longrun: `output/train/deepmimic_newton_longrun_20260211_214747_dualfix_e256`
- tmux session: `mk_deepmimic_newton_dual_fixed`
- active command used `--devices cuda:0 cuda:1 --num_envs 256`

Cross-method dual-device smoke tests (same NCCL profile):
- AMP humanoid: PASS (`args/amp_humanoid_args.txt`, `--mode test --num_envs 4 --test_episodes 1`)
- ASE humanoid: PASS (`args/ase_humanoid_args.txt`, `--mode test --num_envs 4 --test_episodes 1`)
- ADD humanoid: PASS (`args/add_humanoid_args.txt`, `--mode test --num_envs 4 --test_episodes 1`)

## Dual-GPU Saturation Tuning (2026-02-11)

### Key finding

On this host, increasing `num_envs` alone scales throughput a lot, but does not fully saturate both GPUs.

Observed from sweeps:
- `num_envs=1024` per GPU: avg util about `37%/39%`, throughput about `29k samples/s`
- `num_envs=4096` per GPU (default agent): avg util about `44%/44%`, throughput about `79k samples/s`

To push utilization higher, the agent update workload must also increase.

### High-util profile (recommended)

Use:
- `num_envs=4096` per GPU
- `actor_net=fc_3layers_1024units`
- `critic_net=fc_3layers_1024units`
- `update_epochs=40`
- `batch_size=2`

Validated short-run result:
- avg util about `69%/70%` (including startup)
- steady-state (skip first ~30 samples) about `77%/79%`
- instantaneous peaks around `93%/92%`
- throughput about `23k samples/s`

Tradeoff:
- much higher GPU utilization
- lower samples/s vs high-throughput baseline

Agentized files in repo:
- throughput baseline: `data/agents/deepmimic_humanoid_ppo_agent_throughput.yaml`
- high-util profile: `data/agents/deepmimic_humanoid_ppo_agent_hiutil.yaml`

### Create high-util agent config (runtime file)

```bash
cat > /tmp/mk_agent_extreme_4096.yaml <<'EOF'
agent_name: "PPO"

model:
  actor_net: "fc_3layers_1024units"
  actor_init_output_scale: 0.01
  actor_std_type: "FIXED"
  action_std: 0.05
  critic_net: "fc_3layers_1024units"

optimizer:
    type: "SGD"
    learning_rate: 5e-5

discount: 0.99
steps_per_iter: 32
iters_per_output: 100
test_episodes: 32
normalizer_samples: 100000000

update_epochs: 40
batch_size: 2
td_lambda: 0.95
ppo_clip_ratio: 0.2
norm_adv_clip: 4.0
action_bound_weight: 10.0
action_entropy_weight: 0.0
action_reg_weight: 0.0
critic_loss_weight: 1.0
EOF
```

Or directly use repo profile:
```bash
--agent_config data/agents/deepmimic_humanoid_ppo_agent_hiutil.yaml
```

### Launch high-util dual longrun

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/train/deepmimic_newton_longrun_${TS}_dual_hiutil_e4096"
MASTER_PORT=$(shuf -i 20000-45000 -n 1)

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config /tmp/mk_agent_extreme_4096.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --master_port "${MASTER_PORT}" \
  --num_envs 4096 \
  --out_dir "${OUT_DIR}"
```

Port note:
- if you see `DistNetworkError ... EADDRINUSE`, restart with a new explicit `--master_port`.

### Utilization-first sweep order

When goal is "eat both GPUs" instead of max samples/s, tune in this order:
1. keep NCCL profile fixed (`P2P_DISABLE`, `IB_DISABLE`, `CUMEM_ENABLE=0`)
2. increase `num_envs` to `4096` per GPU
3. then increase agent compute (`update_epochs`, network depth)
4. if GPUs are imbalanced at `>4096`, prefer backing down to `4096` with heavier agent update

### Artifact locations from this tuning

- env sweep summary: `output/train/deepmimic_dual_sweep_20260211_221414/summary.tsv`
- high env sweep summary: `output/train/deepmimic_dual_sweep_high_20260211_224408/summary.tsv`
- compute sweep summary: `output/train/deepmimic_dual_compute_sweep_20260211_230136/summary.tsv`
