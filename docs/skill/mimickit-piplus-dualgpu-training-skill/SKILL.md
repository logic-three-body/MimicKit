---
name: mimickit-piplus-dualgpu-training
description: Run pi_plus training cases (`add`, `amp`, `deepmimic`) on Newton dual-GPU reliably using validated env ladders, hiutil agent profiles, and tmux/NCCL safeguards. Use when pi_plus fails with OOM/NCCL/negative-shape errors at higher env scales.
---

# MimicKit Pi-plus Dual-GPU Training

## Goal

Run these cases stably on `cuda:0 cuda:1`:
- `args/add_pi_plus_args.txt`
- `args/amp_pi_plus_args.txt`
- `args/deepmimic_pi_plus_ppo_args.txt`

## Precheck

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -c "import torch,newton,warp as wp,scipy,trimesh; print(torch.cuda.is_available(), torch.cuda.device_count(), newton.__version__, wp.__version__)"
nvidia-smi
```

Require:
- `torch.cuda.is_available() == True`
- `device_count >= 2`
- `scipy` and `trimesh` importable

## Dual-GPU Runtime Env

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

## Validated Stable Settings

Use hiutil agentized configs:
- `data/agents/add_pi_plus_agent_hiutil.yaml`
- `data/agents/amp_pi_plus_agent_hiutil.yaml`
- `data/agents/deepmimic_pi_plus_ppo_agent_hiutil.yaml`

Validated per-GPU `num_envs`:
- ADD pi_plus: `40`
- AMP pi_plus: `38`
- DeepMimic pi_plus PPO: `40`

Universal fallback if one shared value is needed:
- `--num_envs 38`

## Launch Commands

ADD:

```bash
python mimickit/run.py \
  --arg_file args/add_pi_plus_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/add_pi_plus_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 40 \
  --max_samples 20480 \
  --out_dir output/train/add_pi_plus_dual_hiutil_e40
```

AMP:

```bash
python mimickit/run.py \
  --arg_file args/amp_pi_plus_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/amp_pi_plus_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 38 \
  --max_samples 19456 \
  --out_dir output/train/amp_pi_plus_dual_hiutil_e38
```

DeepMimic PPO:

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_pi_plus_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/deepmimic_pi_plus_ppo_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 40 \
  --max_samples 20480 \
  --out_dir output/train/deepmimic_pi_plus_ppo_dual_hiutil_e40
```

## Fallback Ladder (if a case fails)

Try only `num_envs` changes first:
- `44 -> 40 -> 39 -> 38 -> 36 -> 32`

Known behavior:
- `amp_pi_plus` may fail at `39` and pass at `38`.

## tmux Keepalive

```bash
tmux new -s mk_piplus_dual
# pane-1 run training, pane-2 monitor GPU
watch -n 1 nvidia-smi
```

Detach / attach:

```bash
# detach: Ctrl+b, d
tmux ls
tmux attach -t mk_piplus_dual
```

## Pass Criteria

- process exits `0` (for bounded runs) or stays alive (for longrun)
- output contains `model.pt` and `log.txt`
- log has advancing `Iteration` and `Samples`
- both GPUs show active utilization during training stage
