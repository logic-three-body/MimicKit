# MimicKit GPU Agent Profiles

## Goal

Provide reusable agent configs for dual-GPU runs without relying on temporary files.

## Profiles

1. Throughput-first PPO (DeepMimic humanoid)
- File: `data/agents/deepmimic_humanoid_ppo_agent_throughput.yaml`
- Characteristics:
  - lighter update workload
  - higher samples/s
  - lower average GPU utilization

2. Utilization-first PPO (DeepMimic humanoid)
- File: `data/agents/deepmimic_humanoid_ppo_agent_hiutil.yaml`
- Characteristics:
  - heavier update workload (`update_epochs=40`, `batch_size=2`, `3-layer` nets)
  - lower samples/s
  - higher average GPU utilization

3. Utilization-first ADD (pi_plus)
- File: `data/agents/add_pi_plus_agent_hiutil.yaml`
- Characteristics:
  - heavier PPO update workload (`update_epochs=20`, `batch_size=2`, `3-layer` actor/critic)
  - validated dual-GPU stable probe at `--num_envs 40` per GPU

4. Utilization-first AMP (pi_plus)
- File: `data/agents/amp_pi_plus_agent_hiutil.yaml`
- Characteristics:
  - same heavier PPO update settings
  - validated dual-GPU stable probe at `--num_envs 38` per GPU

5. Utilization-first DeepMimic PPO (pi_plus)
- File: `data/agents/deepmimic_pi_plus_ppo_agent_hiutil.yaml`
- Characteristics:
  - same heavier PPO update settings
  - validated dual-GPU stable probe at `--num_envs 40` per GPU

## Common Dual-GPU Runtime Env

Use before launch on this host:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

## Example Commands

Throughput-first:

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent_throughput.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 4096 \
  --master_port 40479 \
  --out_dir output/train/deepmimic_humanoid_ppo_dual_throughput
```

Utilization-first:

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 4096 \
  --master_port 40479 \
  --out_dir output/train/deepmimic_humanoid_ppo_dual_hiutil
```

Pi-plus utilization-first (ADD):

```bash
python mimickit/run.py \
  --arg_file args/add_pi_plus_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/add_pi_plus_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 40 \
  --master_port 40479 \
  --max_samples 20480 \
  --out_dir output/train/add_pi_plus_dual_hiutil_e40
```

Pi-plus utilization-first (AMP):

```bash
python mimickit/run.py \
  --arg_file args/amp_pi_plus_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/amp_pi_plus_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 38 \
  --master_port 40479 \
  --max_samples 19456 \
  --out_dir output/train/amp_pi_plus_dual_hiutil_e38
```

Pi-plus utilization-first (DeepMimic PPO):

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_pi_plus_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --agent_config data/agents/deepmimic_pi_plus_ppo_agent_hiutil.yaml \
  --mode train \
  --visualize false \
  --devices cuda:0 cuda:1 \
  --num_envs 40 \
  --master_port 40479 \
  --max_samples 20480 \
  --out_dir output/train/deepmimic_pi_plus_ppo_dual_hiutil_e40
```
