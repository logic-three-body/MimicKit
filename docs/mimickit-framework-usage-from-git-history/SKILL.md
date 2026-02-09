---
name: mimickit-framework-usage-from-git-history
description: Use MimicKit effectively for training/testing across Newton, Isaac Lab, and Isaac Gym backends using a workflow distilled from this repository's git history and change hotspots. Use when a user asks to run MimicKit, choose env/agent/arg configs, switch backends, debug runtime failures, triage regressions with git-derived priorities, or keep long-running training alive across unstable SSH connections with tmux.
---

# MimicKit Framework Usage (Git-History Derived)

## Goal

Apply a consistent workflow distilled from this repository's git commit history to:
- run MimicKit quickly
- choose the right env and agent configs
- switch simulator backends safely
- debug failures using code hotspots observed in git history

## Git-Learned Hotspots

Prioritize these files first during triage:
- `mimickit/engines/isaac_lab_engine.py` (frequent fixes: rendering, device mismatch, contact handling)
- `mimickit/engines/newton_engine.py` and `mimickit/envs/sim_env.py` (Newton physics integration and mass handling)
- `mimickit/engines/isaac_gym_engine.py` (backend-specific body/object lookup issues)
- `mimickit/envs/deepmimic_env.py` and related env files (`amp_env.py`, `ase_env.py`, `add_env.py`)
- `data/envs/*.yaml`, `data/agents/*.yaml`, `args/*.txt` (most user-facing behavior differences come from config)
- `mimickit/run.py`, `mimickit/learning/mp_optimizer.py` (entrypoint and multi-device behavior)

Use these hotspots before broad repository-wide edits.

## Standard Workflow

### 1. Confirm run target

Collect:
- backend: `newton`, `isaac_lab`, or `isaac_gym`
- method: `deepmimic`, `amp`, `ase`, or `add`
- mode: `train` or `test`
- device plan: single device or multi-device

### 2. Start from arg file

Prefer method-specific arg files in `args/`.

Example:
```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --mode test --num_envs 1 --visualize false
```

Use explicit overrides only when needed:
- `--engine_config`
- `--env_config`
- `--agent_config`
- `--devices`

### 3. Backend switch rule

Switch backend by changing only `engine_config` first:
- `data/engines/newton_engine.yaml`
- `data/engines/isaac_lab_engine.yaml`
- `data/engines/isaac_gym_engine.yaml`

Do not change env and agent configs at the same time unless required. Isolate variables.

### 4. Validate minimal run before scaling

Run a minimal smoke test:
- `--mode test`
- `--num_envs 1`
- `--visualize false`
- short episode count if available

Scale to training only after this passes.

### 5. Run long training inside tmux

Use tmux by default for `--mode train` jobs.

Start a named session:
```bash
tmux new -s mk_train_newton_exp01
```

Run training inside the tmux window:
```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --engine_config data/engines/newton_engine.yaml --mode train --num_envs 1024 --visualize false
```

Detach safely before disconnecting:
- press `Ctrl+b`, release, then press `d`

Recover after reconnect:
```bash
tmux ls
tmux attach -t mk_train_newton_exp01
```

Use split panes for monitoring:
- `Ctrl+b` then `%` for left/right split
- `Ctrl+b` then `"` for top/bottom split
- run `watch -n 1 nvidia-smi` or `nvtop` in a second pane

## Method-to-Config Map

Use these default pairs first:
- DeepMimic: `data/envs/deepmimic_*_env.yaml` + `data/agents/deepmimic_*_agent.yaml`
- AMP: `data/envs/amp_*_env.yaml` + `data/agents/amp_*_agent.yaml`
- ASE: `data/envs/ase_*_env.yaml` + `data/agents/ase_*_agent.yaml`
- ADD: `data/envs/add_*_env.yaml` + `data/agents/add_*_agent.yaml`

For task-conditioned AMP variants, use:
- env: `data/envs/amp_location_*` or `data/envs/amp_steering_*`
- agent: `data/agents/amp_task_*_agent.yaml`

## Debug Playbook

### Failure class A: Backend boot/import/runtime error

Do this:
1. Re-run with `--num_envs 1 --visualize false`.
2. Verify backend-specific package/environment.
3. Check engine hotspot file for recent changes:
   - Isaac Lab: `mimickit/engines/isaac_lab_engine.py`
   - Newton: `mimickit/engines/newton_engine.py`
   - Isaac Gym: `mimickit/engines/isaac_gym_engine.py`

### Failure class B: Config mismatch or wrong behavior

Do this:
1. Print effective `arg_file`, `engine_config`, `env_config`, `agent_config`.
2. Compare with known-good arg files in `args/`.
3. Diff related YAMLs in `data/envs/` and `data/agents/`.

### Failure class C: Multi-device or multiprocessing issues

Do this:
1. Validate single-device first.
2. Enable multi-device with `--devices cuda:0 cuda:1` only after baseline passes.
3. Inspect:
   - `mimickit/run.py`
   - `mimickit/learning/mp_optimizer.py`
   - backend engine tensor/device placement

### Failure class D: Rendering lag or UI anomalies

Do this:
1. Disable visualize and confirm core simulation/training works.
2. Re-enable visualize and inspect backend rendering logic.
3. Prioritize `mimickit/engines/isaac_lab_engine.py` for Isaac Lab issues.

### Failure class E: Missing motion data/assets

Do this:
1. Confirm required file exists under `data/motions/` or dataset yaml under `data/datasets/`.
2. Confirm env `motion_file` points to a valid path.
3. Run `view_motion` args to validate data quickly.

## Command Patterns

Use these patterns for fast validation.

Single-run smoke test:
```bash
python mimickit/run.py --arg_file args/amp_humanoid_args.txt --mode test --num_envs 1 --visualize false
```

Train with explicit backend override:
```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --engine_config data/engines/newton_engine.yaml --mode train --num_envs 1024 --visualize false
```

Multi-device training:
```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --devices cuda:0 cuda:1 --mode train
```

Train in tmux (recommended for long jobs):
```bash
tmux new -s mk_train_amp_exp01
python mimickit/run.py --arg_file args/amp_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --mode train --num_envs 1024 --visualize false
```

## Output Checklist

When completing a MimicKit support task, report:
- exact command used
- backend and config files used
- whether minimal smoke test passed
- if failed, which hotspot file was inspected and why
- next minimal change to try

