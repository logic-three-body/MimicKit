---
name: mimickit-wsl-newton-setup
description: Configure MimicKit in WSL with Newton backend only (no Isaac Gym/Isaac Lab). Use when cloning MimicKit into Ubuntu WSL, creating a clean conda environment, installing Newton + MuJoCo/Warp dependencies, validating GPU import/runtime, running long training in tmux, and handling known blockers such as conda ToS and missing motion data pack.
---

# MimicKit WSL Newton Setup

## Overview

Use this workflow to build a reproducible MimicKit runtime on Ubuntu WSL2 with Newton as the only simulator backend.

Prefer this workflow when:
- starting from a clean WSL image
- `pip`/`conda` is missing
- MimicKit must run with Newton and CUDA

Do not include Isaac Gym or Isaac Lab in this setup.

## Inputs

Set these variables first:

```bash
export MIMICKIT_DIR=/root/Project/MimicKit
export NEWTON_DIR=/root/Project/newton
export MINICONDA_DIR=/root/miniconda3
export ENV_NAME=mimickit
```

Optional proxy:

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
```

## Workflow

### 1. Precheck

Run:

```bash
uname -a
python3 --version
git --version
nvidia-smi
```

Confirm:
- WSL2 Ubuntu is available
- NVIDIA GPU is visible in WSL
- network/proxy can reach GitHub and package indexes

### 2. Install Miniconda (if missing)

```bash
wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
"$MINICONDA_DIR/bin/conda" init bash
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
```

### 3. Clone repositories

```bash
git clone https://github.com/xbpeng/MimicKit "$MIMICKIT_DIR"
git clone https://github.com/newton-physics/newton.git --recurse-submodules "$NEWTON_DIR"
git -C "$NEWTON_DIR" checkout 510f16b4c83ee662c03325c2a960a924e0b5f03e
git -C "$NEWTON_DIR" submodule sync --recursive
git -C "$NEWTON_DIR" submodule update --init --recursive
```

Note: MimicKit README states Newton was tested on commit `510f16b4...`.

### 4. Create conda environment

Accept conda channel terms once:

```bash
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Create env:

```bash
"$MINICONDA_DIR/bin/conda" create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install MimicKit + Newton runtime dependencies

Install MimicKit base requirements:

```bash
cd "$MIMICKIT_DIR"
python -m pip install -r requirements.txt
```

Install Newton dependency chain in this order:

```bash
python -m pip install mujoco --pre -f https://py.mujoco.org/
python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
python -m pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
python -m pip install -e "$NEWTON_DIR"
python -m pip install pyglet
```

If `torch` is still missing:

```bash
python -m pip install torch
```

### 6. Persist environment hooks

Create activation scripts:

```bash
install -d "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d" "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/deactivate.d"
cat > "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d/mimickit_newton.sh" <<'EOF'
#!/usr/bin/env bash
export MIMICKIT_ROOT="/root/Project/MimicKit"
export NEWTON_ROOT="/root/Project/newton"
if [ -f "$NEWTON_ROOT/build/newton_hlc_path.sh" ]; then
  . "$NEWTON_ROOT/build/newton_hlc_path.sh"
fi
EOF
cat > "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/deactivate.d/mimickit_newton.sh" <<'EOF'
#!/usr/bin/env bash
unset MIMICKIT_ROOT
unset NEWTON_ROOT
EOF
chmod +x "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d/mimickit_newton.sh" "$MINICONDA_DIR/envs/$ENV_NAME/etc/conda/deactivate.d/mimickit_newton.sh"
```

`newton_hlc_path.sh` is optional. Newer Newton versions often do not provide it.

### 7. Smoke test

Run:

```bash
conda activate "$ENV_NAME"
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import newton, warp as wp, mimickit; print(newton.__version__, wp.__version__)"
python -m newton.examples basic_pendulum --viewer null
```

Expected:
- imports succeed
- CUDA available is `True`
- Newton example completes without crash

## MimicKit Runtime Check

Use Newton engine config explicitly:

```bash
cd "$MIMICKIT_DIR"
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --num_envs 1 \
  --visualize false \
  --mode test \
  --test_episodes 1 \
  --devices cuda:0
```

If runtime fails with missing file under `data/motions/...`, install data pack first.

## Long-Run Training in tmux

Use tmux for any Newton training that is expected to run longer than a short smoke test.

Start a named session:
```bash
tmux new -s mk_newton_train_exp01
```

Run training inside tmux:
```bash
conda activate "$ENV_NAME"
cd "$MIMICKIT_DIR"
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --num_envs 1024 \
  --visualize false \
  --devices cuda:0
```

Detach safely before closing SSH:
- press `Ctrl+b`, release, then press `d`

Re-attach after reconnect:
```bash
tmux ls
tmux attach -t mk_newton_train_exp01
```

Use two panes for stability checks:
- training pane: `python mimickit/run.py ...`
- monitor pane: `watch -n 1 nvidia-smi` or `nvtop`

Read old logs in tmux copy mode:
- press `Ctrl+b`, then `[`
- scroll with arrows/PageUp/PageDown
- press `q` to exit copy mode

## Known blockers and fixes

1. `CondaToSNonInteractiveError`
- Accept ToS with the two `conda tos accept` commands in Step 4.

2. `No matching distribution found for mujoco>=...dev...`
- Install `mujoco` from `https://py.mujoco.org/` with `--pre` before installing Newton.

3. `warp-lang` missing or wrong version
- Install from `https://pypi.nvidia.com/warp-lang/` with `--pre -U`.

4. MimicKit data missing (`humanoid_spinkick.pkl` not found)
- Download `MimicKit_Data.zip` from the project link in README and extract into `data/`.
- SharePoint link may require browser login and cannot always be fetched non-interactively.

5. CUDA not detected in sandboxed run but detected in real shell
- Re-run in normal WSL shell with GPU access, not in restricted sandbox.

6. SSH disconnected during long training
- This is expected when the job is in tmux; reconnect and run `tmux attach -t <session>`.
- If a session is stuck and unusable, run `tmux kill-session -t <session>` and restart training.

## Output checklist

Record:
- MimicKit commit hash
- Newton commit hash
- Python version
- `torch` / `newton` / `warp-lang` versions
- result of Newton smoke test
- whether motion data pack is installed
- tmux session name used for long-running training
