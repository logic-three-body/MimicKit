# BaselineConfig

This baseline locks the first deliverable for `UE5 x MimicKit` integration.

## Locked Baseline

- Arg case: `args/deepmimic_humanoid_ppo_args.txt`
- Engine config: `data/engines/newton_engine.yaml`
- Environment config: `data/envs/deepmimic_humanoid_env.yaml`
- Agent config: `data/agents/deepmimic_humanoid_ppo_agent.yaml`
- Control mode: `pos`
- Control frequency: `30 Hz`
- Sim frequency: `240 Hz`

## Evidence

- `mimickit/run.py`
  - `load_args`, `build_env`, `build_agent`, `train`, `test`
- `mimickit/envs/char_env.py`
  - `compute_char_obs`, `_apply_action`
- `mimickit/engines/newton_engine.py`
  - `set_cmd`, `control_freq`, `sim_freq`
- `mimickit/learning/base_agent.py`
  - `train_model`, `test_model`, `save`, `load`

## Observation Contract (UE Side)

From `mimickit/envs/char_env.py::compute_char_obs`:

`[root_h?] + root_rot_obs + root_vel_obs + root_ang_vel_obs + joint_rot_obs + dof_vel + key_pos_flat?`

Required schema fields exported by `tools/ue_bridge/export_schema.py`:

- `obs_dim`
- `act_dim`
- `action_low` / `action_high`
- `global_obs`
- `root_height_obs`
- `enable_tar_obs`
- `tar_obs_steps`

## Output Layout

Per training run, bridge artifacts are stored under:

`output/train/<run_name>/ue_export/`

- `schema.json`
- `policy_actor.onnx`
- `onnx_export_meta.json`
- optional trace conversions

## Explicit Gaps (Evidence-based)

- No built-in ONNX exporter in MimicKit core (`onnx`, `torch.onnx` search under `MimicKit/**`)
- No built-in online RPC service (`socket`, `grpc`, `server` search under `mimickit/**`)
- No direct UE LearningAgents trainer protocol support (`LearningAgents` search under `MimicKit/**`)

Bridge tooling under `tools/ue_bridge/` is required to close these gaps.
