# MimicKit UE Bridge Tools

This folder contains the UE bridge tooling for the `MimicKit` -> `UE5` workflow:

- `export_schema.py`
- `export_actor_onnx.py`
- `serve_inference.py`
- `convert_ue_trace_to_mimickit.py`
- `build_mimickit_render_sequences.py`

## Evidence Anchors

- Train/test entry: `mimickit/run.py`
- Environment dispatch: `mimickit/envs/env_builder.py`
- Agent dispatch: `mimickit/learning/agent_builder.py`
- Training artifact save/load (`model.pt`): `mimickit/learning/base_agent.py`
- Action application: `mimickit/envs/char_env.py::_apply_action` -> `mimickit/engines/newton_engine.py::set_cmd`
- Observation contract: `mimickit/envs/char_env.py::compute_char_obs`

## Quick Start

Run from repository root:

```bash
cd /root/Project/MimicKit
```

### 1) Export schema for UE bridge

```bash
python tools/ue_bridge/export_schema.py \
  --arg-file args/deepmimic_humanoid_ppo_args.txt \
  --engine-config data/engines/newton_engine.yaml \
  --model-file output/train/<run_name>/model.pt
```

Output:

- `output/train/<run_name>/ue_export/schema.json`

### 2) Export ONNX policy from model.pt

```bash
python tools/ue_bridge/export_actor_onnx.py \
  --arg-file args/deepmimic_humanoid_ppo_args.txt \
  --engine-config data/engines/newton_engine.yaml \
  --model-file output/train/<run_name>/model.pt \
  --verify
```

Outputs:

- `output/train/<run_name>/ue_export/policy_actor.onnx`
- `output/train/<run_name>/ue_export/onnx_export_meta.json`

### 3) Run fallback socket inference service

```bash
python tools/ue_bridge/serve_inference.py \
  --arg-file args/deepmimic_humanoid_ppo_args.txt \
  --engine-config data/engines/newton_engine.yaml \
  --model-file output/train/<run_name>/model.pt \
  --host 127.0.0.1 --port 18080
```

Protocol (JSON-line):

- Request: `{"obs": [ ... ]}`
- Response: `{"ok": true, "action": [ ... ]}`

### 4) Convert UE trace to offline MimicKit dataset

```bash
python tools/ue_bridge/convert_ue_trace_to_mimickit.py \
  --input /path/to/ue_trace.jsonl \
  --output-dir output/train/<run_name>/ue_export \
  --output-name ue_trace_dataset
```

Outputs:

- `ue_trace_dataset.npz`
- `ue_trace_dataset.jsonl`
- `ue_trace_dataset_summary.json`

### 5) Build per-case inference render frame sequences

```bash
python tools/ue_bridge/build_mimickit_render_sequences.py \
  --train-root output/train \
  --img-root output/img \
  --root-scope all \
  --frames 300 \
  --frame-stride 5 \
  --device cuda:0 \
  --num-envs 1
```

Key behavior:

- Discover from `output/train/*/best_by_case.tsv` and keep only `final_ok=1`.
- Mirror run hierarchy from `output/train/<root>/...` into `output/img/<root>/...`.
- Export render frames at `frame_XXXXXX.png` every `frame_stride` steps.
- Write per-root and global indices:
  - `output/img/<root>/infer_viz_index.tsv`
  - `output/img/render_all_roots.tsv`

Useful flags:

- `--dry-run`: discovery/index planning only.
- `--resume` / `--force`: incremental rerun or full rebuild.
- `--roots` / `--cases`: whitelist execution scope.

Runtime notes:

- Inference renderer uses headless viewer mode by default (`MIMICKIT_VIEWER_HEADLESS=1`).
- If Isaac Gym engine config is resolved but `isaacgym` is unavailable, render export falls back to `data/engines/newton_engine.yaml`.

## Explicit Gaps

Repository evidence still shows no built-in:

- ONNX export pipeline in core MimicKit
- Online inference RPC service in core MimicKit
- Direct compatibility with UE LearningAgents trainer protocol

These tools provide the missing bridge layer without changing MimicKit core training code.
