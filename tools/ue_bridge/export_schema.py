#!/usr/bin/env python3
"""Export a UE-facing schema.json from a MimicKit arg_file + config set."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium.spaces as spaces

from _bridge_common import build_runtime_context, choose_export_dir, save_json, to_list


OBS_LAYOUT_TOKEN = (
    "[root_h?] + root_rot_obs + root_vel_obs + root_ang_vel_obs + "
    "joint_rot_obs + dof_vel + key_pos_flat?"
)



def main() -> int:
    parser = argparse.ArgumentParser(description="Export MimicKit schema.json for UE bridge")
    parser.add_argument("--arg-file", required=True, help="Path to MimicKit arg_file (e.g. args/deepmimic_humanoid_ppo_args.txt)")
    parser.add_argument("--env-config", default="", help="Override env config path")
    parser.add_argument("--engine-config", default="", help="Override engine config path")
    parser.add_argument("--agent-config", default="", help="Override agent config path")
    parser.add_argument("--model-file", default="", help="Optional model.pt path for traceability")
    parser.add_argument("--out-dir", default="", help="Output directory (default: <model_dir>/ue_export or output/train/ue_export)")
    parser.add_argument("--device", default="cpu", help="Torch device used to build env/agent")
    parser.add_argument("--num-envs", type=int, default=1, help="num_envs override for schema probing")

    args = parser.parse_args()

    ctx = build_runtime_context(
        arg_file=args.arg_file,
        overrides=args,
        device=args.device,
        visualize=False,
        load_model=False,
    )

    obs_space = ctx.env.get_obs_space()
    act_space = ctx.env.get_action_space()

    obs_dim = int(obs_space.shape[0]) if len(obs_space.shape) == 1 else int(obs_space.shape[-1])

    schema = {
        "source": {
            "arg_file": str(ctx.arg_file),
            "env_config": ctx.args.parse_string("env_config"),
            "engine_config": ctx.args.parse_string("engine_config"),
            "agent_config": ctx.args.parse_string("agent_config"),
            "model_file": ctx.args.parse_string("model_file", ""),
        },
        "env": {
            "env_name": ctx.env_config.get("env_name", ""),
            "global_obs": bool(ctx.env_config.get("global_obs", True)),
            "root_height_obs": bool(ctx.env_config.get("root_height_obs", True)),
            "enable_tar_obs": bool(ctx.env_config.get("enable_tar_obs", False)),
            "tar_obs_steps": to_list(ctx.env_config.get("tar_obs_steps", [])),
            "control_mode": ctx.engine_config.get("control_mode", "none"),
            "control_freq": int(ctx.engine_config.get("control_freq", 10)),
            "sim_freq": int(ctx.engine_config.get("sim_freq", 60)),
            "obs_layout_token": OBS_LAYOUT_TOKEN,
        },
        "model": {
            "agent_name": ctx.agent_config.get("agent_name", "unknown"),
            "obs_dim": obs_dim,
        },
    }

    if isinstance(act_space, spaces.Box):
        act_dim = int(act_space.shape[0]) if len(act_space.shape) == 1 else int(act_space.shape[-1])
        schema["model"]["action_space_type"] = "Box"
        schema["model"]["act_dim"] = act_dim
        schema["model"]["action_low"] = [float(x) for x in act_space.low.tolist()]
        schema["model"]["action_high"] = [float(x) for x in act_space.high.tolist()]
    elif isinstance(act_space, spaces.Discrete):
        schema["model"]["action_space_type"] = "Discrete"
        schema["model"]["act_dim"] = 1
        schema["model"]["num_actions"] = int(act_space.n)
        schema["model"]["action_low"] = [0.0]
        schema["model"]["action_high"] = [float(act_space.n - 1)]
    else:
        raise TypeError(f"unsupported action space: {type(act_space)}")

    out_dir = choose_export_dir(out_dir=args.out_dir, model_file=ctx.args.parse_string("model_file", ""))
    out_path = Path(out_dir) / "schema.json"
    save_json(out_path, schema)

    print(f"[OK] schema exported: {out_path}")
    print(f"      obs_dim={schema['model']['obs_dim']} act_dim={schema['model']['act_dim']}")
    print(f"      control_mode={schema['env']['control_mode']} control_freq={schema['env']['control_freq']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
