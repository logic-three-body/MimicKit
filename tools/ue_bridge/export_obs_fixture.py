#!/usr/bin/env python3
"""Export UE-facing observation fixture and reference actions from MimicKit model."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import gymnasium.spaces as spaces

from _bridge_common import build_runtime_context, choose_export_dir, save_json


class ActorPolicyExportWrapper(torch.nn.Module):
    """raw_obs -> obs_norm -> model.eval_actor(mode) -> action_unnorm"""

    def __init__(self, agent):
        super().__init__()
        self.obs_norm = agent._obs_norm
        self.action_norm = agent._a_norm
        self.model = agent._model

        signature = inspect.signature(self.model.eval_actor)
        self._eval_actor_arity = len(signature.parameters)

        if self._eval_actor_arity > 1:
            if hasattr(agent, "_latent_buf") and torch.is_tensor(agent._latent_buf) and agent._latent_buf.numel() > 0:
                z0 = agent._latent_buf[0:1].detach().clone().to(dtype=torch.float32)
            elif hasattr(self.model, "get_latent_dim"):
                z_dim = int(self.model.get_latent_dim())
                z0 = torch.zeros((1, z_dim), dtype=torch.float32)
            else:
                raise RuntimeError("actor requires latent input but latent dimension is unavailable")
            self.register_buffer("_latent_seed", z0)
        else:
            self.register_buffer("_latent_seed", torch.zeros((1, 0), dtype=torch.float32))

    def _deterministic_action(self, norm_obs: torch.Tensor) -> torch.Tensor:
        if self._eval_actor_arity > 1:
            z = self._latent_seed.expand(norm_obs.shape[0], -1)
            dist = self.model.eval_actor(norm_obs, z)
        else:
            dist = self.model.eval_actor(norm_obs)

        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "sample"):
            return dist.sample()

        raise RuntimeError(f"unsupported actor distribution output: {type(dist)}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_norm.normalize(obs)
        norm_action = self._deterministic_action(norm_obs)
        return self.action_norm.unnormalize(norm_action)


def _to_numpy_row(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MimicKit obs fixture + reference actions for UE parity")
    parser.add_argument("--arg-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--env-config", default="", help="Override env config path")
    parser.add_argument("--engine-config", default="", help="Override engine config path")
    parser.add_argument("--agent-config", default="", help="Override agent config path")
    parser.add_argument("--out-dir", default="", help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    ctx = build_runtime_context(
        arg_file=args.arg_file,
        overrides=args,
        device=args.device,
        visualize=False,
        load_model=True,
    )

    wrapper = ActorPolicyExportWrapper(ctx.agent)
    wrapper.eval()

    action_low_tensor = None
    action_high_tensor = None
    action_space = ctx.env.get_action_space()
    if isinstance(action_space, spaces.Box):
        action_low_tensor = torch.from_numpy(np.asarray(action_space.low, dtype=np.float32).reshape(1, -1))
        action_high_tensor = torch.from_numpy(np.asarray(action_space.high, dtype=np.float32).reshape(1, -1))

    obs, info = ctx.env.reset(None)
    if obs.ndim != 2:
        raise RuntimeError(f"unexpected obs rank: {obs.shape}")

    obs_rows: list[dict] = []
    action_rows: list[dict] = []

    episode_index = 0

    with torch.no_grad():
        for frame_index in range(int(args.frames)):
            curr_obs = obs[0:1]
            action = wrapper(curr_obs)
            if action_low_tensor is not None and action_high_tensor is not None:
                action = torch.maximum(action, action_low_tensor.to(device=action.device, dtype=action.dtype))
                action = torch.minimum(action, action_high_tensor.to(device=action.device, dtype=action.dtype))

            obs_np = _to_numpy_row(curr_obs)
            act_np = _to_numpy_row(action)

            obs_rows.append(
                {
                    "frame": frame_index,
                    "episode": episode_index,
                    "obs": [float(x) for x in obs_np.tolist()],
                }
            )
            action_rows.append(
                {
                    "frame": frame_index,
                    "episode": episode_index,
                    "action": [float(x) for x in act_np.tolist()],
                }
            )

            next_obs, reward, done, next_info = ctx.env.step(action)
            obs = next_obs
            info = next_info

            if done.shape[0] > 0 and bool(done[0].item()):
                episode_index += 1
                obs, info = ctx.env.reset(None)

    obs_dim = len(obs_rows[0]["obs"]) if obs_rows else 0
    act_dim = len(action_rows[0]["action"]) if action_rows else 0

    out_dir = choose_export_dir(out_dir=args.out_dir, model_file=args.model_file)

    obs_path = Path(out_dir) / "obs_fixture.jsonl"
    act_path = Path(out_dir) / "ref_actions.jsonl"
    meta_path = Path(out_dir) / "fixture_meta.json"

    _write_jsonl(obs_path, obs_rows)
    _write_jsonl(act_path, action_rows)

    meta = {
        "source": {
            "arg_file": str(ctx.arg_file),
            "model_file": str(Path(args.model_file).resolve()),
            "env_config": ctx.args.parse_string("env_config", ""),
            "engine_config": ctx.args.parse_string("engine_config", ""),
            "agent_config": ctx.args.parse_string("agent_config", ""),
        },
        "runtime": {
            "device": args.device,
            "frames": int(args.frames),
            "seed": int(args.seed),
            "episodes_recorded": int(episode_index + 1),
        },
        "shape": {
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "count": int(len(obs_rows)),
        },
        "files": {
            "obs_fixture": str(obs_path.resolve()),
            "ref_actions": str(act_path.resolve()),
        },
    }

    save_json(meta_path, meta)

    print(f"[OK] obs fixture exported: {obs_path}")
    print(f"[OK] reference actions exported: {act_path}")
    print(f"[OK] metadata: {meta_path}")
    print(f"     count={meta['shape']['count']} obs_dim={meta['shape']['obs_dim']} act_dim={meta['shape']['act_dim']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

