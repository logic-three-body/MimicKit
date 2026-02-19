#!/usr/bin/env python3
"""Export dummy fixture + constant-action ONNX policy for non-policy MimicKit cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium.spaces as spaces
import torch

from _bridge_common import build_runtime_context, choose_export_dir, save_json


class ConstantPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, action_values: list[float]):
        super().__init__()
        self.proj = torch.nn.Linear(int(obs_dim), int(len(action_values)), bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.copy_(torch.tensor(action_values, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.proj(obs)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')


def _export_constant_policy_onnx(obs_dim: int, action_values: list[float], out_path: Path, opset: int) -> None:
    model = ConstantPolicy(obs_dim=obs_dim, action_values=action_values)
    model.eval()
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)

    kwargs = {
        'export_params': True,
        'opset_version': int(opset),
        'do_constant_folding': True,
        'input_names': ['obs'],
        'output_names': ['action'],
        'dynamic_axes': {'obs': {0: 'batch'}, 'action': {0: 'batch'}},
    }

    try:
        torch.onnx.export(model, dummy, str(out_path), external_data=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kwargs)

    sidecar = Path(str(out_path) + '.data')
    if sidecar.exists():
        sidecar.unlink()


def _build_default_action(action_low: list[float], action_high: list[float]) -> list[float]:
    values: list[float] = []
    for low, high in zip(action_low, action_high):
        v = 0.0
        if v < low:
            v = low
        if v > high:
            v = high
        values.append(float(v))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description='Export dummy fixture/schema/onnx for non-policy cases')
    parser.add_argument('--arg-file', required=True)
    parser.add_argument('--model-file', default='')
    parser.add_argument('--env-config', default='')
    parser.add_argument('--engine-config', default='')
    parser.add_argument('--agent-config', default='')
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError('--frames must be > 0')

    torch.manual_seed(int(args.seed))

    ctx = build_runtime_context(
        arg_file=args.arg_file,
        overrides=args,
        device=args.device,
        visualize=False,
        load_model=False,
    )

    out_dir = choose_export_dir(out_dir=args.out_dir, model_file=args.model_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_space = ctx.env.get_obs_space()
    act_space = ctx.env.get_action_space()
    obs_dim = int(obs_space.shape[0]) if len(obs_space.shape) == 1 else int(obs_space.shape[-1])

    action_low: list[float]
    action_high: list[float]
    default_action: list[float]
    action_dtype = torch.float32

    if isinstance(act_space, spaces.Box):
        act_dim = int(act_space.shape[0]) if len(act_space.shape) == 1 else int(act_space.shape[-1])
        action_low = [float(x) for x in act_space.low.reshape(-1).tolist()]
        action_high = [float(x) for x in act_space.high.reshape(-1).tolist()]
        default_action = _build_default_action(action_low, action_high)
    elif isinstance(act_space, spaces.Discrete):
        act_dim = 1
        action_low = [0.0]
        action_high = [float(act_space.n - 1)]
        default_action = [0.0]
    else:
        raise TypeError(f'unsupported action space: {type(act_space)}')

    obs_rows: list[dict] = []
    action_rows: list[dict] = []

    obs, _ = ctx.env.reset(None)
    episode_index = 0

    with torch.no_grad():
        for frame_index in range(int(args.frames)):
            curr_obs = obs[0:1]
            obs_np = curr_obs.detach().cpu().numpy().reshape(-1).astype('float32')

            obs_rows.append({'frame': frame_index, 'episode': episode_index, 'obs': [float(x) for x in obs_np.tolist()]})
            action_rows.append({'frame': frame_index, 'episode': episode_index, 'action': default_action})

            if isinstance(act_space, spaces.Box):
                act_tensor = torch.tensor([default_action], device=curr_obs.device, dtype=action_dtype)
            else:
                # Follow DummyAgent discrete branch shape contract [N, 0]
                act_tensor = torch.zeros((1, 0), device=curr_obs.device, dtype=action_dtype)

            next_obs, _, done, _ = ctx.env.step(act_tensor)
            obs = next_obs

            if done.shape[0] > 0 and bool(done[0].item()):
                episode_index += 1
                obs, _ = ctx.env.reset(None)

    schema = {
        'source': {
            'arg_file': str(ctx.arg_file),
            'env_config': ctx.args.parse_string('env_config', ''),
            'engine_config': ctx.args.parse_string('engine_config', ''),
            'agent_config': ctx.args.parse_string('agent_config', ''),
            'model_file': ctx.args.parse_string('model_file', ''),
        },
        'env': {
            'env_name': ctx.env_config.get('env_name', ''),
            'global_obs': bool(ctx.env_config.get('global_obs', True)),
            'root_height_obs': bool(ctx.env_config.get('root_height_obs', True)),
            'enable_tar_obs': bool(ctx.env_config.get('enable_tar_obs', False)),
            'tar_obs_steps': ctx.env_config.get('tar_obs_steps', []),
            'control_mode': ctx.engine_config.get('control_mode', 'none'),
            'control_freq': int(ctx.engine_config.get('control_freq', 10)),
            'sim_freq': int(ctx.engine_config.get('sim_freq', 60)),
            'obs_layout_token': 'dummy_constant_action_path',
        },
        'model': {
            'agent_name': 'Dummy',
            'obs_dim': int(obs_dim),
            'act_dim': int(act_dim),
            'action_space_type': 'Box' if isinstance(act_space, spaces.Box) else 'Discrete',
            'action_low': action_low,
            'action_high': action_high,
        },
    }

    obs_path = out_dir / 'obs_fixture.jsonl'
    ref_path = out_dir / 'ref_actions.jsonl'
    schema_path = out_dir / 'schema.json'
    meta_path = out_dir / 'fixture_meta.json'
    onnx_path = out_dir / 'policy_actor.onnx'
    onnx_meta_path = out_dir / 'onnx_export_meta.json'

    _write_jsonl(obs_path, obs_rows)
    _write_jsonl(ref_path, action_rows)
    save_json(schema_path, schema)

    fixture_meta = {
        'source': schema['source'],
        'runtime': {
            'device': args.device,
            'frames': int(args.frames),
            'seed': int(args.seed),
            'episodes_recorded': int(episode_index + 1),
            'policy_kind': 'dummy_constant_action',
        },
        'shape': {'obs_dim': int(obs_dim), 'act_dim': int(act_dim), 'count': int(len(obs_rows))},
        'files': {'obs_fixture': str(obs_path.resolve()), 'ref_actions': str(ref_path.resolve())},
    }
    save_json(meta_path, fixture_meta)

    _export_constant_policy_onnx(obs_dim=obs_dim, action_values=default_action, out_path=onnx_path, opset=int(args.opset))
    save_json(
        onnx_meta_path,
        {
            'arg_file': str(ctx.arg_file),
            'onnx_file': str(onnx_path.resolve()),
            'obs_dim': int(obs_dim),
            'act_dim': int(act_dim),
            'opset': int(args.opset),
            'policy_kind': 'dummy_constant_action',
            'default_action': default_action,
        },
    )

    print(f'[OK] dummy schema exported: {schema_path}')
    print(f'[OK] dummy fixture exported: {obs_path}')
    print(f'[OK] dummy ref actions exported: {ref_path}')
    print(f'[OK] dummy onnx exported: {onnx_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
