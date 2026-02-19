#!/usr/bin/env python3
"""Export MimicKit actor policy (model.pt) to ONNX for UE/NNE runtime."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from _bridge_common import build_runtime_context, choose_export_dir, save_json, set_seed


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


def _verify_export(wrapper: torch.nn.Module, onnx_path: Path, obs_dim: int, seed: int, tolerance: float) -> float:
    try:
        import onnxruntime as ort
    except Exception as ex:
        print(f"[WARN] onnxruntime not available, skip verification: {ex}")
        return float("nan")

    set_seed(seed)
    first_param = next(wrapper.parameters(), None)
    if first_param is None:
        device = torch.device("cpu")
    else:
        device = first_param.device

    obs = torch.randn((1, obs_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        torch_out = wrapper(obs).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: obs.detach().cpu().numpy()},
    )[0]

    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"[VERIFY] max_abs_diff={max_abs:.8f} tolerance={tolerance:.1e}")

    if max_abs > tolerance:
        raise RuntimeError(f"ONNX output drift exceeded tolerance: {max_abs} > {tolerance}")

    return max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model.pt actor to policy_actor.onnx")
    parser.add_argument("--arg-file", required=True)
    parser.add_argument("--model-file", required=True, help="Path to model.pt")
    parser.add_argument("--env-config", default="", help="Override env config path")
    parser.add_argument("--engine-config", default="", help="Override engine config path")
    parser.add_argument("--agent-config", default="", help="Override agent config path")
    parser.add_argument("--out-dir", default="", help="Output directory")
    parser.add_argument("--out-file", default="policy_actor.onnx", help="ONNX filename")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-device", default="cpu", help="Device for torch.onnx.export and parity verification")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output against torch output")
    parser.add_argument("--verify-tolerance", type=float, default=1e-4)

    args = parser.parse_args()

    ctx = build_runtime_context(
        arg_file=args.arg_file,
        overrides=args,
        device=args.device,
        visualize=False,
        load_model=True,
    )

    obs_space = ctx.env.get_obs_space()
    obs_dim = int(obs_space.shape[0]) if len(obs_space.shape) == 1 else int(obs_space.shape[-1])

    wrapper = ActorPolicyExportWrapper(ctx.agent)
    wrapper.eval()
    export_device = torch.device(args.export_device)
    wrapper = wrapper.to(export_device)

    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=export_device)
    out_dir = choose_export_dir(out_dir=args.out_dir, model_file=args.model_file)
    onnx_path = Path(out_dir) / args.out_file

    torch.onnx.export(
        wrapper,
        dummy_obs,
        str(onnx_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        external_data=False,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    # Ensure stale external-data sidecar is removed; UE NNE runner loads single-file ONNX bytes.
    sidecar_path = Path(str(onnx_path) + ".data")
    if sidecar_path.exists():
        sidecar_path.unlink()

    max_abs = float("nan")
    if args.verify:
        max_abs = _verify_export(
            wrapper=wrapper,
            onnx_path=onnx_path,
            obs_dim=obs_dim,
            seed=int(args.seed),
            tolerance=float(args.verify_tolerance),
        )

    meta = {
        "arg_file": str(ctx.arg_file),
        "model_file": str(Path(args.model_file).resolve()),
        "onnx_file": str(onnx_path.resolve()),
        "obs_dim": obs_dim,
        "opset": int(args.opset),
        "verify": bool(args.verify),
        "verify_max_abs": max_abs,
    }

    meta_path = Path(out_dir) / "onnx_export_meta.json"
    save_json(meta_path, meta)

    print(f"[OK] ONNX exported: {onnx_path}")
    print(f"     meta: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



