#!/usr/bin/env python3
"""Simple JSON-line TCP inference service for MimicKit model.pt.

Request examples:
  {"obs": [..float obs..]}\n
  {"batch_obs": [[..],[..]]}\n
  {"ping": true}\n
Response examples:
  {"ok": true, "action": [..]}\n
  {"ok": true, "batch_action": [[..],[..]]}\n
  {"ok": false, "error": "..."}\n
"""

from __future__ import annotations

import argparse
import json
import socketserver
import threading
from pathlib import Path
from typing import Any, Dict, List

import gymnasium.spaces as spaces
import numpy as np
import torch

from _bridge_common import build_runtime_context


class ActorPolicyExportWrapper(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.obs_norm = agent._obs_norm
        self.action_norm = agent._a_norm
        self.actor_layers = agent._model._actor_layers
        self.action_dist_builder = agent._model._action_dist

    def _deterministic_action(self, hidden: torch.Tensor) -> torch.Tensor:
        if hasattr(self.action_dist_builder, "_mean_net"):
            return self.action_dist_builder._mean_net(hidden)
        if hasattr(self.action_dist_builder, "_logit_net"):
            logits = self.action_dist_builder._logit_net(hidden)
            return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=hidden.dtype)
        raise RuntimeError(f"unsupported action distribution builder: {type(self.action_dist_builder)}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_norm.normalize(obs)
        hidden = self.actor_layers(norm_obs)
        norm_action = self._deterministic_action(hidden)
        return self.action_norm.unnormalize(norm_action)


class InferenceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, policy_wrapper, obs_dim, act_dim, action_low, action_high):
        super().__init__(server_address, request_handler_class)
        self.policy_wrapper = policy_wrapper
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.action_low = action_low
        self.action_high = action_high
        self.lock = threading.Lock()

    def infer(self, obs_batch: np.ndarray) -> np.ndarray:
        if obs_batch.ndim != 2 or obs_batch.shape[1] != self.obs_dim:
            raise ValueError(f"obs shape mismatch: got {obs_batch.shape}, expected [N,{self.obs_dim}]")

        with self.lock:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_batch.astype(np.float32, copy=False))
                out_tensor = self.policy_wrapper(obs_tensor)
                out = out_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

        if self.action_low is not None and self.action_high is not None:
            out = np.minimum(np.maximum(out, self.action_low), self.action_high)

        return out


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            raw = self.rfile.readline()
            if not raw:
                return

            try:
                req = json.loads(raw.decode("utf-8"))
                resp = self._process(req)
            except Exception as ex:
                resp = {"ok": False, "error": str(ex)}

            payload = (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(payload)

    def _process(self, req: Dict[str, Any]) -> Dict[str, Any]:
        srv: InferenceServer = self.server  # type: ignore[assignment]

        if req.get("ping", False):
            return {"ok": True, "pong": True, "obs_dim": srv.obs_dim, "act_dim": srv.act_dim}

        if "obs" in req:
            obs = np.asarray(req["obs"], dtype=np.float32).reshape(1, -1)
            action = srv.infer(obs)
            return {"ok": True, "action": action[0].tolist()}

        if "batch_obs" in req:
            batch = np.asarray(req["batch_obs"], dtype=np.float32)
            action = srv.infer(batch)
            return {"ok": True, "batch_action": action.tolist()}

        return {"ok": False, "error": "expected one of keys: ping|obs|batch_obs"}



def main() -> int:
    parser = argparse.ArgumentParser(description="Run MimicKit policy socket inference service")
    parser.add_argument("--arg-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--env-config", default="")
    parser.add_argument("--engine-config", default="")
    parser.add_argument("--agent-config", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)

    args = parser.parse_args()

    ctx = build_runtime_context(
        arg_file=args.arg_file,
        overrides=args,
        device=args.device,
        visualize=False,
        load_model=True,
    )

    obs_space = ctx.env.get_obs_space()
    act_space = ctx.env.get_action_space()

    obs_dim = int(obs_space.shape[0]) if len(obs_space.shape) == 1 else int(obs_space.shape[-1])

    action_low = None
    action_high = None
    if isinstance(act_space, spaces.Box):
        act_dim = int(act_space.shape[0]) if len(act_space.shape) == 1 else int(act_space.shape[-1])
        action_low = np.asarray(act_space.low, dtype=np.float32).reshape(1, -1)
        action_high = np.asarray(act_space.high, dtype=np.float32).reshape(1, -1)
    elif isinstance(act_space, spaces.Discrete):
        act_dim = 1
    else:
        raise TypeError(f"unsupported action space for server: {type(act_space)}")

    wrapper = ActorPolicyExportWrapper(ctx.agent)
    wrapper.eval()

    server = InferenceServer(
        server_address=(args.host, int(args.port)),
        request_handler_class=JsonLineHandler,
        policy_wrapper=wrapper,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
    )

    print("[MimicKitBridge] socket inference service ready")
    print(f"  host={args.host} port={args.port}")
    print(f"  obs_dim={obs_dim} act_dim={act_dim}")
    print(f"  model={Path(args.model_file).resolve()}")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[MimicKitBridge] received Ctrl+C, shutting down...")
    finally:
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
