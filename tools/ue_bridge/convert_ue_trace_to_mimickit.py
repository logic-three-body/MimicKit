#!/usr/bin/env python3
"""Convert UE trace files into MimicKit-friendly offline dataset artifacts.

Input supports:
- JSONL: one json object per line
- JSON: list[object] or {"records": list[object]}

Default field contract:
- obs: list[float]
- action: list[float]
- reward: float
- done: bool
- episode: int (optional)
- time: float (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np



def _load_records(path: Path, fmt: str) -> List[Dict[str, Any]]:
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() == ".jsonl" else "json"

    if fmt == "jsonl":
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
        return records

    if fmt == "json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("records", None), list):
            return obj["records"]
        raise ValueError("JSON input must be list[...] or {'records': [...]}.")

    raise ValueError(f"unsupported format: {fmt}")



def _to_float_list(value: Any) -> List[float]:
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    raise TypeError(f"expected list-like numeric value, got: {type(value)}")



def main() -> int:
    parser = argparse.ArgumentParser(description="Convert UE traces to MimicKit offline dataset")
    parser.add_argument("--input", required=True, help="Path to UE trace json/jsonl")
    parser.add_argument("--format", choices=["auto", "json", "jsonl"], default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="ue_trace_dataset")

    parser.add_argument("--obs-key", default="obs")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--reward-key", default="reward")
    parser.add_argument("--done-key", default="done")
    parser.add_argument("--episode-key", default="episode")
    parser.add_argument("--time-key", default="time")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(path=input_path, fmt=args.format)
    if len(records) == 0:
        raise RuntimeError("input trace is empty")

    obs_list: List[List[float]] = []
    act_list: List[List[float]] = []
    reward_list: List[float] = []
    done_list: List[int] = []
    episode_list: List[int] = []
    time_list: List[float] = []

    skipped = 0
    auto_episode = 0

    for i, rec in enumerate(records):
        try:
            obs = _to_float_list(rec[args.obs_key])
            act = _to_float_list(rec[args.action_key])
            rew = float(rec.get(args.reward_key, 0.0))
            done = bool(rec.get(args.done_key, False))

            if args.episode_key in rec:
                episode = int(rec[args.episode_key])
            else:
                episode = auto_episode

            if args.time_key in rec:
                t = float(rec[args.time_key])
            else:
                t = float(i)

            obs_list.append(obs)
            act_list.append(act)
            reward_list.append(rew)
            done_list.append(1 if done else 0)
            episode_list.append(episode)
            time_list.append(t)

            if done and args.episode_key not in rec:
                auto_episode += 1

        except Exception:
            skipped += 1
            continue

    if len(obs_list) == 0:
        raise RuntimeError("all records failed conversion")

    obs_dim = len(obs_list[0])
    act_dim = len(act_list[0])

    for row in obs_list:
        if len(row) != obs_dim:
            raise RuntimeError(f"obs dim mismatch detected, expected {obs_dim}")
    for row in act_list:
        if len(row) != act_dim:
            raise RuntimeError(f"action dim mismatch detected, expected {act_dim}")

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    rew_arr = np.asarray(reward_list, dtype=np.float32)
    done_arr = np.asarray(done_list, dtype=np.int32)
    ep_arr = np.asarray(episode_list, dtype=np.int32)
    time_arr = np.asarray(time_list, dtype=np.float32)

    npz_path = out_dir / f"{args.output_name}.npz"
    np.savez_compressed(
        npz_path,
        obs=obs_arr,
        action=act_arr,
        reward=rew_arr,
        done=done_arr,
        episode=ep_arr,
        time=time_arr,
    )

    jsonl_path = out_dir / f"{args.output_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for idx in range(obs_arr.shape[0]):
            item = {
                "obs": obs_arr[idx].tolist(),
                "action": act_arr[idx].tolist(),
                "reward": float(rew_arr[idx]),
                "done": int(done_arr[idx]),
                "episode": int(ep_arr[idx]),
                "time": float(time_arr[idx]),
            }
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "output_npz": str(npz_path),
        "output_jsonl": str(jsonl_path),
        "records_total": len(records),
        "records_converted": int(obs_arr.shape[0]),
        "records_skipped": int(skipped),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "episode_count": int(len(set(episode_list))),
    }

    summary_path = out_dir / f"{args.output_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"[OK] converted {obs_arr.shape[0]}/{len(records)} records")
    print(f"     npz: {npz_path}")
    print(f"     summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
