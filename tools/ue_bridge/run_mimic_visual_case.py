#!/usr/bin/env python3
"""Run MimicKit case inference + export chain and generate visualization artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/Project/MimicKit').resolve()
RUN_PY = ROOT / 'mimickit' / 'run.py'
EXPORT_SCHEMA_PY = ROOT / 'tools' / 'ue_bridge' / 'export_schema.py'
EXPORT_ONNX_PY = ROOT / 'tools' / 'ue_bridge' / 'export_actor_onnx.py'
EXPORT_FIXTURE_PY = ROOT / 'tools' / 'ue_bridge' / 'export_obs_fixture.py'
EXPORT_DUMMY_PY = ROOT / 'tools' / 'ue_bridge' / 'export_dummy_fixture.py'


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as fp:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fp, stderr=subprocess.STDOUT)
        proc.wait()
    text = log_path.read_text(encoding='utf-8', errors='ignore') if log_path.exists() else ''
    return int(proc.returncode or 0), text


def load_action_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('action', None), list):
            rows.append([float(x) for x in obj['action']])
    return rows


def write_heatmap_ppm(path: Path, actions: list[list[float]], max_frames: int = 256) -> bool:
    if not actions:
        return False

    frame_count = min(len(actions), max_frames)
    act_dim = min(len(actions[0]), 256)
    if frame_count <= 0 or act_dim <= 0:
        return False

    matrix: list[list[float]] = []
    for i in range(frame_count):
        row = actions[i][:act_dim]
        if len(row) < act_dim:
            row = row + [0.0] * (act_dim - len(row))
        matrix.append(row)

    flat = [v for row in matrix for v in row]
    lo = min(flat)
    hi = max(flat)
    span = hi - lo if hi > lo else 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        fp.write(f'P3\n{act_dim} {frame_count}\n255\n')
        for row in matrix:
            px: list[str] = []
            for value in row:
                t = (value - lo) / span
                r = int(255 * t)
                g = int(255 * (1.0 - abs(t - 0.5) * 2.0))
                b = int(255 * (1.0 - t))
                px.append(f'{r} {g} {b}')
            fp.write(' '.join(px) + '\n')

    return True


def build_common_export_args(args: argparse.Namespace, out_dir: Path) -> list[str]:
    out: list[str] = [
        '--arg-file',
        args.arg_file,
        '--engine-config',
        args.engine_config,
        '--out-dir',
        str(out_dir),
        '--device',
        args.device,
        '--num-envs',
        str(int(args.num_envs)),
    ]
    if args.env_config:
        out.extend(['--env-config', args.env_config])
    if args.agent_config:
        out.extend(['--agent-config', args.agent_config])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Run MimicKit single-case inference/export and visualization artifacts')
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--arg-file', required=True)
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--engine-config', default='data/engines/newton_engine.yaml')
    parser.add_argument('--env-config', default='')
    parser.add_argument('--agent-config', default='')
    parser.add_argument('--case-kind', choices=['policy', 'dummy'], default='policy')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--test-episodes', type=int, default=2)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_log = out_dir / 'mimic_test.log'
    export_log = out_dir / 'mimic_export.log'
    ref_actions = out_dir / 'ref_actions.jsonl'
    obs_fixture = out_dir / 'obs_fixture.jsonl'
    fixture_meta = out_dir / 'fixture_meta.json'
    schema_path = out_dir / 'schema.json'
    onnx_path = out_dir / 'policy_actor.onnx'
    visual_image = out_dir / 'mimic_action_heatmap.ppm'
    summary_path = out_dir / 'mimic_case_summary.json'

    test_rc = 0
    test_text = ''
    mimic_infer_ok = False

    if not args.skip_test:
        test_cmd = [
            sys.executable,
            str(RUN_PY),
            '--arg_file',
            args.arg_file,
            '--engine_config',
            args.engine_config,
            '--mode',
            'test',
            '--visualize',
            'false',
            '--devices',
            args.device,
            '--num_envs',
            str(int(args.num_envs)),
            '--test_episodes',
            str(int(args.test_episodes)),
        ]
        if args.env_config:
            test_cmd.extend(['--env_config', args.env_config])
        if args.agent_config:
            test_cmd.extend(['--agent_config', args.agent_config])
        if args.case_kind == 'policy':
            test_cmd.extend(['--model_file', args.model_file])

        test_rc, test_text = run_cmd(test_cmd, ROOT, test_log)
        if args.case_kind == 'policy':
            mimic_infer_ok = test_rc == 0 and ('Mean Return:' in test_text and 'Episodes:' in test_text)
        else:
            # Dummy path accepts successful test execution even without policy return signal.
            mimic_infer_ok = test_rc == 0
    else:
        mimic_infer_ok = True

    export_cmds: list[list[str]] = []
    common_args = build_common_export_args(args, out_dir)

    if args.case_kind == 'policy':
        export_cmds.append([
            sys.executable,
            str(EXPORT_SCHEMA_PY),
            *common_args,
            '--model-file',
            args.model_file,
        ])
        onnx_cmd = [
            sys.executable,
            str(EXPORT_ONNX_PY),
            *common_args,
            '--model-file',
            args.model_file,
            '--export-device',
            'cpu',
        ]
        if not args.skip_verify:
            onnx_cmd.append('--verify')
        export_cmds.append(onnx_cmd)
        export_cmds.append([
            sys.executable,
            str(EXPORT_FIXTURE_PY),
            *common_args,
            '--model-file',
            args.model_file,
            '--frames',
            str(int(args.frames)),
            '--seed',
            str(int(args.seed)),
        ])
    else:
        export_cmds.append([
            sys.executable,
            str(EXPORT_DUMMY_PY),
            *common_args,
            '--model-file',
            args.model_file,
            '--frames',
            str(int(args.frames)),
            '--seed',
            str(int(args.seed)),
        ])

    export_rc = 0
    export_texts: list[str] = []
    export_log.parent.mkdir(parents=True, exist_ok=True)
    with export_log.open('w', encoding='utf-8') as fp:
        for cmd in export_cmds:
            fp.write('[CMD] ' + ' '.join(cmd) + '\n')
            rc, txt = run_cmd(cmd, ROOT, out_dir / f'_tmp_export_{len(export_texts)}.log')
            export_texts.append(txt)
            fp.write(txt + '\n')
            if rc != 0:
                export_rc = rc
                break

    rows = load_action_rows(ref_actions)
    viz_ok = write_heatmap_ppm(visual_image, rows)

    fixture_ok = (
        export_rc == 0
        and ref_actions.exists()
        and obs_fixture.exists()
        and fixture_meta.exists()
        and schema_path.exists()
        and onnx_path.exists()
    )

    summary = {
        'case_id': args.case_id,
        'case_kind': args.case_kind,
        'arg_file': args.arg_file,
        'engine_config': args.engine_config,
        'env_config': args.env_config,
        'agent_config': args.agent_config,
        'model_file': str(Path(args.model_file).resolve()),
        'device': args.device,
        'mimic_infer_ok': bool(mimic_infer_ok),
        'mimic_viz_ok': bool(viz_ok),
        'fixture_ok': bool(fixture_ok),
        'test_exit_code': int(test_rc),
        'export_exit_code': int(export_rc),
        'mimic_test_log': str(test_log),
        'export_log': str(export_log),
        'obs_fixture': str(obs_fixture),
        'ref_actions': str(ref_actions),
        'fixture_meta': str(fixture_meta),
        'schema': str(schema_path),
        'onnx': str(onnx_path),
        'mimic_visual_image': str(visual_image),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))

    if summary['mimic_infer_ok'] and summary['fixture_ok'] and summary['mimic_viz_ok']:
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

