#!/usr/bin/env python3
"""Discover MimicKit output cases for UE all-cases pipeline (model-dir truth first)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path('/root/Project/MimicKit').resolve()
DEFAULT_TRAIN_ROOT = ROOT / 'output' / 'train'
ARGS_ROOT = ROOT / 'args'
POLICY_AGENT_NAMES = {'PPO', 'AWR', 'AMP', 'ASE', 'ADD'}


@dataclass
class CaseRow:
    case_id: str
    model_pt: str
    family: str
    arg_file: str
    env_cfg: str
    engine_cfg: str
    agent_cfg: str
    motion_src: str
    obs_dim: int
    act_dim: int
    control_freq: int
    ue_viz_mode: str
    actual_engine: str
    case_kind: str
    arg_source: str
    status: str


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _safe_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _parse_arg_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out

    toks: list[str] = []
    for raw in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        toks.extend(line.split())

    i = 0
    while i < len(toks):
        token = toks[i]
        if token.startswith('--'):
            key = token[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                out[key] = toks[i + 1]
                i += 2
            else:
                out[key] = 'true'
                i += 1
        else:
            i += 1
    return out


def _normalize_path_text(path_text: str) -> str:
    if not path_text:
        return ''
    p = Path(path_text)
    if p.is_absolute():
        return str(p.resolve())
    return str((ROOT / p).resolve())


def _infer_family(text: str) -> str:
    low = text.lower()
    if 'go2' in low:
        return 'go2'
    if re.search(r'(^|[_/\\])g1([_/\\]|$)', low):
        return 'g1'
    if 'smpl' in low:
        return 'smpl'
    if 'humanoid' in low or 'pi_plus' in low:
        return 'humanoid'
    return 'other'


def _default_viz_mode(family: str) -> str:
    return 'Skeletal' if family == 'humanoid' else 'DebugRig'


def _infer_arg_from_path(model_path: Path, arg_names: list[str]) -> str:
    text = str(model_path).lower()
    for name in arg_names:
        stem = name[:-4].lower()
        if stem in text:
            return f'args/{name}'

    parts = model_path.parts
    if 'runs' in parts:
        idx = parts.index('runs')
        if idx + 1 < len(parts):
            guess = f'args/{parts[idx + 1]}.txt'
            if (ROOT / guess).exists():
                return guess
    return ''


def _case_id_from_model(model_path: Path, train_root: Path) -> str:
    rel = model_path.relative_to(train_root)
    return rel.parent.as_posix().replace('/', '__')


def _collect_metadata(model_dir: Path, train_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    cursor = model_dir
    while True:
        for probe in (
            cursor / 'fixture_meta.json',
            cursor / 'onnx_export_meta.json',
            cursor / 'schema.json',
            cursor / 'ue_export' / 'fixture_meta.json',
            cursor / 'ue_export' / 'onnx_export_meta.json',
            cursor / 'ue_export' / 'schema.json',
        ):
            if not probe.exists():
                continue
            data = _safe_json(probe)
            if not isinstance(data, dict):
                continue
            source = data.get('source', {})
            if isinstance(source, dict):
                for key, value in source.items():
                    metadata.setdefault(key, value)
            model = data.get('model', {})
            if isinstance(model, dict):
                metadata.setdefault('obs_dim', model.get('obs_dim', 0))
                metadata.setdefault('act_dim', model.get('act_dim', 0))
            env = data.get('env', {})
            if isinstance(env, dict):
                metadata.setdefault('control_freq', env.get('control_freq', 0))

        if cursor == train_root or train_root not in cursor.parents:
            break
        cursor = cursor.parent
    return metadata


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _read_motion_source(env_cfg_abs: str) -> str:
    if not env_cfg_abs:
        return ''
    env_path = Path(env_cfg_abs)
    if not env_path.exists():
        return ''
    env_cfg = _safe_yaml(env_path)
    if not env_cfg:
        return ''
    for key in ('motion_file', 'motion_path', 'motion_data'):
        if key in env_cfg and isinstance(env_cfg[key], str):
            return env_cfg[key]
    motions = env_cfg.get('motions', None)
    if isinstance(motions, list) and motions:
        first = motions[0]
        if isinstance(first, str):
            return first
    return ''


def _actual_engine_name(engine_cfg_abs: str) -> str:
    if not engine_cfg_abs:
        return 'unknown'
    eng = _safe_yaml(Path(engine_cfg_abs))
    if eng:
        name = eng.get('engine_name', '')
        if isinstance(name, str) and name:
            return name
    text = Path(engine_cfg_abs).read_text(encoding='utf-8', errors='ignore').lower() if Path(engine_cfg_abs).exists() else ''
    if 'newton' in text:
        return 'newton'
    if 'isaac_gym' in text:
        return 'isaac_gym'
    if 'isaac_lab' in text:
        return 'isaac_lab'
    return 'unknown'


def _agent_name(agent_cfg_abs: str) -> str:
    if not agent_cfg_abs:
        return 'Dummy'
    cfg = _safe_yaml(Path(agent_cfg_abs))
    name = cfg.get('agent_name', 'Dummy') if isinstance(cfg, dict) else 'Dummy'
    return str(name) if name else 'Dummy'


def discover_cases(train_root: Path) -> list[CaseRow]:
    if not train_root.exists():
        return []

    arg_files = sorted(ARGS_ROOT.glob('*.txt'))
    arg_names = [p.name for p in arg_files]

    rows: list[CaseRow] = []
    for model_file in sorted(train_root.rglob('model.pt')):
        model_dir = model_file.parent
        metadata = _collect_metadata(model_dir, train_root)

        # model-dir truth first
        local_engine = model_dir / 'engine_config.yaml'
        local_env = model_dir / 'env_config.yaml'
        local_agent = model_dir / 'agent_config.yaml'

        arg_rel = ''
        if isinstance(metadata.get('arg_file'), str):
            candidate = str(metadata['arg_file']).strip()
            if candidate and (ROOT / candidate).exists():
                arg_rel = candidate
        if not arg_rel:
            arg_rel = _infer_arg_from_path(model_file, arg_names)

        arg_abs = (ROOT / arg_rel).resolve() if arg_rel else Path('')
        arg_table = _parse_arg_file(arg_abs) if arg_rel and arg_abs.exists() else {}

        if local_engine.exists():
            engine_cfg = str(local_engine.resolve())
        else:
            engine_cfg = _normalize_path_text(str(metadata.get('engine_config', '')))
            if not engine_cfg:
                engine_cfg = _normalize_path_text(arg_table.get('engine_config', ''))

        if local_env.exists():
            env_cfg = str(local_env.resolve())
        else:
            env_cfg = _normalize_path_text(str(metadata.get('env_config', '')))
            if not env_cfg:
                env_cfg = _normalize_path_text(arg_table.get('env_config', ''))

        if local_agent.exists():
            agent_cfg = str(local_agent.resolve())
        else:
            agent_cfg = _normalize_path_text(str(metadata.get('agent_config', '')))
            if not agent_cfg:
                agent_cfg = _normalize_path_text(arg_table.get('agent_config', ''))

        obs_dim = _to_int(metadata.get('obs_dim', 0))
        act_dim = _to_int(metadata.get('act_dim', 0))
        control_freq = _to_int(metadata.get('control_freq', 0))
        if control_freq <= 0 and engine_cfg:
            engine_cfg_dict = _safe_yaml(Path(engine_cfg))
            control_freq = _to_int(engine_cfg_dict.get('control_freq', 0))

        actual_engine = _actual_engine_name(engine_cfg)
        agent_name = _agent_name(agent_cfg)
        case_kind = 'policy' if agent_name in POLICY_AGENT_NAMES else 'dummy'
        arg_source = 'args_file' if arg_rel and arg_abs.exists() else 'reconstructed'

        motion_src = _read_motion_source(env_cfg)

        family_text = ' '.join([
            arg_rel,
            env_cfg,
            agent_cfg,
            str(model_file),
            motion_src,
        ])
        family = _infer_family(family_text)

        status = 'ready'
        if not engine_cfg or not Path(engine_cfg).exists():
            status = 'missing_engine_cfg'
        elif not env_cfg or not Path(env_cfg).exists():
            status = 'missing_env_cfg'
        elif case_kind == 'policy' and (not agent_cfg or not Path(agent_cfg).exists()):
            status = 'missing_agent_cfg'

        rows.append(
            CaseRow(
                case_id=_case_id_from_model(model_file, train_root),
                model_pt=str(model_file.resolve()),
                family=family,
                arg_file=arg_rel,
                env_cfg=env_cfg,
                engine_cfg=engine_cfg,
                agent_cfg=agent_cfg,
                motion_src=motion_src,
                obs_dim=obs_dim,
                act_dim=act_dim,
                control_freq=control_freq,
                ue_viz_mode=_default_viz_mode(family),
                actual_engine=actual_engine,
                case_kind=case_kind,
                arg_source=arg_source,
                status=status,
            )
        )

    return rows


def write_jsonl(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        for row in rows:
            fp.write(json.dumps(asdict(row), ensure_ascii=False) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Discover MimicKit output cases for UE all-cases pipeline')
    parser.add_argument('--train-root', default=str(DEFAULT_TRAIN_ROOT))
    parser.add_argument('--output', required=True, help='Output case_manifest.jsonl path')
    parser.add_argument('--summary', default='', help='Output summary json path')
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    rows = discover_cases(train_root)

    output_path = Path(args.output).resolve()
    write_jsonl(output_path, rows)

    summary = {
        'train_root': str(train_root),
        'total_cases': len(rows),
        'ready_cases': sum(1 for r in rows if r.status == 'ready'),
        'families': {},
        'status': {},
        'actual_engine': {},
        'case_kind': {},
        'arg_source': {},
    }

    for row in rows:
        summary['families'][row.family] = summary['families'].get(row.family, 0) + 1
        summary['status'][row.status] = summary['status'].get(row.status, 0) + 1
        summary['actual_engine'][row.actual_engine] = summary['actual_engine'].get(row.actual_engine, 0) + 1
        summary['case_kind'][row.case_kind] = summary['case_kind'].get(row.case_kind, 0) + 1
        summary['arg_source'][row.arg_source] = summary['arg_source'].get(row.arg_source, 0) + 1

    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
