#!/usr/bin/env python3
import argparse
import csv
import glob
import importlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / 'mimickit' / 'run.py'
TRAIN_PY = Path(os.environ.get('MIMICKIT_TRAIN_PY', sys.executable))

NCCL_ENV = {
    'NCCL_P2P_DISABLE': '1',
    'NCCL_IB_DISABLE': '1',
    'NCCL_CUMEM_ENABLE': '0',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
}

MESA_ENV = {
    'MESA_GL_VERSION_OVERRIDE': '3.3',
    'MESA_GLSL_VERSION_OVERRIDE': '330',
}

HIUTIL_AGENT_BY_CASE = {
    'add_pi_plus_args.txt': 'data/agents/add_pi_plus_agent_hiutil.yaml',
    'amp_pi_plus_args.txt': 'data/agents/amp_pi_plus_agent_hiutil.yaml',
    'deepmimic_pi_plus_ppo_args.txt': 'data/agents/deepmimic_pi_plus_ppo_agent_hiutil.yaml',
}

AMP_PI_PLUS_CASE = 'amp_pi_plus_args.txt'


def parse_arg_file(path: Path):
    toks = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        toks.extend(line.split())

    out = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith('--'):
            k = t[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                out[k] = toks[i + 1]
                i += 2
            else:
                out[k] = 'true'
                i += 1
        else:
            i += 1
    return out


def parse_csv(text):
    vals = []
    for token in text.split(','):
        t = token.strip()
        if t:
            vals.append(t)
    return vals


def parse_int_csv(text):
    vals = []
    for t in parse_csv(text):
        try:
            v = int(t)
        except Exception:
            continue
        if v > 0:
            vals.append(v)

    out = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def parse_int_or(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def normalize_case_name(name: str):
    base = os.path.basename(name)
    if not base.endswith('.txt'):
        base = f'{base}.txt'
    return base


def check_python_deps(modules):
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append((m, str(e)))
    return missing


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_skip_log(path: Path, msg: str):
    ensure_parent(path)
    path.write_text(msg + '\n')


def run_cmd(cmd, env, cwd: Path, log_path: Path, timeout_s: int):
    ensure_parent(log_path)
    start = time.time()
    with log_path.open('w') as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    timed_out = False
    try:
        if timeout_s and timeout_s > 0:
            proc.wait(timeout=timeout_s)
        else:
            proc.wait()
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    rc = proc.returncode if proc.returncode is not None else 124
    if timed_out:
        rc = 124

    elapsed = time.time() - start
    text = log_path.read_text(errors='ignore') if log_path.exists() else ''
    return rc, timed_out, text, elapsed


def parse_signal_name(name: str):
    sig = getattr(signal, name, None)
    if sig is None:
        return signal.SIGINT
    return sig


def run_cmd_time_budget(cmd, env, cwd: Path, log_path: Path, budget_s: float, stop_sig, grace_s: int):
    ensure_parent(log_path)
    start = time.time()
    with log_path.open('w') as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    reached_budget = False
    stop_reason = 'long_exit'

    try:
        if budget_s and budget_s > 0:
            proc.wait(timeout=float(budget_s))
        else:
            proc.wait()
    except subprocess.TimeoutExpired:
        reached_budget = True
        stop_reason = f'long_budget_reached_{signal.Signals(stop_sig).name}'
        try:
            os.killpg(os.getpgid(proc.pid), stop_sig)
        except Exception:
            stop_reason = f'long_budget_signal_failed_{signal.Signals(stop_sig).name}'

        try:
            proc.wait(timeout=max(1, int(grace_s)))
        except subprocess.TimeoutExpired:
            stop_reason = 'long_budget_grace_term'
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                stop_reason = 'long_budget_force_kill'
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass

    rc = proc.returncode if proc.returncode is not None else 124
    elapsed = time.time() - start
    text = log_path.read_text(errors='ignore') if log_path.exists() else ''
    return rc, reached_budget, stop_reason, text, elapsed


def has_train_iter_row(text: str):
    for ln in text.splitlines():
        if re.match(r'^\s*\d+\s+', ln):
            return True
    return False


def has_fatal_train_marker(text: str, include_nccl: bool = True):
    t = text.lower()
    if 'array shapes must be non-negative' in text:
        return True
    if 'out of memory' in t or 'failed to allocate' in t or "cuda failure 2 'out of memory'" in t:
        return True
    if include_nccl and ('nccl' in t or 'distbackenderror' in t):
        return True
    return False


def has_metric(text: str):
    return ('Mean Return:' in text) and ('Episodes:' in text)


def load_existing_attempts(root_out: Path):
    out = {}
    runs_dir = root_out / 'runs'
    if not runs_dir.exists():
        return out

    for p in sorted(runs_dir.glob('*/attempts.json')):
        case_name = normalize_case_name(f'{p.parent.name}.txt')
        try:
            arr = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(arr, list) or not arr:
            continue

        final = arr[-1] if isinstance(arr[-1], dict) else {}
        out[case_name] = {
            'attempts': arr,
            'final': final,
        }

    return out


def load_case_attempts(case_run_dir: Path):
    p = case_run_dir / 'attempts.json'
    if not p.exists():
        return []
    try:
        arr = json.loads(p.read_text())
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict)]


def save_case_attempts(case_run_dir: Path, attempts):
    case_run_dir.mkdir(parents=True, exist_ok=True)
    with (case_run_dir / 'attempts.json').open('w') as f:
        json.dump(attempts, f, indent=2)


def pick_agent_variants(case_name: str, base_agent: str):
    out = []
    hi = HIUTIL_AGENT_BY_CASE.get(case_name, '')
    if hi:
        hi_path = ROOT / hi
        if hi_path.exists():
            out.append(('hiutil', hi))

    out.append(('default', base_agent))

    dedup = []
    seen = set()
    for vname, agent in out:
        if agent in seen:
            continue
        dedup.append((vname, agent))
        seen.add(agent)
    return dedup


def classify_train_error(text: str, prefix: str):
    t = text.lower()
    if 'array shapes must be non-negative' in text:
        return f'{prefix}_negative_shape'
    if 'out of memory' in t or 'failed to allocate' in t or "cuda failure 2 'out of memory'" in t:
        return f'{prefix}_oom'
    if 'nccl' in t or 'distbackenderror' in t:
        return f'{prefix}_nccl'
    return f'{prefix}_failed'


def load_allocation_profile(path: Path):
    rows = {}
    if not path.exists():
        return rows

    with path.open('r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            case_name = normalize_case_name(str(row.get('case', '')).strip())
            if not case_name:
                continue
            rows[case_name] = dict(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-config', default='data/engines/newton_engine.yaml')
    ap.add_argument('--cases', default='')
    ap.add_argument('--devices-train', default='cuda:0,cuda:1')
    ap.add_argument('--devices-nontrainable', default='cuda:0')
    ap.add_argument('--default-ladder', default='512,256,128,64,32')
    ap.add_argument('--pi-plus-ladder', default='40,39,38,36,32,24,16,8,4,2,1')
    ap.add_argument('--amp-pi-plus-ladder', default='38,36,32,24,16,8,4,2,1')
    ap.add_argument('--include-nontrainable', dest='include_nontrainable', action='store_true', default=True)
    ap.add_argument('--exclude-nontrainable', dest='include_nontrainable', action='store_false')
    ap.add_argument('--long-mode', choices=['max_samples', 'time_budget'], default='max_samples')
    ap.add_argument('--long-max-samples', type=int, default=30_000_000)
    ap.add_argument('--long-budget-hours', type=float, default=24.0)
    ap.add_argument('--long-budget-signal', choices=['SIGINT', 'SIGTERM'], default='SIGINT')
    ap.add_argument('--long-budget-grace-sec', type=int, default=300)
    ap.add_argument('--long-success-policy', choices=['strict_exit', 'budget_checkpoint'], default='strict_exit')
    ap.add_argument('--probe-iters', type=int, default=8)
    ap.add_argument('--probe-timeout', type=int, default=900)
    ap.add_argument('--long-timeout', type=int, default=0)
    ap.add_argument('--test-timeout', type=int, default=600)
    ap.add_argument('--viz-timeout', type=int, default=600)
    ap.add_argument('--test-episodes', type=int, default=10)
    ap.add_argument('--viz-episodes', type=int, default=1)
    ap.add_argument('--nontrainable-num-envs', type=int, default=1)
    ap.add_argument('--nontrainable-max-samples', type=int, default=128)
    ap.add_argument('--allocation-profile-tsv', default='')
    ap.add_argument('--allocation-fallback', choices=['ladder', 'strict'], default='ladder')
    ap.add_argument('--root-out', default='')
    ap.add_argument('--resume-skip-status', choices=['ok', 'all', 'none'], default='ok')
    ap.add_argument('--require-deps', default='torch,newton,warp,scipy,trimesh,pyglet')
    args = ap.parse_args()

    case_filter = {normalize_case_name(x) for x in parse_csv(args.cases)} if args.cases.strip() else set()

    default_ladder = parse_int_csv(args.default_ladder)
    pi_plus_ladder = parse_int_csv(args.pi_plus_ladder)
    amp_pi_plus_ladder = parse_int_csv(args.amp_pi_plus_ladder)

    if not default_ladder:
        default_ladder = [512, 256, 128, 64, 32]
    if not pi_plus_ladder:
        pi_plus_ladder = [40, 39, 38, 36, 32, 24, 16, 8, 4, 2, 1]
    if not amp_pi_plus_ladder:
        amp_pi_plus_ladder = [38, 36, 32, 24, 16, 8, 4, 2, 1]

    if args.long_mode == 'time_budget' and args.long_budget_hours <= 0:
        print('[ERROR] --long-budget-hours must be > 0 when --long-mode=time_budget')
        return 2
    if args.long_budget_grace_sec < 1:
        print('[ERROR] --long-budget-grace-sec must be >= 1')
        return 2

    long_budget_target_sec = max(0, int(round(args.long_budget_hours * 3600.0)))
    long_budget_signal = parse_signal_name(args.long_budget_signal)

    req_mods = parse_csv(args.require_deps)
    missing = check_python_deps(req_mods)
    if missing:
        print('[ERROR] missing deps:')
        for m, e in missing:
            print(f'  - {m}: {e}')
        return 2

    ts = time.strftime('%Y%m%d_%H%M%S')
    if args.root_out:
        root_name = args.root_out
    else:
        root_name = f'case_ultralong_{ts}' if args.long_mode == 'time_budget' else f'case_longcycle_{ts}'
    root_out = ROOT / 'output' / 'train' / root_name
    root_out.mkdir(parents=True, exist_ok=True)

    allocation_profile_path = None
    allocation_profile_rows = {}
    if args.allocation_profile_tsv.strip():
        allocation_profile_path = Path(args.allocation_profile_tsv.strip())
        if not allocation_profile_path.is_absolute():
            allocation_profile_path = (ROOT / allocation_profile_path).resolve()
        if not allocation_profile_path.exists():
            print(f'[ERROR] allocation profile not found: {allocation_profile_path}')
            return 2
        allocation_profile_rows = load_allocation_profile(allocation_profile_path)

    manifest = []
    selected = []
    trainable_count = 0
    nontrainable_count = 0

    for p in sorted(glob.glob(str(ROOT / 'args' / '*.txt'))):
        path = Path(p)
        name = path.name
        cfg = parse_arg_file(path)
        mode = cfg.get('mode', 'train')
        agent = cfg.get('agent_config', '')
        is_trainable = (mode == 'train' and bool(agent))
        case_type = 'trainable' if is_trainable else 'nontrainable'

        row = {
            'case': name,
            'mode': mode,
            'agent_config': agent,
            'env_config': cfg.get('env_config', ''),
            'method': name.split('_')[0],
            'case_type': case_type,
        }
        manifest.append(row)

        if is_trainable:
            trainable_count += 1
        else:
            nontrainable_count += 1

        if case_filter and name not in case_filter:
            continue

        if is_trainable or args.include_nontrainable:
            selected.append(row)

    with (root_out / 'case_manifest.tsv').open('w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['case', 'method', 'case_type', 'mode', 'agent_config', 'env_config'],
            delimiter='\t',
        )
        w.writeheader()
        w.writerows(manifest)

    train_devices = parse_csv(args.devices_train)
    if not train_devices:
        train_devices = ['cuda:0', 'cuda:1']

    nontrain_devices = parse_csv(args.devices_nontrainable)
    if not nontrain_devices:
        nontrain_devices = ['cuda:0']

    print(f'[INFO] root_out={root_out}', flush=True)
    print(
        f'[INFO] manifest trainable={trainable_count} nontrainable={nontrainable_count} '
        f'selected={len(selected)} include_nontrainable={int(args.include_nontrainable)}',
        flush=True,
    )
    print(
        f'[INFO] default_ladder={default_ladder} pi_plus_ladder={pi_plus_ladder} '
        f'amp_pi_plus_ladder={amp_pi_plus_ladder}',
        flush=True,
    )
    print(f'[INFO] train_devices={train_devices}', flush=True)
    print(f'[INFO] nontrainable_devices={nontrain_devices}', flush=True)
    print(
        f'[INFO] long_mode={args.long_mode} long_max_samples={args.long_max_samples} '
        f'long_budget_hours={args.long_budget_hours:.3f} long_budget_signal={args.long_budget_signal} '
        f'long_budget_grace_sec={args.long_budget_grace_sec} long_success_policy={args.long_success_policy}',
        flush=True,
    )
    print(
        f'[INFO] probe_iters={args.probe_iters} probe_timeout={args.probe_timeout}s '
        f'long_timeout={args.long_timeout}s',
        flush=True,
    )
    print(f'[INFO] resume_skip_status={args.resume_skip_status}', flush=True)
    if allocation_profile_path is not None:
        print(
            f'[INFO] allocation_profile={allocation_profile_path} '
            f'rows={len(allocation_profile_rows)} fallback={args.allocation_fallback}',
            flush=True,
        )

    existing_attempts = load_existing_attempts(root_out)
    if existing_attempts:
        existing_ok = 0
        for row in existing_attempts.values():
            final = row.get('final', {})
            if int(final.get('final_ok', 0)) == 1:
                existing_ok += 1
        print(
            f'[INFO] existing_attempts={len(existing_attempts)} '
            f'existing_ok={existing_ok} existing_fail_or_unfinished={len(existing_attempts) - existing_ok}',
            flush=True,
        )

    with (root_out / 'progress.json').open('w') as f:
        json.dump(
            {
                'done': 0,
                'total': len(selected),
                'last_case': '',
                'status': 'starting',
            },
            f,
            indent=2,
        )

    results = []

    for idx, case in enumerate(selected, 1):
        case_name = case['case']
        case_type = case.get('case_type', 'trainable')
        is_trainable = (case_type == 'trainable')
        arg_rel = f'args/{case_name}'
        base_agent = case['agent_config']
        existing = existing_attempts.get(case_name)

        with (root_out / 'progress.json').open('w') as f:
            json.dump(
                {
                    'done': idx - 1,
                    'total': len(selected),
                    'last_case': case_name,
                    'status': 'running_case',
                },
                f,
                indent=2,
            )

        if existing and args.resume_skip_status != 'none':
            prev_final = dict(existing.get('final', {}))
            prev_ok = int(prev_final.get('final_ok', 0)) == 1
            should_reuse = (args.resume_skip_status == 'all') or (args.resume_skip_status == 'ok' and prev_ok)
            if should_reuse:
                prev_final.setdefault('case', case_name)
                prev_final.setdefault('method', case['method'])
                prev_final.setdefault('case_type', case_type)
                prev_final.setdefault('note', 'reused')
                prev_final.setdefault('long_mode', (args.long_mode if is_trainable else 'bootstrap'))
                prev_final.setdefault('long_budget_hours', 0.0)
                prev_final.setdefault('long_budget_target_sec', 0)
                prev_final.setdefault('long_budget_elapsed_sec', 0.0)
                prev_final.setdefault('long_budget_reached', 0)
                prev_final.setdefault('long_budget_signal', '')
                prev_final.setdefault('long_stop_reason', '')
                prev_final.setdefault('alloc_profile_used', 0)
                prev_final.setdefault('alloc_variant', '')
                prev_final.setdefault('alloc_num_envs', -1)
                prev_final.setdefault('alloc_agent_config', '')
                prev_final.setdefault('alloc_source', '')
                results.append(prev_final)

                print(f"\n[CASE {idx}/{len(selected)}] {case_name} ({case_type})", flush=True)
                print(
                    f"  [RESUME] reused existing result final_ok={int(prev_ok)} "
                    f"note={prev_final.get('note', '')}",
                    flush=True,
                )

                with (root_out / 'progress.json').open('w') as f:
                    json.dump(
                        {
                            'done': idx,
                            'total': len(selected),
                            'last_case': case_name,
                            'status': 'reused_case',
                        },
                        f,
                        indent=2,
                    )
                continue

        if is_trainable:
            if case_name == AMP_PI_PLUS_CASE:
                ladder = amp_pi_plus_ladder
            elif 'pi_plus' in case_name:
                ladder = pi_plus_ladder
            else:
                ladder = default_ladder

            variants = pick_agent_variants(case_name, base_agent)
            case_devices = train_devices
            alloc_row = allocation_profile_rows.get(case_name, {})
            has_alloc_row = bool(alloc_row)
            alloc_status = str(alloc_row.get('status', '')).strip().lower() if alloc_row else ''
            alloc_variant = str(alloc_row.get('variant', '')).strip() if alloc_row else ''
            alloc_num_envs = parse_int_or(alloc_row.get('num_envs', ''), -1) if alloc_row else -1
            alloc_agent_cfg = str(alloc_row.get('agent_config', '')).strip() if alloc_row else ''
            if alloc_row and not alloc_agent_cfg:
                alloc_agent_cfg = base_agent
            profile_usable = bool(has_alloc_row and alloc_status == 'ok' and alloc_num_envs > 0 and alloc_agent_cfg)
        else:
            ladder = [max(1, int(args.nontrainable_num_envs))]
            variants = [('default', base_agent)]
            case_devices = nontrain_devices
            alloc_row = {}
            has_alloc_row = False
            alloc_status = ''
            alloc_variant = ''
            alloc_num_envs = -1
            alloc_agent_cfg = ''
            profile_usable = False

        attempt_plan = []
        seen_attempts = set()

        def add_attempt(variant_name, agent_cfg, num_envs, alloc_profile_used, alloc_source):
            key = (variant_name, agent_cfg, int(num_envs))
            if key in seen_attempts:
                return
            seen_attempts.add(key)
            attempt_plan.append(
                {
                    'variant': variant_name,
                    'agent_config': agent_cfg,
                    'num_envs': int(num_envs),
                    'alloc_profile_used': int(alloc_profile_used),
                    'alloc_variant': variant_name if alloc_profile_used else '',
                    'alloc_num_envs': int(num_envs) if alloc_profile_used else -1,
                    'alloc_agent_config': agent_cfg if alloc_profile_used else '',
                    'alloc_source': alloc_source,
                }
            )

        if is_trainable and profile_usable:
            selected_variant = alloc_variant or 'default'
            add_attempt(selected_variant, alloc_agent_cfg, alloc_num_envs, 1, 'profile')
            if args.allocation_fallback == 'ladder':
                for variant_name, agent_cfg in variants:
                    for num_envs in ladder:
                        add_attempt(variant_name, agent_cfg, num_envs, 0, 'ladder_fallback')
        elif is_trainable:
            if has_alloc_row:
                print(
                    f"  [ALLOC] profile row unusable status={alloc_status or 'empty'} "
                    f"num_envs={alloc_num_envs} agent={bool(alloc_agent_cfg)}",
                    flush=True,
                )
            for variant_name, agent_cfg in variants:
                for num_envs in ladder:
                    add_attempt(variant_name, agent_cfg, num_envs, 0, 'auto')
        else:
            for variant_name, agent_cfg in variants:
                for num_envs in ladder:
                    add_attempt(variant_name, agent_cfg, num_envs, 0, 'nontrainable')

        print(f"\n[CASE {idx}/{len(selected)}] {case_name} ({case_type})", flush=True)
        print(f"  [LADDER] {ladder}", flush=True)
        print(f"  [AGENTS] {[x[0] for x in variants]}", flush=True)
        print(f"  [DEVICES] {case_devices}", flush=True)
        if is_trainable and has_alloc_row:
            print(
                f"  [ALLOC] profile_found=1 usable={int(profile_usable)} "
                f"variant={alloc_variant or '-'} num_envs={alloc_num_envs}",
                flush=True,
            )
        print(f"  [ATTEMPTS] {len(attempt_plan)}", flush=True)

        final = None
        case_run_dir = root_out / 'runs' / case_name.replace('.txt', '')
        if args.resume_skip_status == 'none':
            all_attempts = []
        else:
            all_attempts = load_case_attempts(case_run_dir)
        if all_attempts:
            print(f"  [RESUME] loaded {len(all_attempts)} prior attempts", flush=True)

        for planned in attempt_plan:
            variant_name = planned['variant']
            agent_cfg = planned['agent_config']
            num_envs = int(planned['num_envs'])
            alloc_profile_used = int(planned.get('alloc_profile_used', 0))
            alloc_variant_used = str(planned.get('alloc_variant', '') or '')
            alloc_num_envs_used = int(planned.get('alloc_num_envs', -1))
            alloc_agent_cfg_used = str(planned.get('alloc_agent_config', '') or '')
            alloc_source = str(planned.get('alloc_source', '') or '')

            if is_trainable and alloc_profile_used and args.allocation_fallback == 'strict':
                print(f"  [ALLOC] strict profile mode active", flush=True)

            attempt_start = time.time()
            out_dir = root_out / 'runs' / case_name.replace('.txt', '') / f'{variant_name}_e{num_envs}'
            out_dir.mkdir(parents=True, exist_ok=True)

            probe_out = out_dir / 'probe_train'
            long_out = out_dir / 'long_train'
            probe_log = out_dir / 'probe_train.log'
            long_log = out_dir / 'long_train.log'
            test_log = out_dir / 'test.log'
            viz_log = out_dir / 'viz.log'

            if is_trainable:
                probe_samples = num_envs * len(case_devices) * 32 * args.probe_iters
            else:
                probe_samples = max(1, int(args.nontrainable_max_samples))

            env = os.environ.copy()
            env.update(NCCL_ENV)
            env['PYTHONUNBUFFERED'] = '1'

            attempt_idx = None
            for i in range(len(all_attempts) - 1, -1, -1):
                a = all_attempts[i]
                if (
                    a.get('variant') == variant_name
                    and int(a.get('num_envs', -1)) == num_envs
                    and str(a.get('agent_config', '')) == str(agent_cfg)
                ):
                    attempt_idx = i
                    break

            if attempt_idx is None:
                attempt = {
                    'case': case_name,
                    'method': case['method'],
                    'case_type': case_type,
                    'variant': variant_name,
                    'agent_config': agent_cfg,
                    'num_envs': num_envs,
                    'probe_rc': -1,
                    'probe_ok': 0,
                    'probe_timeout': 0,
                    'long_rc': -1,
                    'long_ok': 0,
                    'long_timeout': 0,
                    'long_mode': (args.long_mode if is_trainable else 'bootstrap'),
                    'long_max_samples': int(args.long_max_samples if (is_trainable and args.long_mode == 'max_samples') else 0),
                    'long_budget_hours': float(args.long_budget_hours if (is_trainable and args.long_mode == 'time_budget') else 0.0),
                    'long_budget_target_sec': int(long_budget_target_sec if (is_trainable and args.long_mode == 'time_budget') else 0),
                    'long_budget_elapsed_sec': 0.0,
                    'long_budget_reached': 0,
                    'long_budget_signal': (args.long_budget_signal if (is_trainable and args.long_mode == 'time_budget') else ''),
                    'long_stop_reason': '',
                    'alloc_profile_used': int(alloc_profile_used),
                    'alloc_variant': alloc_variant_used,
                    'alloc_num_envs': int(alloc_num_envs_used),
                    'alloc_agent_config': alloc_agent_cfg_used,
                    'alloc_source': alloc_source,
                    'test_rc': -1,
                    'test_ok': 0,
                    'viz_rc': -1,
                    'viz_ok': 0,
                    'probe_elapsed_sec': 0.0,
                    'long_elapsed_sec': 0.0,
                    'test_elapsed_sec': 0.0,
                    'viz_elapsed_sec': 0.0,
                    'stage_elapsed_sec': 0.0,
                    'final_ok': 0,
                    'note': 'in_progress_probe',
                    'out_dir': str(out_dir),
                    'probe_out_dir': str(probe_out),
                    'long_out_dir': str(long_out if is_trainable else ''),
                    'probe_train_log': str(probe_log),
                    'long_train_log': str(long_log),
                    'test_log': str(test_log),
                    'viz_log': str(viz_log),
                }
                all_attempts.append(attempt)
                attempt_idx = len(all_attempts) - 1
            else:
                attempt = dict(all_attempts[attempt_idx])

            attempt['case'] = case_name
            attempt['method'] = case['method']
            attempt['case_type'] = case_type
            attempt['variant'] = variant_name
            attempt['agent_config'] = agent_cfg
            attempt['num_envs'] = num_envs
            attempt['long_mode'] = (args.long_mode if is_trainable else 'bootstrap')
            attempt['long_max_samples'] = int(args.long_max_samples if (is_trainable and args.long_mode == 'max_samples') else 0)
            attempt['long_budget_hours'] = float(args.long_budget_hours if (is_trainable and args.long_mode == 'time_budget') else 0.0)
            attempt['long_budget_target_sec'] = int(long_budget_target_sec if (is_trainable and args.long_mode == 'time_budget') else 0)
            if is_trainable and args.long_mode == 'time_budget':
                attempt['long_budget_elapsed_sec'] = float(attempt.get('long_budget_elapsed_sec', 0.0) or 0.0)
                attempt['long_budget_reached'] = int(attempt.get('long_budget_reached', 0) or 0)
            else:
                attempt['long_budget_elapsed_sec'] = 0.0
                attempt['long_budget_reached'] = 0
            attempt['long_budget_signal'] = (args.long_budget_signal if (is_trainable and args.long_mode == 'time_budget') else '')
            attempt['long_stop_reason'] = str(attempt.get('long_stop_reason', '') or '')
            attempt['alloc_profile_used'] = int(alloc_profile_used)
            attempt['alloc_variant'] = alloc_variant_used
            attempt['alloc_num_envs'] = int(alloc_num_envs_used)
            attempt['alloc_agent_config'] = alloc_agent_cfg_used
            attempt['alloc_source'] = alloc_source
            attempt['out_dir'] = str(out_dir)
            attempt['probe_out_dir'] = str(probe_out)
            attempt['long_out_dir'] = str(long_out if is_trainable else '')
            attempt['probe_train_log'] = str(probe_log)
            attempt['long_train_log'] = str(long_log)
            attempt['test_log'] = str(test_log)
            attempt['viz_log'] = str(viz_log)

            prev_stage_elapsed = float(attempt.get('stage_elapsed_sec', 0.0) or 0.0)
            all_attempts[attempt_idx] = attempt
            save_case_attempts(case_run_dir, all_attempts)

            probe_ready = (
                int(attempt.get('probe_ok', 0)) == 1
                and (probe_out / 'model.pt').exists()
                and (probe_out / 'log.txt').exists()
            )
            if not probe_ready:
                attempt['note'] = 'in_progress_probe'
                all_attempts[attempt_idx] = attempt
                save_case_attempts(case_run_dir, all_attempts)

                probe_cmd = [
                    str(TRAIN_PY), str(RUN_PY),
                    '--arg_file', arg_rel,
                    '--engine_config', args.engine_config,
                    '--mode', 'train',
                    '--visualize', 'false',
                    '--devices', *case_devices,
                    '--num_envs', str(num_envs),
                    '--max_samples', str(probe_samples),
                    '--out_dir', str(probe_out),
                    '--master_port', str(random.randint(20000, 45000)),
                ]
                if agent_cfg:
                    probe_cmd.extend(['--agent_config', agent_cfg])

                probe_rc, probe_to, probe_text, probe_elapsed = run_cmd(
                    probe_cmd,
                    env,
                    ROOT,
                    probe_log,
                    args.probe_timeout,
                )
                probe_ok = (
                    probe_rc == 0
                    and (probe_out / 'model.pt').exists()
                    and (probe_out / 'log.txt').exists()
                )

                attempt['probe_rc'] = probe_rc
                attempt['probe_ok'] = int(probe_ok)
                attempt['probe_timeout'] = int(probe_to)
                attempt['probe_elapsed_sec'] = round(float(attempt.get('probe_elapsed_sec', 0.0) or 0.0) + probe_elapsed, 2)

                if not probe_ok:
                    note = classify_train_error(probe_text, 'probe')
                    if probe_rc == 124 or probe_to:
                        note = 'probe_timeout'
                    attempt['note'] = note
                    attempt['stage_elapsed_sec'] = round(prev_stage_elapsed + (time.time() - attempt_start), 2)
                    all_attempts[attempt_idx] = attempt
                    save_case_attempts(case_run_dir, all_attempts)
                    print(f"  [TRY {variant_name} e{num_envs}] {note}", flush=True)
                    continue
            else:
                print(f"  [TRY {variant_name} e{num_envs}] resume: skip probe", flush=True)

            if is_trainable:
                long_ready = (
                    int(attempt.get('long_ok', 0)) == 1
                    and (long_out / 'model.pt').exists()
                    and (long_out / 'log.txt').exists()
                )
                if not long_ready:
                    attempt['note'] = 'in_progress_long'
                    all_attempts[attempt_idx] = attempt
                    save_case_attempts(case_run_dir, all_attempts)

                    long_model_file = long_out / 'model.pt'
                    if not long_model_file.exists():
                        long_model_file = probe_out / 'model.pt'

                    long_cmd = [
                        str(TRAIN_PY), str(RUN_PY),
                        '--arg_file', arg_rel,
                        '--engine_config', args.engine_config,
                        '--mode', 'train',
                        '--visualize', 'false',
                        '--devices', *case_devices,
                        '--num_envs', str(num_envs),
                        '--out_dir', str(long_out),
                        '--model_file', str(long_model_file),
                        '--master_port', str(random.randint(20000, 45000)),
                    ]
                    if args.long_mode == 'max_samples':
                        long_cmd.extend(['--max_samples', str(args.long_max_samples)])
                    if agent_cfg:
                        long_cmd.extend(['--agent_config', agent_cfg])

                    long_stop_reason = ''
                    if args.long_mode == 'max_samples':
                        long_rc, long_to, long_text, long_elapsed = run_cmd(
                            long_cmd,
                            env,
                            ROOT,
                            long_log,
                            args.long_timeout,
                        )
                        long_ok = (
                            long_rc == 0
                            and (long_out / 'model.pt').exists()
                            and (long_out / 'log.txt').exists()
                        )
                        long_stop_reason = f'long_exit_rc_{long_rc}'
                    else:
                        prev_budget_elapsed = float(attempt.get('long_budget_elapsed_sec', 0.0) or 0.0)
                        target_budget_sec = float(attempt.get('long_budget_target_sec', 0.0) or 0.0)
                        remaining_budget_sec = max(0.0, target_budget_sec - prev_budget_elapsed)

                        long_to = False
                        long_rc = 0
                        long_text = ''
                        long_elapsed = 0.0
                        reached_budget = False

                        if remaining_budget_sec > 0:
                            long_rc, reached_budget, long_stop_reason, long_text, long_elapsed = run_cmd_time_budget(
                                long_cmd,
                                env,
                                ROOT,
                                long_log,
                                remaining_budget_sec,
                                long_budget_signal,
                                args.long_budget_grace_sec,
                            )
                        else:
                            long_stop_reason = 'long_budget_already_reached'
                            reached_budget = bool(int(attempt.get('long_budget_reached', 0)))

                        budget_elapsed = prev_budget_elapsed + float(long_elapsed)
                        attempt['long_budget_elapsed_sec'] = round(budget_elapsed, 2)
                        if reached_budget or budget_elapsed >= max(0.0, target_budget_sec - 1e-6):
                            attempt['long_budget_reached'] = 1

                        long_train_text = (long_out / 'log.txt').read_text(errors='ignore') if (long_out / 'log.txt').exists() else ''
                        checkpoint_ok = (
                            (long_out / 'model.pt').exists()
                            and (long_out / 'log.txt').exists()
                            and has_train_iter_row(long_train_text)
                        )
                        budget_reached_now = bool(int(attempt.get('long_budget_reached', 0)))
                        fatal_marker = has_fatal_train_marker(long_text, include_nccl=(not budget_reached_now))

                        if args.long_success_policy == 'strict_exit':
                            long_ok = (long_rc == 0 and checkpoint_ok and (not fatal_marker))
                        else:
                            long_ok = (int(attempt.get('long_budget_reached', 0)) == 1 and checkpoint_ok and (not fatal_marker))

                    attempt['long_rc'] = long_rc
                    attempt['long_ok'] = int(long_ok)
                    attempt['long_timeout'] = int(long_to)
                    attempt['long_elapsed_sec'] = round(float(attempt.get('long_elapsed_sec', 0.0) or 0.0) + long_elapsed, 2)
                    attempt['long_stop_reason'] = long_stop_reason

                    if not long_ok:
                        note = classify_train_error(long_text, 'long')
                        if args.long_mode == 'max_samples':
                            if long_rc == 124 or long_to:
                                note = 'long_timeout'
                        else:
                            budget_reached_now = bool(int(attempt.get('long_budget_reached', 0)))
                            if has_fatal_train_marker(long_text, include_nccl=(not budget_reached_now)):
                                note = classify_train_error(long_text, 'long')
                            elif int(attempt.get('long_budget_reached', 0)) == 1:
                                note = 'long_budget_invalid'
                            elif long_rc != 0:
                                note = 'long_failed'
                            else:
                                note = 'long_budget_incomplete'
                        attempt['note'] = note
                        attempt['stage_elapsed_sec'] = round(prev_stage_elapsed + (time.time() - attempt_start), 2)
                        all_attempts[attempt_idx] = attempt
                        save_case_attempts(case_run_dir, all_attempts)
                        print(f"  [TRY {variant_name} e{num_envs}] probe=ok {note}", flush=True)
                        continue
                    else:
                        print(f"  [TRY {variant_name} e{num_envs}] resume: skip long", flush=True)

                    eval_model = long_out / 'model.pt'
                else:
                    write_skip_log(long_log, 'SKIPPED: nontrainable case has no long train stage.')
                    attempt['long_rc'] = 0
                    attempt['long_ok'] = 1
                    attempt['long_timeout'] = 0
                    attempt['long_stop_reason'] = 'bootstrap_skipped_long'
                    eval_model = probe_out / 'model.pt'

                test_ready = False
                if int(attempt.get('test_ok', 0)) == 1 and test_log.exists():
                    test_ready = has_metric(test_log.read_text(errors='ignore'))

                if not test_ready:
                    attempt['note'] = 'in_progress_test'
                    all_attempts[attempt_idx] = attempt
                    save_case_attempts(case_run_dir, all_attempts)

                    test_cmd = [
                        str(TRAIN_PY), str(RUN_PY),
                        '--arg_file', arg_rel,
                        '--engine_config', args.engine_config,
                        '--mode', 'test',
                        '--visualize', 'false',
                        '--devices', 'cuda:0',
                        '--num_envs', '1',
                        '--test_episodes', str(args.test_episodes),
                        '--model_file', str(eval_model),
                    ]
                    if agent_cfg:
                        test_cmd.extend(['--agent_config', agent_cfg])

                    test_rc, test_to, test_text, test_elapsed = run_cmd(
                        test_cmd,
                        env,
                        ROOT,
                        test_log,
                        args.test_timeout,
                    )
                    test_ok = (test_rc == 0 and has_metric(test_text))
                    attempt['test_rc'] = test_rc
                    attempt['test_ok'] = int(test_ok)
                    attempt['test_elapsed_sec'] = round(float(attempt.get('test_elapsed_sec', 0.0) or 0.0) + test_elapsed, 2)

                    if not test_ok:
                        attempt['note'] = 'test_timeout' if (test_rc == 124 or test_to) else 'test_failed'
                        attempt['stage_elapsed_sec'] = round(prev_stage_elapsed + (time.time() - attempt_start), 2)
                        all_attempts[attempt_idx] = attempt
                        save_case_attempts(case_run_dir, all_attempts)
                        print(f"  [TRY {variant_name} e{num_envs}] probe=ok long=ok test=fail", flush=True)
                        continue
                else:
                    print(f"  [TRY {variant_name} e{num_envs}] resume: skip test", flush=True)

                viz_ready = False
                if int(attempt.get('viz_ok', 0)) == 1 and viz_log.exists():
                    viz_ready = has_metric(viz_log.read_text(errors='ignore'))

                if not viz_ready:
                    attempt['note'] = 'in_progress_viz'
                    all_attempts[attempt_idx] = attempt
                    save_case_attempts(case_run_dir, all_attempts)

                    viz_cmd = [
                        str(TRAIN_PY), str(RUN_PY),
                        '--arg_file', arg_rel,
                        '--engine_config', args.engine_config,
                        '--mode', 'test',
                        '--visualize', 'true',
                        '--devices', 'cuda:0',
                        '--num_envs', '1',
                        '--test_episodes', str(args.viz_episodes),
                        '--model_file', str(eval_model),
                    ]
                    if agent_cfg:
                        viz_cmd.extend(['--agent_config', agent_cfg])

                    viz_env = dict(env)
                    viz_env.update(MESA_ENV)
                    viz_rc, viz_to, viz_text, viz_elapsed = run_cmd(
                        viz_cmd,
                        viz_env,
                        ROOT,
                        viz_log,
                        args.viz_timeout,
                    )
                    viz_ok = (viz_rc == 0 and has_metric(viz_text))
                    attempt['viz_rc'] = viz_rc
                    attempt['viz_ok'] = int(viz_ok)
                    attempt['viz_elapsed_sec'] = round(float(attempt.get('viz_elapsed_sec', 0.0) or 0.0) + viz_elapsed, 2)

                    if not viz_ok:
                        note = 'viz_timeout' if (viz_rc == 124 or viz_to) else 'viz_failed'
                        if 'GLSL 1.50 is not supported' in viz_text:
                            note = 'viz_glsl_version'
                        attempt['note'] = note
                        attempt['stage_elapsed_sec'] = round(prev_stage_elapsed + (time.time() - attempt_start), 2)
                        all_attempts[attempt_idx] = attempt
                        save_case_attempts(case_run_dir, all_attempts)
                        print(f"  [TRY {variant_name} e{num_envs}] probe=ok long=ok test=ok viz=fail({note})", flush=True)
                        continue
                else:
                    print(f"  [TRY {variant_name} e{num_envs}] resume: skip viz", flush=True)

                attempt['final_ok'] = 1
                attempt['note'] = 'ok'
                attempt['stage_elapsed_sec'] = round(prev_stage_elapsed + (time.time() - attempt_start), 2)
                all_attempts[attempt_idx] = attempt
                save_case_attempts(case_run_dir, all_attempts)
                final = attempt
                print(f"  [TRY {variant_name} e{num_envs}] final=ok", flush=True)
                break

            if final is not None:
                break

        if final is None:
            if all_attempts:
                final = all_attempts[-1]
            else:
                final = {
                    'case': case_name,
                    'method': case['method'],
                    'case_type': case_type,
                    'variant': '',
                    'agent_config': base_agent,
                    'num_envs': -1,
                    'probe_rc': -1,
                    'probe_ok': 0,
                    'probe_timeout': 0,
                    'long_rc': -1,
                    'long_ok': 0,
                    'long_timeout': 0,
                    'long_mode': (args.long_mode if is_trainable else 'bootstrap'),
                    'long_max_samples': int(args.long_max_samples if (is_trainable and args.long_mode == 'max_samples') else 0),
                    'long_budget_hours': float(args.long_budget_hours if (is_trainable and args.long_mode == 'time_budget') else 0.0),
                    'long_budget_target_sec': int(long_budget_target_sec if (is_trainable and args.long_mode == 'time_budget') else 0),
                    'long_budget_elapsed_sec': 0.0,
                    'long_budget_reached': 0,
                    'long_budget_signal': (args.long_budget_signal if (is_trainable and args.long_mode == 'time_budget') else ''),
                    'long_stop_reason': '',
                    'alloc_profile_used': 0,
                    'alloc_variant': '',
                    'alloc_num_envs': -1,
                    'alloc_agent_config': '',
                    'alloc_source': '',
                    'test_rc': -1,
                    'test_ok': 0,
                    'viz_rc': -1,
                    'viz_ok': 0,
                    'probe_elapsed_sec': 0.0,
                    'long_elapsed_sec': 0.0,
                    'test_elapsed_sec': 0.0,
                    'viz_elapsed_sec': 0.0,
                    'stage_elapsed_sec': 0.0,
                    'final_ok': 0,
                    'note': 'unrun',
                    'out_dir': '',
                    'probe_out_dir': '',
                    'long_out_dir': '',
                    'probe_train_log': '',
                    'long_train_log': '',
                    'test_log': '',
                    'viz_log': '',
                }

        results.append(final)
        save_case_attempts(case_run_dir, all_attempts)

        with (root_out / 'progress.json').open('w') as f:
            json.dump(
                {
                    'done': idx,
                    'total': len(selected),
                    'last_case': case_name,
                    'status': 'completed_case',
                },
                f,
                indent=2,
            )

    fields = [
        'case', 'method', 'case_type', 'variant', 'agent_config', 'num_envs',
        'probe_rc', 'probe_ok', 'probe_timeout',
        'long_rc', 'long_ok', 'long_timeout', 'long_mode', 'long_max_samples',
        'long_budget_hours', 'long_budget_target_sec', 'long_budget_elapsed_sec',
        'long_budget_reached', 'long_budget_signal', 'long_stop_reason',
        'alloc_profile_used', 'alloc_variant', 'alloc_num_envs', 'alloc_agent_config', 'alloc_source',
        'test_rc', 'test_ok',
        'viz_rc', 'viz_ok',
        'probe_elapsed_sec', 'long_elapsed_sec', 'test_elapsed_sec', 'viz_elapsed_sec', 'stage_elapsed_sec',
        'final_ok', 'note',
        'out_dir', 'probe_out_dir', 'long_out_dir',
        'probe_train_log', 'long_train_log', 'test_log', 'viz_log',
    ]

    with (root_out / 'best_by_case.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for row in results:
            w.writerow(row)

    ok = sum(1 for r in results if int(r.get('final_ok', 0)) == 1)
    total = len(results)
    print(f'\n[DONE] ok={ok}/{total} results={root_out / "best_by_case.tsv"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
