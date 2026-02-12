#!/usr/bin/env python3
import argparse
import csv
import glob
import importlib
import json
import os
import random
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


def has_metric(text: str):
    return ('Mean Return:' in text) and ('Episodes:' in text)


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
    ap.add_argument('--long-max-samples', type=int, default=30_000_000)
    ap.add_argument('--probe-iters', type=int, default=8)
    ap.add_argument('--probe-timeout', type=int, default=900)
    ap.add_argument('--long-timeout', type=int, default=0)
    ap.add_argument('--test-timeout', type=int, default=600)
    ap.add_argument('--viz-timeout', type=int, default=600)
    ap.add_argument('--test-episodes', type=int, default=10)
    ap.add_argument('--viz-episodes', type=int, default=1)
    ap.add_argument('--nontrainable-num-envs', type=int, default=1)
    ap.add_argument('--nontrainable-max-samples', type=int, default=128)
    ap.add_argument('--root-out', default='')
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

    req_mods = parse_csv(args.require_deps)
    missing = check_python_deps(req_mods)
    if missing:
        print('[ERROR] missing deps:')
        for m, e in missing:
            print(f'  - {m}: {e}')
        return 2

    ts = time.strftime('%Y%m%d_%H%M%S')
    root_out = ROOT / 'output' / 'train' / (args.root_out if args.root_out else f'case_longcycle_{ts}')
    root_out.mkdir(parents=True, exist_ok=True)

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
        f'[INFO] long_max_samples={args.long_max_samples} probe_iters={args.probe_iters} '
        f'probe_timeout={args.probe_timeout}s long_timeout={args.long_timeout}s',
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

        if is_trainable:
            if case_name == AMP_PI_PLUS_CASE:
                ladder = amp_pi_plus_ladder
            elif 'pi_plus' in case_name:
                ladder = pi_plus_ladder
            else:
                ladder = default_ladder

            variants = pick_agent_variants(case_name, base_agent)
            case_devices = train_devices
        else:
            ladder = [max(1, int(args.nontrainable_num_envs))]
            variants = [('default', base_agent)]
            case_devices = nontrain_devices

        print(f"\n[CASE {idx}/{len(selected)}] {case_name} ({case_type})", flush=True)
        print(f"  [LADDER] {ladder}", flush=True)
        print(f"  [AGENTS] {[x[0] for x in variants]}", flush=True)
        print(f"  [DEVICES] {case_devices}", flush=True)

        final = None
        all_attempts = []

        for variant_name, agent_cfg in variants:
            for num_envs in ladder:
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

                attempt = {
                    'case': case_name,
                    'method': case['method'],
                    'case_type': case_type,
                    'variant': variant_name,
                    'agent_config': agent_cfg,
                    'num_envs': num_envs,
                    'probe_rc': probe_rc,
                    'probe_ok': int(probe_ok),
                    'probe_timeout': int(probe_to),
                    'long_rc': -1,
                    'long_ok': 0,
                    'long_timeout': 0,
                    'long_max_samples': int(args.long_max_samples if is_trainable else 0),
                    'test_rc': -1,
                    'test_ok': 0,
                    'viz_rc': -1,
                    'viz_ok': 0,
                    'probe_elapsed_sec': round(probe_elapsed, 2),
                    'long_elapsed_sec': 0.0,
                    'test_elapsed_sec': 0.0,
                    'viz_elapsed_sec': 0.0,
                    'stage_elapsed_sec': 0.0,
                    'final_ok': 0,
                    'note': '',
                    'out_dir': str(out_dir),
                    'probe_out_dir': str(probe_out),
                    'long_out_dir': str(long_out if is_trainable else ''),
                    'probe_train_log': str(probe_log),
                    'long_train_log': str(long_log),
                    'test_log': str(test_log),
                    'viz_log': str(viz_log),
                }

                if not probe_ok:
                    note = classify_train_error(probe_text, 'probe')
                    if probe_rc == 124 or probe_to:
                        note = 'probe_timeout'
                    attempt['note'] = note
                    attempt['stage_elapsed_sec'] = round(time.time() - attempt_start, 2)
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] {note}", flush=True)
                    continue

                if is_trainable:
                    long_cmd = [
                        str(TRAIN_PY), str(RUN_PY),
                        '--arg_file', arg_rel,
                        '--engine_config', args.engine_config,
                        '--mode', 'train',
                        '--visualize', 'false',
                        '--devices', *case_devices,
                        '--num_envs', str(num_envs),
                        '--max_samples', str(args.long_max_samples),
                        '--out_dir', str(long_out),
                        '--model_file', str(probe_out / 'model.pt'),
                        '--master_port', str(random.randint(20000, 45000)),
                    ]
                    if agent_cfg:
                        long_cmd.extend(['--agent_config', agent_cfg])

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
                    attempt['long_rc'] = long_rc
                    attempt['long_ok'] = int(long_ok)
                    attempt['long_timeout'] = int(long_to)
                    attempt['long_elapsed_sec'] = round(long_elapsed, 2)

                    if not long_ok:
                        note = classify_train_error(long_text, 'long')
                        if long_rc == 124 or long_to:
                            note = 'long_timeout'
                        attempt['note'] = note
                        attempt['stage_elapsed_sec'] = round(time.time() - attempt_start, 2)
                        all_attempts.append(attempt)
                        print(f"  [TRY {variant_name} e{num_envs}] probe=ok {note}", flush=True)
                        continue

                    eval_model = long_out / 'model.pt'
                else:
                    write_skip_log(long_log, 'SKIPPED: nontrainable case has no long train stage.')
                    attempt['long_rc'] = 0
                    attempt['long_ok'] = 1
                    attempt['long_timeout'] = 0
                    attempt['long_elapsed_sec'] = 0.0
                    eval_model = probe_out / 'model.pt'

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
                attempt['test_elapsed_sec'] = round(test_elapsed, 2)

                if not test_ok:
                    attempt['note'] = 'test_timeout' if (test_rc == 124 or test_to) else 'test_failed'
                    attempt['stage_elapsed_sec'] = round(time.time() - attempt_start, 2)
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] probe=ok long=ok test=fail", flush=True)
                    continue

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
                attempt['viz_elapsed_sec'] = round(viz_elapsed, 2)

                if not viz_ok:
                    note = 'viz_timeout' if (viz_rc == 124 or viz_to) else 'viz_failed'
                    if 'GLSL 1.50 is not supported' in viz_text:
                        note = 'viz_glsl_version'
                    attempt['note'] = note
                    attempt['stage_elapsed_sec'] = round(time.time() - attempt_start, 2)
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] probe=ok long=ok test=ok viz=fail({note})", flush=True)
                    continue

                attempt['final_ok'] = 1
                attempt['note'] = 'ok'
                attempt['stage_elapsed_sec'] = round(time.time() - attempt_start, 2)
                all_attempts.append(attempt)
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
                    'long_max_samples': int(args.long_max_samples if is_trainable else 0),
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

        case_run_dir = root_out / 'runs' / case_name.replace('.txt', '')
        case_run_dir.mkdir(parents=True, exist_ok=True)
        with (case_run_dir / 'attempts.json').open('w') as f:
            json.dump(all_attempts, f, indent=2)

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
        'long_rc', 'long_ok', 'long_timeout', 'long_max_samples',
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
