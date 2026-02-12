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
    # dedupe while preserving order
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


def run_cmd(cmd, env, cwd: Path, log_path: Path, timeout_s: int):
    ensure_parent(log_path)
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
        proc.wait(timeout=timeout_s)
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
    text = log_path.read_text(errors='ignore') if log_path.exists() else ''
    return rc, timed_out, text


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
    # dedupe by agent path
    dedup = []
    seen = set()
    for vname, agent in out:
        if agent in seen:
            continue
        dedup.append((vname, agent))
        seen.add(agent)
    return dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-config', default='data/engines/newton_engine.yaml')
    ap.add_argument('--cases', default='')
    ap.add_argument('--devices-train', default='cuda:0')
    ap.add_argument('--devices-nontrainable', default='cuda:0')
    ap.add_argument('--default-ladder', default='512,256,128,64,32')
    ap.add_argument('--pi-plus-ladder', default='40,39,38,36,32,24,16,8,4,2,1')
    ap.add_argument('--train-iters', type=int, default=4)
    ap.add_argument('--include-nontrainable', action='store_true')
    ap.add_argument('--nontrainable-num-envs', type=int, default=1)
    ap.add_argument('--nontrainable-max-samples', type=int, default=128)
    ap.add_argument('--train-timeout', type=int, default=360)
    ap.add_argument('--test-timeout', type=int, default=240)
    ap.add_argument('--viz-timeout', type=int, default=240)
    ap.add_argument('--test-episodes', type=int, default=10)
    ap.add_argument('--viz-episodes', type=int, default=1)
    ap.add_argument('--root-out', default='')
    ap.add_argument('--require-deps', default='torch,newton,warp,scipy,trimesh,pyglet')
    args = ap.parse_args()

    case_filter = {normalize_case_name(x) for x in parse_csv(args.cases)} if args.cases.strip() else set()
    default_ladder = parse_int_csv(args.default_ladder)
    pi_plus_ladder = parse_int_csv(args.pi_plus_ladder)
    if not default_ladder:
        default_ladder = [512, 256, 128, 64, 32]
    if not pi_plus_ladder:
        pi_plus_ladder = [40, 39, 38, 36, 32, 24, 16, 8, 4, 2, 1]

    req_mods = parse_csv(args.require_deps)
    missing = check_python_deps(req_mods)
    if missing:
        print('[ERROR] missing deps:')
        for m, e in missing:
            print(f'  - {m}: {e}')
        return 2

    ts = time.strftime('%Y%m%d_%H%M%S')
    root_out = ROOT / 'output' / 'train' / (args.root_out if args.root_out else f'case_e2e_{ts}')
    root_out.mkdir(parents=True, exist_ok=True)

    # discover cases
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
        train_devices = ['cuda:0']

    nontrain_devices = parse_csv(args.devices_nontrainable)
    if not nontrain_devices:
        nontrain_devices = ['cuda:0']

    print(f'[INFO] root_out={root_out}', flush=True)
    print(
        f'[INFO] manifest trainable={trainable_count} nontrainable={nontrainable_count} '
        f'selected={len(selected)} include_nontrainable={int(args.include_nontrainable)}',
        flush=True,
    )
    print(f'[INFO] default_ladder={default_ladder} pi_plus_ladder={pi_plus_ladder}', flush=True)
    print(f'[INFO] train_devices={train_devices}', flush=True)
    print(f'[INFO] nontrainable_devices={nontrain_devices}', flush=True)

    results = []
    for idx, case in enumerate(selected, 1):
        case_name = case['case']
        case_type = case.get('case_type', 'trainable')
        is_trainable = (case_type == 'trainable')
        arg_rel = f'args/{case_name}'
        base_agent = case['agent_config']

        if is_trainable:
            ladder = pi_plus_ladder if 'pi_plus' in case_name else default_ladder
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
                out_dir = root_out / 'runs' / case_name.replace('.txt', '') / f'{variant_name}_e{num_envs}'
                out_dir.mkdir(parents=True, exist_ok=True)

                # stage 1: train
                train_out = out_dir / 'train'
                if is_trainable:
                    max_samples = num_envs * len(case_devices) * 32 * args.train_iters
                else:
                    max_samples = max(1, int(args.nontrainable_max_samples))
                master_port = random.randint(20000, 45000)
                train_log = out_dir / 'train.log'

                train_cmd = [
                    str(TRAIN_PY), str(RUN_PY),
                    '--arg_file', arg_rel,
                    '--engine_config', args.engine_config,
                    '--mode', 'train',
                    '--visualize', 'false',
                    '--devices', *case_devices,
                    '--num_envs', str(num_envs),
                    '--max_samples', str(max_samples),
                    '--out_dir', str(train_out),
                    '--master_port', str(master_port),
                ]
                if agent_cfg:
                    train_cmd.extend(['--agent_config', agent_cfg])

                env = os.environ.copy()
                env.update(NCCL_ENV)
                env['PYTHONUNBUFFERED'] = '1'

                rc_train, to_train, text_train = run_cmd(train_cmd, env, ROOT, train_log, args.train_timeout)
                has_model = (train_out / 'model.pt').exists()
                has_log = (train_out / 'log.txt').exists()
                train_ok = (rc_train == 0 and has_model and has_log)

                attempt = {
                    'case': case_name,
                    'method': case['method'],
                    'case_type': case_type,
                    'variant': variant_name,
                    'agent_config': agent_cfg,
                    'num_envs': num_envs,
                    'train_rc': rc_train,
                    'train_timeout': int(to_train),
                    'train_ok': int(train_ok),
                    'test_rc': -1,
                    'test_ok': 0,
                    'viz_rc': -1,
                    'viz_ok': 0,
                    'final_ok': 0,
                    'out_dir': str(out_dir),
                    'train_out_dir': str(train_out),
                    'train_log': str(train_log),
                    'test_log': '',
                    'viz_log': '',
                    'note': '',
                }

                if not train_ok:
                    note = 'train_failed'
                    if 'Array shapes must be non-negative' in text_train:
                        note = 'train_negative_shape'
                    elif 'out of memory' in text_train.lower() or 'failed to allocate' in text_train.lower():
                        note = 'train_oom'
                    elif 'nccl' in text_train.lower() or 'distbackenderror' in text_train.lower():
                        note = 'train_nccl'
                    attempt['note'] = note
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] train={note}", flush=True)
                    continue

                # stage 2: inference
                test_log = out_dir / 'test.log'
                test_cmd = [
                    str(TRAIN_PY), str(RUN_PY),
                    '--arg_file', arg_rel,
                    '--engine_config', args.engine_config,
                    '--mode', 'test',
                    '--visualize', 'false',
                    '--devices', 'cuda:0',
                    '--num_envs', '1',
                    '--test_episodes', str(args.test_episodes),
                    '--model_file', str(train_out / 'model.pt'),
                ]
                if agent_cfg:
                    test_cmd.extend(['--agent_config', agent_cfg])
                rc_test, to_test, text_test = run_cmd(test_cmd, env, ROOT, test_log, args.test_timeout)
                test_ok = (rc_test == 0 and has_metric(text_test))
                attempt['test_rc'] = rc_test
                attempt['test_ok'] = int(test_ok)
                attempt['test_log'] = str(test_log)
                if not test_ok:
                    attempt['note'] = 'test_failed'
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] train=ok test=fail", flush=True)
                    continue

                # stage 3: visualization
                viz_log = out_dir / 'viz.log'
                viz_cmd = [
                    str(TRAIN_PY), str(RUN_PY),
                    '--arg_file', arg_rel,
                    '--engine_config', args.engine_config,
                    '--mode', 'test',
                    '--visualize', 'true',
                    '--devices', 'cuda:0',
                    '--num_envs', '1',
                    '--test_episodes', str(args.viz_episodes),
                    '--model_file', str(train_out / 'model.pt'),
                ]
                if agent_cfg:
                    viz_cmd.extend(['--agent_config', agent_cfg])
                viz_env = dict(env)
                viz_env.update(MESA_ENV)
                rc_viz, to_viz, text_viz = run_cmd(viz_cmd, viz_env, ROOT, viz_log, args.viz_timeout)
                viz_ok = (rc_viz == 0 and has_metric(text_viz))
                attempt['viz_rc'] = rc_viz
                attempt['viz_ok'] = int(viz_ok)
                attempt['viz_log'] = str(viz_log)

                if not viz_ok:
                    note = 'viz_failed'
                    if 'GLSL 1.50 is not supported' in text_viz:
                        note = 'viz_glsl_version'
                    attempt['note'] = note
                    all_attempts.append(attempt)
                    print(f"  [TRY {variant_name} e{num_envs}] train=ok test=ok viz=fail({note})", flush=True)
                    continue

                attempt['final_ok'] = 1
                attempt['note'] = 'ok'
                all_attempts.append(attempt)
                final = attempt
                print(f"  [TRY {variant_name} e{num_envs}] final=ok", flush=True)
                break

            if final is not None:
                break

        if final is None:
            # pick best attempt for reporting
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
                    'train_rc': -1,
                    'train_timeout': 0,
                    'train_ok': 0,
                    'test_rc': -1,
                    'test_ok': 0,
                    'viz_rc': -1,
                    'viz_ok': 0,
                    'final_ok': 0,
                    'out_dir': '',
                    'train_out_dir': '',
                    'train_log': '',
                    'test_log': '',
                    'viz_log': '',
                    'note': 'unrun',
                }

        results.append(final)

        # dump attempts for this case
        with (root_out / 'runs' / case_name.replace('.txt', '') / 'attempts.json').open('w') as f:
            json.dump(all_attempts, f, indent=2)

        with (root_out / 'progress.json').open('w') as f:
            json.dump({'done': idx, 'total': len(selected), 'last_case': case_name}, f, indent=2)

    fields = [
        'case', 'method', 'case_type', 'variant', 'agent_config', 'num_envs',
        'train_rc', 'train_timeout', 'train_ok',
        'test_rc', 'test_ok',
        'viz_rc', 'viz_ok',
        'final_ok', 'note',
        'out_dir', 'train_out_dir', 'train_log', 'test_log', 'viz_log',
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
