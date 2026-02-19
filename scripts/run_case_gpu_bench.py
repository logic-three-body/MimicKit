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

try:
    import yaml
except Exception:
    print("PyYAML is required", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / 'mimickit' / 'run.py'
TRAIN_PY = Path(os.environ.get('MIMICKIT_TRAIN_PY', sys.executable))
ENGINE_NEWTON = os.environ.get('MIMICKIT_ENGINE_CONFIG', 'data/engines/newton_engine.yaml')
DEVICES = 'cuda:0 cuda:1'
ENV_LADDER = [1024, 512, 256]
PI_PLUS_ENV_LADDER = [1024, 768, 512, 384, 256, 192, 128, 96, 64, 48, 44, 40, 39, 38, 36, 32]
AMP_PI_PLUS_ENV_LADDER = [40, 39, 38, 36, 32, 24, 16, 8, 4, 2, 1]
AMP_PI_PLUS_CASE = 'amp_pi_plus_args.txt'
MAX_SECONDS = 420
ITER_TARGET = 8

NCCL_ENV = {
    'NCCL_P2P_DISABLE': '1',
    'NCCL_IB_DISABLE': '1',
    'NCCL_CUMEM_ENABLE': '0',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
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


def parse_int_ladder(text: str):
    vals = []
    for token in text.split(','):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    out = []
    seen = set()
    for v in vals:
        if v <= 0:
            continue
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def parse_csv_names(text: str):
    vals = []
    for token in text.split(','):
        t = token.strip()
        if not t:
            continue
        vals.append(t)
    return vals


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


def build_hiutil_agent(src_rel: str, dst: Path):
    src = ROOT / src_rel
    cfg = yaml.safe_load(src.read_text())

    model = cfg.get('model', {})
    if model.get('actor_net') == 'fc_2layers_1024units':
        model['actor_net'] = 'fc_3layers_1024units'
    if model.get('critic_net') == 'fc_2layers_1024units':
        model['critic_net'] = 'fc_3layers_1024units'

    if 'update_epochs' in cfg:
        cfg['update_epochs'] = max(int(cfg['update_epochs']), 20)
    if 'batch_size' in cfg:
        cfg['batch_size'] = min(int(cfg['batch_size']), 2)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))


def read_gpu_once():
    cmd = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits',
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return None

    rows = [x.strip() for x in out.strip().splitlines() if x.strip()]
    if len(rows) < 2:
        return None

    def parse_line(line):
        parts = [p.strip() for p in line.split(',')]
        return {
            'util': float(parts[0]),
            'mem': float(parts[1]),
            'mem_total': float(parts[2]),
            'power': float(parts[3]),
        }

    return parse_line(rows[0]), parse_line(rows[1])


def summarize_gpu(samples, skip=20):
    if not samples:
        return {}
    used = samples[skip:] if len(samples) > skip else samples
    if not used:
        used = samples

    def avg(xs):
        return sum(xs) / len(xs)

    u0 = [s[0]['util'] for s in used]
    u1 = [s[1]['util'] for s in used]
    m0 = [s[0]['mem'] for s in used]
    m1 = [s[1]['mem'] for s in used]
    p0 = [s[0]['power'] for s in used]
    p1 = [s[1]['power'] for s in used]

    return {
        'n_samples': len(used),
        'avg_util0': avg(u0),
        'avg_util1': avg(u1),
        'min_avg_util': min(avg(u0), avg(u1)),
        'max_util0': max(u0),
        'max_util1': max(u1),
        'max_mem0': max(m0),
        'max_mem1': max(m1),
        'max_power0': max(p0),
        'max_power1': max(p1),
    }


def kill_proc_tree(proc):
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


def cleanup_outdir_processes(out_dir: Path):
    marker = str(out_dir)
    try:
        out = subprocess.check_output(['ps', '-eo', 'pid,cmd'], text=True)
    except Exception:
        return
    for line in out.splitlines():
        if marker not in line:
            continue
        if 'mimickit/run.py' not in line and 'multiprocessing.spawn' not in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def run_one(case_name, arg_rel, agent_rel, variant, num_envs, out_dir):
    max_samples = num_envs * 2 * 32 * ITER_TARGET
    master_port = random.randint(20000, 45000)
    console = out_dir / 'console.log'
    gpu_csv = out_dir / 'gpu.csv'

    cmd = [
        str(TRAIN_PY), str(RUN_PY),
        '--arg_file', arg_rel,
        '--engine_config', ENGINE_NEWTON,
        '--agent_config', agent_rel,
        '--mode', 'train',
        '--visualize', 'false',
        '--devices', 'cuda:0', 'cuda:1',
        '--master_port', str(master_port),
        '--num_envs', str(num_envs),
        '--max_samples', str(max_samples),
        '--out_dir', str(out_dir),
    ]

    env = os.environ.copy()
    env.update(NCCL_ENV)
    env['PYTHONUNBUFFERED'] = '1'

    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with console.open('w') as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    gpu_samples = []
    timed_out = False
    while True:
        if proc.poll() is not None:
            break
        now = time.time()
        if now - start > MAX_SECONDS:
            timed_out = True
            kill_proc_tree(proc)
            break
        g = read_gpu_once()
        if g is not None:
            gpu_samples.append(g)
            with gpu_csv.open('a') as gf:
                gf.write(f"{g[0]['util']},{g[0]['mem']},{g[0]['mem_total']},{g[0]['power']},{g[1]['util']},{g[1]['mem']},{g[1]['mem_total']},{g[1]['power']}\\n")
        time.sleep(1)

    rc = proc.returncode if proc.returncode is not None else 124
    elapsed = int(time.time() - start)
    cleanup_outdir_processes(out_dir)

    console_text = console.read_text(errors='ignore') if console.exists() else ''
    has_model = (out_dir / 'model.pt').exists()
    has_log = (out_dir / 'log.txt').exists()

    samples_last = 0
    iter_last = -1
    for line in console_text.splitlines():
        if '|              Samples |' in line:
            try:
                samples_last = int(line.split('|')[2].strip())
            except Exception:
                pass
        if '|            Iteration |' in line:
            try:
                iter_last = int(line.split('|')[2].strip())
            except Exception:
                pass

    gpu_stat = summarize_gpu(gpu_samples, skip=20)

    if rc == 0 and has_model and has_log:
        status = 'ok'
    elif timed_out:
        status = 'timeout'
    elif 'out of memory' in console_text.lower() or 'distbackenderror' in console_text.lower() or 'nccl' in console_text.lower():
        status = 'oom_or_nccl'
    else:
        status = 'fail'

    sps = samples_last / elapsed if elapsed > 0 else 0

    return {
        'case': case_name,
        'arg_file': arg_rel,
        'variant': variant,
        'num_envs': num_envs,
        'status': status,
        'rc': rc,
        'elapsed_s': elapsed,
        'iteration': iter_last,
        'samples': samples_last,
        'samples_per_s': round(sps, 2),
        **{k: round(v, 2) if isinstance(v, float) else v for k, v in gpu_stat.items()},
        'out_dir': str(out_dir),
        'agent_config': agent_rel,
        'master_port': master_port,
    }


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def choose_case_ladder(case_name: str, default_ladder, pi_plus_ladder, amp_pi_plus_ladder):
    if case_name == AMP_PI_PLUS_CASE:
        return list(amp_pi_plus_ladder)
    if 'pi_plus' in case_name:
        return list(pi_plus_ladder)
    return list(default_ladder)


def select_case_result(case_runs, objective: str, util_floor: float):
    if not case_runs:
        return {'case': '', 'status': 'unrun'}, 'no_runs', 0.0

    ok_runs = [x for x in case_runs if x.get('status') == 'ok']
    if not ok_runs:
        selected = max(
            case_runs,
            key=lambda x: (to_float(x.get('min_avg_util', -1.0), -1.0), to_float(x.get('samples_per_s', -1.0), -1.0)),
        )
        return selected, 'no_ok_pick_highest_util_from_all', to_float(selected.get('min_avg_util', 0.0), 0.0)

    if objective == 'util':
        selected = max(
            ok_runs,
            key=lambda x: (to_float(x.get('min_avg_util', 0.0), 0.0), to_float(x.get('samples_per_s', 0.0), 0.0)),
        )
        return selected, 'util_max_min_avg_util', to_float(selected.get('min_avg_util', 0.0), 0.0)

    if objective == 'sps':
        selected = max(
            ok_runs,
            key=lambda x: (to_float(x.get('samples_per_s', 0.0), 0.0), to_float(x.get('min_avg_util', 0.0), 0.0)),
        )
        return selected, 'sps_max_samples_per_s', to_float(selected.get('samples_per_s', 0.0), 0.0)

    floor_ok = [x for x in ok_runs if to_float(x.get('min_avg_util', 0.0), 0.0) >= util_floor]
    if floor_ok:
        selected = max(
            floor_ok,
            key=lambda x: (to_float(x.get('samples_per_s', 0.0), 0.0), to_float(x.get('min_avg_util', 0.0), 0.0)),
        )
        return selected, f'balanced_util_floor_{util_floor:g}_then_max_sps', to_float(selected.get('samples_per_s', 0.0), 0.0)

    selected = max(
        ok_runs,
        key=lambda x: (to_float(x.get('min_avg_util', 0.0), 0.0), to_float(x.get('samples_per_s', 0.0), 0.0)),
    )
    return selected, f'balanced_no_floor_hit_{util_floor:g}_fallback_max_util', to_float(selected.get('min_avg_util', 0.0), 0.0)


def main():
    global MAX_SECONDS, ITER_TARGET, ENGINE_NEWTON

    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='', help='comma-separated case names (e.g. amp_humanoid_args.txt,deepmimic_pi_plus_ppo_args.txt)')
    ap.add_argument('--env-ladder', default=','.join(str(x) for x in ENV_LADDER), help='default per-GPU env ladder')
    ap.add_argument('--pi-plus-ladder', default=','.join(str(x) for x in PI_PLUS_ENV_LADDER), help='per-GPU env ladder for pi_plus cases')
    ap.add_argument('--amp-pi-plus-ladder', default=','.join(str(x) for x in AMP_PI_PLUS_ENV_LADDER), help='per-GPU env ladder for amp_pi_plus case')
    ap.add_argument('--objective', choices=['balanced', 'util', 'sps'], default='balanced', help='selection objective for final case allocation')
    ap.add_argument('--util-floor', type=float, default=50.0, help='minimum min_avg_util for balanced objective')
    ap.add_argument('--scan-policy', choices=['first_ok', 'full_scan'], default='full_scan', help='probe stop policy per case')
    ap.add_argument('--max-seconds', type=int, default=MAX_SECONDS, help='max runtime per probe')
    ap.add_argument('--iter-target', type=int, default=ITER_TARGET, help='target iterations for bounded probe')
    ap.add_argument('--engine-config', default=ENGINE_NEWTON, help='engine config path')
    ap.add_argument('--root-out', default='', help='custom output root under output/train')
    ap.add_argument('--require-deps', default='trimesh,scipy', help='comma-separated Python modules required before run; empty disables check')
    args = ap.parse_args()

    MAX_SECONDS = int(args.max_seconds)
    ITER_TARGET = int(args.iter_target)
    ENGINE_NEWTON = args.engine_config

    case_filter = {normalize_case_name(x) for x in parse_csv_names(args.cases)} if args.cases.strip() else set()
    default_ladder = parse_int_ladder(args.env_ladder)
    pi_plus_ladder = parse_int_ladder(args.pi_plus_ladder)
    amp_pi_plus_ladder = parse_int_ladder(args.amp_pi_plus_ladder)
    if not default_ladder:
        default_ladder = list(ENV_LADDER)
    if not pi_plus_ladder:
        pi_plus_ladder = list(PI_PLUS_ENV_LADDER)
    if not amp_pi_plus_ladder:
        amp_pi_plus_ladder = list(AMP_PI_PLUS_ENV_LADDER)

    required_deps = parse_csv_names(args.require_deps)
    if required_deps:
        missing = check_python_deps(required_deps)
        if missing:
            print('[ERROR] Missing Python dependencies required for this benchmark:', file=sys.stderr)
            for mod, err in missing:
                print(f'  - {mod}: {err}', file=sys.stderr)
            sys.exit(2)

    ts = time.strftime('%Y%m%d_%H%M%S')
    if args.root_out:
        root_out = ROOT / 'output' / 'train' / args.root_out
    else:
        root_out = ROOT / 'output' / 'train' / f'case_gpu_bench_{ts}'
    root_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    train_cases = []

    for p in sorted(glob.glob(str(ROOT / 'args' / '*.txt'))):
        rel = os.path.relpath(p, ROOT)
        name = os.path.basename(p)
        case_cfg = parse_arg_file(Path(p))
        mode = case_cfg.get('mode', 'train')
        agent = case_cfg.get('agent_config', '')
        env = case_cfg.get('env_config', '')
        method = name.split('_')[0]
        row = {
            'case': name,
            'method': method,
            'mode': mode,
            'agent_config': agent,
            'env_config': env,
            'engine_config': case_cfg.get('engine_config', ''),
        }
        manifest_rows.append(row)
        if mode == 'train' and agent:
            if case_filter and name not in case_filter:
                continue
            train_cases.append(row)

    manifest_tsv = root_out / 'case_manifest.tsv'
    with manifest_tsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'method', 'mode', 'agent_config', 'env_config', 'engine_config'], delimiter='\t')
        w.writeheader()
        w.writerows(manifest_rows)

    results = []
    print(f"[INFO] total_cases={len(manifest_rows)} trainable_cases={len(train_cases)}", flush=True)
    print(f"[INFO] root_out={root_out}", flush=True)
    print(
        f"[INFO] default_env_ladder={default_ladder} "
        f"pi_plus_env_ladder={pi_plus_ladder} "
        f"amp_pi_plus_env_ladder={amp_pi_plus_ladder}",
        flush=True,
    )
    print(
        f"[INFO] objective={args.objective} util_floor={args.util_floor:g} scan_policy={args.scan_policy}",
        flush=True,
    )
    print(f"[INFO] max_seconds={MAX_SECONDS} iter_target={ITER_TARGET} engine_config={ENGINE_NEWTON}", flush=True)
    if case_filter:
        print(f"[INFO] case_filter={sorted(case_filter)}", flush=True)

    for idx, case in enumerate(train_cases, 1):
        case_name = case['case']
        arg_rel = f"args/{case_name}"
        base_agent = case['agent_config']
        case_ladder = choose_case_ladder(case_name, default_ladder, pi_plus_ladder, amp_pi_plus_ladder)

        print(f"\\n[CASE {idx}/{len(train_cases)}] {case_name}", flush=True)
        print(f"  [LADDER] {case_ladder}", flush=True)

        hi_agent_path = root_out / 'agent_variants' / case_name.replace('.txt', '_hiutil.yaml')
        try:
            build_hiutil_agent(base_agent, hi_agent_path)
            hi_rel = os.path.relpath(hi_agent_path, ROOT)
        except Exception as e:
            print(f"  [WARN] build hiutil agent failed: {e}", flush=True)
            hi_rel = base_agent

        case_runs = []
        run_plan = []
        for num_envs in case_ladder:
            run_plan.append(('hiutil', hi_rel, num_envs))
        for num_envs in case_ladder:
            run_plan.append(('default', base_agent, num_envs))

        for variant, agent_cfg, num_envs in run_plan:
            out_dir = root_out / 'runs' / case_name.replace('.txt', '') / f'{variant}_e{num_envs}'
            res = run_one(case_name, arg_rel, agent_cfg, variant, num_envs, out_dir)
            case_runs.append(res)
            print(
                f"  [TRY {variant} e{num_envs}] status={res['status']} "
                f"min_avg={res.get('min_avg_util')} sps={res['samples_per_s']}",
                flush=True,
            )
            if args.scan_policy == 'first_ok' and res['status'] == 'ok':
                break

        selected, selection_reason, score = select_case_result(case_runs, args.objective, float(args.util_floor))

        selected = dict(selected)
        selected['method'] = case['method']
        selected['base_agent'] = base_agent
        selected['objective'] = args.objective
        selected['selection_reason'] = selection_reason
        selected['score'] = round(float(score), 4)
        results.append(selected)

        with (root_out / 'progress.json').open('w') as f:
            json.dump({'done': idx, 'total': len(train_cases), 'last_case': case_name}, f, indent=2)

    results_tsv = root_out / 'best_by_case.tsv'
    fields = [
        'case', 'method', 'status', 'variant', 'num_envs', 'elapsed_s', 'iteration', 'samples', 'samples_per_s',
        'avg_util0', 'avg_util1', 'min_avg_util', 'max_util0', 'max_util1',
        'max_mem0', 'max_mem1', 'max_power0', 'max_power1',
        'agent_config', 'base_agent', 'objective', 'selection_reason', 'score', 'out_dir'
    ]
    with results_tsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)

    alloc_tsv = root_out / 'allocation_profile.tsv'
    alloc_fields = [
        'case', 'method', 'status', 'variant', 'num_envs',
        'agent_config', 'base_agent', 'min_avg_util', 'samples_per_s',
        'objective', 'selection_reason', 'score', 'out_dir',
    ]
    with alloc_tsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=alloc_fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\\n[DONE] results={results_tsv}", flush=True)
    print(f"[DONE] allocation_profile={alloc_tsv}", flush=True)


if __name__ == '__main__':
    main()
