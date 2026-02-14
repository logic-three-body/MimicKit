---
name: mimickit-multicase-gpu-util
description: Benchmark MimicKit trainable cases for dual-GPU utilization on Newton backend, pick best per-case config from env ladder and hiutil/default variants, and summarize by method class with reproducible TSV outputs.
---

# MimicKit Multi-Case GPU Utilization

## Goal

Run all trainable cases from `args/*.txt` with a consistent dual-GPU benchmark protocol and produce:
- per-case best config/result
- method-level classification summary
- reusable command pattern for replay

## Inputs

- Case list source: `args/*.txt`
- Benchmark script: `scripts/run_case_gpu_bench.py`
- Case catalog doc: `docs/benchmarks/README_MimicKit_CaseCatalog.md`

## Runtime Baseline (this host)

Always export before dual-device runs:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

Install dependency required by some Newton assets:

```bash
/root/miniconda3/envs/mimickit/bin/pip install trimesh
/root/miniconda3/envs/mimickit/bin/pip install scipy
```

## Benchmark Procedure

1. Run the script in `mimickit` env:

```bash
cd /root/Project/MimicKit
/root/miniconda3/envs/mimickit/bin/python -u scripts/run_case_gpu_bench.py
```

2. Script behavior:
- discovers trainable cases (`mode=train` and `agent_config` exists)
- creates per-case hiutil agent variant (`update_epochs>=20`, `batch_size<=2`, optional 3-layer net upgrade)
- tries env ladder:
  - default cases: `1024 -> 512 -> 256` (per GPU)
  - `pi_plus` cases: `1024 -> 768 -> 512 -> 384 -> 256 -> 192 -> 128 -> 96 -> 64 -> 48 -> 44 -> 40 -> 39 -> 38 -> 36 -> 32`
- falls back to default agent ladder if hiutil fails
- writes `best_by_case.tsv`

3. Output location:
- `output/train/case_gpu_bench_<timestamp>/case_manifest.tsv`
- `output/train/case_gpu_bench_<timestamp>/best_by_case.tsv`
- per-run logs under `output/train/case_gpu_bench_<timestamp>/runs/...`

4. Useful script options:

```bash
python scripts/run_case_gpu_bench.py --help
```

Key options:
- `--cases`: run only selected cases (comma-separated)
- `--env-ladder`: default case ladder
- `--pi-plus-ladder`: dedicated ladder for `pi_plus`
- `--max-seconds`: timeout per probe
- `--iter-target`: bounded probe iterations
- `--root-out`: deterministic output folder

## Classification Summary

After a run, aggregate by method:

```bash
python3 - <<'PY'
import csv, glob, os
from collections import defaultdict

root = sorted(glob.glob('output/train/case_gpu_bench_*'))[-1]
path = os.path.join(root, 'best_by_case.tsv')
agg = defaultdict(list)
with open(path) as f:
    r = csv.DictReader(f, delimiter='\\t')
    for row in r:
        agg[row['method']].append(row)

for m in sorted(agg):
    rows = agg[m]
    ok = [x for x in rows if x.get('status') == 'ok']
    if not ok:
        print(f'{m}: no success')
        continue
    avg = sum(float(x.get('min_avg_util', 0) or 0) for x in ok) / len(ok)
    print(f'{m}: ok={len(ok)}/{len(rows)} avg_min_util={avg:.2f}')
PY
```

## Reporting Checklist

When reporting results, include:
- benchmark root directory
- total trainable cases and success count
- top/bottom cases by `min_avg_util`
- per-method success and average utilization
- failure pattern categories (dependency, OOM, NCCL, timeout, asset mismatch)

## Pi-plus Recovery Workflow (validated)

When baseline benchmark marks `*_pi_plus*` as `oom_or_nccl`, run a focused subset:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -u scripts/run_case_gpu_bench.py \
  --cases add_pi_plus_args.txt,amp_pi_plus_args.txt,deepmimic_pi_plus_ppo_args.txt \
  --env-ladder 44,40,39,38,36,32 \
  --pi-plus-ladder 44,40,39,38,36,32 \
  --max-seconds 420 \
  --iter-target 8 \
  --root-out case_gpu_bench_piplus_final_<timestamp>
```

Validated result on this host (`2026-02-12`):
- `add_pi_plus_args.txt`: dual ok at `num_envs=40` (hiutil)
- `amp_pi_plus_args.txt`: dual ok at `num_envs=38` (hiutil)
- `deepmimic_pi_plus_ppo_args.txt`: dual ok at `num_envs=40` (hiutil)

Use `e38` as the universal safe fallback for all `pi_plus` methods when one common value is required.
