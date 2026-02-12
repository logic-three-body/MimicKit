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
- Case catalog doc: `docs/README_MimicKit_CaseCatalog.md`

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
- tries env ladder: `1024 -> 512 -> 256` (per GPU)
- falls back to default agent ladder if hiutil fails
- writes `best_by_case.tsv`

3. Output location:
- `output/train/case_gpu_bench_<timestamp>/case_manifest.tsv`
- `output/train/case_gpu_bench_<timestamp>/best_by_case.tsv`
- per-run logs under `output/train/case_gpu_bench_<timestamp>/runs/...`

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

## Current Observations (2026-02-12, in-progress)

- `add` family on Newton dual:
  - `add_g1`, `add_go2`, `add_humanoid`, `add_smpl`: hiutil `e1024` runs succeeded
  - `add_pi_plus`: hiutil and default ladders (`1024/512/256`) all ended with `oom_or_nccl`
- Keep this section updated from `docs/README_MimicKit_GPUCaseBenchmark.md`.
