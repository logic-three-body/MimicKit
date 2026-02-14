---
name: mimickit-allcase-e2e
description: Run all MimicKit args cases through a reproducible end-to-end flow (train -> inference -> visualization), auto-handle trainable and nontrainable cases, and report per-case pass/fail with logs and best config outputs.
---

# MimicKit All-Case E2E (Train -> Inference -> Visualization)

## Goal

Run all `args/*.txt` cases with one workflow and produce:
- per-case E2E status (`train`, `test`, `viz`)
- per-case logs and artifacts
- a single machine-readable summary (`best_by_case.tsv`)

This skill is for full-regression validation, not for longrun training.

For long-cycle training across all cases, use:
- `docs/skill/mimickit-allcase-longcycle-skill/SKILL.md`
- script: `scripts/run_case_longcycle.py`

## Scope

Current repository split:
- trainable cases: 25
- nontrainable cases: 7 (`dof_test_*`, `view_motion_*`)
- total: 32

`scripts/run_case_e2e.py` supports both groups.

## Prerequisites

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit
```

Check runtime deps:
```bash
python -c "import torch,newton,warp,trimesh,scipy,pyglet; print('deps_ok')"
```

## One-Command Full Run (32 cases)

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -u scripts/run_case_e2e.py \
  --engine-config data/engines/newton_engine.yaml \
  --devices-train cuda:0,cuda:1 \
  --include-nontrainable \
  --root-out case_e2e_all_$(date +%Y%m%d_%H%M%S)
```

Notes:
- script applies NCCL env profile internally for train stage:
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_IB_DISABLE=1`
  - `NCCL_CUMEM_ENABLE=0`
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- script applies Mesa overrides internally for visualization stage:
  - `MESA_GL_VERSION_OVERRIDE=3.3`
  - `MESA_GLSL_VERSION_OVERRIDE=330`
- for `pi_plus` cases, script uses dedicated env ladder and hiutil agent variant when available.
- script uses case-specific default for `amp_pi_plus`:
  - `--amp-pi-plus-ladder 38,36,32,24,16,8,4,2,1`
  - avoids known `hiutil e40/e39` OOM on this host

## What the Script Does

Per case:
1. Stage-1 train:
   - trainable cases: env-ladder probe + bounded `max_samples`
   - nontrainable cases: forced short train bootstrap (`mode=train`) to create model artifact
2. Stage-2 inference:
   - `mode=test`, `visualize=false`, verify metrics in log
3. Stage-3 visualization:
   - `mode=test`, `visualize=true`, verify metrics in log

Pass condition per case:
- `final_ok = 1` in summary TSV.

## Outputs

Given root `output/train/<root_out>/`:
- `case_manifest.tsv`: discovered case inventory
- `best_by_case.tsv`: final result per case
- `progress.json`: execution progress
- `runs/<case>/...`: per-attempt logs
  - `train.log`
  - `test.log`
  - `viz.log`
  - `attempts.json`

## Acceptance Check

Use this check after run:

```bash
python - <<'PY'
import csv,glob,os
root=sorted(glob.glob('output/train/case_e2e_all_*'))[-1]
path=os.path.join(root,'best_by_case.tsv')
rows=list(csv.DictReader(open(path),delimiter='\t'))
ok=sum(1 for r in rows if str(r.get('final_ok')).strip()=='1')
print('root',root)
print('ok',ok,'total',len(rows))
print('all_pass',ok==len(rows))
PY
```

Target: `all_pass True`.

## Failure Handling

If some cases fail:
1. Re-run only failed subset:
```bash
python -u scripts/run_case_e2e.py \
  --cases case_a.txt,case_b.txt \
  --include-nontrainable \
  --root-out case_e2e_retry_$(date +%Y%m%d_%H%M%S)
```
2. Read `runs/<case>/attempts.json` and corresponding `*.log`.
3. For OOM/NCCL failures:
   - lower env ladder (`--default-ladder` / `--pi-plus-ladder`)
4. For visualization GLSL issues:
   - confirm Mesa overrides are present in viz command logs.

## Latest Verified Snapshot (2026-02-12)

Latest rerun (single-root 32-case):
- rerun root: `output/train/case_e2e_all_rerun_20260212_164601`
- result: `32/32` cases `final_ok=1`

Earlier split-run snapshot (also fully passed):
- trainable batch root: `output/train/case_e2e_all_20260212_1325`
- nontrainable batch root: `output/train/case_e2e_nontrainable_20260212_142059`
- combined result: `32/32` cases `final_ok=1`

Observed stable pi-plus behavior on this host:
- `add_pi_plus`: `hiutil e40` pass
- `deepmimic_pi_plus`: `hiutil e40` pass
- `amp_pi_plus`: `hiutil e40/e39` OOM, `hiutil e38` pass
- practical fallback default for `amp_pi_plus`: `--amp-pi-plus-ladder 38,36,32,24,...`

Important runtime fix included in repo:
- `mimickit/learning/base_agent.py`
  - `_sync_optimizer()` now skips optimizer sync when agent has no `_optimizer` (required for DummyAgent-based paths).
