---
name: mimickit-allcase-longcycle
description: Run all MimicKit cases through a long-cycle full pipeline on Newton backend. For trainable cases, execute probe train -> long train -> inference -> visualization; for nontrainable cases, execute bootstrap train -> inference -> visualization. Generate reproducible per-case logs and summary TSV outputs.
---

# MimicKit All-Case Longcycle (Train -> Inference -> Visualization)

## Goal

Run `args/*.txt` with a long-cycle, full-flow policy:
- trainable cases: probe train -> long train -> inference -> visualization
- nontrainable cases: bootstrap train -> inference -> visualization

This skill is for long-cycle execution and reproducible result snapshots, not quick regression checks.

## Scope

Current repository case split:
- trainable: 25
- nontrainable: 7
- total: 32

## Entry Script

Use:
- `scripts/run_case_longcycle.py`

Core behavior:
- serial single-queue scheduling
- Newton engine default
- NCCL stable env profile for train stages
- Mesa GL/GLSL overrides for visualization stage
- per-case adaptive env ladders with `amp_pi_plus` special ladder default

## Longcycle One-Command Run

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -u scripts/run_case_longcycle.py \
  --engine-config data/engines/newton_engine.yaml \
  --devices-train cuda:0,cuda:1 \
  --include-nontrainable \
  --long-max-samples 30000000 \
  --root-out case_longcycle_$(date +%Y%m%d_%H%M%S)
```

## Parameter Defaults

- `--long-max-samples 30000000` (trainable only)
- `--probe-iters 8`
- `--probe-timeout 900`
- `--long-timeout 0` (no timeout)
- `--test-timeout 600`
- `--viz-timeout 600`
- `--default-ladder 512,256,128,64,32`
- `--pi-plus-ladder 40,39,38,36,32,24,16,8,4,2,1`
- `--amp-pi-plus-ladder 38,36,32,24,16,8,4,2,1`
- `--resume-skip-status ok` (reuse finished cases in existing `root_out`)

`--resume-skip-status` modes:
- `ok` (default): skip only cases whose previous `final_ok=1`
- `all`: skip any case with existing `attempts.json` (including failed)
- `none`: ignore existing attempts and rerun everything in selection

## Output Structure

For root `output/train/<root_out>/`:
- `case_manifest.tsv`
- `best_by_case.tsv`
- `progress.json`
- `runs/<case>/<variant>_e<num_envs>/`
  - `probe_train.log`
  - `long_train.log`
  - `test.log`
  - `viz.log`
  - `probe_train/` (stage output dir)
  - `long_train/` (stage output dir for trainable)
- `runs/<case>/attempts.json`
  - now written incrementally per stage (not only case end)
  - `note` transitions: `in_progress_probe` -> `in_progress_long` -> `in_progress_test` -> `in_progress_viz` -> `ok`

## Interruption Resume (Power Loss / Reboot / SSH Drop)

If execution is interrupted, resume with the same `root_out`.

Example:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -u scripts/run_case_longcycle.py \
  --engine-config data/engines/newton_engine.yaml \
  --devices-train cuda:0,cuda:1 \
  --include-nontrainable \
  --long-max-samples 30000000 \
  --root-out case_longcycle_full_20260212_211318 \
  --resume-skip-status ok
```

Behavior:
- script scans `output/train/<root_out>/runs/*/attempts.json`
- cases with `final_ok=1` are reused and skipped
- unfinished/failed cases continue from stage-level:
  - if `probe_ok=1`, resume starts from long stage
  - if `long_ok=1`, resume starts from test stage
  - if `test_ok=1`, resume starts from viz stage
  - if currently `in_progress_probe`, resume restarts probe stage

When to use other modes:
- force full rerun: `--resume-skip-status none`
- freeze current failed states and continue remaining: `--resume-skip-status all`

## Summary Fields (`best_by_case.tsv`)

Includes:
- `probe_ok`, `probe_rc`
- `long_ok`, `long_rc`
- `long_max_samples`
- `stage_elapsed_sec`
- `final_ok`, `note`

## 3-Min Monitor Template (GPU + progress + ETA hint)

```bash
TS=$(date +%Y%m%d_%H%M%S)
ROOT="output/train/case_longcycle_<your_ts>"
LOG="/tmp/mk_longcycle_report_${TS}.log"

while true; do
  {
    echo "=== $(date '+%F %T') ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    cat "${ROOT}/progress.json" 2>/dev/null || echo "progress not ready"
    tail -n 20 "${ROOT}/best_by_case.tsv" 2>/dev/null || true
    echo
  } | tee -a "${LOG}"
  sleep 180
done
```

## Acceptance Check

```bash
python - <<'PY'
import csv, glob, os
root = sorted(glob.glob('output/train/case_longcycle_*'))[-1]
path = os.path.join(root, 'best_by_case.tsv')
rows = list(csv.DictReader(open(path), delimiter='\t'))
ok = sum(1 for r in rows if str(r.get('final_ok')).strip() == '1')
print('root', root)
print('rows', len(rows), 'ok', ok, 'fail', len(rows) - ok)
print('trainable_ok', sum(1 for r in rows if r.get('case_type') == 'trainable' and str(r.get('final_ok')).strip() == '1'))
print('nontrainable_ok', sum(1 for r in rows if r.get('case_type') == 'nontrainable' and str(r.get('final_ok')).strip() == '1'))
PY
```

Target:
- `rows = 32`
- `ok = 32` (or explicit failed case diagnostics in `attempts.json` + `note`)

## Failure Handling

If a case fails:
1. inspect `runs/<case>/attempts.json`
2. inspect stage logs in failed attempt dir
3. rerun failed subset only:

```bash
python -u scripts/run_case_longcycle.py \
  --cases case_a.txt,case_b.txt \
  --include-nontrainable \
  --long-max-samples 30000000 \
  --root-out case_longcycle_retry_$(date +%Y%m%d_%H%M%S)
```

## Known Host Behavior

- `amp_pi_plus` is stable with `hiutil e38`
- `amp_pi_plus` often OOM at `hiutil e40/e39` on this host
- `add_pi_plus` and `deepmimic_pi_plus` can pass at `hiutil e40`
