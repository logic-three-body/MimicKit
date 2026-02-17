# MimicKit All-Case Longcycle Snapshot (Newton, dual-GPU)

## Final Snapshot

- Run root: `output/train/case_longcycle_full_20260212_211318`
- Scope: `args/*.txt` all cases (`32` total)
- Final result: `32/32` `final_ok=1`
- Status file: `output/train/case_longcycle_full_20260212_211318/progress.json`
- Summary file: `output/train/case_longcycle_full_20260212_211318/best_by_case.tsv`

## Case Split Result

- Trainable: `25/25` pass
- Nontrainable: `7/7` pass

Method breakdown:

- `add`: `5/5`
- `amp`: `9/9`
- `ase`: `2/2`
- `deepmimic`: `7/7`
- `dof`: `1/1`
- `vault`: `2/2`
- `view`: `6/6`

## Key Runtime Decisions (Validated)

- Backend: `data/engines/newton_engine.yaml`
- Train devices: `cuda:0,cuda:1`
- Serial queue execution: one case at a time
- Trainable flow: `probe_train -> long_train(30000000) -> test -> viz`
- Nontrainable flow: `bootstrap_train(128) -> test -> viz`
- NCCL stability env enabled in train stage:
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_IB_DISABLE=1`
  - `NCCL_CUMEM_ENABLE=0`
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- Viz Mesa env enabled in visualization stage:
  - `MESA_GL_VERSION_OVERRIDE=3.3`
  - `MESA_GLSL_VERSION_OVERRIDE=330`

## Pi-Plus Stable Env Picks

- `add_pi_plus_args.txt`: `hiutil e40`
- `deepmimic_pi_plus_ppo_args.txt`: `hiutil e40`
- `amp_pi_plus_args.txt`: `hiutil e36`
  - note: this host repeatedly hit OOM at `e40/e39`, stable at `e36`

## Artifact Layout And Reuse

Under `output/train/case_longcycle_full_20260212_211318/`:

- `case_manifest.tsv`: discovered case inventory
- `best_by_case.tsv`: final selected attempt per case
- `runs/<case>/<variant>_e<num_envs>/`:
  - `probe_train/`, `long_train/`
  - `probe_train.log`, `long_train.log`, `test.log`, `viz.log`
- `runs/<case>/attempts.json`: all attempted ladders and stage outcomes

Fast lookup example (inspect one case best run):

```bash
python - <<'PY'
import csv
path='output/train/case_longcycle_full_20260212_211318/best_by_case.tsv'
case='deepmimic_humanoid_ppo_args.txt'
for r in csv.DictReader(open(path), delimiter='\t'):
    if r['case']==case:
        print('out_dir=', r['out_dir'])
        print('long_out_dir=', r['long_out_dir'])
        print('test_log=', r['test_log'])
        print('viz_log=', r['viz_log'])
        break
PY
```

## One-Command Reproduce

Fresh longcycle run:

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

Resume after interruption (same root):

```bash
python -u scripts/run_case_longcycle.py \
  --engine-config data/engines/newton_engine.yaml \
  --devices-train cuda:0,cuda:1 \
  --include-nontrainable \
  --long-max-samples 30000000 \
  --root-out case_longcycle_full_20260212_211318 \
  --resume-skip-status ok
```

## Acceptance Check

```bash
python - <<'PY'
import csv
path='output/train/case_longcycle_full_20260212_211318/best_by_case.tsv'
rows=list(csv.DictReader(open(path), delimiter='\t'))
ok=sum(1 for r in rows if str(r.get('final_ok','')).strip()=='1')
print('total=', len(rows), 'ok=', ok, 'fail=', len(rows)-ok)
print('all_pass=', ok==len(rows)==32)
PY
```
