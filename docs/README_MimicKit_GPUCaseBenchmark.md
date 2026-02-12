# MimicKit GPU Case Benchmark (Newton, dual-GPU)

## Runs

1. Baseline full benchmark (`25` trainable cases):
- Root: `output/train/case_gpu_bench_20260212_002319`
- Result: `22/25` success
- Failing cases: `add_pi_plus_args.txt`, `amp_pi_plus_args.txt`, `deepmimic_pi_plus_ppo_args.txt`

2. Pi-plus focused recovery benchmark:
- Root: `output/train/case_gpu_bench_piplus_final_20260212_1158`
- Result: `3/3` success
- Method summary:
  - `add`: `1/1`, avg min util `45.35`
  - `amp`: `1/1`, avg min util `51.54`
  - `deepmimic`: `1/1`, avg min util `52.03`

## Final Status

After applying pi-plus-specific dual-GPU ladder and hiutil agent variants:

- Total trainable cases: `25`
- Dual-GPU runnable cases: `25`
- Final success ratio: `100%`

Method-level final status:
- `add`: `5/5`
- `amp`: `9/9`
- `ase`: `2/2`
- `deepmimic`: `7/7`
- `vault`: `2/2`

## Pi-plus Recommended Dual-GPU Settings

Per-GPU `num_envs` (validated):

| case | variant | num_envs | min_avg_util | samples/s |
|---|---|---:|---:|---:|
| `add_pi_plus_args.txt` | `hiutil` | `40` | `45.35` | `193.21` |
| `amp_pi_plus_args.txt` | `hiutil` | `38` | `51.54` | `192.63` |
| `deepmimic_pi_plus_ppo_args.txt` | `hiutil` | `40` | `52.03` | `238.14` |

If one shared value is needed for all pi-plus methods, use:
- `--num_envs 38` (universal safe fallback)

## Failure Signatures (High env on pi-plus)

Common patterns before recovery:
- `ValueError: Array shapes must be non-negative, got -...`
- `RuntimeError: Failed to allocate ... bytes on device 'cuda:x'`
- NCCL teardown warning after rank crash (`destroy_process_group()` not called)

These occurred most often at:
- `num_envs >= 44` for pi-plus on dual 4090 setup

## Repro Commands

Full benchmark:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit

python -u scripts/run_case_gpu_bench.py
```

Pi-plus focused recovery:

```bash
python -u scripts/run_case_gpu_bench.py \
  --cases add_pi_plus_args.txt,amp_pi_plus_args.txt,deepmimic_pi_plus_ppo_args.txt \
  --env-ladder 44,40,39,38,36,32 \
  --pi-plus-ladder 44,40,39,38,36,32 \
  --max-seconds 420 \
  --iter-target 8 \
  --root-out case_gpu_bench_piplus_final_20260212_1158
```
