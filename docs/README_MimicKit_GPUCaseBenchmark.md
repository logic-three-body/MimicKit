# Case GPU Benchmark Summary

- root: `output/train/case_gpu_bench_20260212_002319`
- total cases: `25`

## Method Summary

| method | ok/total | avg min util | avg samples/s |
|---|---:|---:|---:|
| `add` | `4/5` | `39.60` | `4938.80` |
| `amp` | `8/9` | `52.63` | `5585.05` |
| `ase` | `2/2` | `66.46` | `5090.65` |
| `deepmimic` | `6/7` | `44.50` | `6980.74` |
| `vault` | `2/2` | `42.17` | `6500.99` |

## Case Details

| case | method | status | variant | num_envs | min_avg_util | samples/s | out_dir |
|---|---|---|---|---:|---:|---:|---|
| `add_g1_args.txt` | `add` | `ok` | `hiutil` | `1024` | `35.66` | `4128.25` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/add_g1_args/hiutil_e1024` |
| `add_go2_args.txt` | `add` | `ok` | `hiutil` | `1024` | `27.44` | `4161.02` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/add_go2_args/hiutil_e1024` |
| `add_humanoid_args.txt` | `add` | `ok` | `hiutil` | `1024` | `49.19` | `6472.69` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/add_humanoid_args/hiutil_e1024` |
| `add_pi_plus_args.txt` | `add` | `oom_or_nccl` | `default` | `256` | `0.95` | `0.00` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/add_pi_plus_args/default_e256` |
| `add_smpl_args.txt` | `add` | `ok` | `hiutil` | `1024` | `46.11` | `4993.22` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/add_smpl_args/hiutil_e1024` |
| `amp_g1_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `43.42` | `4405.78` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_g1_args/hiutil_e1024` |
| `amp_go2_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `32.71` | `3826.92` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_go2_args/hiutil_e1024` |
| `amp_humanoid_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `59.45` | `6898.53` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_humanoid_args/hiutil_e1024` |
| `amp_location_humanoid_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `56.55` | `6168.09` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_location_humanoid_args/hiutil_e1024` |
| `amp_location_humanoid_sword_shield_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `49.67` | `5295.84` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_location_humanoid_sword_shield_args/hiutil_e1024` |
| `amp_pi_plus_args.txt` | `amp` | `oom_or_nccl` | `default` | `256` | `0.68` | `0.00` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_pi_plus_args/default_e256` |
| `amp_smpl_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `65.77` | `5890.88` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_smpl_args/hiutil_e1024` |
| `amp_steering_humanoid_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `55.98` | `6168.09` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_steering_humanoid_args/hiutil_e1024` |
| `amp_steering_humanoid_sword_shield_args.txt` | `amp` | `ok` | `hiutil` | `1024` | `57.52` | `6026.30` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/amp_steering_humanoid_sword_shield_args/hiutil_e1024` |
| `ase_humanoid_args.txt` | `ase` | `ok` | `hiutil` | `1024` | `66.92` | `5140.08` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/ase_humanoid_args/hiutil_e1024` |
| `ase_humanoid_sword_shield_args.txt` | `ase` | `ok` | `hiutil` | `1024` | `66.00` | `5041.23` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/ase_humanoid_sword_shield_args/hiutil_e1024` |
| `deepmimic_g1_ppo_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `32.64` | `4993.22` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_g1_ppo_args/hiutil_e1024` |
| `deepmimic_go2_ppo_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `24.96` | `4854.52` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_go2_ppo_args/hiutil_e1024` |
| `deepmimic_humanoid_awr_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `52.24` | `8738.13` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_humanoid_awr_args/hiutil_e1024` |
| `deepmimic_humanoid_ppo_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `49.75` | `8456.26` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_humanoid_ppo_args/hiutil_e1024` |
| `deepmimic_humanoid_sword_shield_ppo_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `51.95` | `7943.76` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_humanoid_sword_shield_ppo_args/hiutil_e1024` |
| `deepmimic_pi_plus_ppo_args.txt` | `deepmimic` | `oom_or_nccl` | `default` | `256` | `0.62` | `0.00` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_pi_plus_ppo_args/default_e256` |
| `deepmimic_smpl_ppo_args.txt` | `deepmimic` | `ok` | `hiutil` | `1024` | `55.48` | `6898.53` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/deepmimic_smpl_ppo_args/hiutil_e1024` |
| `vault_g1_args.txt` | `vault` | `ok` | `hiutil` | `1024` | `33.46` | `4809.98` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/vault_g1_args/hiutil_e1024` |
| `vault_humanoid_args.txt` | `vault` | `ok` | `hiutil` | `1024` | `50.89` | `8192.00` | `/root/Project/MimicKit/output/train/case_gpu_bench_20260212_002319/runs/vault_humanoid_args/hiutil_e1024` |
