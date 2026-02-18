# MimicKit GPU 利用模式总结（基于历史日志与案例基准）

## 结论（先回答你的问题）

是的，确实存在两类不同情况，而且会同时出现：

1. 有些案例可以把显存吃到接近满（24GB），但 GPU 利用率不一定满。
2. 有些案例 GPU 利用率可以到较高水平（60%+，局部瞬时更高），但显存只在中等区间（约 4-8GB）。

换句话说，`显存占用高` 和 `GPU util 高` 不是同一件事，不同 case 的瓶颈不同。

## 数据来源（本机历史）

结构化 benchmark 结果（主依据）：

- `output/train/case_gpu_bench_20260212_002319/best_by_case.tsv`（25 个 trainable 基线）
- `output/train/case_gpu_bench_piplus_final_20260212_1158/best_by_case.tsv`（3 个 pi_plus 修复结果）

长周期监控日志（运行态辅助）：

- `/tmp/mk_ultralong_monitor_20260217_134108.log`

早期 3 分钟采样日志（辅助）：

- `/tmp/mk_3min_report_20260211_215800.log`
- `/tmp/mk_gpu_report_dual_longrun_20260211_215352.log`

## 全案例（25 trainable）最终画像

统计口径：把 pi_plus 三例替换为 `case_gpu_bench_piplus_final_20260212_1158` 的最终通过配置。

- 总数：`25`
- 全部可双卡运行：`25/25`

按类型分组：

1. `显存吃满 + 中高利用率`（3 例，pi_plus）
2. `算力主导高利用率`（3 例）
3. `均衡型`（12 例）
4. `吞吐/环境主导低利用率`（7 例）

## 1) 显存吃满 + 中高利用率（VRAM-bound mixed）

这类案例显存接近 24GB 上限，利用率中高但通常不到 90% 持续满载。

| case | 配置 | min_avg_util | max_mem0/max_mem1 |
|---|---|---:|---:|
| `add_pi_plus_args.txt` | `hiutil e40` | `45.35` | `24037 / 24093` MiB |
| `amp_pi_plus_args.txt` | `hiutil e38` | `51.54` | `24052 / 24115` MiB |
| `deepmimic_pi_plus_ppo_args.txt` | `hiutil e40` | `52.03` | `23802 / 24110` MiB |

补充：在基线默认配置（`default e256`）下，这三例曾出现 `oom_or_nccl`，并且显存已冲到 `~21.6-24.0GB`，但 `min_avg_util < 1%`，典型“先撞显存墙”。

## 2) 算力主导高利用率（Compute-heavy）

这类案例显存中等，但利用率明显更高。

| case | 配置 | min_avg_util | max_mem0/max_mem1 |
|---|---|---:|---:|
| `ase_humanoid_args.txt` | `hiutil e1024` | `66.92` | `4826 / 5480` MiB |
| `ase_humanoid_sword_shield_args.txt` | `hiutil e1024` | `66.00` | `5566 / 6265` MiB |
| `amp_smpl_args.txt` | `hiutil e1024` | `65.77` | `6684 / 7276` MiB |

## 3) 均衡型（Balanced）

共 `12` 例，`min_avg_util` 大致在 `46-59`，显存在 `~2.6-7.3GB` 区间。代表：

- `amp_humanoid_args.txt`（`59.45`）
- `amp_location_humanoid_args.txt`（`56.55`）
- `deepmimic_humanoid_ppo_args.txt`（`49.75`）
- `vault_humanoid_args.txt`（`50.89`）

## 4) 吞吐/环境主导低利用率（Throughput-or-env-bound）

共 `7` 例，`min_avg_util` 大致在 `25-43`，并非显存瓶颈。代表：

- `add_go2_args.txt`：`27.44`，`3873 / 7412` MiB
- `deepmimic_go2_ppo_args.txt`：`24.96`，`2980 / 3807` MiB
- `vault_g1_args.txt`：`33.46`，`2538 / 3785` MiB

## 长周期日志中的实证（不是短基准）

来自 `/tmp/mk_ultralong_monitor_20260217_134108.log` 的聚合：

- `add_g1_args.txt`（长训阶段）：
  - 采样点 `n=481`
  - 平均 util：`gpu0=50.73%`，`gpu1=53.85%`
  - 峰值 util：`gpu0=81%`，`gpu1=72%`
  - 峰值显存：`4225 / 2057` MiB

- `add_go2_args.txt`（长训阶段）：
  - 采样点 `n=86`
  - 平均 util：`gpu0=46.05%`，`gpu1=49.43%`
  - 峰值 util：`gpu0=66%`，`gpu1=65%`
  - 峰值显存：`4248 / 2094` MiB

这和基准结论一致：`add` 类很多时候不是显存吃满型，而是中低 util 的吞吐型。

## 方法级平均（最终 25 案例）

| method | 平均 min_avg_util | 平均 max_mem(双卡取大) |
|---|---:|---:|
| `ase` | `66.46` | `5872.5` MiB |
| `amp` | `52.51` | `8587.0` MiB |
| `deepmimic` | `45.58` | `6592.7` MiB |
| `vault` | `42.17` | `3841.0` MiB |
| `add` | `40.75` | `10051.2` MiB（受 `add_pi_plus` 高显存拉高） |

## 经验化建议

1. 想“显存吃满”：优先看 `pi_plus`，并使用已验证档位：`add/deepmimic=e40`，`amp=e38`。
2. 想“利用率更高”：优先 `ase`、`amp_smpl` 这类算力主导场景。
3. util 不高不一定是坏事：很多 `add/deepmimic` 案例吞吐仍然高，瓶颈在环境/同步而非显存。
4. 调参顺序建议：
   `先保稳定(不过OOM/NCCL) -> 再抬util -> 最后看样本吞吐与回报`。
