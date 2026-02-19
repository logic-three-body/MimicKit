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

## 为什么每个案例会不一样（根因拆解）

下面是本仓库里导致“同机不同案例表现差异大”的主要原因：

1. 环境计算负载不同（仿真侧差异）
- 不同任务/机器人带来的接触、约束、状态维度不同，导致每步仿真成本不同。
- 同样 `num_envs` 下，有的 case 更偏仿真吞吐，有的更偏训练更新计算。

2. 算法结构不同（学习侧差异）
- `deepmimic/ppo` 主要是 actor-critic 更新。
- `amp/add/ase` 还包含 discriminator（`disc_*`）相关开销。
- `ase` 额外有 encoder/latent（`enc_net`, `latent_dim`），计算更重，常见更高 util。

3. agent 配置强度不同（同方法也会差很多）
- 默认配置通常：`fc_2layers_1024units + update_epochs=5 + batch_size=4`。
- hiutil 配置通常：`fc_3layers_1024units + update_epochs=20/40 + batch_size=2`。
- 这会直接改变每轮反向传播负载，导致 util 曲线显著变化。

4. 显存占用结构不同（buffer 与 obs/action 形状）
- `steps_per_iter=32` 固定，但 obs/action 维度、方法额外缓存（如 disc buffer）不同。
- `pi_plus` 在本机表现为典型高显存场景（稳定档位也接近 24GB）。
- 高显存并不保证高 util，可能是“内存压满但算力没满”。

5. 多卡同步与通信占比不同
- 双卡下每轮都有同步成本，轻量计算任务更容易被通信开销“稀释” util。
- 这也是部分 `add/deepmimic` case util 中低而吞吐仍可接受的原因之一。

6. 训练阶段不同会改变观测
- `probe`、`long`、`test/viz` 的 util 与显存特征完全不同，不能混在一起判断。
- 长周期里建议只用 long 阶段统计做瓶颈判断。

## 如何更高效利用机器（实操手册）

先明确目标，再调参：

1. 目标 A：最大化“整机样本产出/天”
2. 目标 B：尽量抬高单任务 GPU 利用率
3. 目标 C：稳定优先（长周期不中断）

### 快速决策表

| 观测症状（按 long 阶段） | 判定 | 优先动作 |
|---|---|---|
| 显存 `>23GB` 且 util `45-55%` | VRAM 受限（pi_plus 常见） | 保持稳定档位，避免继续加 `num_envs`；优先保证不中断 |
| 显存 `<8GB` 且 util `<45%` | 吞吐/环境主导 | 若目标是整机吞吐，考虑两张卡分开跑两个单卡任务 |
| util `>60%` 且显存中等 | 计算主导 | 该配置已接近“高利用率区”，优先保持稳定运行 |
| util 波动大、平均不高但 samples/s 高 | 通信/阶段性波动 | 以 samples/s 与训练回报为主，不只盯 util |
| 频繁 OOM/NCCL | 稳定性不足 | 先降 `num_envs` 档位，再谈利用率 |

### 参数调优顺序（推荐固定流程）

1. 先稳定：
- 用已验证 ladder（默认与 pi_plus 专用 ladder）。
- 出现 OOM/NCCL 先降档，不要直接加重模型。

2. 再抬利用率：
- 对非 pi_plus：可以尝试 hiutil agent（更深网络、更高 `update_epochs`、更小 `batch_size`）。
- 对 pi_plus：先守住稳定显存档位（`add/deepmimic e40`, `amp e38`），再小步试探。

3. 最后看整机效率：
- 如果长期 `util<45%` 且显存明显没吃满，优先考虑“单卡并行双任务”而不是继续强行堆同一任务。
- 如果已经显存接近满载，不要盲目并行，优先保证主任务连续性。

### 针对本机（双 4090）可直接执行的策略

1. `pi_plus` 三例：
- 维持当前稳定档位：
  - `add_pi_plus`: `hiutil e40`
  - `amp_pi_plus`: `hiutil e38`
  - `deepmimic_pi_plus`: `hiutil e40`
- 目标优先级：稳定 > 利用率。

2. `ase/amp_smpl`：
- 这些 case 本身已偏高 util，继续双卡长训收益稳定。

3. `add_go2/deepmimic_go2/vault_g1` 这类低 util case：
- 若你的目标是“整机吞吐最大化”，可把一条双卡任务拆成两条单卡任务并行跑不同 case。
- 若目标是“单案例最快收敛”，保留双卡但不要只用 util 评估好坏。

## 监控口径建议（避免误判）

1. 统一用 long 阶段统计，不混 probe/test/viz。
2. 同时看三项：
- `min_avg_util` 或窗口平均 util
- `max_mem0/max_mem1`
- `samples_per_s`（吞吐）
3. 当三者冲突时，优先级建议：
- 长周期稳定性 > 吞吐 > util 表象。
