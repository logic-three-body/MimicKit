# MimicKit 新手学习与使用文档（全量 25 案例，论文对照）

## 00_零基础先认识 10 个概念

| 概念 | 一句话解释 |
|---|---|
| 环境（Env） | 机器人所在仿真世界，负责出观测、算奖励、判定结束 |
| 智能体（Agent） | 学习控制策略的算法模块（PPO/AMP/ASE/ADD 等） |
| 策略（Policy） | 把观测映射为动作的函数（通常是神经网络） |
| 观测（Observation） | 每一步输入给策略的信息 |
| 动作（Action） | 每一步由策略输出给仿真的控制量 |
| 奖励（Reward） | 每一步“做得好不好”的分数 |
| 步（Step） | 仿真推进一次 |
| 回合（Episode） | 从 reset 到 done 的完整过程 |
| 推理（Test） | 固定模型参数，只评估表现，不更新模型 |
| 可视化（Visualize） | 打开渲染窗口，肉眼检查动作是否自然稳定 |

建议新手先记住一个原则：  
训练看“指标是否持续变好”，推理看“是否稳定复现”，可视化看“动作是否自然”。

## 01_学习入口与阅读顺序

建议按下面顺序学习：

1. 框架总览与安装：`README.md`
2. 方法级基础说明：
   - `docs/methods/README_DeepMimic.md`
   - `docs/methods/README_AMP.md`
   - `docs/methods/README_ASE.md`
   - `docs/methods/README_ADD.md`
3. 论文-代码对照（公式到函数）：
   - `docs/paper_code/README_DeepMimic_TheoryCode.md`
   - `docs/paper_code/README_AMP_TheoryCode.md`
   - `docs/paper_code/README_ASE_TheoryCode.md`
   - `docs/paper_code/README_ADD_TheoryCode.md`
4. 再读本文，完成“训练 -> 推理 -> 可视化 -> 日志查看”的新手闭环。

---

## 02_运行主链路（源码定位）

以下是从命令行到训练循环的固定主链路（可直接按行号定位）：

- 入口与参数：
  - `mimickit/run.py:22` (`load_args`)
  - `mimickit/run.py:95` (`run`)
  - `mimickit/run.py:132` (`main`)
- Env 分发：`mimickit/envs/env_builder.py:8`
- Engine 分发：`mimickit/engines/engine_builder.py:6`
- Agent 分发：`mimickit/learning/agent_builder.py:5`
- 训练/测试循环：
  - `mimickit/learning/base_agent.py:51` (`train_model`)
  - `mimickit/learning/base_agent.py:92` (`test_model`)
- 动作数据加载：
  - `mimickit/anim/motion_lib.py:149` (`_load_motion_pkl`)
  - `mimickit/anim/motion_lib.py:251` (`_fetch_motion_files`)
- 纯动作可视化（不依赖策略）：
  - `mimickit/envs/view_motion_env.py:39` (`_load_motions`)
  - `mimickit/envs/view_motion_env.py:53` (`_sync_motion`)

---

## 03_引擎使用：训练/推理/可视化统一模板

### 3.1 引擎配置文件

- `data/engines/isaac_gym_engine.yaml`
- `data/engines/isaac_lab_engine.yaml`
- `data/engines/newton_engine.yaml`

说明：

- `args/*.txt` 默认写的是 Isaac Gym 引擎（`--engine_config data/engines/isaac_gym_engine.yaml`）。
- 你可以在命令行覆盖为 Isaac Lab 或 Newton，不需要改 `args` 文件。

### 3.2 统一训练模板

```bash
python mimickit/run.py \
  --arg_file args/<case_args>.txt \
  --engine_config data/engines/<isaac_gym_engine.yaml|isaac_lab_engine.yaml|newton_engine.yaml> \
  --mode train \
  --visualize false \
  --out_dir output/train/<run_name>
```

### 3.3 统一推理模板（策略测试）

```bash
python mimickit/run.py \
  --arg_file args/<case_args>.txt \
  --engine_config data/engines/<isaac_gym_engine.yaml|isaac_lab_engine.yaml|newton_engine.yaml> \
  --mode test \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/<run_name>/model.pt
```

### 3.4 统一可视化模板（策略可视化）

```bash
python mimickit/run.py \
  --arg_file args/<case_args>.txt \
  --engine_config data/engines/<isaac_gym_engine.yaml|isaac_lab_engine.yaml|newton_engine.yaml> \
  --mode test \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/<run_name>/model.pt
```

### 3.5 纯参考动作可视化（不跑策略）

```bash
python mimickit/run.py \
  --arg_file args/view_motion_humanoid_args.txt \
  --engine_config data/engines/<isaac_gym_engine.yaml|isaac_lab_engine.yaml|newton_engine.yaml> \
  --mode test \
  --visualize true
```

---

## 04_训练数据说明（motion 与 dataset）

### 4.1 两种 `motion_file` 形式

1. 单动作文件（`.pkl`）  
示例：`data/motions/humanoid/humanoid_spinkick.pkl`

2. 多动作数据集（`.yaml`）  
示例：`data/datasets/dataset_humanoid_locomotion.yaml`

`MotionLib` 会自动判断：

- 如果是 `.pkl`，按单动作读取；
- 如果是 `.yaml`，解析 `motions: [file, weight]` 列表并按权重采样。  
对应源码：`mimickit/anim/motion_lib.py:251`。

### 4.2 当前仓库动作数据概况

- `data/motions` 下共有 `145` 个 `.pkl`（按当前仓库统计）。
- 核心子目录：
  - `data/motions/humanoid`
  - `data/motions/go2`
  - `data/motions/g1`
  - `data/motions/hightorque_pi_plus`
  - `data/motions/smpl`
  - `data/motions/reallusion`

### 4.3 典型 dataset 文件

- `data/datasets/dataset_humanoid_locomotion.yaml`
- `data/datasets/dataset_humanoid_sword_shield.yaml`
- `data/datasets/dataset_humanoid_sword_shield_locomotion.yaml`
- `data/datasets/dataset_go2_locomotion.yaml`

---

## 05_全量25案例映射（分方法）

字段：`case | method | paper | env_config | agent_config | motion_file`

### 5.1 ADD（5）

| case | method | paper | env_config | agent_config | motion_file |
|---|---|---|---|---|---|
| add_g1_args.txt | add | ADD 2025 | data/envs/add_g1_env.yaml | data/agents/add_g1_agent.yaml | data/motions/g1/g1_walk.pkl |
| add_go2_args.txt | add | ADD 2025 | data/envs/add_go2_env.yaml | data/agents/add_go2_agent.yaml | data/motions/go2/go2_pace.pkl |
| add_humanoid_args.txt | add | ADD 2025 | data/envs/add_humanoid_env.yaml | data/agents/add_humanoid_agent.yaml | data/motions/humanoid/humanoid_spinkick.pkl |
| add_pi_plus_args.txt | add | ADD 2025 | data/envs/add_pi_plus_env.yaml | data/agents/add_pi_plus_agent.yaml | data/motions/hightorque_pi_plus/pi_plus_walk.pkl |
| add_smpl_args.txt | add | ADD 2025 | data/envs/add_smpl_env.yaml | data/agents/add_smpl_agent.yaml | data/motions/smpl/smpl_walk.pkl |

### 5.2 AMP（9）

| case | method | paper | env_config | agent_config | motion_file |
|---|---|---|---|---|---|
| amp_g1_args.txt | amp | AMP 2021 | data/envs/amp_g1_env.yaml | data/agents/amp_g1_agent.yaml | data/motions/g1/g1_walk.pkl |
| amp_go2_args.txt | amp | AMP 2021 | data/envs/amp_go2_env.yaml | data/agents/amp_go2_agent.yaml | data/motions/go2/go2_pace.pkl |
| amp_humanoid_args.txt | amp | AMP 2021 | data/envs/amp_humanoid_env.yaml | data/agents/amp_humanoid_agent.yaml | data/motions/humanoid/humanoid_spinkick.pkl |
| amp_location_humanoid_args.txt | amp | AMP 2021 | data/envs/amp_location_humanoid_env.yaml | data/agents/amp_task_humanoid_agent.yaml | data/datasets/dataset_humanoid_locomotion.yaml |
| amp_location_humanoid_sword_shield_args.txt | amp | AMP 2021 | data/envs/amp_location_humanoid_sword_shield_env.yaml | data/agents/amp_task_humanoid_agent.yaml | data/datasets/dataset_humanoid_sword_shield_locomotion.yaml |
| amp_pi_plus_args.txt | amp | AMP 2021 | data/envs/amp_pi_plus_env.yaml | data/agents/amp_pi_plus_agent.yaml | data/motions/hightorque_pi_plus/pi_plus_walk.pkl |
| amp_smpl_args.txt | amp | AMP 2021 | data/envs/amp_smpl_env.yaml | data/agents/amp_smpl_agent.yaml | data/motions/smpl/smpl_walk.pkl |
| amp_steering_humanoid_args.txt | amp | AMP 2021 | data/envs/amp_steering_humanoid_env.yaml | data/agents/amp_task_humanoid_agent.yaml | data/datasets/dataset_humanoid_locomotion.yaml |
| amp_steering_humanoid_sword_shield_args.txt | amp | AMP 2021 | data/envs/amp_steering_humanoid_sword_shield_env.yaml | data/agents/amp_task_humanoid_agent.yaml | data/datasets/dataset_humanoid_sword_shield_locomotion.yaml |

### 5.3 ASE（2）

| case | method | paper | env_config | agent_config | motion_file |
|---|---|---|---|---|---|
| ase_humanoid_args.txt | ase | ASE 2022 | data/envs/ase_humanoid_env.yaml | data/agents/ase_humanoid_agent.yaml | data/datasets/dataset_humanoid_locomotion.yaml |
| ase_humanoid_sword_shield_args.txt | ase | ASE 2022 | data/envs/ase_humanoid_sword_shield_env.yaml | data/agents/ase_humanoid_agent.yaml | data/datasets/dataset_humanoid_sword_shield.yaml |

### 5.4 DeepMimic（7）

| case | method | paper | env_config | agent_config | motion_file |
|---|---|---|---|---|---|
| deepmimic_g1_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_g1_env.yaml | data/agents/deepmimic_g1_ppo_agent.yaml | data/motions/g1/g1_walk.pkl |
| deepmimic_go2_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_go2_env.yaml | data/agents/deepmimic_go2_ppo_agent.yaml | data/motions/go2/go2_pace.pkl |
| deepmimic_humanoid_awr_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_humanoid_env.yaml | data/agents/deepmimic_humanoid_awr_agent.yaml | data/motions/humanoid/humanoid_spinkick.pkl |
| deepmimic_humanoid_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_humanoid_env.yaml | data/agents/deepmimic_humanoid_ppo_agent.yaml | data/motions/humanoid/humanoid_spinkick.pkl |
| deepmimic_humanoid_sword_shield_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_humanoid_sword_shield_env.yaml | data/agents/deepmimic_humanoid_ppo_agent.yaml | data/motions/reallusion/RL_Avatar_Atk_2xCombo01_Motion.pkl |
| deepmimic_pi_plus_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_pi_plus_env.yaml | data/agents/deepmimic_pi_plus_ppo_agent.yaml | data/motions/hightorque_pi_plus/pi_plus_walk.pkl |
| deepmimic_smpl_ppo_args.txt | deepmimic | DeepMimic 2018 | data/envs/deepmimic_smpl_env.yaml | data/agents/deepmimic_smpl_ppo_agent.yaml | data/motions/smpl/smpl_walk.pkl |

### 5.5 Vault 扩展（2）

| case | method | paper | env_config | agent_config | motion_file |
|---|---|---|---|---|---|
| vault_g1_args.txt | vault | DeepMimic 风格扩展示例（仓库 docs 未给独立论文） | data/envs/vault_g1_env.yaml | data/agents/deepmimic_g1_ppo_agent.yaml | data/motions/g1/g1_double_kong.pkl |
| vault_humanoid_args.txt | vault | DeepMimic 风格扩展示例（仓库 docs 未给独立论文） | data/envs/vault_humanoid_env.yaml | data/agents/deepmimic_humanoid_ppo_agent.yaml | data/motions/humanoid/humanoid_speed_vault.pkl |

---

## 06_论文-代码-配置对照（DeepMimic/AMP/ASE/ADD）

本节是“先读论文，再看代码”的最短索引；详细推导见 `docs/paper_code/README_*_TheoryCode.md`。

### 6.1 DeepMimic（2018）

论文关键词：指数型跟踪奖励、RSI、ET、PPO/AWR。

- 奖励：`mimickit/envs/deepmimic_env.py:788`
- 观察：
  - `mimickit/envs/deepmimic_env.py:330`
  - `mimickit/envs/deepmimic_env.py:681`
- RSI/重置：
  - `mimickit/envs/deepmimic_env.py:174`
  - `mimickit/envs/deepmimic_env.py:276`
- ET：
  - `mimickit/envs/deepmimic_env.py:463`
  - `mimickit/envs/deepmimic_env.py:725`

常用配置文件：

- 环境：`data/envs/deepmimic_humanoid_env.yaml`
- Agent（PPO）：`data/agents/deepmimic_humanoid_ppo_agent.yaml`
- Agent（AWR）：`data/agents/deepmimic_humanoid_awr_agent.yaml`

### 6.2 AMP（2021）

论文关键词：任务奖励 + 风格奖励（判别器）。

- 判别器观测：
  - `mimickit/envs/amp_env.py:94`
  - `mimickit/envs/amp_env.py:194`
  - `mimickit/envs/amp_env.py:328`
- 判别器损失与奖励：
  - `mimickit/learning/amp_agent.py:130`
  - `mimickit/learning/amp_agent.py:209`
  - `mimickit/learning/amp_agent.py:101`

任务版（location/steering）补充：

- location 任务观测/奖励：
  - `mimickit/envs/task_location_env.py:126`
  - `mimickit/envs/task_location_env.py:144`
  - `mimickit/envs/task_location_env.py:192`
  - `mimickit/envs/task_location_env.py:200`
- steering 任务观测/奖励：
  - `mimickit/envs/task_steering_env.py:202`
  - `mimickit/envs/task_steering_env.py:222`
  - `mimickit/envs/task_steering_env.py:251`
  - `mimickit/envs/task_steering_env.py:271`

常用配置文件：

- 纯模仿：`data/agents/amp_humanoid_agent.yaml`（`task_reward_weight=0.0`, `disc_reward_weight=1.0`）
- 任务+模仿：`data/agents/amp_task_humanoid_agent.yaml`（`task_reward_weight=0.5`, `disc_reward_weight=0.5`）

### 6.3 ASE（2022）

论文关键词：latent 条件策略 + 编码器互信息 + 多样性。

- latent 条件策略：
  - `mimickit/learning/ase_model.py:14`
  - `mimickit/learning/ase_model.py:20`
- 编码器与奖励/损失：
  - `mimickit/learning/ase_model.py:26`
  - `mimickit/learning/ase_agent.py:212`
  - `mimickit/learning/ase_agent.py:309`
- latent 调度与多样性：
  - `mimickit/learning/ase_agent.py:95`
  - `mimickit/learning/ase_agent.py:116`
  - `mimickit/learning/ase_agent.py:327`

常用配置文件：

- `data/agents/ase_humanoid_agent.yaml`  
重点参数：`latent_dim`, `enc_loss_weight`, `enc_reward_weight`, `diversity_weight`。

### 6.4 ADD（2025）

论文关键词：差分向量判别器、零向量正样本、差分归一化。

- 差分观测：
  - `mimickit/envs/add_env.py:33`
  - `mimickit/envs/add_env.py:67`
  - `mimickit/envs/add_env.py:154`
- 零向量正样本与判别器：
  - `mimickit/learning/add_agent.py:21`
  - `mimickit/learning/add_agent.py:74`
- 差分奖励：
  - `mimickit/learning/add_agent.py:50`

常用配置文件：

- `data/agents/add_humanoid_agent.yaml`  
重点参数：`disc_grad_penalty`, `disc_reward_scale`, `disc_reward_weight`。

---

## 07_新手最小闭环示例（4条）

下面每条都包含：训练、推理、策略可视化、日志查看。

### 7.1 DeepMimic Humanoid PPO

```bash
# 训练
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --out_dir output/train/deepmimic_humanoid_ppo_demo

# 推理
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/deepmimic_humanoid_ppo_demo/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/deepmimic_humanoid_ppo_demo/model.pt
```

### 7.2 AMP Humanoid

```bash
# 训练
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --out_dir output/train/amp_humanoid_demo

# 推理
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/amp_humanoid_demo/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/amp_humanoid_demo/model.pt
```

### 7.3 ASE Humanoid

```bash
# 训练
python mimickit/run.py \
  --arg_file args/ase_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --out_dir output/train/ase_humanoid_demo

# 推理
python mimickit/run.py \
  --arg_file args/ase_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/ase_humanoid_demo/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/ase_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/ase_humanoid_demo/model.pt
```

### 7.4 ADD Humanoid

```bash
# 训练
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode train \
  --visualize false \
  --out_dir output/train/add_humanoid_demo

# 推理
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/add_humanoid_demo/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
  --engine_config data/engines/newton_engine.yaml \
  --mode test \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/add_humanoid_demo/model.pt
```

### 7.5 日志查看（统一）

```bash
# TensorBoard
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```

文本日志位置：

- 仓库内示例日志：`data/logs/*.txt`
- 你自己的训练日志：`output/train/<run_name>/log.txt`

---

## 08_常见问题与排查

### 8.1 `args` 文件和命令行参数冲突时，谁生效？

`run.py` 先解析命令行，再加载 `--arg_file`。  
因此建议把“最终要覆盖的参数”放在命令行（例如 `--engine_config`、`--mode`、`--model_file`），避免歧义。

### 8.2 `motion_file` 用 `.pkl` 还是 `.yaml`？

- 想训练单一技能：用 `.pkl`。
- 想训练多技能分布：用 dataset `.yaml`。

### 8.3 可视化打不开或很慢？

- 先用 `--visualize false` 验证训练/推理链路。
- 可视化优先用 `--num_envs 1`。
- 区分两类可视化：
  - 策略可视化：`--mode test --model_file ... --visualize true`
  - 参考动作可视化：`args/view_motion_*_args.txt`

### 8.4 预训练模型找不到？

- 优先使用你自己训练输出的 `output/train/<run_name>/model.pt`。
- 仓库自带模型在 `data/models/`，但不是每个案例都提供。

---

## 09_验收清单

### 9.1 命令可执行性检查

- [ ] 模板命令参数都来自 `mimickit/run.py` 参数链路。
- [ ] `--mode train/test`、`--engine_config`、`--arg_file`、`--model_file` 用法一致。

### 9.2 案例映射检查

- [ ] 表格总数为 `25`（仅可训练案例）。
- [ ] 每行 `env_config` 与 `args/*_args.txt` 一致。
- [ ] 每行 `motion_file` 与对应 `data/envs/*.yaml` 一致。
- [ ] 每行 `agent_config` 与 `args/*_args.txt` 一致。

### 9.3 论文对应检查

- [ ] 方法级说明与 `docs/paper_code/README_*_TheoryCode.md` 一致。
- [ ] 函数行号可在仓库中定位。

### 9.4 可视化路径检查

- [ ] 策略可视化路径已覆盖：`--mode test --visualize true --model_file ...`
- [ ] 参考动作可视化路径已覆盖：`args/view_motion_*_args.txt`
- [ ] 文档中明确区分两者用途。

---

## 假设与默认值

- 已按 `README.md` 安装依赖并下载 `data/` 资源。
- 默认沿用 `args/*.txt` 基线配置（Isaac Gym），通过 `--engine_config` 可覆盖为 Isaac Lab/Newton。
- 若案例无仓库预训练模型，推理默认用训练输出 `output/.../model.pt`。
- 全量案例范围固定为 `25` 个可训练案例。
