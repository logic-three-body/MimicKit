# DeepMimic（含 Vault 扩展示例）

![DeepMimic](../../images/DeepMimic_teaser.png)

论文主页："DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"  
https://xbpeng.github.io/projects/DeepMimic/index.html

## 0. RL零基础预备知识（先看这一节）

| 术语 | 一句话解释 |
|---|---|
| 环境（Env） | 机器人所在的仿真世界，负责给观测、算奖励、判断回合结束 |
| 智能体（Agent） | 学习控制策略的模块（例如 PPO、AWR） |
| 观测（Obs） | 每一步输入给策略的信息（姿态、速度、目标信息等） |
| 动作（Action） | 策略输出给角色的控制量（关节目标/力矩相关量） |
| 奖励（Reward） | 对“这一步做得好不好”的打分，训练目标是长期累计奖励最大 |
| 步（Step） | 仿真推进一次 |
| 回合（Episode） | 从 reset 到 done 的一段完整交互 |
| 收敛 | 指标不再大幅波动，策略质量趋于稳定 |

本仓库 3 种常用运行模式：
- 训练：`--mode train`，用于更新模型参数。
- 推理：`--mode test --visualize false`，用于客观评估模型表现。
- 可视化：`--mode test --visualize true`，用于肉眼检查动作质量。

## 1. 论文导读（新手版）

### 1.1 论文要解决什么问题

DeepMimic 的核心目标是：
- 让物理角色在仿真中复现参考动作（imitation）；
- 同时保留任务目标（task）能力；
- 在复杂技能中保证训练稳定性（RSI、ET）。

论文中的典型目标可写为：
- 总奖励：`r_t = w^I * r_t^I + w^G * r_t^G`
- 模仿奖励由多个误差项指数化后加权组合（姿态、速度、根部、关键点等）。

### 1.2 读论文时重点看什么

建议优先看这 4 点：
1. 指数型跟踪奖励为什么比线性误差更稳定。  
2. RSI（Reference State Initialization）如何降低长序列起步难度。  
3. ET（Early Termination）如何避免“错误状态继续滚雪球”。  
4. PPO/AWR 在模仿任务中的区别（PPO 稳定、AWR 样本利用率高）。

## 2. 代码导读（论文机制 -> 源码位置）

### 2.1 运行主链路

- 参数入口：`mimickit/run.py:22`、`mimickit/run.py:95`、`mimickit/run.py:132`
- 环境分发：`mimickit/envs/env_builder.py:8`
- Agent 分发：`mimickit/learning/agent_builder.py:5`
- 训练/测试循环：`mimickit/learning/base_agent.py:51`、`mimickit/learning/base_agent.py:92`

### 2.2 DeepMimic 关键函数

| 论文机制 | 代码入口 |
|---|---|
| 模仿奖励 | `mimickit/envs/deepmimic_env.py:788` |
| 观测构造 | `mimickit/envs/deepmimic_env.py:330`, `mimickit/envs/deepmimic_env.py:681` |
| 目标观测采样 | `mimickit/envs/deepmimic_env.py:564` |
| RSI（参考状态初始化） | `mimickit/envs/deepmimic_env.py:174`, `mimickit/envs/deepmimic_env.py:276` |
| ET（提前终止） | `mimickit/envs/deepmimic_env.py:463`, `mimickit/envs/deepmimic_env.py:725` |
| PPO 优化器 | `mimickit/learning/ppo_agent.py` |
| AWR 优化器 | `mimickit/learning/awr_agent.py` |

更细公式对照：`docs/paper_code/README_DeepMimic_TheoryCode.md`

## 3. 训练参数导读（参数意义）

### 3.1 环境参数（`data/envs/deepmimic*_env.yaml` / `data/envs/vault*_env.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `motion_file` | 参考动作来源（单 `.pkl` 或数据集 `.yaml`） | 单技能先用 `.pkl`，多技能再换 dataset |
| `rand_reset` | RSI，重置到参考轨迹随机时刻 | 技能长且难启动时保持 `True` |
| `enable_early_termination` | ET 总开关 | 建议训练期保持开启 |
| `pose_termination` / `pose_termination_dist` | 姿态偏差终止阈值 | 失败判定太严会学不动，太松会学坏姿态 |
| `enable_tar_obs` / `tar_obs_steps` | 给策略“未来目标帧提示” | DeepMimic/Vault 默认开启，适合跟踪类模仿 |
| `reward_*_w` | 各奖励分项权重 | 先保持默认配比，再按误差日志微调 |
| `reward_*_scale` | 各误差项指数斜率 | 增大后对误差更敏感，训练更“挑剔” |
| `key_bodies` / `contact_bodies` | 关键点误差与接触约束 | 武器/体操动作需要按角色结构定制 |

### 3.2 Agent 参数（`data/agents/deepmimic*_agent.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `agent_name` | 训练算法（`PPO` 或 `AWR`） | PPO 先起步，AWR 做对比实验 |
| `optimizer.learning_rate` | 学习率 | Go2 常用更高 `2e-4`，SMPL 常用更低 `3e-5` |
| `model.action_std` | 初始动作噪声（探索强度） | 大关节/四足可适当更大（如 `0.1`） |
| `steps_per_iter` / `update_epochs` / `batch_size` | 每轮采样与更新强度 | 显存有限优先降 `batch_size` |
| `ppo_clip_ratio`（PPO） | 策略更新幅度限制 | 过大易不稳，过小学习慢 |
| `awr_temp` / `a_weight_clip`（AWR） | AWR 优势加权温度与截断 | 温度高更激进，clip 太大会不稳 |
| `td_lambda` / `discount` | 时序信用分配 | 长动作通常保留默认 `0.95/0.99` |

### 3.3 与训练预算相关参数

- `args/deepmimic_pi_plus_ppo_args.txt` 中 `--max_samples 150000000`：
  - 含义：总样本上限更高，给 PI Plus 留更长收敛时间。

## 4. 案例覆盖（DeepMimic 7 + Vault 扩展 2 = 9）

### 4.1 DeepMimic 论文主线案例（7）

| case | env_config | agent_config | motion_file | 参数导读（意义） |
|---|---|---|---|---|
| `deepmimic_g1_ppo_args.txt` | `data/envs/deepmimic_g1_env.yaml` | `data/agents/deepmimic_g1_ppo_agent.yaml` | `data/motions/g1/g1_walk.pkl` | 机器人基础步态；`pose_termination=True` 保持姿态约束 |
| `deepmimic_go2_ppo_args.txt` | `data/envs/deepmimic_go2_env.yaml` | `data/agents/deepmimic_go2_ppo_agent.yaml` | `data/motions/go2/go2_pace.pkl` | `action_std=0.1` + `lr=2e-4`，更强探索与更快更新 |
| `deepmimic_humanoid_awr_args.txt` | `data/envs/deepmimic_humanoid_env.yaml` | `data/agents/deepmimic_humanoid_awr_agent.yaml` | `data/motions/humanoid/humanoid_spinkick.pkl` | AWR 路线，关注 `awr_temp` 与 `a_weight_clip` |
| `deepmimic_humanoid_ppo_args.txt` | `data/envs/deepmimic_humanoid_env.yaml` | `data/agents/deepmimic_humanoid_ppo_agent.yaml` | `data/motions/humanoid/humanoid_spinkick.pkl` | Humanoid PPO 基线，适合先跑通全流程 |
| `deepmimic_humanoid_sword_shield_ppo_args.txt` | `data/envs/deepmimic_humanoid_sword_shield_env.yaml` | `data/agents/deepmimic_humanoid_ppo_agent.yaml` | `data/motions/reallusion/RL_Avatar_Atk_2xCombo01_Motion.pkl` | 武器动作，`key_bodies` 包含 `sword`，接触约束更关键 |
| `deepmimic_pi_plus_ppo_args.txt` | `data/envs/deepmimic_pi_plus_env.yaml` | `data/agents/deepmimic_pi_plus_ppo_agent.yaml` | `data/motions/hightorque_pi_plus/pi_plus_walk.pkl` | `--max_samples=150000000`，长预算训练；PI Plus 接地高度单独配置 |
| `deepmimic_smpl_ppo_args.txt` | `data/envs/deepmimic_smpl_env.yaml` | `data/agents/deepmimic_smpl_ppo_agent.yaml` | `data/motions/smpl/smpl_walk.pkl` | `lr=3e-5` 更保守，适合高维人体参数稳定优化 |

### 4.2 Vault 扩展示例（2，DeepMimic 风格）

| case | env_config | agent_config | motion_file | 参数导读（意义） |
|---|---|---|---|---|
| `vault_g1_args.txt` | `data/envs/vault_g1_env.yaml` | `data/agents/deepmimic_g1_ppo_agent.yaml` | `data/motions/g1/g1_double_kong.pkl` | `env_name=static_objects`，加入 `objects` 做跨越/翻越 |
| `vault_humanoid_args.txt` | `data/envs/vault_humanoid_env.yaml` | `data/agents/deepmimic_humanoid_ppo_agent.yaml` | `data/motions/humanoid/humanoid_speed_vault.pkl` | 同样是静态障碍扩展，重点检查接触体与障碍尺寸匹配 |

## 5. 训练/推理/可视化模板

```bash
# 训练
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode train \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --out_dir output/train/<run_name>

# 推理（无渲染）
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/<run_name>/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/<run_name>/model.pt
```

## 6. 每个案例推理与可视化讲解

统一命令（将 `<case_args>` 替换为下表对应案例）：

```bash
# 推理（指标验证）
python mimickit/run.py \
  --arg_file args/<case_args>.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/<run_name>/model.pt

# 可视化（动作质检）
python mimickit/run.py \
  --arg_file args/<case_args>.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize true \
  --num_envs 1 \
  --test_episodes 1 \
  --model_file output/train/<run_name>/model.pt
```

| case | 推荐 `run_name` | 推理讲解（看什么） | 可视化讲解（看什么） |
|---|---|---|---|
| `deepmimic_g1_ppo_args.txt` | `deepmimic_g1_ppo` | 看 episode 成功率与跟踪误差是否稳定下降 | 观察躯干是否稳定、步态是否连贯 |
| `deepmimic_go2_ppo_args.txt` | `deepmimic_go2_ppo` | 看速度保持与摔倒率（四足节律） | 观察 pace 节奏与落足时序是否一致 |
| `deepmimic_humanoid_awr_args.txt` | `deepmimic_humanoid_awr` | 对比 PPO：看收敛速度与最终误差 | 观察动作是否更“锐利”且无抖动 |
| `deepmimic_humanoid_ppo_args.txt` | `deepmimic_humanoid_ppo` | 作为人形基线，先确认可稳定复现 spinkick | 看起跳、旋转、落地三阶段连续性 |
| `deepmimic_humanoid_sword_shield_ppo_args.txt` | `deepmimic_humanoid_sword_shield_ppo` | 看武器动作轨迹与身体协调是否一致 | 重点看 sword 末端轨迹与接触时机 |
| `deepmimic_pi_plus_ppo_args.txt` | `deepmimic_pi_plus_ppo` | 长训练后看是否显著降低姿态误差 | 看下肢接地稳定性与摆臂同步性 |
| `deepmimic_smpl_ppo_args.txt` | `deepmimic_smpl_ppo` | 看高维角色是否存在慢性漂移 | 观察全身关节是否自然、无关节爆震 |
| `vault_g1_args.txt` | `vault_g1` | 看越障成功率与落地后恢复能力 | 重点看过障高度、离地时机、落地姿态 |
| `vault_humanoid_args.txt` | `vault_humanoid` | 看跨越后是否能快速恢复到稳定步态 | 观察障碍接近-起跳-跨越-落地的完整链路 |

## Citation

```bibtex
@article{
	2018-TOG-deepMimic,
	author = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
	title = {DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2018},
	volume = {37},
	number = {4},
	month = jul,
	year = {2018},
	issn = {0730-0301},
	pages = {143:1--143:14},
	articleno = {143},
	numpages = {14},
	url = {http://doi.acm.org/10.1145/3197517.3201311},
	doi = {10.1145/3197517.3201311},
	acmid = {3201311},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
}
```
