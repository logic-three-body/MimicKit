# AMP

![AMP](../../images/AMP_teaser.png)

论文主页："AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"  
https://xbpeng.github.io/projects/AMP/index.html

## 0. RL零基础预备知识（先看这一节）

| 术语 | 一句话解释 |
|---|---|
| 环境（Env） | 仿真世界，定义任务目标、观测、奖励、终止条件 |
| 策略（Policy） | 把观测映射成动作的函数（神经网络） |
| 判别器（Discriminator） | 区分“像数据风格”还是“不像”的网络（AMP 核心） |
| 任务奖励 | 要做成目标本身的得分（如到点、转向） |
| 风格奖励 | 动作是否“像参考数据分布”的得分 |
| 回放（Replay） | 用历史样本稳定判别器训练 |
| 推理 | 固定模型参数，只评估效果，不再学习 |
| 可视化 | 打开渲染窗口，观察动作质感与稳定性 |

本仓库 3 种常用运行模式：
- 训练：`--mode train`。
- 推理：`--mode test --visualize false`。
- 可视化：`--mode test --visualize true`。

## 1. 论文导读（新手版）

### 1.1 论文核心思想

AMP 的关键是把“做什么”和“怎么做”拆开：
- 任务奖励 `r^G`：决定目标完成度；
- 风格奖励 `r^S`：由判别器学习动作分布风格。

典型组合形式：`r_t = w_G * r_t^G + w_S * r_t^S`。

与 DeepMimic 相比，AMP 不强依赖逐帧对齐参考相位，而是用对抗学习让策略动作“整体像数据集风格”。

### 1.2 读论文重点

1. 判别器输入为什么用状态转移（`s_t, s_{t+1}`）而不是单帧。  
2. 风格奖励与任务奖励如何平衡。  
3. 对抗训练稳定化技巧（replay、梯度惩罚、归一化）。

## 2. 代码导读（论文机制 -> 源码位置）

### 2.1 运行主链路

- 参数入口：`mimickit/run.py:22`、`mimickit/run.py:95`、`mimickit/run.py:132`
- 环境分发：`mimickit/envs/env_builder.py:8`
- Agent 分发：`mimickit/learning/agent_builder.py:5`
- 训练/测试循环：`mimickit/learning/base_agent.py:51`、`mimickit/learning/base_agent.py:92`

### 2.2 AMP 核心函数

| 论文机制 | 代码入口 |
|---|---|
| 判别器 demo 采样 | `mimickit/envs/amp_env.py:63` |
| 判别器观测缓存构建 | `mimickit/envs/amp_env.py:94` |
| 判别器观测在线更新 | `mimickit/envs/amp_env.py:194` |
| 判别器观测拼接 | `mimickit/envs/amp_env.py:328` |
| 策略奖励融合 | `mimickit/learning/amp_agent.py:101` |
| 判别器损失 | `mimickit/learning/amp_agent.py:130` |
| 风格奖励计算 | `mimickit/learning/amp_agent.py:209` |

### 2.3 任务环境（location / steering）

- `task_location` 奖励与观测：`mimickit/envs/task_location_env.py:126`, `mimickit/envs/task_location_env.py:144`, `mimickit/envs/task_location_env.py:192`, `mimickit/envs/task_location_env.py:200`
- `task_steering` 奖励与观测：`mimickit/envs/task_steering_env.py:202`, `mimickit/envs/task_steering_env.py:222`, `mimickit/envs/task_steering_env.py:251`, `mimickit/envs/task_steering_env.py:271`

更细公式对照：`docs/paper_code/README_AMP_TheoryCode.md`

## 3. 训练参数导读（参数意义）

### 3.1 环境参数（`data/envs/amp*_env.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `num_disc_obs_steps` | 判别器时序窗口长度 | AMP 默认 `10`，动作复杂可增加但显存更高 |
| `motion_file` | 风格先验来源（`.pkl` 或 dataset） | 任务型 AMP 建议 dataset（覆盖风格变化） |
| `pose_termination` | 姿态偏差是否直接终止 | AMP 默认 `False`，避免过早截断风格探索 |
| `rand_reset` | 参考状态随机初始化 | 建议保持 `True`，稳定早期训练 |
| `reward_*_w` / `reward_*_scale` | 模仿误差项基底 | 与判别器奖励共同作用，不建议一开始大改 |
| `tar_speed*` / `tar_change_time*` / `tar_dist_max` | 任务目标动态范围（location/steering） | 任务奖励学不动时先缩窄目标范围 |
| `reward_steering_tar_w` / `reward_steering_face_w` | 转向任务速度/朝向权重 | 朝向不稳可提高 `face_w` |

### 3.2 Agent 参数（`data/agents/amp*_agent.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `task_reward_weight` | 任务奖励占比（what to do） | 纯模仿设 `0.0`，任务型常设 `0.5` |
| `disc_reward_weight` | 风格奖励占比（how to do） | 与 `task_reward_weight` 成对调节 |
| `disc_reward_scale` | 风格奖励缩放 | 风格不足可增大，但过大易压制任务项 |
| `disc_grad_penalty` | 判别器梯度惩罚强度 | 太小判别器会过陡，太大学习变慢 |
| `disc_replay_samples` / `disc_buffer_size` | 判别器 replay 稳定项 | 风格分布漂移大时优先提高 replay |
| `disc_batch_size` | 判别器每步样本量 | 增大更稳但更耗显存 |
| `optimizer.learning_rate` | 学习率 | Go2 常用 `2e-4`，SMPL 常用 `3e-5` |
| `model.action_std` | 探索噪声 | 四足场景常用 `0.1`，人形常用 `0.05` |

## 4. 案例覆盖（AMP 9）

| case | env_config | agent_config | motion_file | 参数导读（意义） |
|---|---|---|---|---|
| `amp_g1_args.txt` | `data/envs/amp_g1_env.yaml` | `data/agents/amp_g1_agent.yaml` | `data/motions/g1/g1_walk.pkl` | 纯风格模仿，`task_reward_weight=0`, `disc_reward_weight=1` |
| `amp_go2_args.txt` | `data/envs/amp_go2_env.yaml` | `data/agents/amp_go2_agent.yaml` | `data/motions/go2/go2_pace.pkl` | `action_std=0.1` + `lr=2e-4`，适合四足快速探索 |
| `amp_humanoid_args.txt` | `data/envs/amp_humanoid_env.yaml` | `data/agents/amp_humanoid_agent.yaml` | `data/motions/humanoid/humanoid_spinkick.pkl` | 人形纯模仿基线 |
| `amp_location_humanoid_args.txt` | `data/envs/amp_location_humanoid_env.yaml` | `data/agents/amp_task_humanoid_agent.yaml` | `data/datasets/dataset_humanoid_locomotion.yaml` | 任务+风格 5:5，`env_name=task_location` |
| `amp_location_humanoid_sword_shield_args.txt` | `data/envs/amp_location_humanoid_sword_shield_env.yaml` | `data/agents/amp_task_humanoid_agent.yaml` | `data/datasets/dataset_humanoid_sword_shield_locomotion.yaml` | 武器 locomotion 数据集，任务与风格均衡 |
| `amp_pi_plus_args.txt` | `data/envs/amp_pi_plus_env.yaml` | `data/agents/amp_pi_plus_agent.yaml` | `data/motions/hightorque_pi_plus/pi_plus_walk.pkl` | `--max_samples=120000000`，更长训练预算 |
| `amp_smpl_args.txt` | `data/envs/amp_smpl_env.yaml` | `data/agents/amp_smpl_agent.yaml` | `data/motions/smpl/smpl_walk.pkl` | `lr=3e-5`，高维人体更保守学习率 |
| `amp_steering_humanoid_args.txt` | `data/envs/amp_steering_humanoid_env.yaml` | `data/agents/amp_task_humanoid_agent.yaml` | `data/datasets/dataset_humanoid_locomotion.yaml` | `env_name=task_steering`，关注 `reward_steering_*` |
| `amp_steering_humanoid_sword_shield_args.txt` | `data/envs/amp_steering_humanoid_sword_shield_env.yaml` | `data/agents/amp_task_humanoid_agent.yaml` | `data/datasets/dataset_humanoid_sword_shield_locomotion.yaml` | 武器转向任务，`rand_face_dir=True` 更强调朝向控制 |

## 5. 训练/推理/可视化模板

```bash
# 训练
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
  --mode train \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --out_dir output/train/<run_name>

# 推理（无渲染）
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/<run_name>/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/amp_humanoid_args.txt \
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

# 可视化（风格质检）
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
| `amp_g1_args.txt` | `amp_g1` | 看风格奖励均值是否稳定，episode 是否无塌陷 | 观察动作是否保持目标风格且不僵硬 |
| `amp_go2_args.txt` | `amp_go2` | 看四足步态成功率与速度稳定性 | 观察 pace 风格是否连续、无跳变 |
| `amp_humanoid_args.txt` | `amp_humanoid` | 纯模仿基线：看风格分数与跟踪误差 | 看 spinkick 风格是否自然、节奏连贯 |
| `amp_location_humanoid_args.txt` | `amp_location_humanoid` | 看到点任务成功率与风格分维持情况 | 观察“到目标点”与动作风格是否兼顾 |
| `amp_location_humanoid_sword_shield_args.txt` | `amp_location_humanoid_sword_shield` | 看武器角色到点成功率与稳定性 | 观察移动过程中武器姿态与身体协调 |
| `amp_pi_plus_args.txt` | `amp_pi_plus` | 长预算后看风格稳定性与收敛平滑度 | 观察 PI Plus 关节协同与落足稳定性 |
| `amp_smpl_args.txt` | `amp_smpl` | 看高维角色在低学习率下是否稳收敛 | 观察动作自然度，避免高频抖动 |
| `amp_steering_humanoid_args.txt` | `amp_steering_humanoid` | 看转向任务奖励与风格奖励是否同时增长 | 观察转向时身体朝向与移动方向一致性 |
| `amp_steering_humanoid_sword_shield_args.txt` | `amp_steering_humanoid_sword_shield` | 看复杂转向+武器场景下失败率 | 观察急转向时步态是否断裂、武器是否抖动 |

## Citation

```bibtex
@article{
	2021-TOG-AMP,
	author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
	title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2021},
	volume = {40},
	number = {4},
	month = jul,
	year = {2021},
	articleno = {1},
	numpages = {15},
	url = {http://doi.acm.org/10.1145/3450626.3459670},
	doi = {10.1145/3450626.3459670},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
}
```
