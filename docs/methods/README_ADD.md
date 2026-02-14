# ADD

![ADD](../../images/ADD_teaser.png)

论文主页："Physics-Based Motion Imitation with Adversarial Differential Discriminators"  
https://xbpeng.github.io/projects/ADD/index.html

## 0. RL零基础预备知识（先看这一节）

| 术语 | 一句话解释 |
|---|---|
| 差分向量（Delta） | 策略状态与参考状态的差值，是 ADD 的核心判别对象 |
| 正样本/负样本 | 正样本固定零向量；负样本是实际差分向量 |
| 判别器 | 判断差分是否“接近零”的网络 |
| 梯度惩罚（GP） | 约束判别器梯度，防止训练不稳定 |
| 奖励缩放 | 调节风格奖励强度（过大易不稳） |
| 推理 | 不再学习，只测试模型是否稳定 |
| 可视化 | 看动作是否平滑、是否仍有明显差分误差 |
| 收敛 | 模型表现趋于稳定，不再剧烈波动 |

本仓库 3 种常用运行模式：
- 训练：`--mode train`。
- 推理：`--mode test --visualize false`。
- 可视化：`--mode test --visualize true`。

## 1. 论文导读（新手版）

### 1.1 论文核心思想

ADD 关注的问题是：传统多项奖励手工加权难调。  
它把判别对象从“状态本身”换成“差分向量 `Delta`”：
- 正样本固定为零向量；
- 负样本是策略状态与参考状态的差分。

在模仿场景中可理解为：
- 差得越远，判别器越容易识别；
- 策略通过对抗训练把差分拉回接近零。

### 1.2 读论文重点

1. 为什么“零向量正样本”能简化多目标权重调参。  
2. 梯度惩罚在差分判别中的稳定作用。  
3. 差分归一化对训练稳定性的影响。

## 2. 代码导读（论文机制 -> 源码位置）

### 2.1 运行主链路

- 参数入口：`mimickit/run.py:22`、`mimickit/run.py:95`、`mimickit/run.py:132`
- 环境分发：`mimickit/envs/env_builder.py:8`
- Agent 分发：`mimickit/learning/agent_builder.py:5`
- 训练/测试循环：`mimickit/learning/base_agent.py:51`、`mimickit/learning/base_agent.py:92`

### 2.2 ADD 核心函数

| 论文机制 | 代码入口 |
|---|---|
| 策略侧判别观测更新 | `mimickit/envs/add_env.py:33` |
| demo 侧判别观测更新 | `mimickit/envs/add_env.py:67` |
| 差分观测拼接 | `mimickit/envs/add_env.py:154` |
| 零向量正样本构建 | `mimickit/learning/add_agent.py:21` |
| 奖励融合（demo-policy 差分） | `mimickit/learning/add_agent.py:50` |
| 判别器损失（含 GP） | `mimickit/learning/add_agent.py:74` |
| 风格奖励变换（基类） | `mimickit/learning/amp_agent.py:209` |
| 差分归一化器 | `mimickit/learning/diff_normalizer.py` |

更细公式对照：`docs/paper_code/README_ADD_TheoryCode.md`

## 3. 训练参数导读（参数意义）

### 3.1 环境参数（`data/envs/add*_env.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `num_disc_obs_steps` | 差分判别时序窗口 | ADD 默认 `1`，聚焦单步差分 |
| `enable_tar_obs` / `tar_obs_steps` | 目标提示观测 | ADD 默认开启，帮助对齐参考动作 |
| `pose_termination` / `pose_termination_dist` | 姿态偏差终止阈值 | 过紧会频繁中断，过松会容忍坏姿态 |
| `motion_file` | 参考动作来源 | 可先单动作，再扩展到 dataset |
| `reward_*_w` / `reward_*_scale` | 模仿奖励底座 | 先保默认，重点调判别器相关参数 |

### 3.2 Agent 参数（`data/agents/add*_agent.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `disc_grad_penalty` | 判别器梯度惩罚强度 | ADD 稳定性关键参数，PI Plus 常设更高 |
| `disc_reward_scale` | 差分风格奖励缩放 | 过大可能抑制策略探索 |
| `disc_loss_weight` | 判别器损失权重 | 提高可增强判别器学习，但会增加对抗不稳定风险 |
| `disc_replay_samples` / `disc_buffer_size` | replay 稳定器 | 负样本分布波动大时优先检查 |
| `task_reward_weight` / `disc_reward_weight` | 任务与风格融合 | 当前 ADD 案例均是纯模仿（`0/1`） |
| `optimizer.learning_rate` | 学习率 | Go2 常用更高，PI Plus 常用更低 |
| `model.action_std` | 探索噪声 | 四足可更高（`0.1`），人形常见 `0.05` |

### 3.3 与训练预算相关参数

- `args/add_pi_plus_args.txt` 中 `--max_samples 120000000`：
  - 含义：PI Plus 动作复杂度更高，训练步数预算加长。

## 4. 案例覆盖（ADD 5）

| case | env_config | agent_config | motion_file | 参数导读（意义） |
|---|---|---|---|---|
| `add_g1_args.txt` | `data/envs/add_g1_env.yaml` | `data/agents/add_g1_agent.yaml` | `data/motions/g1/g1_walk.pkl` | 基础双足机器人 ADD 模仿基线；`disc_grad_penalty=1` |
| `add_go2_args.txt` | `data/envs/add_go2_env.yaml` | `data/agents/add_go2_agent.yaml` | `data/motions/go2/go2_pace.pkl` | `action_std=0.1` + `lr=2e-4`，四足探索更强 |
| `add_humanoid_args.txt` | `data/envs/add_humanoid_env.yaml` | `data/agents/add_humanoid_agent.yaml` | `data/motions/humanoid/humanoid_spinkick.pkl` | 人形 ADD 基线，`task_reward_weight=0`, `disc_reward_weight=1` |
| `add_pi_plus_args.txt` | `data/envs/add_pi_plus_env.yaml` | `data/agents/add_pi_plus_agent.yaml` | `data/motions/hightorque_pi_plus/pi_plus_walk.pkl` | `disc_grad_penalty=5.5` + `--max_samples=120000000`，强调稳定与长训练 |
| `add_smpl_args.txt` | `data/envs/add_smpl_env.yaml` | `data/agents/add_smpl_agent.yaml` | `data/motions/smpl/smpl_walk.pkl` | 高维人体骨架，保持默认 GP 与较稳学习节奏 |

## 5. 训练/推理/可视化模板

```bash
# 训练
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
  --mode train \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --out_dir output/train/<run_name>

# 推理（无渲染）
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
  --mode test \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --visualize false \
  --num_envs 1 \
  --test_episodes 10 \
  --model_file output/train/<run_name>/model.pt

# 策略可视化
python mimickit/run.py \
  --arg_file args/add_humanoid_args.txt \
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
| `add_g1_args.txt` | `add_g1` | 看回报与失败率是否稳定，避免训练后期抖动 | 观察步态连续性和落足稳定性 |
| `add_go2_args.txt` | `add_go2` | 看四足场景是否存在突然摔倒或节奏断裂 | 观察 pace 节律是否均匀、转身是否平滑 |
| `add_humanoid_args.txt` | `add_humanoid` | 看基线人形案例是否能稳定复现动作 | 观察 spinkick 动作的起跳-旋转-落地完整性 |
| `add_pi_plus_args.txt` | `add_pi_plus` | 长训练后重点看收益是否持续改进 | 观察高 GP 设置下动作是否更稳、是否减少抖动 |
| `add_smpl_args.txt` | `add_smpl` | 看高维骨架下是否维持稳定回合长度 | 观察全身关节协调性和自然度 |

## Citation

```bibtex
@inproceedings{
	zhang2025ADD,
    author={Zhang, Ziyu and Bashkirov, Sergey and Yang, Dun and Shi, Yi and Taylor, Michael and Peng, Xue Bin},
    title = {Physics-Based Motion Imitation with Adversarial Differential Discriminators},
    year = {2025},
    booktitle = {SIGGRAPH Asia 2025 Conference Papers (SIGGRAPH Asia '25 Conference Papers)}
}
```
