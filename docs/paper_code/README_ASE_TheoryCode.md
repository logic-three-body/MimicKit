# ASE：理论-代码精细对照

## 1. 论文主问题与目标
参考论文：ASE (TOG 2022)  
- 论文链接：https://arxiv.org/abs/2205.01906

ASE 的核心是学习可复用技能嵌入：低层策略 \(\pi(a|s,z)\) 学“技能库”，高层策略 \(\omega(z|s,g)\) 在新任务上调用技能。

论文预训练目标可写成：
\[
\max_\pi \; -D_{JS}(d_\pi, d_M) + \beta I((s,s');z)
\]
- 第一项：对抗模仿，让行为分布贴近动作数据集
- 第二项：互信息项，确保不同 latent 对应可区分行为

## 2. 论文关键机制与代码映射

### 2.1 Skill-conditioned policy（\(\pi(a|s,z)\)）
代码中 actor/critic 都显式拼接 `obs` 与 `z`：
- actor：`mimickit/learning/ase_model.py:14` `eval_actor`
- critic：`mimickit/learning/ase_model.py:20` `eval_critic`

这与论文“latent 条件策略”完全对应。

### 2.2 互信息下界与编码器目标
论文通过变分分布 \(q(z|s,s')\) 近似互信息下界。MimicKit 对应：
- 编码器：`mimickit/learning/ase_model.py:26` `eval_enc`
- 编码器损失：`mimickit/learning/ase_agent.py:309` `_compute_enc_loss`
- 编码器奖励：`mimickit/learning/ase_agent.py:212` `_calc_enc_rewards`

实现要点：
- `enc_pred = eval_enc(disc_obs)`
- `enc_err = -<z, enc_pred>`（点积负号形式）
- `enc_loss = mean(enc_err)`
- `enc_reward = clamp_min(-enc_err, 0)`

即：编码器越能从状态转移恢复 z，reward 越高。

### 2.3 对抗模仿项（沿用 AMP 路径）
ASE 在代码上复用 AMP 的对抗分支：
- discriminator 更新与 reward：来自 `AMPAgent`
- ASE 在 `ASEAgent._compute_rewards` 中将其与 enc reward 合并

对应函数：
- `mimickit/learning/ase_agent.py:186` `_compute_rewards`

### 2.4 多样性目标（diversity）
论文在技能可区分性上额外加约束。MimicKit 对应：
- `mimickit/learning/ase_agent.py:327` `_compute_diversity_loss`

实现思想：
- 同一观测下采样新 latent `new_z`
- 比较动作分布均值差异 `a_diff`
- 用 latent 差异 `z_diff` 归一
- 拉向 `diversity_tar`

### 2.5 Latent 时域调度与复用
论文中 latent 在时域段内保持一段时间再切换。MimicKit 对应：
- 采样 latent：`_sample_latents` (`mimickit/learning/ase_agent.py:126`)
- 设定重采样时刻：`_reset_latents` (`mimickit/learning/ase_agent.py:95`)
- 按环境时间触发更新：`_update_latents`

对应配置：`latent_time_min`, `latent_time_max`。

## 3. 论文项 -> 配置映射

| 论文项 | 作用 | MimicKit 参数 |
|---|---|---|
| \(\beta\)（MI 权重） | 编码器/技能可辨识 | `enc_reward_weight`, `enc_loss_weight` |
| adversarial imitation 权重 | 动作分布贴近数据 | `disc_reward_weight`, `disc_loss_weight` |
| task 项权重 | 下游目标 | `task_reward_weight` |
| latent 维度 | 技能容量 | `model.latent_dim` |
| encoder 网络 | 互信息近似器 | `model.enc_net` |
| diversity 系数 | 防 mode collapse | `diversity_weight`, `diversity_tar` |
| latent 切换时间 | 技能片段时长 | `latent_time_min`, `latent_time_max` |

主要配置文件：`data/agents/ase_humanoid_agent.yaml`。

## 4. 训练链路（实现顺序）
1. rollout 时记录 `obs/action/reward/disc_obs/latents`。
2. 计算混合奖励：`task + disc + enc`。
3. PPO 更新 actor/critic（conditioned on z）。
4. 额外叠加 `enc_loss` 与 `diversity_loss`。
5. 按时间重采样 latent，持续探索技能空间。

## 5. 与论文一致/差异点
- 一致：对抗模仿 + 互信息 + 多样性 + latent 条件策略。
- 工程实现差异：互信息项采用可高效训练的点积误差形式（`_calc_enc_error`），并将部分项以 reward、部分项以 loss 注入优化。
