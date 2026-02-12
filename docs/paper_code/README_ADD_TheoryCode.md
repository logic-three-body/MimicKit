# ADD：论文-代码精细对照（更新版）

## 1. 论文核心问题与目标
参考论文：ADD (SIGGRAPH Asia 2025)
- arXiv: https://arxiv.org/abs/2505.04961

ADD针对多目标优化里“手工加权难调”的问题，提出对抗差分判别：
- 传统：\(\min_\theta \sum_i w_i l_i(\theta)\)
- ADD：先构造差分向量 \(\Delta\)，再做对抗优化

论文核心形式（简化写法）：
\[
\min_\theta \max_D\; \log D(\mathbf{0}) + \mathbb{E}[\log(1-D(\Delta))] - \lambda^{GP}\mathcal{L}^{GP}(D)
\]
其中唯一正样本是 \(\Delta=\mathbf{0}\)，负样本是当前策略产生的差分向量。

在运动模仿场景，论文写作：
\[
\Delta_t = \phi(\hat{s}_t) \ominus \phi(s_t),\quad
r_t = -\log(1-D(\Delta_t))
\]

## 2. 从论文到 MimicKit 的实现映射

### 2.1 差分向量构造
- policy观测：`mimickit/envs/add_env.py:33` (`_update_disc_obs`)
- demo观测：`mimickit/envs/add_env.py:67` (`_update_disc_obs_demo`)
- 差分构造：`mimickit/learning/add_agent.py:50` (`_compute_rewards` 中 `obs_diff = disc_obs_demo - disc_obs`)

### 2.2 “单一正样本=零向量”判别器
- 零向量模板：`mimickit/learning/add_agent.py:21` (`_build_pos_diff`)
- 判别器主损失：`mimickit/learning/add_agent.py:74` (`_compute_disc_loss`)

实现上：
- 正样本：`pos_diff = 0`
- 负样本：`diff_obs = tar_disc_obs - disc_obs`（含replay混合）

### 2.3 梯度惩罚（GP）
- 代码在负样本差分上对判别器输出求梯度惩罚（`norm_diff_obs` 分支），与论文“负样本GP重要”结论一致。
- 入口同样在：`mimickit/learning/add_agent.py:74`（函数内部GP段）。

### 2.4 差分归一化（DiffNormalizer）
- 构建：`mimickit/learning/add_agent.py:27` (`_build_normalizers`)
- 模块：`mimickit/learning/diff_normalizer.py`

它解决不同差分分量量纲差异问题，是ADD稳定训练的关键工程点之一。

### 2.5 策略奖励路径
- ADD重写奖励融合：`mimickit/learning/add_agent.py:50` (`_compute_rewards`)
- 判别器奖励函数沿用AMP基类实现：`mimickit/learning/amp_agent.py:209` (`_calc_disc_rewards`)

说明：
- 旧文档里将 `_calc_disc_rewards` 写在 `add_agent.py`，实际当前仓库是在AMP基类里。

## 3. 论文项 -> 配置映射

| 论文项 | 含义 | MimicKit 参数 |
|---|---|---|
| \(\lambda^{GP}\) | 梯度惩罚强度 | `disc_grad_penalty` |
| 判别器正则 | 稳定边界/防过拟合 | `disc_logit_reg`, `disc_weight_decay` |
| 判别器loss权重 | 对抗分支权重 | `disc_loss_weight` |
| 差分奖励强度 | 对策略的信号强度 | `disc_reward_scale` |
| 任务/风格融合 | 是否叠加外部任务奖励 | `task_reward_weight`, `disc_reward_weight` |
| replay稳态化 | 历史负样本注入 | `disc_buffer_size`, `disc_replay_samples`, `disc_batch_size` |

主要配置文件：`data/agents/add_humanoid_agent.yaml`（其他机器人同结构）。

## 4. 与论文一致点与实现差异

一致点：
- 以差分向量 \(\Delta\) 为判别对象。
- 正样本只用零向量。
- 策略奖励由 \(-\log(1-D(\Delta))\) 思路驱动。

实现差异（工程化）：
- 训练时加入replay与归一化器实现稳定化。
- 判别器损失以工程可训练实现组织（logits + 正则 + GP），而非只保留最小数学形式。

## 5. 读代码时的核对清单
- `obs_diff` 是否确实来自 demo-policy 差分。
- `disc_grad_penalty` 是否在负样本分支起作用。
- DiffNormalizer是否更新正常（否则训练容易抖动）。
- 奖励路径是否经过 ADD 的 `_compute_rewards` + AMP基类 `_calc_disc_rewards`。
