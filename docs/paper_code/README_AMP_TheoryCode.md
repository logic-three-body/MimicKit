# AMP：论文-代码精细对照（更新版）

## 1. 论文核心问题与目标
参考论文：AMP (TOG 2021)
- arXiv: https://arxiv.org/abs/2104.02180

AMP将任务目标与风格目标解耦：
\[
r_t = w_G r_t^G + w_S r_t^S
\]
其中：
- \(r_t^G\)：任务奖励（what to do）
- \(r_t^S\)：风格奖励（how to do），来自adversarial motion prior

论文关键思想：不再强依赖逐帧跟踪某个参考动作，而是让判别器学习“该动作分布是否像数据集中的风格”。

## 2. 论文关键公式与 MimicKit 对照

### 2.1 判别器训练目标（论文）
论文采用 state-transition 判别器，并给出 least-squares GAN 风格目标（含梯度正则）。
风格奖励形式可写为（论文形式之一）：
\[
r^S(s_t,s_{t+1}) = \max\left(0, 1 - 0.25\,(D-1)^2\right)
\]

### 2.2 仓库实现路径
- 判别器网络：`mimickit/learning/amp_model.py:12` (`eval_disc`)
- 判别器损失：`mimickit/learning/amp_agent.py:130` (`_compute_disc_loss`)
- 风格奖励：`mimickit/learning/amp_agent.py:209` (`_calc_disc_rewards`)
- 奖励融合：`mimickit/learning/amp_agent.py:101` (`_compute_rewards`)

实现细节（非常关键）：
- 当前代码里判别器损失是 `BCEWithLogitsLoss`（正样本=demo，负样本=policy+replay），而不是直接按论文LSGAN公式逐字实现。
- 风格奖励采用 logit 概率变换：
  \[
  r^S = -\log(1-\sigma(\text{logit}))\times \texttt{disc_reward_scale}
  \]
- 因此：该实现与论文思想一致，但损失形式属于工程上的等价替代与稳定化实现。

### 2.3 判别器输入 \(\Phi(s_t,s_{t+1})\)
- demo片段采样：`mimickit/envs/amp_env.py:63` (`_fetch_disc_demo_data`)
- 轨迹buffer：`mimickit/envs/amp_env.py:94` (`_build_disc_obs_buffers`)
- 在线更新：`mimickit/envs/amp_env.py:194` (`_update_disc_obs`)
- 特征拼接：`mimickit/envs/amp_env.py:328` (`compute_disc_obs`)

`num_disc_obs_steps` 对应论文里的时序窗口长度。

### 2.4 训练稳定化（论文思想 -> 代码）
- 判别器观测归一化：`mimickit/learning/amp_agent.py:44`
- replay样本注入：`mimickit/learning/amp_agent.py:86`
- gradient penalty：`mimickit/learning/amp_agent.py:130`（函数内GP项）

## 3. 公式/概念 -> 配置映射

| 论文概念 | 含义 | MimicKit 参数 |
|---|---|---|
| \(w_G\) | 任务奖励权重 | `task_reward_weight` |
| \(w_S\) | 风格奖励权重 | `disc_reward_weight` |
| 判别器batch | 对抗训练规模 | `disc_batch_size` |
| replay注入 | 历史分布稳态化 | `disc_buffer_size`, `disc_replay_samples` |
| GP强度 | 判别器平滑正则 | `disc_grad_penalty` |
| 判别器正则 | 防过拟合/过陡边界 | `disc_logit_reg`, `disc_weight_decay` |
| 风格奖励缩放 | 风格信号强度 | `disc_reward_scale` |
| 时序窗口 | 判别器输入长度 | `num_disc_obs_steps`（env yaml） |

常用配置文件：
- `data/agents/amp_humanoid_agent.yaml`（纯模仿）
- `data/agents/amp_task_humanoid_agent.yaml`（任务+模仿）

## 4. 与论文一致点与实现差异

一致点：
- style prior 来自判别器而非手工风格奖励。
- 核心目标仍是 `task + style`。
- 使用时序状态转移特征而非必须依赖同步相位。

实现差异（需要明确）：
- 论文展示了LSGAN表达；仓库判别器采用 BCE-logits 实现。
- 奖励变换也采用工程化的 `-log(1-sigmoid(logit))`，不完全等式照搬论文样式。

## 5. 读代码时的核对清单
- `task_reward_weight` 与 `disc_reward_weight` 是否符合实验目标。
- `disc_grad_penalty`、replay、normalizer 是否启用（稳定性关键）。
- `num_disc_obs_steps` 与动作数据复杂度是否匹配。
