# AMP：理论-代码精细对照

## 1. 论文主问题与目标
参考论文：AMP (TOG 2021)  
- 论文链接：https://arxiv.org/abs/2104.02180

AMP 的核心是把“风格”从手工 reward 中剥离出来，用判别器学习 style prior：
\[
r_t = w_G r_t^G + w_S r_t^S
\]
- \(r_t^G\)：任务目标（what）
- \(r_t^S\)：风格目标（how），由 adversarial discriminator 给出

## 2. 论文关键公式与 MimicKit 对照

### 2.1 判别器与风格奖励
论文采用 state-transition 判别器（而非必须 state-action），并给策略风格奖励。MimicKit 对应：
- 判别器网络：`mimickit/learning/amp_model.py:12` `eval_disc`
- 判别器训练：`mimickit/learning/amp_agent.py:130` `_compute_disc_loss`
- 风格奖励计算：`mimickit/learning/amp_agent.py:209` `_calc_disc_rewards`
- 总奖励融合：`mimickit/learning/amp_agent.py:101` `_compute_rewards`

实现说明：
- 论文正文给出 LSGAN 版本与 GP。
- MimicKit 采用 BCE-logits + reward shaping（`-log(1-sigmoid(logit))`）实现同类 adversarial reward 信号。

### 2.2 判别器输入特征 \(\Phi(s_t,s_{t+1})\)
论文强调判别器观测需要包含速度/姿态等运动学关键信息。MimicKit 对应：
- 轨迹片段缓存：`mimickit/envs/amp_env.py:94` `_build_disc_obs_buffers`
- 在线更新片段：`mimickit/envs/amp_env.py:194` `_update_disc_obs`
- demo 片段生成：`mimickit/envs/amp_env.py:63` `_fetch_disc_demo_data`
- feature 组装：`mimickit/envs/amp_env.py:328` `compute_disc_obs`

`num_disc_obs_steps` 直接对应论文的“时序状态转移窗口”。

### 2.3 稳定训练：Replay + Normalizer + GP
论文强调 adversarial 训练稳定性。MimicKit 的稳定化路径：
- replay buffer：`_store_disc_replay_data` (`mimickit/learning/amp_agent.py:86`)
- 判别器观测归一化：`_build_normalizers` (`mimickit/learning/amp_agent.py:44`)
- gradient penalty：`_compute_disc_loss` 内 `disc_grad_penalty` (`mimickit/learning/amp_agent.py:155`)

### 2.4 AMP 与 DeepMimic 的目标差异
- DeepMimic：对齐特定参考帧（强同步 tracking）。
- AMP：匹配行为分布（style prior），不要求逐帧严格同步。

这也是为什么 AMP 的 env 默认 `enable_tar_obs: False`，而 DeepMimic 常开 `enable_tar_obs`。

## 3. 公式/概念 -> 配置映射

| 论文概念 | 含义 | MimicKit 配置 |
|---|---|---|
| \(w_G\) | 任务奖励权重 | `task_reward_weight` |
| \(w_S\) | 风格奖励权重 | `disc_reward_weight` |
| 判别器 batch | 对抗训练规模 | `disc_batch_size` |
| replay 采样 | 历史策略样本注入 | `disc_buffer_size`, `disc_replay_samples` |
| GP 系数 | 判别器平滑约束 | `disc_grad_penalty` |
| 判别器正则 | 防止过拟合/爆炸 | `disc_logit_reg`, `disc_weight_decay` |
| style reward 缩放 | 风格信号强度 | `disc_reward_scale` |
| 时序窗口 | 判别器观测长度 | `data/envs/amp_*.yaml` 的 `num_disc_obs_steps` |

主要文件：`data/agents/amp_humanoid_agent.yaml`, `data/agents/amp_task_humanoid_agent.yaml`。

## 4. 训练链路（实现顺序）
1. env 提供 `disc_obs`（策略轨迹）和 `fetch_disc_obs_demo`（参考轨迹）。
2. agent 组装 batch：policy obs + demo obs + replay obs。
3. 更新 discriminator（BCE + GP + regularization）。
4. 用 disc logit 生成 style reward，与 task reward 加权。
5. PPO 更新 actor/critic。

## 5. 代码阅读重点
- 判别器输入是否包含速度/旋转/末端信息（`compute_disc_obs`）。
- `task_reward_weight` 与 `disc_reward_weight` 是否按实验目的设置。
- GP 和 replay 是否打开（决定稳定性）。
