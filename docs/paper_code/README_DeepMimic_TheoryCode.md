# DeepMimic：论文-代码精细对照（更新版）

## 1. 论文核心问题与目标
参考论文：DeepMimic (TOG 2018)
- arXiv: https://arxiv.org/abs/1804.02717

DeepMimic将控制问题写成标准RL目标：
\[
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_t\gamma^t r_t\right]
\]
并把每步奖励拆成模仿项与任务项：
\[
r_t=\omega^I r_t^I + \omega^G r_t^G
\]
论文还强调了三点对动态技能学习很关键：
- phase variable（用于与参考动作同步）
- RSI（Reference State Initialization）
- ET（Early Termination）

在论文实验中，任务版常见设定示例是 \(\omega^I=0.7\), \(\omega^G=0.3\)。

## 2. 关键机制在 MimicKit 的代码映射

### 2.1 模仿奖励（论文） -> `compute_reward`（实现）
核心实现：`mimickit/envs/deepmimic_env.py:788`

实现形式是“误差指数化 + 加权和”，与论文一致：
- `pose_r = exp(-pose_scale * pose_err)`
- `vel_r = exp(-vel_scale * vel_err)`
- `root_pose_r = exp(-root_pose_scale * (...))`
- `root_vel_r = exp(-root_vel_scale * (...))`
- `key_pos_r = exp(-key_pos_scale * key_pos_err)`
- 总奖励：`pose_w*pose_r + vel_w*vel_r + root_pose_w*... + root_vel_w*... + key_pos_w*...`

### 2.2 观测构造：phase与目标提示
- 观测入口：`mimickit/envs/deepmimic_env.py:330` (`_compute_obs`)
- 观测组装：`mimickit/envs/deepmimic_env.py:681` (`compute_deepmimic_obs`)
- 目标帧采样：`mimickit/envs/deepmimic_env.py:564` (`_fetch_tar_obs_data`)

对照关系：
- 论文 phase variable -> `enable_phase_obs` + motion phase计算
- 论文目标姿态提示 -> `enable_tar_obs` + `tar_obs_steps`

### 2.3 RSI（Reference State Initialization）
- 采样参考时间：`mimickit/envs/deepmimic_env.py:276` (`_sample_motion_times`)
- reset到参考动作状态：`mimickit/envs/deepmimic_env.py:174` (`_reset_ref_motion`)
- 常用开关：`data/envs/deepmimic_*.yaml` 的 `rand_reset`

### 2.4 ET（Early Termination）
- done更新入口：`mimickit/envs/deepmimic_env.py:463` (`_update_done`)
- 终止规则：`mimickit/envs/deepmimic_env.py:725` (`compute_done`)

主要包含：
- 非白名单部位触地失败
- 姿态偏差失败（`pose_termination`）
- 动作到结尾成功（非wrap动作）

### 2.5 优化器
- PPO：`mimickit/learning/ppo_agent.py`（clip objective）
- AWR：`mimickit/learning/awr_agent.py`（advantage-weighted）

这对应仓库里 DeepMimic 可切换 PPO/AWR 的设计（`args/deepmimic_humanoid_ppo_args.txt` 与 `args/deepmimic_humanoid_awr_args.txt`）。

### 2.6 论文中的多动作整合 vs 本仓库路径
论文讨论了多种多动作整合方式（如 multi-clip reward、composite policy）。

MimicKit 主路径更偏工程统一配置：
- 通过 `motion_file` 直接指向单动作 `.pkl`
- 或指向 dataset `.yaml`（多动作加权采样）

对应数据读取：`mimickit/anim/motion_lib.py:251`。

## 3. 论文符号 -> 配置项映射

| 论文概念 | 含义 | MimicKit 对应 |
|---|---|---|
| \(\omega^I\), \(\omega^G\) | 模仿/任务权重 | DeepMimic里多通过奖励分项与任务环境共同体现 |
| 分项权重 \(w_i\) | 奖励项重要性 | `reward_pose_w`, `reward_vel_w`, `reward_root_pose_w`, `reward_root_vel_w`, `reward_key_pos_w` |
| 分项尺度 \(\alpha_i\) | 指数惩罚斜率 | `reward_pose_scale`, `reward_vel_scale`, `reward_root_pose_scale`, `reward_root_vel_scale`, `reward_key_pos_scale` |
| RSI | 参考状态初始化 | `rand_reset` + `_sample_motion_times` |
| ET | 提前终止 | `enable_early_termination`, `pose_termination`, `pose_termination_dist` |
| PPO clip | 策略更新约束 | `ppo_clip_ratio` |
| GAE/TD(\(\lambda\)) | 优势估计 | `td_lambda` |

## 4. 与论文一致点与实现差异

一致点：
- 核心仍是“模仿奖励 + 任务奖励”的RL框架。
- RSI/ET 在训练稳定性中的作用与论文叙述一致。
- 模仿奖励采用指数误差项并加权融合。

实现差异（工程化）：
- 奖励项命名与论文符号不完全同名（如 `root_pose`、`key_pos` 等工程字段）。
- 论文中多动作整合方法在仓库里统一为 `motion_file` 数据驱动配置，不强调单独的“composite policy模块”。

## 5. 读代码时的核对清单
- 奖励是否由 `compute_reward` 主导，且权重/尺度与yaml匹配。
- 训练是否开启 RSI/ET（对高动态动作影响很大）。
- 任务实验是否显式引入 `r^G`（例如 task env），避免只训练纯模仿。
