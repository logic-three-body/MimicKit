# DeepMimic：理论-代码精细对照

## 1. 论文主问题与目标
参考论文：DeepMimic (TOG 2018)  
- 论文链接：https://arxiv.org/abs/1804.02717

论文把控制问题建模为 RL：
\[
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_t\gamma^t r_t\right]
\]
并将即时奖励拆成“模仿目标 + 任务目标”：
\[
r_t=\omega^I r_t^I + \omega^G r_t^G
\]
其中模仿项采用指数型跟踪误差（论文中是 pose/velocity/end-effector/center-of-mass 等项）。

## 2. 关键论文机制在 MimicKit 的对应

### 2.1 模仿奖励分解（论文） vs `compute_reward`（实现）
论文中的核心思想是每个跟踪误差项经过指数映射后再加权求和。MimicKit 在 `mimickit/envs/deepmimic_env.py:788` 的 `compute_reward` 完整实现了该思路：
- `pose_r = exp(-pose_scale * pose_err)`
- `vel_r = exp(-vel_scale * vel_err)`
- `root_pose_r = exp(-root_pose_scale * (...))`
- `root_vel_r = exp(-root_vel_scale * (...))`
- `key_pos_r = exp(-key_pos_scale * key_pos_err)`
- 最终线性加权 `r = pose_w*... + vel_w*... + ...`

这与论文“指数化误差 + 加权组合”一一对应，只是 MimicKit 的分项命名更工程化（`root_pose/root_vel/key_pos`）。

### 2.2 观测构造：相位与目标未来帧
论文强调策略既要看到当前动力学状态，也要有足够的运动上下文。MimicKit 对应在：
- `mimickit/envs/deepmimic_env.py:330` `_compute_obs`
- `mimickit/envs/deepmimic_env.py:681` `compute_deepmimic_obs`
- `mimickit/envs/deepmimic_env.py:564` `_fetch_tar_obs_data`

对照关系：
- 论文中的 motion phase -> `enable_phase_obs` + `calc_motion_phase`
- 论文中的目标姿态/未来轨迹提示 -> `enable_tar_obs` + `tar_obs_steps`

### 2.3 RSI（Reference State Initialization）
论文将 RSI 作为动态技能学习关键机制之一。MimicKit 对应路径：
- motion 时间采样：`mimickit/envs/deepmimic_env.py:276` `_sample_motion_times`
- reset 到参考轨迹状态：`mimickit/envs/deepmimic_env.py` 中 `_reset_ref_motion`（由 `_reset_char` 调用）

配置开关：`data/envs/deepmimic_humanoid_env.yaml` 的 `rand_reset`。

### 2.4 ET（Early Termination）
论文将 ET 作为避免坏局部最优（倒地“装样子”）的重要设计。MimicKit 对应：
- 终止逻辑入口：`mimickit/envs/deepmimic_env.py:463` `_update_done`
- 终止规则实现：`mimickit/envs/deepmimic_env.py:725` `compute_done`

包含：
- 触地失败（非接触白名单）
- 姿态偏差失败（`pose_termination`）
- 动作播放结束成功（非 wrap）

### 2.5 PPO / AWR 优化
论文主实验使用 PPO。MimicKit 同时支持 PPO 与 AWR：
- PPO：`mimickit/learning/ppo_agent.py:198`（clip surrogate）
- AWR：`mimickit/learning/awr_agent.py:201`（advantage-weighted log-prob）

这对应了 README 中“同一环境可切换不同 RL 优化器”的设计。

## 3. 论文符号 -> 代码参数映射

| 论文概念/符号 | 含义 | MimicKit 参数/位置 |
|---|---|---|
| \(\omega^I\), \(\omega^G\) | 模仿/任务权重 | DeepMimic 环境通常由奖励分项直接编码；AMP/ASE/ADD 显式使用 `task_reward_weight` |
| 跟踪分项权重 \(w_i\) | 各误差项重要性 | `data/envs/deepmimic_humanoid_env.yaml`: `reward_pose_w` `reward_vel_w` `reward_root_pose_w` `reward_root_vel_w` `reward_key_pos_w` |
| 跟踪分项尺度 \(\alpha_i\) | 指数惩罚斜率 | `reward_pose_scale` `reward_vel_scale` `reward_root_pose_scale` `reward_root_vel_scale` `reward_key_pos_scale` |
| RSI | 参考状态初始化 | `rand_reset` + `_sample_motion_times` |
| ET | 提前终止 | `enable_early_termination` `pose_termination` `pose_termination_dist` |
| PPO clip | 策略更新约束 | `data/agents/deepmimic_humanoid_ppo_agent.yaml`: `ppo_clip_ratio` |
| GAE/TD(lambda) | 优势与目标值估计 | `td_lambda`（PPO/AWR 共用） |

## 4. 训练链路（代码执行顺序）
1. `mimickit/run.py:95` `run` 读取 `env_config + agent_config`。
2. `deepmimic_env` 每步更新参考帧、观测、奖励、终止。
3. `ppo_agent` 或 `awr_agent` 收集 rollout，计算 advantage/target。
4. 策略与价值网络迭代更新。

## 5. 你在读代码时应重点核对的点
- 奖励是否真的是“指数误差 + 加权和”（`compute_reward`）。
- 训练是否打开 RSI/ET（很多动态动作成败关键）。
- 任务版实验是否正确设置了 task reward（例如 heading/location 任务 env）。
