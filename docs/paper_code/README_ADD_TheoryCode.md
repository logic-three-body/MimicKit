# ADD：理论-代码精细对照

## 1. 论文主问题与目标
参考论文：ADD (SIGGRAPH Asia 2025)  
- 论文预印本：https://arxiv.org/abs/2505.04961

ADD 把多目标优化从“手工加权和”改成“对抗差分聚合”。

传统形式：
\[
\min_\theta \sum_i w_i l_i(\theta)
\]
ADD 形式：把各目标误差拼成差分向量
\[
\Delta = [l_1(\theta),...,l_n(\theta)]
\]
再做 adversarial min-max：
\[
\min_\theta \max_D \; \log D(0) + \log(1-D(\Delta)) - \lambda_{GP}L_{GP}
\]
其中正样本只有一个：\(\Delta=0\)（理想零误差）。

## 2. 从论文到 MimicKit 的实现对照

### 2.1 差分向量构造（核心）
论文 motion imitation 里定义：\(\Delta_t = \phi(\hat s_t) \ominus \phi(s_t)\)。

MimicKit 对应：
- policy 分支观测：`mimickit/envs/add_env.py:33` `_update_disc_obs`
- demo 分支观测：`mimickit/envs/add_env.py:67` `_update_disc_obs_demo`
- 差分计算：`mimickit/learning/add_agent.py:55` `obs_diff = disc_obs_demo - disc_obs`

### 2.2 “单一正样本”判别器训练
论文强调正样本仅用零向量。MimicKit 完整实现：
- 零向量模板：`_build_pos_diff` (`mimickit/learning/add_agent.py:21`)
- 正样本 logit：`disc_pos_logit = eval_disc(pos_diff)`
- 负样本 logit：来自 `diff_obs = tar_disc_obs - disc_obs`
- 组合损失：`_compute_disc_loss` (`mimickit/learning/add_agent.py:74`)

### 2.3 梯度惩罚（对负样本）
论文在 ADD 中强调 GP 对稳定性关键，且重点作用于负样本差分。MimicKit 对应：
- `mimickit/learning/add_agent.py:107` 开始对 `norm_diff_obs` 求梯度
- `disc_loss += disc_grad_penalty * ...`

### 2.4 差分归一化（DiffNormalizer）
ADD 的关键工程难点是各目标量纲差异。MimicKit 专门引入：
- `mimickit/learning/diff_normalizer.py`
- `record` 统计绝对值均值
- `normalize` 用 `x / mean_abs` 做缩放并裁剪

agent 中启用位置：
- `mimickit/learning/add_agent.py:27` `_build_normalizers`

### 2.5 策略奖励
论文中策略奖励为 \(-\log(1-D(\Delta_t))\)。MimicKit 对应：
- `mimickit/learning/add_agent.py:50` `_compute_rewards`
- `mimickit/learning/add_agent.py:59` `_calc_disc_rewards`

实现上与 AMP 的 logit->prob 转换保持一致，再乘 `disc_reward_scale`。

## 3. 特征映射 \(\phi(\cdot)\) 与代码
论文中的 \(\phi\) 是“用于比较参考/当前状态的特征提取”。MimicKit 对应：
- `mimickit/envs/add_env.py:101` `compute_pos_obs`
- `mimickit/envs/add_env.py:137` `compute_disc_vel_obs`
- `mimickit/envs/add_env.py:154` `compute_disc_obs`

包含 root/global pose、joint rotation、body position、vel 等，最后 flatten 成判别器输入。

## 4. 论文项 -> 配置映射

| 论文项 | 含义 | MimicKit 参数 |
|---|---|---|
| \(\lambda_{GP}\) | GP 强度 | `disc_grad_penalty` |
| 判别器正则 | 控制过拟合/陡峭边界 | `disc_logit_reg`, `disc_weight_decay` |
| 对抗损失权重 | 判别器在总 loss 中权重 | `disc_loss_weight` |
| 差分奖励缩放 | policy 训练信号强度 | `disc_reward_scale` |
| 任务/风格融合 | 是否混入外部 task reward | `task_reward_weight`, `disc_reward_weight` |
| replay 样本 | 稳定判别器训练 | `disc_buffer_size`, `disc_replay_samples`, `disc_batch_size` |

主要配置：`data/agents/add_humanoid_agent.yaml`。

## 5. ADD 与 AMP 的关键区别（代码视角）
- AMP 判别器输入：`disc_obs` 与 `demo_obs` 分别分类。  
- ADD 判别器输入：`demo_obs - disc_obs` 的差分向量，仅以零向量作为正样本。  

对应代码差异：
- AMP：`mimickit/learning/amp_agent.py:130`  
- ADD：`mimickit/learning/add_agent.py:74`

## 6. 训练链路（实现顺序）
1. env 同时输出 policy 与 demo 判别观测。  
2. agent 构造差分向量并更新差分归一化器。  
3. discriminator 以 `0`（正）和 `diff`（负）训练，含 GP。  
4. 用判别器输出生成 reward，回传给 PPO 主干。  
