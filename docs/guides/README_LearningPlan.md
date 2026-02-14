# MimicKit 学习方案（论文-代码对照版）

## 1. 目标
- 建立一条可追踪链路：`论文公式 -> MimicKit函数 -> YAML参数 -> 训练结果`。
- 在同一框架下理解 `DeepMimic / AMP / ASE / ADD` 的目标差异与实现差异。

## 2. 统一入口（先读）

| 主题 | 代码入口 | 作用 |
|---|---|---|
| 运行主入口 | `mimickit/run.py` | 解析参数、构建 env/agent、启动 train/test |
| 环境分发 | `mimickit/envs/env_builder.py` | `env_name` -> `deepmimic/amp/ase/add/...` |
| 算法分发 | `mimickit/learning/agent_builder.py` | `agent_name` -> `PPO/AWR/AMP/ASE/ADD` |
| 动作数据 | `mimickit/anim/motion_lib.py` | 读取 motion clip/dataset，采样与插值 |

建议先用 `view_motion` 验证动作数据与引擎可用：

```bash
python mimickit/run.py --mode test --arg_file args/view_motion_humanoid_args.txt --visualize true
```

## 3. 方法级精细文档（论文已展开到公式级）

- DeepMimic：`docs/paper_code/README_DeepMimic_TheoryCode.md`
- AMP：`docs/paper_code/README_AMP_TheoryCode.md`
- ASE：`docs/paper_code/README_ASE_TheoryCode.md`
- ADD：`docs/paper_code/README_ADD_TheoryCode.md`
- 文档索引：`docs/paper_code/README.md`

这些文档已按以下结构展开：
- 论文目标函数与关键机制（含符号级解释）
- MimicKit 对应函数（到文件与函数粒度）
- 论文变量到 YAML 参数映射
- 论文实现与仓库实现的一致点/实现差异

## 4. 四方法快速对照

| 方法 | 论文核心目标 | 关键代码 |
|---|---|---|
| DeepMimic | 指数型跟踪奖励 + PPO，依赖 RSI/ET 提升动态动作学习 | `mimickit/envs/deepmimic_env.py`, `mimickit/learning/ppo_agent.py` |
| AMP | 任务奖励 + 对抗风格奖励（motion prior） | `mimickit/envs/amp_env.py`, `mimickit/learning/amp_agent.py` |
| ASE | 对抗模仿 + 互信息技能发现 + 多样性约束 | `mimickit/learning/ase_agent.py`, `mimickit/learning/ase_model.py` |
| ADD | 差分向量判别器（零向量正样本）替代手工多项加权 | `mimickit/learning/add_agent.py`, `mimickit/envs/add_env.py` |

## 5. 统一实验记录模板

| 论文机制 | 代码位置 | 配置项 | 改动 | 预期 | 实际 |
|---|---|---|---|---|---|
| 示例：AMP 的 task/style 权衡 | `mimickit/learning/amp_agent.py` | `task_reward_weight`, `disc_reward_weight` | 0.5/0.5 -> 0.8/0.2 | 任务更强、风格可能下降 | 待填 |
