---
name: mimickit-gpu-efficiency-optimization
description: Diagnose why GPU memory/utilization differs across MimicKit cases and apply stable, high-efficiency tuning actions on dual-4090 Newton runs.
---

# MimicKit GPU Efficiency Optimization Skill

## Goal

解决两个核心问题：

1. 为什么不同案例在同一台机器上会出现不同的显存/利用率表现。
2. 如何在稳定前提下更高效利用机器（吞吐、利用率、连续运行）。

## When To Use

适用于以下场景：

1. 双卡训练中，有的 case 显存接近满但 util 不高。
2. 有的 case util 一直不高，不确定是否需要调参。
3. 需要在“稳定性 / 吞吐 / 利用率”之间做可执行取舍。

## Inputs

优先使用结构化输出：

1. `output/train/case_gpu_bench_*/best_by_case.tsv`
2. `output/train/case_longcycle_*/best_by_case.tsv` 或 `output/train/case_ultralong_*/best_by_case.tsv`
3. `/tmp/mk_*monitor*.log`（运行态监控）

## Step 1: Quick Classification

先做 case 分类（按 long 或 benchmark 的稳定阶段统计）：

1. `VRAM-bound mixed`：
- 典型特征：`max_mem ~ 23-24GB`，`min_avg_util` 中等（约 `45-55%`）。
- 代表：`pi_plus` 三例。

2. `Compute-heavy`：
- 典型特征：显存中等，但 util 高（`>=60%`）。
- 代表：`ase_humanoid*`、`amp_smpl`。

3. `Throughput/Env-bound`：
- 典型特征：显存不高（常 `<8GB`），util 偏低（常 `<45%`）。
- 代表：`add_go2`、`deepmimic_go2`、`vault_g1`。

## Step 2: Root-Cause Checklist

按顺序排查：

1. 环境成本：
- 是否是接触/约束更重的场景。

2. 算法成本：
- `deepmimic/ppo` vs `amp/add/ase`（是否有 `disc` / `enc` 额外开销）。

3. Agent 强度：
- 默认配置通常 `2-layer + update_epochs=5 + batch_size=4`。
- hiutil 常见 `3-layer + update_epochs=20/40 + batch_size=2`。

4. 多卡同步：
- 轻量更新任务更容易被通信成本稀释 util。

5. 阶段口径：
- 不要混用 `probe/test/viz` 去判断 long 训练瓶颈。

## Step 3: Tuning Policy (Decision Table)

1. 如果显存已接近满载（`>23GB`）：
- 不再盲目抬 `num_envs`。
- 优先稳定档位（防 OOM/NCCL）。

2. 如果 util 低且显存也低（`<8GB`）：
- 优先把目标改成“整机吞吐最大化”。
- 可考虑双卡拆分成两条单卡任务并行（不同 case）。

3. 如果 util 已高（`>=60%`）：
- 以稳定连续训练为主，不建议频繁改配置。

4. 若频繁 OOM/NCCL：
- 先降 `num_envs` 档位，再谈 util 优化。

## Validated Host Settings (Dual 4090 + Newton)

### Runtime Env (required)

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

### Pi-plus Stable Picks

1. `add_pi_plus_args.txt`: `hiutil e40`
2. `amp_pi_plus_args.txt`: `hiutil e38`
3. `deepmimic_pi_plus_ppo_args.txt`: `hiutil e40`

## Fast Commands

### A) 分类统计（从 latest benchmark root）

```bash
python - <<'PY'
import csv,glob,os
root=sorted(glob.glob('output/train/case_gpu_bench_*'))[-1]
path=os.path.join(root,'best_by_case.tsv')
rows=list(csv.DictReader(open(path),delimiter='\t'))
ok=[r for r in rows if r.get('status')=='ok']
for r in sorted(ok,key=lambda x:float(x.get('min_avg_util',0))):
    mem=max(float(r.get('max_mem0',0)),float(r.get('max_mem1',0)))
    print(r['case'],r['method'],'util',r['min_avg_util'],'max_mem',int(mem),'variant',r['variant'],'e',r['num_envs'])
PY
```

### B) 运行态统计（从 monitor log 聚合）

```bash
python - <<'PY'
import re
from collections import defaultdict
log='/tmp/mk_ultralong_monitor_20260217_134108.log'
cur={'u0':None,'u1':None,'c':None}
agg=defaultdict(lambda:[0,0,0])
for ln in open(log,errors='ignore'):
    s=ln.strip()
    m=re.match(r'^0,\\s*(\\d+),',s)
    if m: cur['u0']=int(m.group(1)); continue
    m=re.match(r'^1,\\s*(\\d+),',s)
    if m: cur['u1']=int(m.group(1)); continue
    if s.startswith('current_case '):
        c=s.split()[1]
        if cur['u0'] is not None and cur['u1'] is not None:
            agg[c][0]+=1; agg[c][1]+=cur['u0']; agg[c][2]+=cur['u1']
for c,(n,u0,u1) in agg.items():
    print(c,'n',n,'avg_u0',round(u0/n,2),'avg_u1',round(u1/n,2))
PY
```

## Acceptance Criteria

应用该 skill 后，至少满足：

1. 失败率可控：
- 目标配置下 `oom_or_nccl` 明显减少，或被稳定档位规避。

2. 指标可解释：
- 每个 case 能被归入上述分类之一，并能解释“为何如此”。

3. 机器利用策略清晰：
- 明确当前优先级是稳定、吞吐还是 util，并有对应动作。

## References

1. `docs/benchmarks/README_MimicKit_GPUUtilizationPatterns.md`
2. `docs/benchmarks/README_MimicKit_GPUCaseBenchmark.md`
3. `docs/benchmarks/README_GPU_AgentProfiles.md`
