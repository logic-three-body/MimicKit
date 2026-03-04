# Checkpoint: 2026-02-24 17:15:09 (CST) - Ultralong 长训接力交接

## 1) 当前目标
- 任务：执行 MimicKit 全案例 ultralong 流程（train->infer->viz），root 为 `output/train/case_ultralong_alloc_full_20260219_154546`。
- 策略：`scripts/run_case_longcycle.py` + `--long-mode time_budget --long-budget-hours 24`，trainable 双卡，nontrainable 单卡。

## 2) 当前现场状态（已验证）
- tmux 会话仍在：
  - `mk_longcycle_resume_20260224_151049`（主编排）
  - `mk_progress_dashboard_8787`（Web 监控）
  - `mk_dualgpu_live_monitor`（GPU 监控日志）
- 活跃进程：
  - `python -u scripts/run_case_longcycle.py ... --root-out case_ultralong_alloc_full_20260219_154546 --resume-skip-status ok`
  - 子进程：`mimickit/run.py` 当前在 `add_go2_args` 的 long_train。
  - `python -u scripts/run_progress_dashboard.py --root-out case_ultralong_alloc_full_20260219_154546 --monitor-log /tmp/mk_dualgpu_follow_until_high_20260224_151139.log --host 0.0.0.0 --port 8787`
- 最新进度：`output/train/case_ultralong_alloc_full_20260219_154546/progress.json`
  - `done=1, total=32, last_case=add_go2_args.txt, status=running_case`
- 当前瞬时 GPU（示例快照）：
  - GPU0 ~57%，GPU1 ~66%，显存约 5.1GiB / 3.2GiB（非满载但双卡都在工作）。

## 3) 已完成与未完成
- 已完成：
  - `add_g1_args` 已 `final_ok=1`（成功跑完 long/test/viz）。
- 进行中：
  - `add_go2_args`（多档位失败后继续尝试，当前在 long 阶段）。
- 未开始：
  - 其余 30 个 case。

## 4) 未解决问题（新 agent 需要优先关注）
1. 长周期推进很慢，当前仅 1/32 完成。
- 原因：每个 trainable 案例预算 24h，且部分案例出现多次 `long_nccl`，会触发 ladder fallback，拉长总时长。

2. `add_go2_args` / `add_humanoid_args` 出现多次 `long_nccl`。
- 证据：
  - `output/train/case_ultralong_alloc_full_20260219_154546/runs/add_go2_args/attempts.json`
  - `output/train/case_ultralong_alloc_full_20260219_154546/runs/add_humanoid_args/attempts.json`

3. 监控信息存在“尝试记录滞后”现象。
- `mimickit/run.py` 实际可能在恢复较早 attempt（如 `hiutil_e1024`），
  但监控脚本按 `attempts.json` 最后一条显示，可能显示为旧条目（如 `default_e32 long_nccl`）。
- 结论：监控页 `variant/e/note` 可能偶发不代表当前活跃 worker；需结合进程命令行核对。

4. `best_by_case.tsv` 在脚本末尾统一写出，运行中不存在。
- 这是当前实现行为，不是文件丢失。
- 影响：dashboard 在中途无法给出完整 summary（只可靠 `progress.json` + monitor log）。

## 5) 关键路径与文件导航（给 0 记忆 agent）
- 主脚本：`scripts/run_case_longcycle.py`
- 短训资源画像：`scripts/run_case_gpu_bench.py`
- 实时监控站：`scripts/run_progress_dashboard.py`
- 关键输出 root：`output/train/case_ultralong_alloc_full_20260219_154546`
- allocation profile：`output/train/case_gpu_alloc_full_20260219_154546/allocation_profile.tsv`
- 主编排日志：`/tmp/case_ultralong_alloc_full_20260219_154546.log`
- GPU 监控日志：`/tmp/mk_dualgpu_follow_until_high_20260224_151139.log`
- 监控脚本：`/tmp/mk_dualgpu_live_monitor_20260224_151139.sh`
- 相关 skill：
  - `docs/skill/mimickit-progress-dashboard-skill/SKILL.md`
  - `docs/skill/mimickit-allcase-longcycle-skill/SKILL.md`
  - `docs/skill/mimickit-multicase-gpu-util-skill/SKILL.md`

## 6) 0 记忆 agent 接手 SOP（最小命令集）
1. 先确认现场
```bash
cd /root/Project/MimicKit
tmux ls
pgrep -af 'run_case_longcycle.py|mimickit/run.py|run_progress_dashboard.py'
cat output/train/case_ultralong_alloc_full_20260219_154546/progress.json
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
```

2. 如主流程挂了，按原参数恢复（不要新建 root）
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit
python -u scripts/run_case_longcycle.py \
  --engine-config data/engines/newton_engine.yaml \
  --devices-train cuda:0,cuda:1 \
  --include-nontrainable \
  --allocation-profile-tsv output/train/case_gpu_alloc_full_20260219_154546/allocation_profile.tsv \
  --allocation-fallback ladder \
  --long-mode time_budget \
  --long-budget-hours 24 \
  --long-budget-signal SIGINT \
  --long-budget-grace-sec 300 \
  --long-success-policy budget_checkpoint \
  --root-out case_ultralong_alloc_full_20260219_154546 \
  --resume-skip-status ok
```

3. 如 dashboard 挂了，恢复
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mimickit
cd /root/Project/MimicKit
python -u scripts/run_progress_dashboard.py \
  --root-out case_ultralong_alloc_full_20260219_154546 \
  --monitor-log /tmp/mk_dualgpu_follow_until_high_20260224_151139.log \
  --host 0.0.0.0 --port 8787
```
- 页面：`http://<host>:8787/`

4. 安全停止（如必须）
```bash
pkill -INT -f 'scripts/run_case_longcycle.py --engine-config data/engines/newton_engine.yaml'
```
- 先 `SIGINT`，避免直接 `kill -9` 破坏 stage 状态。

## 7) 建议下一步（优先级）
1. 继续跑当前 root 至至少 1~2 个新增 case 完成，确认 resume 路径稳定。
2. 修复监控“活跃 attempt 识别滞后”：
- 优先从运行中的 `mimickit/run.py --out_dir` 反推当前 variant/env，而不是仅读 `attempts.json` 最后一条。
3. 增量可观测性优化：
- 在 `run_case_longcycle.py` 中增加运行中 `best_by_case.partial.tsv`（每完成一个 case 刷新一次），减少中途盲区。
4. 若 `long_nccl` 在特定 case 持续复现：
- 对该 case 固定到已知稳定 env 档位，记录到 allocation profile 或 case override，优先保证流程连贯性。

## 8) 仓库脏状态提醒
当前 `main` 分支存在未提交修改（接手前先 `git status --short`）：
- `scripts/run_case_e2e.py`
- `scripts/run_case_gpu_bench.py`
- `scripts/run_progress_dashboard.py`
- `scripts/summarize_case_gpu_bench.py`
- `tools/ue_bridge/discover_all_cases.py`
- `tools/ue_bridge/export_obs_fixture.py`
- `tools/ue_bridge/run_mimic_visual_case.py`

建议：接手 agent 先“读状态、不中断、最小改动”，避免在不清楚改动来源时覆盖已有工作。
