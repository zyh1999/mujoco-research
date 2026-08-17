Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-DUAL5060-SWIMMER-RERUN-20260817-03

# 唯一目标

在保持 M1/M2、严格匹配矩阵和历史 provenance 不变的前提下，修复 dual-5060 环境缺失 `mujoco_py` 的基础设施问题，并仅补跑当前 35-cell batch 中失败的 5 个 Swimmer-v3 cells 至既定的 `9,994,240` steps。

# 已验证起始状态

- CSF3 当前无 live MuJoCo job。
- Bede array `1072326_0-17` scheduler-complete，但科学 artifacts 当前不可访问；本任务不补跑、不据此作科学结论。
- dual-5060 batch：`30/35` 已完成到 `9,994,240` steps。
- 剩余 5 个均为 Swimmer-v3 dependency failures，根因是缺少 `mujoco_py`。
- 两张 5060 GPU 在状态采集时均空闲，但执行前必须刷新。
- 历史失败 `18302268_10` 仍未映射；必须保留为 unresolved historical provenance，不得猜测归属。

# 范围与不变量

1. 仅处理已失败的 5 个 Swimmer-v3 cells，不新增环境、方法、seed、预算或超参。
2. 从 batch manifest、失败日志和现有结果表恢复 5 个 cell 的 exact：
   - M1/M2 身份
   - method/baseline 身份
   - seed
   - config
   - code commit
   - environment/version
   - command
   - output path
3. 除使 Swimmer-v3 可加载所必需的依赖修复外，所有字段必须与原 batch 严格一致。
4. M1 与 M2 分开记录、分开验证；禁止跨线替代或汇总为同一方法。
5. CSF3 是决策控制平面；计算优先使用当前空闲的 dual-5060。
6. 禁止 `.54` / `ws4090-31`。

# 执行步骤与允许动作

## A. 执行前刷新

- 先读取最新 `.agent/GOAL.md`、`STATE.md`、`TASK.md`、`AGENT_REPORT.md`。
- 刷新 CSF3、Bede 和 dual-5060 的 scheduler、GPU、进程、日志、reward/KL、artifact 与错误扫描。
- 核实没有重复运行这 5 个 cells，且两张 5060 仍可用。
- 记录开始 HEAD、工作树和时间戳；保留所有无关改动。

## B. 恢复严格匹配清单

为 5 个失败 cell 建立 immutable launch table，逐项对照原 35-cell manifest及成功 matched cells。任何关键字段无法恢复时：

- 不得猜测或启动该 cell；
- 标记 `BLOCKED-METADATA`;
- 报告缺失字段及查找证据。

## C. 最小依赖修复与 preflight

- 从仓库 lockfile、既有成功环境或历史日志确定兼容的 `mujoco_py`、MuJoCo、Gym、Python及相关依赖版本。
- 仅在隔离、可回滚环境中安装缺失依赖；不得宽泛升级依赖或修改共享环境。
- 在 M1、M2 各执行一次无学习 Swimmer-v3 smoke test：import、env construction、reset、少量 step、模型初始化。
- smoke test 必须记录版本、输出 shape、API 行为和 exit code。
- 任一线 preflight 失败，则不得启动该线的正式 cells。

## D. 补跑

- 仅启动通过 metadata 核验和 preflight 的失败 cells。
- 使用原始命令语义、seed、预算、commit、日志及 artifact 命名规则。
- 在两张 5060 上安全分配，但不得改变单 run 资源语义来追求并行度。
- 每个 run 必须从头按原语义重跑；除非原配置明确允许且存在属于该 cell 的有效 checkpoint，否则禁止拼接或借用 checkpoint。
- 监控至完成、明确失败或本轮时间边界；不得启动额外 sweep。

# 必需证据

- 执行前后的 scheduler/GPU/process 快照。
- 5-cell immutable launch table及其 manifest/log来源。
- dependency 修复前后的精确 package diff。
- M1/M2 smoke-test 原始证据。
- 每个正式 run 的 job/PID、GPU、command、commit、config、seed、日志和 artifact 路径。
- steps、reward、KL及仓库既定核心指标的最新值和 freshness。
- NaN/Inf、OOM、traceback、dependency、磁盘和权限错误扫描。
- 与对应最高严格匹配 baseline 的字段级 matching audit。
- 所有历史失败行，包括：
  - 5 个原 Swimmer-v3 dependency failures
  - 新补跑结果作为关联的新记录，不覆盖旧记录
  - `18302268_10` 保持 unresolved/unmapped
  - Bede `1072326_0-17` 保持 scheduler-complete/artifacts-inaccessible

# 早停与失败分类

- 低于最高严格匹配 baseline 的 `3/5` 或前期明显崩溃时，只标记 `early-stop-candidate`，记录 step、比较值、baseline 和原因；本任务不得因科学表现主动取消。
- 依赖、OOM、断连等分别分类为：
  - algorithmic
  - numerical
  - infrastructure/dependency
  - scheduler/quota waiting
- dependency/infrastructure failure 不得计为算法失败或纳入科学均值。

# Required Outputs

更新 `.agent/STATE.md` 和 `.agent/AGENT_REPORT.md`，至少包含：

1. Task-ID、起止时间、起始/结束 HEAD。
2. refreshed live-state snapshot。
3. 5-cell launch/matching table。
4. dependency root cause、exact fix、package diff及 rollback。
5. M1/M2 preflight 结果。
6. 每个补跑 cell 的状态：
   - completed-valid
   - running
   - failed-algorithmic
   - failed-numerical
   - failed-infrastructure
   - waiting-scheduler/quota
   - blocked-metadata
7. steps、reward/KL、artifact、日志 freshness及错误扫描。
8. batch completeness：分别报告原始 `30/35`、成功补齐数和新的有效完成数；禁止删除原失败计数。
9. strict-matching audit及 early-stop-candidate 评估。
10. Bede artifacts inaccessible 与 `18302268_10` unmapped 的保留状态。
11. unresolved blockers 和唯一 recommended next action。
12. changed files、commit hash、push result。

# Acceptance Criteria

- 5 个目标 cells 均被准确恢复身份，或明确标记 metadata blocker。
- M1/M2 均保持原定义并分别核验。
- 只补跑原失败 Swimmer-v3 cells。
- 正式启动前 `mujoco_py` preflight 成功且依赖变化最小、可回滚。
- 所有已启动 cells 达到 `9,994,240` steps或有完整失败证据。
- 结果仅在字段级 strict matching 通过后标记 `completed-valid`。
- 原 dependency failures、Bede artifact 问题和 `18302268_10` provenance 全部保留。
- 报告提交并推送至 `origin/agent-work`；结束工作树干净或仅保留原有无关改动。

# Prohibited Actions

- 禁止补跑已完成的 30 个 cells。
- 禁止修改算法、M1/M2 边界、seed、预算、超参或匹配矩阵。
- 禁止把 M1 结果替代 M2，反之亦然。
- 禁止覆盖、删除或重写历史失败记录和 artifacts。
- 禁止将 Bede scheduler completion 当作科学完成。
- 禁止映射或猜测 `18302268_10`。
- 禁止启动 Procgen、Isaac或其他 MuJoCo sweep。
- 禁止使用 `.54` / `ws4090-31`。
- 禁止使用 Jupyter；若发现完全空闲超过一小时的相关 Jupyter，按既定规则处理并记录。
- 禁止自动执行科学早停。
- 禁止提交无关代码或用户已有改动。

# 提交与推送

仅提交本任务产生的 `.agent` 报告、必要的最小依赖声明及直接相关诊断文件。

建议提交信息：

`agent: recover five matched Swimmer-v3 cells`

推送至 `origin/agent-work`，并在 `AGENT_REPORT.md` 中记录 commit hash 与 push 结果。
