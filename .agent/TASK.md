Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-READONLY-REFRESH-20260817-01

# 唯一目标

在不启动、停止、修改或续跑任何实验的前提下，刷新 MuJoCo 仓库两条既定研究线的实时状态，形成一份可审计的 `AGENT_REPORT.md`，使 Planner 下一轮能够基于当前证据选择唯一实验任务。

# 范围

仅限：

- 仓库 `https://github.com/zyh1999/mujoco-research`
- 分支 `agent-work`
- `.agent/GOAL.md`
- `.agent/STATE.md`
- `.agent/TASK.md`
- `.agent/AGENT_REPORT.md`
- 上述文件直接引用的 MuJoCo 报告、配置、脚本、日志和 artifact
- CSF3 上与本仓库有关的 scheduler、GPU、进程和文件状态
- 仅当仓库记录显示存在相关 MuJoCo 作业时，检查 Bede、双 5060 或其他允许远端的对应状态

不得涉及 Procgen 或 Isaac。

必须保持仓库既定的：

- 两条 MuJoCo 研究线及其边界
- 环境与软件版本
- seeds
- 总预算及单次预算
- baseline 与方法的严格匹配矩阵语义
- 已记录失败配置和早停记录

CSF3 是决策控制平面。禁止访问或使用 `.54` / `ws4090-31`。

# 允许动作

1. 只读同步并检查远端 `agent-work` 的最新提交；记录远端 HEAD commit、检查时间和工作树状态。
2. 完整读取四份 `.agent` 文件，再按其中引用关系读取必要的报告、配置、代码和日志。
3. 在 CSF3 刷新并记录：
   - 当前及近期相关 scheduler 作业
   - 作业状态、节点、GPU、运行时间和退出码
   - GPU 使用率、显存占用及对应进程
   - 训练进程、父子进程及启动参数
   - 最新日志更新时间和尾部关键内容
   - reward、KL 及仓库定义的其他核心指标
   - checkpoints、评估结果、汇总表及其他预期 artifact
   - NaN/Inf、OOM、traceback、断连、磁盘、权限、环境或依赖错误
4. 若 `.agent` 文件记录了 Bede、双 5060 或其他允许远端上的相关 MuJoCo 作业，可进行同等只读检查。
5. 对发现的每个 run 按以下类别标注：
   - running
   - completed-valid
   - completed-invalid
   - failed-algorithmic
   - failed-numerical
   - failed-infrastructure
   - waiting-scheduler/quota
   - stale/orphaned
   - unknown
6. 只允许更新 `.agent/AGENT_REPORT.md` 以及完成任务所必需的 `.agent` 协作状态字段。
7. 提交并推送上述报告性改动至 `agent-work`。

# 必需证据

报告必须包含带 UTC 或 Europe/London 时间戳的原始证据或准确摘录：

1. `agent-work` 远端 HEAD commit，以及开始和结束时的工作树状态。
2. 四份 `.agent` 文件的当前 commit/内容摘要；明确两条研究线的名称、目标和边界。
3. 两条研究线各自的既定环境、版本、seeds、预算和严格匹配矩阵；只能转录仓库定义，不得自行补全。
4. 所有相关 scheduler 作业：
   - job ID
   - cluster/host
   - partition/queue
   - state/reason
   - node/GPU
   - elapsed/time limit
   - submit/start/end time
   - exit code
5. 所有相关活跃进程及其 run/config/日志映射。
6. GPU 型号、利用率、显存和占用进程；若无相关 GPU 进程也必须明确报告。
7. 每个 run 的：
   - 研究线
   - 方法与 baseline 身份
   - environment/version
   - seed
   - budget/当前进度
   - config 或命令来源
   - 最新日志时间
   - reward/KL/其他仓库核心指标
   - artifact/checkpoint 状态
8. 对日志和 artifact 做 freshness 检查；不得把旧文件的存在当作仍在运行的证据。
9. 对 NaN、Inf、OOM、traceback、error、failed、killed、timeout、NCCL/CUDA、磁盘和权限问题的扫描结果。
10. baseline 严格匹配核验：
    - 列出匹配字段
    - 标记 matched / mismatched / unverifiable
    - 不得把不严格匹配的结果计入 3/5 判断
11. 历史与当前失败配置表。必须保留既有失败项，不得因本轮未复现而删除。
12. 若某配置低于当前最高严格匹配 baseline 的 3/5，或前期明显崩溃，只能标记为 `early-stop-candidate`；记录比较值、步数、匹配依据和原因，本任务不得执行早停。

若某项无法采集，必须写明：

- 缺失项
- 尝试的只读来源
- 具体阻塞原因
- 它阻止了哪些结论

# Required Outputs

更新 `.agent/AGENT_REPORT.md`，至少包含：

## Metadata
- Task-ID
- inspection_start
- inspection_end
- branch
- inspected_remote_HEAD
- report_commit
- control_plane
- inspected_hosts
- explicitly_excluded_hosts

## Agent-file reconciliation
- GOAL summary
- prior STATE summary
- prior TASK status
- prior AGENT_REPORT conclusion
- inconsistencies or stale claims

## Research-line definitions
分别记录两条既定 MuJoCo 研究线的：
- exact name
- objective
- allowed environments/versions
- seeds
- budgets
- matching-matrix semantics
- current recorded phase

## Live inventory
逐项列出：
- scheduler jobs
- GPU/process state
- runs
- configs
- logs
- reward/KL/core metrics
- checkpoints/artifacts
- error-scan results

## Run classification table
每个已知 run 一行，并保留历史失败项：
- line
- run ID
- method
- baseline
- env/version
- seed
- budget/progress
- match status
- latest metrics
- artifact status
- freshness
- classification
- evidence
- failure reason

## Early-stop assessment
- 仅评估，不执行
- strict matched baseline used
- 3/5 threshold
- observed value and step
- candidate yes/no
- evidence and uncertainty

## Failure accounting
分别汇总：
- algorithmic
- numerical
- infrastructure
- scheduler/quota waiting
- stale/unknown

## Planner decision inputs
- facts established
- unresolved questions
- currently running valid work
- free/occupied resources
- highest strictly matched baseline per research line
- exact candidate next actions，按优先级列出，但不得执行
- recommended single next action
- recommendation rationale

## Changes and delivery
- files changed
- confirmation that no experiment/code/config was changed
- commit hash
- push result

# 验收标准

只有同时满足以下条件才算完成：

1. 先读完四份 `.agent` 文件，再检查其引用的材料。
2. 所有易过期状态均在本轮重新采集，并带时间戳。
3. 两条研究线分别报告，未混入 Procgen 或 Isaac。
4. 未改变环境、版本、seeds、预算或匹配矩阵语义。
5. 每个已知 run 都有状态分类和证据；未知项明确标注 unknown。
6. reward/KL、日志、artifact 和错误扫描都有当前结果或明确阻塞说明。
7. 3/5 规则只使用最高的严格匹配 baseline，并且只形成早停候选，不执行早停。
8. 历史失败配置继续保留，且失败类型区分正确。
9. 没有启动、停止、续跑、重排或修改任何实验。
10. `AGENT_REPORT.md` 已提交并推送到远端 `agent-work`，报告中给出 commit hash 和 push 结果。
11. 最终工作树干净；若开始时已有无关改动，必须原样保留并在报告中说明。

# 禁止事项

- 禁止启动、续跑、停止、取消、重排或迁移实验。
- 禁止执行早停，即使达到 3/5 条件。
- 禁止修改训练代码、配置、依赖、环境、数据或 artifact。
- 禁止改变两条研究线的长期方向或合并研究目标。
- 禁止增加环境、版本、seed 或预算。
- 禁止用非严格匹配 baseline 作结论。
- 禁止访问或使用 `.54` / `ws4090-31`。
- 禁止规划或检查 Procgen、Isaac。
- 禁止依据旧 scheduler、旧 GPU 快照或旧日志宣称当前状态。
- 尽量不用 Jupyter；本任务不应新建 Jupyter 会话。若发现本仓库已有完全空闲的 Jupyter 会话，只记录其状态和空闲时长，不得在本只读任务中自行 scancel；将其标记为需按“一小时内 scancel”规则处理的待决项。
- 禁止覆盖、清理或提交用户已有的无关改动。
- 除 `.agent` 报告/协作状态外，禁止产生仓库写入。

# 提交与推送要求

仅暂存本任务实际更新的 `.agent` 报告性文件。

建议提交信息：

`agent: refresh live MuJoCo status for planner`

必须推送至：

`origin/agent-work`

若 push 失败，不得改推其他分支；在报告中记录错误、远端状态和本地 commit hash。
