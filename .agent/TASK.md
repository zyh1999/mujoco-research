Status: RUNNING
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-MLP-FULLEF-FULLGGN-KTRUE-M050708-S01-20260825-05

# 唯一科学目标

启动一项独立 MuJoCo 实验：在 large-batch no-shared MLP、Full-EF actor +
Full-GGN critic 线上，将 `Kaczmarz=true`，严格比较 matched actor/critic
momentum `0.5/0.5`、`0.7/0.7`、`0.8/0.8`。覆盖 canonical 七环境和统一
seeds `0,1`，共 42 个唯一 formal cells。本任务不是 M2 Transformer/FullEmp，
也不是旧 `K=false` 任务的续跑；旧任务只作为 strict-matched parent 和独立
comparison provenance。

# Formal matrix 与启动上限

- 冻结矩阵：3 momentum × 7 canonical environments × 2 seeds = 42 cells。
- 顺序固定为 momentum `0.5 -> 0.7 -> 0.8`、canonical environment 顺序、
  seed `0 -> 1`；不得按表现改变。
- formal launch 最多 42 次；每 cell 最多一次；禁止自动重试、补 seed 或新增
  setting。preflight 最多三个且不计入 42。
- Wave 1 为 manifest 前 24 cells；Wave 2 为剩余 18 cells。Wave 1 全部进入
  terminal scheduler/process 状态、GPU 释放且 artifacts 核验后才能启动 Wave 2；
  两波禁止重叠。

# Canonical matching parent

以刚完成的 `Kaczmarz=false` 同一 large-batch no-shared MLP Full-EF+Full-GGN
formal 配置为唯一 parent。恢复并冻结 source/trainer/config SHA256、七环境
ID/version/wrapper、网络与初始化、rollout、minibatch、epochs、LR、VF 全部
语义、Full-EF/Full-GGN 构造、precision、solver/reduction、KL、评估、nominal
10M 及 9,994,240 logged-step terminal convention。

新实验固定：

- no-shared large-batch MLP；Full-EF actor；Full-GGN critic；
- damping `0.03`；normalization `none`；Kaczmarz `true`；
- parameter L2 clip `0.5`；actor momentum = critic momentum；seeds `0,1`；
- 其余科学字段与 K=false parent 完全一致。

相对对应 K=false cell 允许的科学差异只有 `Kaczmarz false -> true`；三个
K=true setting 间允许的科学差异只有 matched actor/critic momentum。run ID、
路径及非干预 telemetry 不算科学差异。

# 旧任务收口与 provenance

启动前在报告中冻结旧任务：Bede `24/24 completed`；dual-5060 按现有证据
标记 interrupted；保留全部 job/run ID、commit、配置、日志和 artifacts，并
标记 `KFALSE_CLOSED_PRESERVED`。禁止恢复、补跑、覆盖、删除或重新分类旧
dual-5060 cells。新旧结果不得合并为同一 run/seed/统计样本；新实验必须使用
独立非碰撞 root、manifest、日志、checkpoint 和 artifact 路径。

# 唯一允许资源与拓扑

- 只允许 Bede 六张 V100；禁止访问、查询、分配或使用 CSF3、dual-5060、
  其他远端、`.54`、`ws4090-31`；禁止 Jupyter。
- 启动前刷新 Bede scheduler/GPU/process/storage/log 状态，并验证新 root 可写、
  artifact 可读回和空间充足。无法获得六卡或回收验证失败则报告
  `BLOCKED_INFRASTRUCTURE`，不得切换平台。
- 最多六张 V100；每卡最多四个 trainer；全局最多 24；禁止提高每卡并发。

# Mandatory Kaczmarz preflight

正式启动前分别对 momentum `0.5`、`0.7`、`0.8` 完成三个代表环境 seed0
preflight。每个必须以运行时证据证明：

1. `Kaczmarz=true` 被解析；
2. 首次符合条件的 update 建立 SGD momentum history；
3. 至少第二次 update 从非空 momentum buffer 计算并实际传入
   `previous_projection`；不得恒空或每步错误重置；
4. actor Full-EF、critic Full-GGN、matched momentum、d=0.03、normalization
   none、parameter L2 clip=0.5、no-shared MLP 和 parent VF 语义全部匹配；
5. projection norm、solver residual、gradient、curvature、VF、KL telemetry 有限；
6. 保存代码路径、调用顺序和 runtime telemetry。

任一 setting preflight 失败，不得关闭 Kaczmarz、绕过 projection、改变 solver、
damping、momentum、normalization、clip 或 VF；不得启动该 setting 的 14 cells，
标记 `PRECHECK_BLOCKED_KACZMARZ_PATH`。若暴露算法实现缺陷，本任务不授权修复。

# Formal 运行、分析与报告

仅启动 matching audit 和对应 preflight 通过的 cells。每 cell 记录 ID、env、seed、
momentum、Bede job/node/GPU/同卡并发、source/config SHA256、命令、steps、reward、
KL、VF、EF/GGN residual、projection 创建/读取/更新计数和 norm、artifacts、
freshness、exit code 与 NaN/Inf/OOM/traceback/dependency/disk/permission/scheduler
错误扫描。基础设施失败不得算算法失败或进入均值；禁止自动重试。

低于同环境最高 strict-matched K=false baseline 的 3/5 或明显崩溃只标记
`early-stop-candidate`，不得因科学表现自动取消。失败须区分 algorithmic、
numerical、infrastructure、scheduler/quota、matching/preflight。

仅用 strict-valid 同 env/seed/momentum paired cells，输出七环境×三 momentum×
两 seed 的 K=true 表，并与 K=false parent 配对；报告逐 seed difference/ratio、
两 seed mean/std、win/loss/tie、solver/projection/KL/VF 稳定性。跨环境只用仓库
预定义 normalization，不直接平均原始 reward；两 seed 不得声称显著性或最终最优。

更新 `.agent/STATE.md`、`.agent/AGENT_REPORT.md` 和正式结果文件，包含旧任务
closed-preserved 快照、42-cell manifest、matching audit、Bede 资源证据、三个
preflight、两波映射、每 cell 证据、paired analysis、失败分类、未启动 cells、
唯一下一步、changed files、commit 和 push。

# Acceptance / prohibited

- 身份、K=true、三个 exact momentum、d=0.03、normalization none、L2 clip .5、
  七环境 S01、42 个唯一 cells 和 previous_projection 真正使用均须有证据。
- 只用 Bede 六卡、每卡最多四 trainer、全局最多24、严格 24+18 两波。
- 不超过42 formal launches，无自动重试或额外 cells；新旧 provenance 隔离。
- 禁止 M2、shared、非 Full-EF/GGN、Kaczmarz 降级、actor/critic momentum 不同、
  momentum 专属调参、恢复旧 5060、覆盖历史 artifacts 或提交无关改动。
- 报告及必要代码/manifest/results 提交并推送 `origin/agent-work`；记录 assignment、
  source/config、evidence、delivery commit 和 push 验证。完成后唤醒同一 Planner，
  请求恰好一个下一步有界 MuJoCo 任务。
