Status: RUNNING
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R

# 唯一科学目标

在既定 no-shared large-batch MLP、Full-EF actor + Full-GGN critic 线上，建立七环境统一两 seed 的严格匹配 momentum 矩阵，新增并评估 actor/critic momentum `0.7/0.7` 与 `0.8/0.8`，公平比较 canonical low-momentum、`0.5/0.5` 和 `0.9/0.9`。本任务明确不属于 M2 FullEmp 线。

# Canonical scientific identity

以仓库七环境乘五 seed formal low-momentum matrix 为 canonical parent，恢复 exact source commit、trainer SHA256、config 和实际 low-momentum 数值，不得把 `m≈0` 改写为字面 0。固定：large-batch MLP；no-shared actor/critic；Full-EF/Full-GGN；damping `0.03`；normalization `none`；Kaczmarz `false`；parameter L2 clip `0.5`；actor momentum = critic momentum；相同 VF、环境/version/wrapper、初始化、rollout、minibatch、epochs、LR、batch geometry、precision、solver、KL、评估语义、nominal 10M budget 和 terminal convention。除 momentum、run identity/path 和 telemetry 外不得改变科学字段。

# 有界 seed 策略

1. 按 canonical low-momentum manifest 的预定义顺序选择前两个 seed ID 为统一 `S2`，禁止按表现或现有覆盖率选 seed。
2. 五个 momentum、七环境均使用同一 `S2`。
3. low-momentum 与 0.9 只复用 `S2` strict-valid cells，不重跑。
4. 0.5 只复用相对 parent 字段严格匹配且属于 `S2` 的 cells，并补齐缺口。
5. 新运行 `0.7 × 7 env × S2` 和 `0.8 × 7 env × S2`。
6. 启动上限：0.7 最多14 cells，0.8 最多14 cells，0.5 缺口最多14 cells，总计最多42 cells；不得扩展五 seed。

# 执行前证据门与资源放置

- 完整读取四份 `.agent` 文件及 canonical low、0.5、0.9 manifests/configs/logs。
- 刷新 CSF3、Bede、dual-5060 scheduler/quota、GPU/进程、trainer/container、磁盘、日志、reward/KL、artifacts 和错误扫描。
- CSF3 为控制平面；Executor 自主决定 host、partition、GPU 和 concurrency。
- Bede 仅在验证 artifacts 可持续写、读、回收后承载 formal cells。
- 禁止 `.54` / `ws4090-31`；禁止 Jupyter。
- 启动前建立五 momentum × 七环境 × `S2` formal matrix，核验 canonical identity/SHA256、low 实际值、0.9 和 0.5 相对 parent 的字段 diff、environment、seed、budget、terminal convention、artifact 和 provenance。
- 若 low 与 0.9 在 momentum 外存在科学字段差异，标记 `BLOCKED_CANONICAL_REFERENCE`，不得启动或声称因果比较。

# 配置、preflight 与正式运行

- 从同一 parent 机械生成 0.5 缺口、0.7、0.8 配置，diff 仅含 actor/critic momentum、run identity/path 和 telemetry。
- 每个新 setting 至少一次无训练/单-update preflight，验证 MLP/no-shared、Full-EF/Full-GGN、d=0.03、normalization none、K=false、L2 clip=0.5、VF 匹配、momentum 精确解析，且 gradients、curvature、solver residual、VF telemetry 有限。
- preflight 失败不得降级或调参；只启动缺失且过门 cells，使用非碰撞路径，从头训练至 canonical 10M 终点，不借用其他 momentum 状态。
- 每 cell 记录 host/job/PID、container/image、commit/config/SHA256、seed、momentum、steps、reward、KL、VF、EF/GGN/solver telemetry、artifact freshness 和错误扫描。
- 低于最高严格 reference 的 3/5 或明显崩溃只标记 `early-stop-candidate`，不得自动取消。
- 区分 algorithmic、numerical、infrastructure/dependency、scheduler/quota waiting、matching/provenance blocker。

# 科学分析与输出

只使用 strict-valid 同环境同 `S2` cells，输出七环境 × 五 momentum × 两 seed 终点表；计算 0.7/0.8 相对 low/0.5/0.9 的 paired difference、ratio、win/loss/tie 和 rank；报告逐环境两-seed mean/std，但不得声称显著性或最终排名。跨环境汇总只用仓库预定义 normalization，并保留逐环境结果。

更新 `.agent/STATE.md`、`.agent/AGENT_REPORT.md` 和正式结果表，包含 Task-ID、起止时间/HEAD、canonical identity/SHA256、low 实际值和 S2 证据、formal matrix、matching/dedup、placement、preflight、cell 状态/指标/artifacts/error scan、paired analysis、限制、early-stop/失败分类、唯一下一步建议、changed files、commit 和 push。

必须保留且不混入比较：原 M2 30/35 与五个 Swimmer dependency failures、linked rerun 5/5、Bede `1072326_0-17` scientifically unmapped、`18302268_10` unresolved/unmapped 及其他历史 provenance。

# Acceptance Criteria / Prohibited Actions

- 身份必须是 no-shared large-batch MLP Full-EF+Full-GGN，而非 M2 FullEmp。
- 五 momentum 使用同一 `S2` 且除 momentum 外完全一致；0.5 仅补 S2 缺口，0.7/0.8 各最多14 cells。
- 所有启动 cells 完成 canonical 终点或有明确失败证据；只用 strict-valid paired cells 下结论。
- 禁止 M2 momentum 补齐、shared、非 Full-EF/GGN、Kaczmarz、actor/critic momentum 不一致、momentum 专属调参、按表现选 seed、扩展五 seed、新增 momentum、覆盖历史 artifacts、借用状态、猜测 provenance、Procgen/Isaac 和无关提交。
- 报告提交并推送至 `origin/agent-work`；成功后唤醒同一 Planner，请求恰好一个下一步有界 MuJoCo 任务。
