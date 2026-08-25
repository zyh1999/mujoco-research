Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-KTRUE-TELEMETRY-PREFLIGHT-AND-BEDE42-20260825-08

# Unique objective

Make only two reporting/telemetry corrections for the frozen Kaczmarz=true
experiment: write the actual runtime Kaczmarz value to `run_info.txt`, and
expose distinct auditable `actor_solver_residual` and
`critic_solver_residual` values. Then run exactly one new isolated three-cell
preflight. Only if telemetry semantics and all three preflights pass may the
unchanged 42-cell Bede matrix run in frozen 24+18 waves.

# Preserved evidence

Preserve commits `52dcce1`, `f209f22`, `9bdb1c4`, and `32c8aef`; failed GLFW
preflight `1074569_[0-2]`; cancelled zero-elapsed placeholder
`1074570_[0-5]`; and successful runtime-but-telemetry-blocked corrected
preflight `1074573_[0-2]`. Formal launch count remains zero and all 42 cells
remain unstarted. Never overwrite or reclassify any prior root or artifact.

# Frozen scientific identity

- large-batch no-shared MLP, Full-EF actor, Full-GGN critic, Kaczmarz true;
- matched actor/critic momentum exactly 0.5, 0.7, 0.8;
- damping 0.03, normalization none, parameter L2 clip 0.5;
- canonical seven environments, seeds 0,1, exactly 42 unique formal cells;
- identical VF, architecture, initialization, rollout, minibatch, epochs, LR,
  batch geometry, precision, solver, clip, KL, evaluation and 10M endpoint;
- frozen config, manifest and GLFW-corrected launcher semantics do not change.

# Only authorized telemetry modifications

1. Replace the worker's hardcoded `kaczmarz=false` report with the actual
   runtime/config value. K=true must be consistent in config, stdout and
   `run_info.txt`; no other scientific report field may change.
2. Emit separately named `actor_solver_residual` and `critic_solver_residual`
   for the actual Full-EF and Full-GGN systems. Define the residual equation,
   numerator, denominator, epsilon and measurement point. Prefer already
   computed values; otherwise compute detached/read-only after the direction
   is fixed using the existing matrix/operator, RHS and direction.
3. Diagnostics must not change direction, optimizer/parameters,
   `previous_projection`, RNG, solver iterations/stopping/precision, gradients,
   algorithm state or training control flow. No placeholder/proxy residuals.

If both residuals cannot be exposed without scientific change, stop as
`BLOCKED_TELEMETRY_IMPLEMENTATION`.

# Telemetry-only semantics gate

- Preserve before/after SHA256 and line diff of every modified worker/trainer.
- The diff may contain only actual Kaczmarz reporting, detached residual
  computation/exposure and corresponding log/artifact fields.
- Before the new preflight, run a fixed-seed/fixed-input A/B regression and
  compare actor/critic parameters, optimizer state, directions,
  `previous_projection` buffers, RNG state, losses, KL and VF update.
- Other than new telemetry, these must be bitwise identical; if platform
  operations prevent bitwise comparison, document why and provide exact zero
  or machine-precision evidence without inventing a tolerance.
- Failure blocks all new preflight and formal work.

# Resources

Use only Bede V100: at most six GPUs, four trainers per GPU, 24 globally. Do
not access/query/use CSF3, dual-5060, other remotes, `.54`, `ws4090-31`, or
Jupyter. Refresh scheduler/GPU/process/quota/storage and write/read/delete
artifact capability first. Keep the established GLFW export unchanged.

# Exactly one new preflight round

Use a fresh non-colliding root for three HalfCheetah seed-0 cells at the same
81,920-step budget, one launch each for matched momentum 0.5, 0.7 and 0.8.
Each must prove rc=0; environment construction; K=true agreement in runtime
and run_info; frozen method/config identity; one `KACZMARZ_NO_HISTORY`; later
real `KACZMARZ_PROJECTION_USED`; finite separate actor and critic residuals
mapped to their exact systems; finite projection/gradient/curvature/KL/VF;
complete logs/hashes/RC; and clean error scan.

All three must pass. Any failure stops the task: no further telemetry change,
second preflight, Kaczmarz downgrade, or formal launch.

# Conditional formal execution

Do not pre-submit placeholder jobs. After complete gate review:

- Wave 1 is the frozen first 24 cells on six V100s, max four trainers/card.
- Wave 2 is the remaining 18 only after Wave 1 is terminal, GPUs released and
  all logs/artifacts/errors reviewed. Waves cannot overlap.
- At most 42 formal launches, one per cell, no automatic retries or additions.

Record per cell identity, env/seed/momentum, job/node/GPU/concurrency, hashes,
command, steps, reward/KL/VF, separate actor/critic residuals, projection
creation/read/use/update counts and norms, artifacts/freshness, RC and error
scan. Infrastructure failures remain separate. Performance below 3/5 of the
strict matched K=false reference is only an early-stop candidate and must not
be automatically cancelled.

# Outputs and completion

Update `.agent/STATE.md`, `.agent/AGENT_REPORT.md`, telemetry patch/diff, A/B
evidence, preflight evidence, manifest state and conditional formal results.
Preserve all K=false and K=true provenance in isolated roots. Commit and push
only directly relevant files to `origin/agent-work`, record assignment/source/
telemetry/evidence/delivery commits, then wake the same ChatGPT Planner with
the full report and SHA asking for exactly one next bounded MuJoCo task.

# Prohibited

No scientific algorithm/solver/config/manifest change; no changed momentum,
damping, normalization, clip, VF or endpoint; no Kaczmarz bypass; no second
new preflight; no formal retry/extra cell; no non-Bede compute; no historical
artifact overwrite; no unrelated changes.
