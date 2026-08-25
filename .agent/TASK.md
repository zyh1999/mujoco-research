Status: BLOCKED
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-KTRUE-GLFW-PREFLIGHT-AND-BEDE42-20260825-07

# Unique objective

Restore only the known Bede GLFW library path in isolated launcher copies,
then perform exactly one corrected three-setting Kaczmarz preflight. Only if
all three preflights prove real `previous_projection` reuse may the unchanged
42-cell Bede matrix run in frozen 24+18 waves. Do not change algorithm,
trainer, config, manifest, dependencies, environment, or scientific identity.

# Preserved prior evidence

- Keep assignment `f68c1ed`, source/config `5d531d8`, and evidence `32c8aef`.
- Preserve failed preflights `1074569_[0-2]`: momentum 0.5/0.7/0.8, all failed
  before optimizer/update with `ImportError: Failed to load GLFW3 shared library.`
- Preserve cancelled dependent Wave 1 `1074570_[0-5]`, elapsed zero,
  `DependencyNeverSatisfied`; formal launch count remains zero and all 42 cells
  remain unstarted.
- Classify those failures as launcher/infrastructure only, never algorithmic or
  numerical. Never overwrite or reuse their roots, logs, or artifacts.

# Only authorized correction

After canonical `ROOT` is defined and before Python/MuJoCo/GLFW starts, add:

```bash
export LD_LIBRARY_PATH=$ROOT/local/glfw-conda/lib:${LD_LIBRARY_PATH:-}
```

Only launcher environment and non-colliding run/job/output identity may differ.
Produce old/new launcher SHA256 and a line diff. If the functional diff includes
anything besides this export, stop as `BLOCKED_LAUNCHER_DIFF`.

# Frozen scientific identity and matrix

- no-shared large-batch MLP; Full-EF actor; Full-GGN critic; Kaczmarz true;
- matched actor/critic momentum exactly 0.5, 0.7, 0.8;
- damping 0.03; normalization none; parameter L2 clip 0.5;
- canonical seven environments and seeds 0,1;
- same VF, architecture, initialization, rollout, minibatch, epochs, LR, batch
  geometry, precision, solver, clipping, KL, evaluation, nominal 10M and
  9,994,240 logged-step endpoint as the frozen K=false parent;
- relative to K=false the only scientific difference is Kaczmarz false to true;
  between K=true settings the only scientific difference is matched momentum;
- frozen manifest is exactly 3 x 7 x 2 = 42 unique cells and must not change.

# Resource and topology boundary

- Use only Bede V100; do not access/query/use CSF3, dual-5060, other remotes,
  `.54`, `ws4090-31`, or Jupyter.
- Refresh Bede scheduler, GPU/process state, quota/storage, target-root
  write/read/delete ability and artifact readback before launch.
- At most six V100s, at most four trainers per GPU, at most 24 globally.
- Use isolated non-colliding root/log/artifact/tmp paths.

# One corrected preflight attempt

Run exactly three HalfCheetah seed-0 preflights, one each for momentum
0.5/0.5, 0.7/0.7 and 0.8/0.8, with the frozen preflight budget. Each must show:

1. GLFW/MuJoCo import and environment construction succeed.
2. Kaczmarz true, no-shared MLP, Full-EF actor, Full-GGN critic, exact matched
   momentum, damping 0.03, normalization none, L2 clip 0.5 and frozen VF/solver.
3. The first eligible update creates/writes momentum history; a later eligible
   update reads a nonempty momentum buffer and actually passes/uses
   `previous_projection`, without resetting it every step.
4. At least two unambiguous `KACZMARZ_PROJECTION_USED` events (or exact
   equivalent), buffers greater than zero, finite projection norm, gradient,
   curvature, solver residual, VF and KL telemetry.
5. Exit code, call counts, commands, logs, SHA256 and error scan are preserved.

All three must return rc=0 and pass every gate. If any fails, stop immediately;
do not make a second correction/retry and do not submit any formal job.

# Conditional formal execution

Do not pre-submit dependency placeholder jobs. After reviewing all preflight
evidence, if and only if the complete gate passes:

- Wave 1: frozen manifest first 24 cells on six V100s, max four trainers/card.
- Wave 2: remaining 18 cells only after Wave 1 is terminal, GPUs are released,
  and logs/artifacts/errors have been reviewed; no wave overlap.
- At most 42 formal launches total, one per unique cell, no automatic retry.
- Do not change membership based on performance or failure.

For every cell record cell ID, env/seed/momentum, job/node/GPU/concurrency,
hashes and exact command, steps/terminal convention, reward/KL/VF, EF/GGN
residual, projection creation/read/use/update counts and norm, artifact paths
and freshness, exit code and error scans. Infrastructure/dependency failures
must remain separate from scientific results. Performance below 3/5 of the
highest strict-matched K=false reference is only an `early-stop-candidate`;
do not cancel automatically.

# Required report and provenance

Update `.agent/STATE.md`, `.agent/AGENT_REPORT.md`, corrected launcher/diff,
preflight evidence, manifest status and (only if launched) formal result table.
Report task times/HEADs, live Bede refresh, launcher hashes/diff, runtime
`LD_LIBRARY_PATH` and GLFW evidence, complete three-preflight evidence, formal
gate decision, wave mappings, per-cell evidence, failure classes, commits and
push verification. Preserve separately:

1. K=false Bede 24/24 and interrupted dual-5060 as
   `KFALSE_CLOSED_PRESERVED_USER_AUTHORIZED_REPLACEMENT`;
2. failed K=true `1074569_[0-2]` GLFW preflight;
3. cancelled zero-elapsed `1074570_[0-5]` and formal count zero;
4. this corrected preflight;
5. this formal matrix, if the gate passes.

Commit only directly relevant `.agent` records, corrected launchers/diff and
evidence/results. Push to `origin/agent-work`, then wake the same ChatGPT
Planner with the full report and commit SHA, asking for exactly one next
bounded MuJoCo task.

# Prohibited actions

No second corrected-preflight retry; no algorithm/trainer/config/manifest or
dependency changes; no Kaczmarz bypass; no momentum/damping/normalization/clip/
VF/solver change; no extra formal cells/retries; no non-Bede compute; no old
artifact overwrite; no unrelated edits.
