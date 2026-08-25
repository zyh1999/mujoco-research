Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-KTRUE-CONFIGPATH-PREFLIGHT-AND-BEDE42-20260825-09

# Unique objective

Fix only the reporting telemetry config-path resolution so
`configured_kaczmarz()` reads the exact frozen config actually supplied to the
trainer under `ktrue_momentum_stage/`, not an assumed `configurations/` path.
After path-only auditing, run exactly one new isolated three-momentum
preflight. Only if every path, telemetry, Kaczmarz, projection and numerical
gate passes may the frozen 42-cell Bede matrix launch in 24+18 waves.

# Preserved evidence

Preserve assignment `b5464d6`, telemetry source `19f86c8`, report `9efda14`,
prior report `9bdb1c4`, and failed preflight `1074576_[0-2]`. The three cells
failed before environment/optimizer/update because the reader used the wrong
directory; this is reporting-only path evidence. Formal launch count remains
zero. Also preserve `1074569`, `1074570`, successful runtime preflight
`1074573`, old K=false closure and interrupted dual-5060 provenance. Never
overwrite, delete or reclassify any prior root/artifact.

# Only authorized code correction

Pass the exact config path used for this run explicitly to reporting. Resolve
it with `realpath`, read that same file, and record resolved path, SHA256 and
parsed Kaczmarz value. It must be the existing
`ktrue_momentum_stage/rat_mlp_detjc_normnone_ktrue_d003_lrv01_e4_mlp.yaml`.

Do not search directories, glob, fallback/default, copy/move/change the config,
create a second config, or let reporting affect trainer runtime. An explicit
reporting-only absolute path is allowed, but it must be identical to the
trainer's config realpath and frozen SHA256.

# Frozen science

Keep unchanged: large-batch no-shared MLP; Full-EF actor; Full-GGN critic;
Kaczmarz true; matched momentum 0.5/0.7/0.8; damping 0.03; normalization none;
parameter L2 clip 0.5; canonical seven environments; seeds 0,1; all VF,
architecture, initialization, rollout, minibatch, epochs, LR, geometry,
precision, solver, KL, evaluation and 10M endpoint fields; frozen 42-row
manifest; trainer mathematics; residual telemetry; GLFW export.

# Path-only semantics gate

- Preserve before/after worker SHA256 and line diff; diff may only pass/read/
  report the exact config path, realpath and SHA plus non-colliding identity.
- On Bede prove the exact file exists/readable, realpath points to the stage
  config, SHA256 equals the frozen hash, trainer/reporting paths match, and the
  parser returns Kaczmarz true.
- The old wrong path must fail explicitly with no fallback/default.
- Rerun the fixed-input audit: actor/critic parameters, optimizer states and
  momentum, directions, previous projection, RNG, loss/KL/VF and detached
  residuals must remain bitwise identical.
- Path reporting must not enter optimizer, solver or training control.

Any failure blocks preflight as `BLOCKED_CONFIG_PATH_SEMANTICS`.

# Resources

Use only Bede V100, at most six GPUs, four trainers/card and 24 globally. Never
access/query/use CSF3, dual-5060, other remotes, `.54`, `ws4090-31` or Jupyter.
Refresh Bede scheduler/GPU/process/quota/storage and artifact write/read/delete
before launch. Keep the verified GLFW export unchanged.

# Exactly one new preflight round

Use a fresh isolated root for three HalfCheetah seed-0 81,920-step cells, one
each at matched momentum 0.5, 0.7 and 0.8. Each must prove rc=0; GLFW/MuJoCo
construction; identical trainer/reporting config realpath and SHA; run_info
Kaczmarz true; runtime/config/run_info identity; frozen method fields; one
`KACZMARZ_NO_HISTORY`; later real `KACZMARZ_PROJECTION_USED`; finite distinct
actor/critic residuals; finite projection/gradient/curvature/KL/VF; complete
logs/hashes/RC and clean error scan.

All three must pass. Any failure stops the task with no further path change,
second preflight, Kaczmarz bypass or formal launch.

# Conditional formal execution

Do not pre-submit placeholders. After complete gate review:

- Wave 1: frozen first 24 cells, six V100s, max four trainers/card.
- Wave 2: remaining 18 only after Wave 1 is terminal, processes exited, GPUs
  released and logs/artifacts reviewed; no overlap.
- Maximum 42 formal launches, one per cell, no retry/addition/reordering.

Record per cell method identity, config realpath/SHA, env/seed/momentum,
job/node/GPU/concurrency, source/launcher hashes, command, steps, reward/KL/VF,
actor/critic residuals, projection counters/norms, artifacts/freshness, RC and
error scans. Infrastructure/path failures remain separate. Low performance is
only an early-stop candidate and must not be automatically cancelled.

# Outputs

Update `.agent/STATE.md`, `.agent/AGENT_REPORT.md`, path-only patch/diff,
bitwise audit, preflight evidence, manifest state and conditional formal
results. Commit/push only directly relevant files to `origin/agent-work`, then
wake the same ChatGPT Planner with the report and SHA asking for exactly one
next bounded MuJoCo task.

# Prohibited

No second path-corrected preflight; no search/glob/default config; no config,
algorithm, solver, manifest or scientific-field change; no Kaczmarz bypass;
no formal retry/extra cell; no non-Bede compute; no historical overwrite; no
unrelated change.
