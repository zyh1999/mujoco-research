Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: MUJOCO-KTRUE-CONCURRENT-WAVE2-LAUNCH-VERIFY-20260825-11

# Unique objective

Under the user's explicit authorization, while Wave 1 remains live, submit the
frozen Wave 2 eighteen formal cells immediately on six additional, physically
disjoint Bede V100s and perform bounded live startup verification. This task
authorizes only concurrent submission and verification, not retries,
dependency repair, algorithm changes, Wave 1 cancellation or new cells.

# Authorization override

This task replaces the prior serial/no-overlap restriction only as follows:

- Wave 1 and Wave 2 may run concurrently on Bede;
- the Bede limit is twelve V100s total;
- Wave 2 must use six V100s disjoint from Wave 1;
- every GPU remains capped at four trainers.

All other frozen science, manifest, provenance, failure-classification and
platform restrictions remain unchanged.

# Preserved evidence

Continue from delivery `3ae08bc`, assignment `08c56ea` and path-only source
`39a789d`. Preserve passed preflight `1074579_[0-2]`, live Wave 1
`1074582_[0-5]`, its 24 formal launches, and the two momentum-0.5 Swimmer
seed failures as `failed-infrastructure/dependency: No module named
mujoco_py`. Wave 2 has not yet been submitted. The frozen matrix remains 42
unique cells. Never overwrite, delete or reclassify prior evidence.

# Frozen science and launch accounting

Wave 2 is exactly the remaining frozen manifest eighteen cells, in unchanged
order, each launched at most once. Cumulative formal launches are at most 42:
Wave 1=24 and Wave 2=18. Keep large-batch no-shared MLP, Full-EF actor,
Full-GGN critic, Kaczmarz true, matched actor/critic momentum, damping 0.03,
normalization none, parameter L2 clip 0.5, seeds 0/1, all VF, architecture,
initialization, rollout, minibatch, epochs, LR, geometry, precision, solver,
KL, evaluation and 10M endpoint fields, and exact source/config/manifest
hashes and config realpath.

Do not recover momentum-0.5 Swimmer, retry any Wave 1/Wave 2 cell, add a seed,
environment, momentum or cell, submit a placeholder, or change membership or
ordering.

# Bede-only concurrent topology

Wave 1 retains its six allocations. Wave 2 uses six other physical V100s;
the GPU sets must be disjoint. Total Bede use is at most twelve GPUs, no GPU
has more than four trainers, Wave 2 has at most eighteen trainers and combined
trainer count is at most 42. Never access/query/use CSF3, dual-5060, another
remote, `.54`, `ws4090-31` or Jupyter.

# Immediate resource gate

Before submission, refresh only Bede: Wave 1 jobs/nodes/GPU identities and
PID count, idle V100 capacity, the six planned disjoint Wave 2 GPU identities,
quota/storage and artifact write/read/delete, non-colliding Wave 2 paths, and
frozen source/config/launcher/manifest hashes. If six disjoint V100s cannot be
proven, stop as `BLOCKED_CONCURRENT_PLACEMENT`; do not preempt, cancel,
migrate, partially submit or switch platform.

# Wave 2 submission and bounded verification

After the gate, submit all eighteen cells without a Wave 1 dependency. Do not
modify Wave 1. Preserve isolated cell command/PID/log/checkpoint/artifact paths
and unique identities. Verify job/array, scheduler/node/GPU mapping, disjoint
GPU sets, eighteen command artifacts and per-cell PID/status, per-GPU trainer
count, hashes/config realpath, runtime K=true/matched momentum/method identity,
first available steps/Kaczmarz/residual/KL/gradient/VF telemetry and startup
error scan.

Verification finishes when every Wave 2 cell is one of
`RUNNING_VERIFIED`, `PENDING_SCHEDULER_VERIFIED`, or
`FAILED_STARTUP_CLASSIFIED`; do not wait for 10M endpoints in this task.

Wave 2 Swimmer cells must still be submitted. If `mujoco_py` is missing,
classify them as infrastructure/dependency, preserve evidence and do not
retry, install dependencies, change host, count as algorithm failure or put
them into scientific means.

# Wave 1 protection

Do not cancel, pause, restart, migrate, preempt, rebind or otherwise alter
`1074582_[0-5]`. Continue recording its scheduler/GPU/step/log/error state;
record new failures without treating or retrying them.

# Outputs

Update `.agent/STATE.md`, `.agent/AGENT_REPORT.md`, Wave 2 submission/status,
concurrent topology evidence and live startup verification. Record explicit
user authorization, resource snapshot, disjoint six-plus-six GPU mapping,
Wave 2 job, all eighteen commands/PIDs/statuses, per-card/global concurrency,
hashes/config realpath, first telemetry, error classification, Wave 1
nonintervention and launch accounting. Commit/push only relevant evidence to
`origin/agent-work`, then wake the same ChatGPT Planner with full report and
SHA asking for exactly one next bounded MuJoCo task.

# Prohibited

No delayed serial submission after a passed resource gate; no GPU overlap;
no more than four trainers/GPU or twelve Bede GPUs; no partial Wave 2; no
retry, dependency repair, algorithm/config/solver/manifest/scientific change;
no Wave 1 intervention; no non-Bede compute; no historical overwrite; no
unrelated change.
