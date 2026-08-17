# Executor Report

## Metadata

- Task-ID: `MUJOCO-READONLY-REFRESH-20260817-01`
- Inspection window: `2026-08-17T13:25:18Z` to `2026-08-17T13:37:04Z`
- Branch: `agent-work`
- Inspected local/remote HEAD: `7c2356880b863a4f32701207d52021b743ca7694`
- Starting worktree: clean
- Control plane: CSF3 `login1.csf3.man.alces.network`
- Directly inspected: CSF3; dual-5060 `47.114.81.212:60023`
- Bede: attempted, denied noninteractive authentication
- Explicitly excluded: `.54`, `10.49.7.54`, `ws4090-31`

## Agent-file reconciliation

| File | SHA256 before update | Finding |
|---|---|---|
| `.agent/GOAL.md` | `4e1cf39f1b363d9a1ffbf38c64a2013e89240617428e5826b3fa59239609d7b7` | Preserve M1 and M2 as separate matched lines |
| `.agent/STATE.md` | `ec01016a5bbf3b17a234c5063c9f8eba97b51f29b65cabd2ed7e8e373d26ee49` | Dual-5060/Bede handoff status was stale |
| `.agent/TASK.md` | `119e57ea98e6859867337584e03a1b3bb62512225f655e6be4e0d75d1a1a0649` | `READY`, read-only refresh |
| `.agent/AGENT_REPORT.md` | `55091f772f5fd4adeacfffc88b255b1400e0330c5419980bdada7970f65f0e20` | Prior report only described infrastructure setup |

## Research-line definitions and matching

| Line | Methods | Environments/budget | Strict matching boundary |
|---|---|---|---|
| M1 large-batch | Curv256/critic-GGN256, K-true/K-opt, EF255+1, FullEmp/FullGGN | seven MuJoCo tasks; about 500 full updates; accepted terminal boundary 128,450,560 | environment/version, MLP architecture, seed, rollout/minibatch, exact Fisher/GGN and clip semantics, damping/momentum, terminal artifacts, last-10 convention |
| M2 small-batch no-shared Transformer | PPO, K-FAC, Emp256, EF255+1, FullEmp | Ant, HalfCheetah, Hopper, Walker2d, Humanoid, HumanoidStandup, Swimmer-v3; selected batch nominal 10M | environment/version, no-shared Transformer, seed, budget, old-policy/KL semantics, curvature subset, damping/momentum and evaluation convention |

Rows without exact source/config/provenance mapping are `unverifiable`, not
silently matched. Cross-environment decisions must not use Hopper alone.

## Live inventory

### Scheduler and processes

| Host | Job/run | Fresh state | Classification/evidence |
|---|---|---|---|
| CSF3 | current MuJoCo jobs | none running or pending | fresh `squeue` and owned-process scan |
| CSF3 | `18302268_0-9,11-41` | historical terminal | completed-valid at accounting level; exact per-cell artifact table is prior evidence |
| CSF3 | `18302268_10` | FAILED after 57m48s | unknown failure; exact cell/root/cause absent; preserved |
| Bede | `1072326_0-17`, `tf-g7-10m` | all 18 scheduler-COMPLETED, 2h28m05s-10h41m04s | scientific artifacts unknown because direct login was denied |
| dual-5060 | selected 10M batch | no live worker; 30 FINISHED, 5 FAILED | direct status/rc/log/CSV inspection at `13:32Z-13:36Z` |
| dual-5060 | GPUs 0/1 | RTX 5060 Ti, 0%, 33/15 MiB of 16311 MiB | no MuJoCo PID; currently idle telemetry |

No new Jupyter session was created. No current MuJoCo Jupyter process/job was
found, so the one-hour idle-cancellation rule has no current target.

### Dual-5060 run root and provenance

Root:
`/home/zzz/rlstack5060/workspaces/perf_runs/global7env_selected10m_5060_20260816`

The batch has eight worker task files and 35 terminal status files. Each valid
run includes native `progress.csv`, `stdout.log`, `stderr.log`, `time.json`,
`status=FINISHED`, and `rc=0`. All finished rows terminate at 9,994,240 logged
steps. Values below are the final native `eprewmean` and `kl`; `kl` semantics
must remain source-defined and are not relabeled as fixed-behavior KL.

| Method | Environment | Seed | Steps | Final reward | Final KL | Class |
|---|---|---:|---:|---:|---:|---|
| Emp256 | Ant | 1 | 9,994,240 | 3900.77 | 0.00814 | completed-valid |
| Emp256 | Ant | 6 | 9,994,240 | 2172.87 | 0.01592 | completed-valid |
| Emp256 | HalfCheetah | 2 | 9,994,240 | 7438.86 | 0.00419 | completed-valid |
| Emp256 | Hopper | 1 | 9,994,240 | 2421.38 | 0.00736 | completed-valid |
| Emp256 | Hopper | 6 | 9,994,240 | 225.83 | 0.00926 | completed-valid |
| Emp256 | Humanoid | 1 | 9,994,240 | 5323.67 | 0.00843 | completed-valid |
| Emp256 | Humanoid | 6 | 9,994,240 | 5697.98 | 0.00859 | completed-valid |
| Emp256 | HumanoidStandup | 1 | 9,994,240 | 215735.68 | 0.02297 | completed-valid |
| Emp256 | HumanoidStandup | 6 | 9,994,240 | 215849.72 | 0.00389 | completed-valid |
| Emp256 | Walker2d | 0 | 9,994,240 | 3378.36 | 0.00562 | completed-valid |
| Emp256 | Walker2d | 5 | 9,994,240 | 473.48 | 0.00940 | completed-valid |
| EF255+1 | Ant | 1 | 9,994,240 | 2100.79 | 0.01357 | completed-valid |
| EF255+1 | Ant | 6 | 9,994,240 | 1739.97 | 0.01295 | completed-valid |
| EF255+1 | HalfCheetah | 2 | 9,994,240 | 5061.53 | 0.00563 | completed-valid |
| EF255+1 | Hopper | 0 | 9,994,240 | 996.76 | 0.01038 | completed-valid |
| EF255+1 | Hopper | 5 | 9,994,240 | 487.19 | 0.01194 | completed-valid |
| EF255+1 | Humanoid | 2 | 9,994,240 | 5361.66 | 0.00702 | completed-valid |
| EF255+1 | HumanoidStandup | 2 | 9,994,240 | 164456.45 | 0.00954 | completed-valid |
| EF255+1 | Walker2d | 0 | 9,994,240 | 3572.14 | 0.00744 | completed-valid |
| EF255+1 | Walker2d | 5 | 9,994,240 | 4568.56 | 0.00579 | completed-valid |
| FullEmp | Ant | 1 | 9,994,240 | 4411.46 | 0.00837 | completed-valid |
| FullEmp | Ant | 6 | 9,994,240 | 5221.72 | 0.00953 | completed-valid |
| FullEmp | HalfCheetah | 2 | 9,994,240 | 6801.55 | 0.00649 | completed-valid |
| FullEmp | Hopper | 2 | 9,994,240 | 372.71 | 0.00381 | completed-valid |
| FullEmp | Humanoid | 0 | 9,994,240 | 4958.05 | 0.00471 | completed-valid |
| FullEmp | Humanoid | 5 | 9,994,240 | 5944.58 | 0.01346 | completed-valid |
| FullEmp | HumanoidStandup | 0 | 9,994,240 | 240802.64 | 0.01179 | completed-valid |
| FullEmp | HumanoidStandup | 5 | 9,994,240 | 234459.98 | 0.00953 | completed-valid |
| FullEmp | Walker2d | 0 | 9,994,240 | 3238.64 | 0.00617 | completed-valid |
| FullEmp | Walker2d | 5 | 9,994,240 | 4244.70 | 0.01182 | completed-valid |

### Failed selected-10M rows

| Method | Environment | Seed | Progress | Classification | Exact reason |
|---|---|---:|---|---|---|
| Emp256 | Swimmer-v3 | 2 | no training row | failed-infrastructure | missing deprecated `mujoco_py` |
| EF255+1 | Swimmer-v3 | 1 | no training row | failed-infrastructure | missing deprecated `mujoco_py` |
| EF255+1 | Swimmer-v3 | 6 | no training row | failed-infrastructure | missing deprecated `mujoco_py` |
| FullEmp | Swimmer-v3 | 0 | no training row | failed-infrastructure | missing deprecated `mujoco_py` |
| FullEmp | Swimmer-v3 | 5 | no training row | failed-infrastructure | missing deprecated `mujoco_py` |

Every stderr ends with
`gymnasium.error.DependencyNotInstalled: No module named 'mujoco_py'` from
`gymnasium.envs.mujoco.swimmer_v3`. This is a version/environment packaging
failure, not evidence about any optimizer.

## Historical failure accounting

- Algorithmic: none newly established in this read-only cycle.
- Numerical: none newly established; no terminal NaN/Inf/OOM signature found
  in the 30 valid selected runs.
- Infrastructure: five selected Swimmer-v3 dependency failures; prior
  ws4090-76 EF255+1 Humanoid seed4 interruption remains preserved.
- Unknown: `18302268_10`; Bede per-cell scientific validity; old 4090 current
  utilization/provenance.
- Scheduler/quota: no current MuJoCo queue row.

The low Hopper endpoints (for example Emp256 seed6 and FullEmp seed2) are not
declared early-stop candidates because the current evidence does not provide a
fully verified same-seed five-seed highest strict baseline. The 3/5 rule was
evaluated for applicability only; no stop was executed.

## Contradictions and decision inputs

- Prior STATE described eight active 3M missing-cell workers; direct inspection
  found no worker and a newer terminal selected-10M batch.
- Bede's old pending/running snapshot is stale; all 18 elements are now
  scheduler-complete, but scientific completion remains unresolved.
- The dual-5060 GPUs are genuinely idle now, but choosing Swimmer compatibility
  repair versus another cell is a Planner research decision.
- Highest strictly matched baseline per line cannot be recomputed from this
  cycle alone because Bede mapping and exact selected-batch baseline cells are
  missing. Existing M1 128,450,560 terminal artifacts remain valid by policy.
- Sufficient evidence exists for the Planner to assign one bounded next task;
  the Executor did not select or launch it.

## Changes and delivery

- Changed only `.agent/STATE.md` and `.agent/AGENT_REPORT.md`.
- No experiment, code, config, environment, artifact, scheduler or process was
  changed.
- Evidence commit: `a5794e7939ca6c0bb6bb3a9af092b04983d26af7`
- Push target: `origin/agent-work`
- Final worktree/push verification: recorded by the follow-up delivery commit;
  `origin/agent-work` is verified after push.

TASK_COMPLETE
