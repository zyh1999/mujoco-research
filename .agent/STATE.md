# Current Project State

Updated: 2026-08-16

## Established lines

- M1 large-batch: Curv256 + critic GGN256, K-true/K-opt, EnergyFree255+1,
  FullEmp/FullGGN, approximately 500 full updates / 128.45M transitions.
- M2 small-batch no-shared Transformer: PPO, K-FAC, Emp256, EF255+1, and
  FullEmp across Ant, HalfCheetah, Hopper, Walker2d, Humanoid,
  HumanoidStandup, and Swimmer.

## Current distributed work at the latest handoff

- Bede array `1072326_0-17` had completed/running/resource-pending cells, but
  the Bede channel must be refreshed before treating that snapshot as current.
- The dual-5060 host was running eight mapped 3M Transformer missing-cell
  workers under
  `/home/zzz/rlstack5060/workspaces/perf_runs/global7env_missing52_5060_20260816`.
- `ws4090-92` contains completed/failed M1 K-opt and Humanoid evidence and was
  sampled idle; `ws4090-76` contains the interrupted EF255+1 Humanoid anchor.
- CSF3 historical array `18302268` is mostly terminal, with element 10 needing
  exact failure mapping.

## Evidence rules

- Compare the same architecture, task/version, seed set, budget, and last-10
  reward convention.
- Accept 128,450,560 as the final full-update boundary when terminal artifacts
  exist; it is not a rerun trigger for a nominal 130M budget.
- Keep packaging failures, incomplete continuations, numerical failures, and
  completed results distinct.

All live state must be refreshed before execution.
