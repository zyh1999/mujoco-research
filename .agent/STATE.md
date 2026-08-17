# Current Project State

Updated: 2026-08-17T13:37:04Z

## Established lines

- M1 large-batch: Curv256 + critic GGN256, K-true/K-opt, EnergyFree255+1,
  FullEmp/FullGGN, approximately 500 full updates / 128,450,560 transitions.
- M2 small-batch no-shared Transformer: PPO, K-FAC, Emp256, EF255+1 and
  FullEmp over Ant, HalfCheetah, Hopper, Walker2d, Humanoid,
  HumanoidStandup and Swimmer-v3.

## Fresh distributed state

- CSF3 has no running MuJoCo GPU job. Historical array `18302268_0-9,11-41`
  remains terminal-complete; element `18302268_10` remains a preserved failure
  whose exact cell/root/cause is unmapped.
- Bede array `1072326_0-17` is scheduler-complete in the CSF3 controller's
  fresh accounting snapshot. Direct Bede login failed noninteractive
  authentication, so exact cells, rewards, KL and artifacts remain unverified.
- The new dual-5060 endpoint `47.114.81.212:60023` is reachable and idle:
  two RTX 5060 Ti GPUs at 0%, 33 MiB and 15 MiB; no training PID.
- Its `global7env_selected10m_5060_20260816` batch is terminal: 30/35 runs
  finished at 9,994,240 steps and five Swimmer-v3 runs failed before training.
- All five failures have the same infrastructure/dependency cause:
  `gymnasium.error.DependencyNotInstalled` because Swimmer-v3 requires
  deprecated `mujoco_py`. They are not algorithmic or numerical failures.
- `ws4090-92` and `ws4090-76` have only historical controller telemetry in
  this cycle; do not infer free capacity or current process state.
- `ws4090-31` / `10.49.7.54` remains quarantined and is zero capacity.

## Evidence rules and Planner blockers

- A terminal 128,450,560-transition M1 run with terminal artifacts is complete;
  nominal 130M must not cause a rerun.
- The 5060 `progress.csv` `kl` field is preserved as the run's native logged
  KL. Its exact reference-policy semantics require source confirmation before
  cross-family comparison.
- No strict five-seed baseline mapping was available for the selected 10M
  cells, so the 3/5 early-stop rule was not applied.
- Next planning must decide whether to repair Swimmer-v3 compatibility or to
  prioritize another bounded missing-cell/provenance task. The Executor did
  not make that research choice.
