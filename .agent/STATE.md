# Current Project State

Updated: 2026-08-24T13:59:00Z

## Active bounded task

- `MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R` is running.
- Scientific identity is the no-shared large-batch MLP Full-EF actor plus
  Full-GGN critic line, not the M2 Transformer FullEmp line.
- Canonical `S2` is seeds `0,1`, selected by manifest order. Canonical
  low-momentum is the literal PyTorch SGD value `1e-6` from CSF3 job
  `17491617`; damping is `0.03`, normalization `none`, Kaczmarz false and the
  executed full-curvature actor branch uses parameter L2 clipping at `0.5`.
- Existing momentum `0.5` and `0.9` S2 references are strict-complete across
  all seven environments under the CSF3 momentum matrix; no reference rerun
  was launched.
- dual-5060 preflights for momentum `0.7` and `0.8` completed with `rc=0` and
  runtime actor/critic momentum telemetry matched exactly. Formal Ant cells
  are running, with the remaining Ant/Hopper/Swimmer/Walker2d cells queued on
  the two local GPU workers.
- Bede was live and writable, but both submission attempts and scheduler test
  requests were rejected before job creation as `Requested node configuration
  is not available`. Current jobs from other accounts prove the identical
  1-GPU/32-CPU/129872-MiB shape is valid and 23 V100 nodes are idle. The
  `yihe` associations still exist, so this is isolated to current `bdman37g`
  GPU allocation/eligibility rather than code, queue occupancy, or an
  algorithm result.
- At the user's direction, CSF3 jobs `19206549` and `19206550` were cancelled.
  One short preflight element completed before cancellation; no CSF3 formal
  training cell ran. The remaining HalfCheetah/Humanoid/HumanoidStandup cells
  are queued as a tail on dual-5060 after its current four-environment queues.

## Previous completed state (2026-08-17)

## Established lines

- M1 large-batch remains Curv256 + critic GGN256, K-true/K-opt,
  EnergyFree255+1 and FullEmp/FullGGN at the accepted terminal boundary of
  128,450,560 transitions.
- M2 remains the small-batch no-shared single-layer Transformer comparison of
  PPO, K-FAC, Emp256, EF255+1 and FullEmp over seven MuJoCo environments.
- M1 and M2 were not merged or substituted in this task. All five formal
  Swimmer-v3 recovery cells are M2; M1 was exercised only by a zero-learning
  compatibility/model-construction smoke test.

## Fresh distributed state

- CSF3 had no running or pending MuJoCo job at `2026-08-17T13:52:35Z`.
- Direct Bede login became available in this cycle. Array `1072326_0-17`
  remains scheduler-complete, but its exact per-cell artifact/root mapping was
  not reconciled, so it is not treated as scientific completion.
- dual-5060 task `MUJOCO-DUAL5060-SWIMMER-RERUN-20260817-03` completed all five
  authorized M2 Swimmer-v3 reruns with `rc=0` and 9,994,240 logged steps.
- The original batch remains immutable at 30 `FINISHED` and five dependency
  `FAILED` records. The linked quarantine root adds five completed-valid
  records, so the effective matched batch now has 35 valid completions without
  deleting the five historical failures.
- The isolated compatibility image is
  `rlstack5060/mujoco-rat-swimmerv3:cu128`, image ID
  `f1ca97dd1d845b7ab13438f9064bbc7dd0b101a88692781170f797da1e774057`.
  The shared base image was not modified.
- Post-run GPUs 0/1 were idle at 33/15 MiB, with no training process or
  container. Disk remained healthy at 29% used.
- No Jupyter session was created or found. Quarantined `ws4090-31` / `.54` was
  not accessed.

## Evidence and unresolved provenance

- All rerun trainer/config/Transformer SHA256 values match their corresponding
  original failed records exactly. Formal source commit is
  `df9d5e18279d096218a923ad5d4df37c35fdca68`.
- Accessible Bede M2 PPO and K-FAC Swimmer-v3 five-seed artifacts are themselves
  dependency failures. There is therefore no valid highest strict Swimmer-v3
  baseline for the 3/5 early-stop-candidate rule; no stop was applied.
- Historical CSF3 element `18302268_10` remains unresolved and unmapped. It was
  not guessed or reassigned.
- Native `progress.csv` `kl` remains source-defined logged KL and is not
  relabeled as another reference-policy convention.

## Next bounded decision

The only recommended next action is a Planner-authored read-only scientific
audit that first establishes a valid strict five-seed M2 Swimmer-v3 baseline
mapping before interpreting or ranking these five recovered endpoints.
