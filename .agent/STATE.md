# Current Project State

Updated: 2026-08-25T06:51:29Z

## Concurrent Wave 2 launch

- The user explicitly authorized concurrent Wave 2 on six additional Bede
  V100s. ChatGPT Planner task
  `MUJOCO-KTRUE-CONCURRENT-WAVE2-LAUNCH-VERIFY-20260825-11` replaced the prior
  no-overlap limit; assignment commit is `a094ebc`.
- Wave 2 `1074588_[0-5]` launched all 18 remaining frozen cells without a
  Wave 1 dependency. Wave 1 and Wave 2 use disjoint physical GPU identities:
  Wave 1=`gpu024:{0,1,2,3},gpu025:{0,2}`; Wave 2=
  `gpu025:{3},gpu026:{0,1,2,3},gpu027:{0}`.
- All 18 Wave 2 command/PID artifacts exist. Fourteen non-Swimmer cells are
  running-verified with exact matched momentum, Kaczmarz no-history and finite
  actor/critic residual telemetry; their startup error scan is clean. The four
  Wave 2 Swimmer cells are classified failed-infrastructure/dependency due to
  the unchanged missing `mujoco_py`, with no retry or repair.
- Wave 1 remained untouched and live: 22 running, two previously classified
  momentum-0.5 Swimmer dependency failures. Current combined live trainers
  are 36 after the six Swimmer dependency exits; peak formal attempts remain
  bounded at 42 and no GPU exceeded four trainer launches.

## Current config-path-corrected K=true execution

- `MUJOCO-KTRUE-CONFIGPATH-PREFLIGHT-AND-BEDE42-20260825-09` is running.
  Assignment commit is `08c56ea`; path-only source commit is `39a789d`.
- Bede path semantics passed: reporting and trainer both resolve the frozen
  stage config realpath, SHA256 is
  `cfe6e0b87f51b998e4c5b4f2315446a9f591f7cf5a7b8fbfa979c98ba5acffef`,
  and the parsed value is Kaczmarz true. The old `configurations/` path fails
  explicitly with `FileNotFoundError`; there is no search or fallback.
- The repeated fixed-input residual audit passed RC zero with every parameter,
  optimizer/momentum state, direction, previous projection, RNG, loss, KL and
  VF field bitwise equal; actor and critic residuals remained finite.
- The only real new preflight round, `1074579_[0-2]`, completed 3/3 with RC
  zero. Every momentum has matching actor/critic runtime value, K=true identity,
  one no-history marker, four real projection-use markers, four finite actor
  and critic residuals, finite training telemetry and a clean error scan.
- Wave 1 job `1074582_[0-5]` is running on six one-GPU allocations: four V100s
  on `gpu024` and two on `gpu025`. It created 24 trainer commands/PIDs, exactly
  four trainers per allocated card and 24 globally. Twenty-two cells are
  running. The two momentum-0.5 Swimmer seeds failed during environment
  construction because Bede lacks `mujoco_py`; this is the preserved known
  Bede dependency failure, not a Kaczmarz/numerical result. No retry, bypass,
  cancellation or alternate host was used.
- Wave 2 has not been submitted and cannot overlap Wave 1.

## Active bounded task

- `MUJOCO-KTRUE-TELEMETRY-PREFLIGHT-AND-BEDE42-20260825-08` is blocked after
  its single authorized telemetry-corrected preflight round. The fixed-input
  telemetry nonintervention audit passed bitwise for actor/critic parameters,
  optimizer state, directions, previous-projection buffer, RNG, loss, KL and
  VF. Telemetry source commit is `19f86c8`.
- New Bede preflight `1074576_[0-2]` failed before environment construction or
  optimizer/update in 2-3 seconds. The reporting worker looked only under
  `$REPO/configurations/` for the K=true config, but this isolated task keeps
  the config under `$REPO/ktrue_momentum_stage/`. All three elements therefore
  exited with the same `FileNotFoundError`. This is a telemetry/reporting path
  bug, not Kaczmarz, numerical or GPU evidence.
- The task prohibits any further telemetry modification or second preflight
  after a failure. No formal job was submitted; formal launch count remains
  zero and all 42 cells remain unstarted. A new Planner authorization is
  required for a bounded config-path resolution correction and one new
  isolated preflight.
- `MUJOCO-KTRUE-GLFW-PREFLIGHT-AND-BEDE42-20260825-07` is blocked at its formal
  launch gate after its single authorized corrected preflight attempt. The
  launcher-only GLFW correction succeeded and job `1074573_[0-2]` completed
  all three momentum settings with RC zero.
- Runtime evidence for each of momentum 0.5/0.7/0.8 shows exact matched actor
  and critic momentum, one first-update `KACZMARZ_NO_HISTORY`, and four later
  `KACZMARZ_PROJECTION_USED` calls with seven nonempty buffers and finite
  projection norms. GLFW/MuJoCo construction and finite KL, actor-gradient,
  critic-gradient and critic-step telemetry also passed.
- The frozen worker produced no actor/critic solver-residual telemetry, while
  its auxiliary `run_info.txt` still hardcodes `kaczmarz=false` despite
  unambiguous runtime `karzmarz_True` and projection-use evidence. The READY
  task requires finite solver-residual evidence and prohibits worker/trainer
  changes or a second corrected preflight retry, so the formal gate cannot be
  marked PASS. Formal launch count remains zero; all 42 cells are unstarted.
- The corrected isolated root is
  `/nobackup/projects/bdman37/yihe/perf_runs/bede_mlp_fullEF_fullGGN_ktrue_m050708_s01_10m_20260825_glfwfix1`.
- Assignment commit is `52dcce1`; corrected-launcher commit is `f209f22`.
- The new isolated matrix is no-shared large-batch MLP Full-EF actor plus
  Full-GGN critic, Kaczmarz true, damping 0.03, normalization none, parameter
  L2 clip 0.5, matched actor/critic momentum 0.5/0.7/0.8, seven environments
  and seeds 0/1: exactly 42 formal cells, at most one launch per cell.
- Only Bede is authorized. The frozen execution topology is six V100s, at
  most four trainers/card, Wave 1=24 cells and Wave 2=18 cells with no overlap.
- Bede refresh at `2026-08-25T05:20:56+01:00` found no user jobs, many idle
  V100 nodes, 790 TiB available under `/nobackup`, and a fresh write/read/delete
  probe passed. Canonical remote trainer/config hashes remain matched.

## Closed-preserved prior task

- `MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R` is
  `KFALSE_CLOSED_PRESERVED` by explicit user replacement.
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
  runtime actor/critic momentum telemetry matched exactly. The original two
  one-trainer GPU queues remain live and their partial Ant evidence is
  preserved. A proposed destructive reallocation was not performed because
  stopping those advancing cells was not separately authorized.
- Bede rejection was caused by explicit task/CPU topology in the submission
  scripts after the site's Slurm 25.11.7 upgrade, not account eligibility or
  capacity. With `JobSubmitPlugins=lua`, `--gres=gpu:1` passes and derives 32
  CPUs/129872 MiB; adding `--cpus-per-task=32` or even `--ntasks=1` fails.
  Corrected exact-file test-only checks passed. The earlier serial formal job
  `1074306_[0-5]` was snapshotted and cancelled as infrastructure reallocation,
  not algorithm failure, when the user requested four trainers per card.
- The replacement Bede topology is exactly six one-GPU array elements, one
  environment per card, with momentum 0.7 and 0.8 parent workers concurrent
  and seeds 0 and 1 concurrent inside each parent: four trainers per card and
  24 formal cells total. The first 24-way topology preflight (`1074452`) proved
  concurrency but exposed the known Bede-only Swimmer `mujoco_py` dependency
  failure. Swimmer was removed from Bede; retry preflight `1074458_[0-5]`
  completed all 24 non-Swimmer seed runs with `rc=0` and a clean error scan.
  Formal array `1074464_[0-5]` completed all 24 cells with RC zero under the
  isolated retry4 root. The old dual-5060 queue is preserved as interrupted:
  two Ant seed0 cells completed, both seed1 cells stopped near 13%, and later
  environments did not start. None of those cells will be resumed or mixed
  into the K=true matrix.
- At the user's direction, CSF3 jobs `19206549` and `19206550` were cancelled.
  One short preflight element completed before cancellation; no CSF3 formal
  training cell ran. The remaining HalfCheetah/Humanoid/HumanoidStandup cells
  are now assigned to Bede. The duplicate dual-5060 tail was stopped before it
  launched any cell; the original four-environment dual-5060 queues continue.

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
