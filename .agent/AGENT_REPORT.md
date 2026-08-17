# Executor Report

## Metadata

- Task-ID: `MUJOCO-DUAL5060-SWIMMER-RERUN-20260817-03`
- Execution window: `2026-08-17T13:52:35Z` to `2026-08-17T16:10:09Z`
- Starting HEAD: `7639d44200bb9539ddeadb32444a6e9d8f237a07`
- Evidence commit: `70edbbd27761ce5d2826b58995b0b6b2a2de5682`
- Branch/target: source state `agent-work`; push target `origin/agent-work`
- Formal source commit: `df9d5e18279d096218a923ad5d4df37c35fdca68`
- Formal run root:
  `/home/zzz/rlstack5060/workspaces/perf_runs/global7env_selected10m_5060_20260816_swimmerv3_rerun_20260817`
- Original immutable root:
  `/home/zzz/rlstack5060/workspaces/perf_runs/global7env_selected10m_5060_20260816`

## Refreshed live state

| Plane | Pre-run evidence | Post-run evidence |
|---|---|---|
| CSF3 | no running/pending MuJoCo job at `13:52:35Z` | no MuJoCo action taken |
| Bede | direct login restored; `1072326_0-17` scheduler-complete | exact cell-to-artifact mapping still unverified; no scientific completion claim |
| dual-5060 | GPUs 0/1: 0%, 33/15 MiB; no trainer/container; 335 GiB free | GPUs 0/1: 0%, 33/15 MiB; no trainer/container; disk 29% used |
| selected batch | original 30 `FINISHED`, five Swimmer-v3 `FAILED` | original unchanged; five linked reruns `FINISHED`, bundle `rc=0` |

No `.54` / `ws4090-31` access occurred. No Jupyter session was created or
found, so the one-hour idle-Jupyter rule had no target.

## Immutable five-cell launch table

All formal rows are M2 no-shared single-layer Transformer cells with
Swimmer-v3, 10,000,000 requested timesteps / 9,994,240 expected logged steps,
rollout 8,192, actor/critic 4 epochs x 8 minibatches, momentum 0.5/0.5,
damping 0.10, `fisher_l2`, max norm 0.5 and KL target 0.008.

| Line | Method | Seed | GPU | Trainer | Config | Original and linked output suffix |
|---|---|---:|---:|---|---|---|
| M2 | Emp256 | 2 | 0 | `train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py` | `rat_transformer_detjc_emp256_ggn256.yaml` | `emp256_mom0.5_d0.10_fisher_l2_kl008_10m/emp256/swimmer/seed2` |
| M2 | EF255+1 | 1 | 0 | `train_detach_energyfree255p1_criticggn256_batch262144.py` | `rat_transformer_detjc_energyfree255p1_ggn256.yaml` | `energyfree255p1_mom0.5_d0.10_fisher_l2_kl008_10m/energyfree255p1/swimmer/seed1` |
| M2 | EF255+1 | 6 | 1 | same EF255+1 trainer | same EF255+1 config | `energyfree255p1_mom0.5_d0.10_fisher_l2_kl008_10m/energyfree255p1/swimmer/seed6` |
| M2 | FullEmp | 0 | 0 | `train_detach_jointcritic_actor_fisherclip.py` | `rat_transformer_detjc_full_emp_full_ggn.yaml` | `fullemp_mom0.5_d0.10_fisher_l2_kl008_10m/fullemp/swimmer/seed0` |
| M2 | FullEmp | 5 | 1 | same FullEmp trainer | same FullEmp config | `fullemp_mom0.5_d0.10_fisher_l2_kl008_10m/fullemp/swimmer/seed5` |

The executed launcher SHA256 was
`939d7c4a82a54b074c6100dcb28b2c0ccf516d8a9afee50ddb38f91dff4fb3b0`.
The committed version additionally refuses a duplicate launch whenever a
formal status file already exists. Original failure files were never
overwritten.

## Dependency repair and rollback

Root cause was not algorithmic: Gymnasium 1.0 dispatches Swimmer-v3 through
deprecated `mujoco_py`, which was absent from the shared Python 3.11 image.
The first isolated preflight then exposed the archived wheel's stale generated
C code. The final isolated fix used:

- base image ID
  `8c26347d1f50c73192c1fff666a8cc14ca7319f3f78588d878aba0b5c9b7217b`;
- MuJoCo `2.1.0` binaries and `mujoco-py==2.1.2.14`;
- `Cython==0.29.37` and `fasteners==0.20`;
- pinned Ubuntu GL/GLFW/GLEW/OSMesa development/runtime packages plus
  `patchelf`;
- a Python-3.11 header compatibility link and regeneration of `cymj.c` from
  the package's own `cymj.pyx` with pinned Cython.

The shared base was unchanged. Rollback is removal of only
`rlstack5060/mujoco-rat-swimmerv3:cu128`. Exact Python and system package diffs
are preserved in `evidence/package_diff.txt` under the rerun root.

## Preflight

- Environment smoke: Python 3.11.12, Gymnasium 1.0.0, `mujoco-py 2.1.2.14`;
  reset returned `(obs, info)`, observation `(8,)`, action `(2,)`, step API
  length 5, then `ENV_SMOKE_OK`.
- M1 constructor-only smoke: MLP config, value `(1,)`, policy `(1,4)`, 136,968
  parameters, `optimizer_constructed=false`, `learning_steps=0`.
- M2 constructor-only smoke: Transformer config, value `(1,)`, policy `(1,4)`,
  71,752 parameters, `optimizer_constructed=false`, `learning_steps=0`.
- A quarantined attempt showed that the trainer's nominal zero-timestep input
  still enters its floor-plus-two update loop. It was interrupted during the
  second update, never used as a formal cell, and replaced by the verified
  constructor-only diagnostic. Evidence is retained in
  `preflight/m1/zero_timestep_incident.txt`.

## Terminal formal results

The native values below are the final `progress.csv` row at update 1,219.
`kl` is retained with its source-defined semantics.

| Method | Seed | Status | RC | Steps | Final reward | Final KL | GPU | Elapsed s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Emp256 | 2 | completed-valid | 0 | 9,994,240 | 272.9407 | 0.0034261 | 0 | 6,145.94 |
| EF255+1 | 1 | completed-valid | 0 | 9,994,240 | 5.8639 | 0.0179583 | 0 | 6,986.26 |
| EF255+1 | 6 | completed-valid | 0 | 9,994,240 | 24.3793 | 0.0588235 | 1 | 6,860.62 |
| FullEmp | 0 | completed-valid | 0 | 9,994,240 | 15.0778 | 0.0049545 | 0 | 6,468.20 |
| FullEmp | 5 | completed-valid | 0 | 9,994,240 | 59.1063 | 0.0025145 | 1 | 6,320.29 |

Each row has fresh `run_info.txt`, `stdout.log`, `stderr.log`, native
`progress.csv`, TensorBoard events, `rc`, `status` and `finished_at.txt`.
Terminal progress mtimes range from `2026-08-17T23:55:11+08:00` through
`2026-08-18T00:09:12+08:00`.

## Strict matching and failure audit

- Every trainer, config and `utils/mujoco_transformer.py` SHA256 matches the
  corresponding original failed row exactly (`15/15` field checks passed).
- Method, M2 identity, environment/version, seed, architecture, rollout,
  epochs/minibatches, budget and all explicit optimizer/clip/KL arguments match.
- The only intentional difference is the isolated dependency layer and linked
  output root.
- Formal recursive scans found no NaN/Inf, OOM, traceback, assertion,
  `LinAlgError`, dependency, disk or permission marker.
- Accessible Bede PPO/K-FAC Swimmer-v3 baselines have all five seeds failed for
  missing `mujoco_py`; they are infrastructure failures, not valid scientific
  baselines. The 3/5 early-stop-candidate comparison is therefore not
  assessable. No run was scientifically stopped.

Failure classification for this cycle: five historical
`infrastructure/dependency` failures preserved; five linked reruns
`completed-valid`; zero new algorithmic, numerical, scheduler/quota or
infrastructure failures.

## Completeness and preserved provenance

- Original batch: `30/35` finished, five failed records retained.
- Successful linked recovery: `5/5`.
- Effective valid completion count: `35/35`; historical failure count remains
  five and is not deleted or rewritten.
- `18302268_10` remains unresolved/unmapped.
- Bede `1072326_0-17` remains scheduler-complete; direct access is restored but
  exact scientific artifact mapping remains unverified.

## Delivery

Changed files:

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `transformer_noshared_stage/rtx5060/Dockerfile.mujoco-rat-swimmerv3`
- `transformer_noshared_stage/rtx5060/run_swimmerv3_rerun_20260817.sh`
- `transformer_noshared_stage/rtx5060/swimmerv3_model_init_smoke.py`

Evidence commit: `70edbbd27761ce5d2826b58995b0b6b2a2de5682`.
Push result: `origin/agent-work` updated through the evidence and report
commits, then verified against the remote ref.

The only recommended next action is a bounded read-only audit that establishes
a valid strict five-seed M2 Swimmer-v3 baseline mapping before scientific
interpretation of these endpoints.

TASK_COMPLETE
