# Executor Report

## Metadata

- Task-ID: `MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R`
- Status: `RUNNING`
- Inspection/launch start: `2026-08-24T13:14:30Z`
- Starting HEAD: `061c0a0d1b23ae70254fbc788e92afb02295238f`
- Target: `origin/agent-work`
- Planner thread: `6a8309f0-0bd0-83eb-a4ab-3ad1227b2e1c`

## Corrected scientific identity

The first Planner response incorrectly selected the M2 Transformer FullEmp
line. Live audit showed that M2 has no strict seven-environment momentum `0`
or `0.9` reference, while the user's preceding comparison was the no-shared
large-batch MLP Full-EF actor plus Full-GGN critic line. The same ChatGPT
Planner corrected the task before any new formal cell was launched.

Canonical parent is CSF3 job `17491617`:

- root: `/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/perf_runs/csf3_detjc_exact_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491617`;
- seven environments, manifest seeds `0,1,2,3,4`; deterministic `S2=0,1`;
- trainer SHA256 `16b740589dd6960a20c95b28cd1f87623724ff7ca5189b275d80607cc759d05d`;
- config SHA256 `78c38c206c0f32aef90690af18718129bf34a39b974ed3ebcd6655b8fe823d86`;
- actual low momentum is `1e-6` for both independent SGD optimizers;
- MLP 256x2 actor and critic, 32x256 rollout, e4x8, actor LR 0.05,
  critic LR 0.1, damping 0.03, exact kernel, normalization none,
  Kaczmarz false, full 1024 actor/critic rows and 10M nominal budget;
- the executed full-curvature RAT actor branch applies
  `torch.nn.utils.clip_grad_norm_(..., 0.5)`, i.e. parameter L2 clipping,
  despite the inherited configuration label `post_grad=fisher_clip`.

## Existing S2 reference gate

Reference root:
`/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/perf_runs/csf3_detach_smallbatch_momentum05_09_dual255p1_curv256_full_all7_2seed_10m_gpuL_20260807`.

For method `full`, momentum `0.5` and `0.9`, every one of the seven
environment directories is `FINISHED`; all 28 seed RC files for seeds 0 and 1
are zero. Commands fix full actor/critic rows, damping 0.03, exact kernel,
normalization none, GN/SGD critic, LR 0.05/0.1 and common momentum. Reference
trainer SHA256 is
`aa4adb4daba92519e3286d3b735182d72e4c5e22c911b57cd0d91b8e9d91108a`.
Its additional energy-coordinate code is inactive for method `full`.

No low/0.5/0.9 reference was restarted.

## Fresh resource state and placement

At launch refresh:

- CSF3: no MuJoCo job; unrelated Procgen jobs occupied the user's current GPU
  QOS allocation.
- dual-5060: both RTX 5060 Ti cards 0%, 33/15 MiB, no trainer/container,
  316 GiB free.
- Bede: no user jobs, many V100 nodes reported idle, 723 GiB available and a
  write/read/delete probe under the project perf root passed.

Placement keeps both momenta for a given environment on the same host class:

| Host | Environments | Momenta | Seeds | Current state |
|---|---|---|---|---|
| dual-5060 | Ant, Hopper, Swimmer, Walker2d | 0.7, 0.8 | 0,1 | running local queues |
| dual-5060 tail | HalfCheetah, Humanoid, HumanoidStandup | 0.7, 0.8 | 0,1 | queued after primary local queues |

Bede submission was attempted twice. Both `sbatch` calls failed before job-ID
creation with `Requested node configuration is not available`. Historical
successful ReqTRES from `1072326` was account `bdman37g`, 32 CPUs,
129872 MiB and one GPU. The same current request and test-only variants for
`gpu`, typed V100, `full`, `half`, and GH partitions were rejected. A live
audit then showed 23 idle V100 nodes and other accounts running jobs submitted
today with the identical 1-GPU/32-CPU/129872-MiB request. The `yihe`
associations for `bdman37`, `bdman37g`, and `hpcuser` still exist, but even a
five-minute test-only request pinned to idle `gpu008` fails. This isolates the
failure to current `bdman37g` GPU allocation/eligibility rather than the node
shape, source, artifacts, or queue occupancy. Bede is therefore classified
scheduler/account infrastructure unavailable, not scientifically failed.
Empty evidence roots from the two rejected submissions are retained.

## New source and preflight

The new trainer is the prior matched momentum trainer plus one runtime-only
actor/critic momentum print. SHA256:
`04c87fcd0af1e351f91a2ae4b1bbb0dc19fe419e5878e33d374c36c50b15fbdc`.
Worker SHA256:
`1df1ee282892d067832d41c5e3927c18c52d1d1ee80429acb58318807972c69b`.

dual-5060 uses the isolated Swimmer-compatible image
`rlstack5060/mujoco-rat-swimmerv3:cu128`, image ID
`f1ca97dd1d845b7ab13438f9064bbc7dd0b101a88692781170f797da1e774057`,
and canonical CSF3 `utils`, `vec_env` and configuration copied into a
non-colliding task source directory.

Both 81,920-step Ant preflights completed:

| Momentum | Seed | RC | Runtime telemetry | Last logged step | Error scan |
|---:|---:|---:|---|---:|---|
| 0.7 | 970 | 0 | actor=0.7, critic=0.7 | 81,920 | clean |
| 0.8 | 980 | 0 | actor=0.8, critic=0.8 | 81,920 | clean |

They exercised environment construction and twelve full update cycles with
finite KL, actor gradient norm, critic gradient norm and critic step norm.

## Formal live state

dual-5060 root:
`/home/zzz/rlstack5060/workspaces/perf_runs/dual5060_mlp_fullEF_fullGGN_momentum0708_s2_10m_20260824`.

At the latest snapshot, momentum 0.7 Ant seed0 had reached 2.62M steps and
momentum 0.8 Ant seed0 had reached 2.54M steps. Both trainer processes remained
live, and no OOM, NaN/Inf, traceback, dependency, disk or permission marker was
found.
Each cell runs seed0 then seed1; later environments are queued behind the
current cell on each physical GPU.

At the user's direction this batch will not run on CSF3. Jobs `19206549` and
`19206550` were cancelled: preflight element `19206549_0` completed before the
cancellation reached it, while `19206549_1` and all formal elements were
cancelled before training. Thus no CSF3 formal cell ran. The twelve remaining
HalfCheetah/Humanoid/HumanoidStandup formal cells are queued as a non-colliding
dual-5060 tail (PID `19130`) which starts only after the primary two-GPU queues
finish.

## Preserved provenance

- Original M2 selected batch remains 30/35 plus five immutable Swimmer-v3
  dependency failures; linked isolated rerun remains 5/5 completed-valid.
- Bede `1072326_0-17` remains scheduler-complete but scientifically unmapped.
- `18302268_10` remains unresolved/unmapped.
- No `.54` / `ws4090-31`, Jupyter, Procgen or Isaac action was taken.

## Changed files

- `.agent/TASK.md`
- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `momentum_smallbatch_stage/train_detach_smallbatch_momentum.py`
- `momentum0708_mlp_stage/run_full_momentum_cell.py`
- dual-5060 primary/tail launchers and Bede/CSF3 preflight, formal and
  submission scripts

Formal runs remain active. Final metrics, paired analysis, final commit and
Planner callback are pending terminal completion.

TASK_RUNNING
