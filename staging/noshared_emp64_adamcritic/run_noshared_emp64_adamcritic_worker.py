#!/usr/bin/env python3
from pathlib import Path

import run_noshared_emp64_ggn64_worker as base


def command(python, repo, env, seed, timesteps):
    return [
        python,
        "-u",
        str(Path(repo) / "train_detach_jointcritic.py"),
        "--config",
        "rat_mlp.yaml",
        "--env_name",
        env,
        "--seed",
        str(seed),
        "--device",
        "0",
        "--timesteps_per_proc",
        str(timesteps),
        "--pi_epochs",
        "4",
        "--lr_pi",
        "0.05",
        "--cg_damping",
        "0.1",
        "--fisher_kernel",
        "exact",
        "--fisher_kernel_normalization",
        "none",
        "--actor_curvature_subsample",
        "64",
        "--critic_curvature_subsample",
        "0",
        "--critic_update",
        "standard",
        "--critic_optimizer",
        "adam",
        "--actor_subsample_full_batch_gradient",
    ]


base.command = command
base.main()
