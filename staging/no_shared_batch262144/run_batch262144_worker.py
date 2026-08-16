#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import numpy as np


ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
]
METHODS = ["ppo", "curv256_criticggn"]
SEEDS = range(5)
NUM_SHARDS = 4
TOTAL_TIMESTEPS_ARG = 130_809_855
ACTUAL_UPDATES = 500
ACTUAL_SAMPLES = 131_072_000
PPO_REPO = Path("/scratch/h99859yz/trust-region-main")
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
BAD_RE = re.compile(
    r"CUDA out of memory|Traceback \(most recent call last\)|\bnan\b|\binf\b",
    re.IGNORECASE,
)
BENIGN_CLEANUP_RE = re.compile(
    r"Traceback \(most recent call last\):.*?"
    r"OSError: \[Errno 39\] Directory not empty: '[^']+'\s*",
    re.IGNORECASE | re.DOTALL,
)


def all_tasks():
    # Interleaving methods makes shards 0/2 PPO and shards 1/3 curv256. Each
    # method is therefore represented on one A100 and one L40S.
    return [
        (method, env, seed)
        for env in ENVS
        for seed in SEEDS
        for method in METHODS
    ]


def command_for(method, python, repo, env, seed, timesteps):
    if method == "ppo":
        return [
            python,
            "-u",
            str(PPO_REPO / "train_detach.py"),
            "--config",
            "ppo_mlp_detach_fixedlr_public_batch262144.yaml",
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
        ]

    return [
        python,
        "-u",
        str(repo / "train_detach_jointcritic.py"),
        "--config",
        "rat_mlp_detjc_curv256_criticggn_batch262144.yaml",
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
        "0.03",
        "--fisher_kernel",
        "exact",
        "--fisher_kernel_normalization",
        "none",
        "--actor_curvature_subsample",
        "256",
        "--critic_curvature_subsample",
        "256",
        "--critic_update",
        "gn",
        "--critic_optimizer",
        "sgd",
        "--no-actor_subsample_full_batch_gradient",
    ]


def gpu_memory_used_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except Exception:
        return -1


def run_command(command, repo, stdout_path, stderr_path, peak_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        peak_mib = gpu_memory_used_mib()
        while process.poll() is None:
            peak_mib = max(peak_mib, gpu_memory_used_mib())
            time.sleep(2)
        return_code = process.wait()
        peak_mib = max(peak_mib, gpu_memory_used_mib())
    peak_path.write_text(f"{peak_mib}\n")
    return return_code


def parse_last10(path):
    values = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = EPREW_RE.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        return float("nan"), 0
    return float(np.mean(values[-10:])), len(values)


def successful(task_dir):
    rc_path = task_dir / "rc"
    return rc_path.exists() and rc_path.read_text().strip() == "0"


def has_bad_output(text):
    return BAD_RE.search(BENIGN_CLEANUP_RE.sub("", text)) is not None


def run_smoke(method, python, repo, run_root, shard):
    smoke_dir = run_root / "preflight" / f"shard{shard}_{method}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    if successful(smoke_dir):
        text = (smoke_dir / "stdout").read_text(errors="replace") + "\n" + (
            smoke_dir / "stderr"
        ).read_text(errors="replace")
        if not has_bad_output(text):
            (smoke_dir / "status").write_text("FINISHED\n")
            return
    command = command_for(method, python, repo, "halfcheetah", 9000 + shard, 1)
    (smoke_dir / "command.txt").write_text(shlex.join(command) + "\n")
    (smoke_dir / "status").write_text("RUNNING\n")
    task_repo = PPO_REPO if method == "ppo" else repo
    return_code = run_command(
        command,
        task_repo,
        smoke_dir / "stdout",
        smoke_dir / "stderr",
        smoke_dir / "peak_gpu_memory_mib.txt",
    )
    (smoke_dir / "rc").write_text(f"{return_code}\n")
    text = (smoke_dir / "stdout").read_text(errors="replace") + "\n" + (
        smoke_dir / "stderr"
    ).read_text(errors="replace")
    if return_code != 0 or has_bad_output(text):
        (smoke_dir / "status").write_text("FAILED\n")
        raise RuntimeError(f"smoke failed for {method} on shard {shard}, rc={return_code}")
    (smoke_dir / "status").write_text("FINISHED\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shard", required=True, type=int, choices=range(NUM_SHARDS))
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    shard_root = run_root / f"shard{args.shard}"
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "status").write_text("PREFLIGHT\n")
    tasks = all_tasks()[args.shard::NUM_SHARDS]
    methods = sorted({method for method, _, _ in tasks})

    try:
        for method in methods:
            run_smoke(method, args.python, repo, run_root, args.shard)
        (shard_root / "status").write_text("RUNNING\n")
        metric_rows = []

        for task_index, (method, env, seed) in enumerate(tasks):
            task_dir = run_root / method / env / f"seed{seed}"
            task_dir.mkdir(parents=True, exist_ok=True)
            if successful(task_dir):
                value, count = parse_last10(task_dir / "stdout")
                metric_rows.append((method, env, seed, value, count))
                continue

            command = command_for(
                method, args.python, repo, env, seed, TOTAL_TIMESTEPS_ARG
            )
            (task_dir / "command.txt").write_text(shlex.join(command) + "\n")
            (task_dir / "status").write_text("RUNNING\n")
            (shard_root / "current_task.txt").write_text(
                f"{task_index + 1}/{len(tasks)} {method} {env} seed{seed}\n"
            )
            task_repo = PPO_REPO if method == "ppo" else repo
            return_code = run_command(
                command,
                task_repo,
                task_dir / "stdout",
                task_dir / "stderr",
                task_dir / "peak_gpu_memory_mib.txt",
            )
            (task_dir / "rc").write_text(f"{return_code}\n")
            if return_code != 0:
                (task_dir / "status").write_text("FAILED\n")
                raise RuntimeError(
                    f"formal task failed: {method} {env} seed{seed}, rc={return_code}"
                )
            (task_dir / "status").write_text("FINISHED\n")
            value, count = parse_last10(task_dir / "stdout")
            metric_rows.append((method, env, seed, value, count))

        with (shard_root / "seed_metrics.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                ["method", "env", "seed", "mean_last10", "n_eprew_values"]
            )
            writer.writerows(metric_rows)
        (shard_root / "current_task.txt").write_text("none\n")
        (shard_root / "status").write_text("FINISHED\n")
    except Exception as exc:
        (shard_root / "failure.txt").write_text(repr(exc) + "\n")
        (shard_root / "status").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
