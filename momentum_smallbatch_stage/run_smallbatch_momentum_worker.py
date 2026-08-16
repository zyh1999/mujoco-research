#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import re
import subprocess


ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
METHODS = ["dual255p1", "curv256", "full"]
MOMENTA = [0.5, 0.9]
SEEDS = [0, 1]
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
FINITE_RE = re.compile(r"\|\s*(?:kl|actor_free_rho_value|critic_free_rho_value)\s*\|\s*([-+0-9.eE]+)\s*\|")
BAD_MARKERS = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "LinAlgError")


def decode_task(task_id):
    env = ENVS[task_id % len(ENVS)]
    quotient = task_id // len(ENVS)
    momentum = MOMENTA[quotient % len(MOMENTA)]
    method = METHODS[quotient // len(MOMENTA)]
    return method, momentum, env


def command(python, trainer, method, momentum, env, seed, timesteps):
    cmd = [
        python, "-u", trainer,
        "--config", "rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml",
        "--env_name", env,
        "--seed", str(seed),
        "--device", "0",
        "--timesteps_per_proc", str(timesteps),
        "--pi_epochs", "4",
        "--lr_pi", "0.05",
        "--sgd_momentum", str(momentum),
        "--cg_damping", "0.03",
        "--fisher_kernel", "exact",
        "--fisher_kernel_normalization", "none",
        "--critic_update", "gn",
        "--critic_optimizer", "sgd",
    ]
    if method == "dual255p1":
        cmd += [
            "--actor_curvature_subsample", "255",
            "--critic_curvature_subsample", "255",
            "--actor_subsample_free_rho", "anchor",
            "--critic_subsample_free_rho", "anchor",
        ]
    elif method == "curv256":
        cmd += [
            "--actor_curvature_subsample", "256",
            "--critic_curvature_subsample", "256",
            "--actor_subsample_free_rho", "none",
            "--critic_subsample_free_rho", "none",
        ]
    elif method == "full":
        cmd += [
            "--actor_curvature_subsample", "0",
            "--critic_curvature_subsample", "0",
            "--actor_subsample_free_rho", "none",
            "--critic_subsample_free_rho", "none",
        ]
    else:
        raise ValueError(f"unknown method: {method}")
    return cmd


def start(cmd, cwd, stdout_path, stderr_path):
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not prior_pythonpath else f"{cwd}:{prior_pythonpath}"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env)


def validate(stdout_path, stderr_path, method, momentum):
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    combined = stdout + "\n" + stderr
    if any(marker in combined for marker in BAD_MARKERS):
        raise RuntimeError("error marker found")
    values = [float(match.group(1)) for match in FINITE_RE.finditer(stdout)]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("missing or non-finite KL/rho diagnostics")
    if method == "dual255p1":
        for key in ("actor_free_rho", "critic_free_rho"):
            if not re.search(rf"\|\s*{key}\s*\|\s*1(?:\.0+)?\s*\|", stdout):
                raise RuntimeError(f"missing {key}=1")
    if f"--sgd_momentum {momentum}" not in (stdout_path.parent / "command.txt").read_text():
        raise RuntimeError("momentum command provenance mismatch")


def parse_last10(path):
    values = [float(match.group(1)) for match in EPREW_RE.finditer(path.read_text(errors="replace"))]
    if not values:
        raise RuntimeError(f"no eprewmean values in {path}")
    return sum(values[-10:]) / len(values[-10:]), len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--task-id", type=int, required=True, choices=range(42))
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    trainer = str(root / "train_detach_smallbatch_momentum.py")
    method, momentum, env = decode_task(args.task_id)
    task = root / method / f"momentum_{momentum}" / env
    task.mkdir(parents=True, exist_ok=True)
    (task / "status").write_text("PREFLIGHT\n")

    smoke = task / "preflight"
    smoke.mkdir(exist_ok=True)
    smoke_cmd = command(args.python, trainer, method, momentum, env, 0, 81920)
    (smoke / "command.txt").write_text(" ".join(smoke_cmd) + "\n")
    process = start(smoke_cmd, repo, smoke / "stdout", smoke / "stderr")
    rc = process.wait()
    (smoke / "rc").write_text(f"{rc}\n")
    if rc != 0:
        (task / "status").write_text("PREFLIGHT_FAILED\n")
        raise RuntimeError(f"preflight failed rc={rc}")
    validate(smoke / "stdout", smoke / "stderr", method, momentum)
    (smoke / "status").write_text("FINISHED\n")
    (task / "status").write_text("RUNNING\n")

    # Full 1024x1024 actor and critic systems are run one seed at a time to
    # retain L40S memory headroom. Reduced systems safely share the card.
    groups = [[0], [1]] if method == "full" else [[0, 1]]
    for seed_group in groups:
        processes = []
        for seed in seed_group:
            seed_dir = task / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True)
            cmd = command(args.python, trainer, method, momentum, env, seed, 10_000_000)
            (seed_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            process = start(cmd, repo, seed_dir / "stdout", seed_dir / "stderr")
            (seed_dir / "pid").write_text(f"{process.pid}\n")
            processes.append((seed, seed_dir, process))
        failures = []
        for seed, seed_dir, process in processes:
            rc = process.wait()
            (seed_dir / "rc").write_text(f"{rc}\n")
            if rc != 0:
                failures.append((seed, rc))
        if failures:
            (task / "status").write_text("FAILED\n")
            raise RuntimeError(f"formal runs failed: {failures}")

    with (task / "seed_metrics.tsv").open("w") as handle:
        handle.write("method\tmomentum\tenv\tseed\tmean_last10_at10m\tn_eprew_values\n")
        for seed in SEEDS:
            mean, count = parse_last10(task / f"seed{seed}" / "stdout")
            handle.write(f"{method}\t{momentum}\t{env}\t{seed}\t{mean:.9g}\t{count}\n")
    (task / "status").write_text("FINISHED\n")


if __name__ == "__main__":
    main()
