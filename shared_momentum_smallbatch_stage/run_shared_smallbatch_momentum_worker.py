#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import re
import shlex
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


def command(python, trainer_root, method, momentum, env, seed, timesteps):
    if method == "dual255p1":
        trainer = trainer_root / "train_shared_dual255p1_momentum.py"
    else:
        trainer = trainer_root / "train_shared_jointkernel_vfrhs_momentum.py"
    cmd = [
        python, "-u", str(trainer),
        "--config", "rat_mlp_shared.yaml",
        "--env_name", env,
        "--seed", str(seed),
        "--device", "0",
        "--timesteps_per_proc", str(timesteps),
        "--epochs", "4",
        "--lr", "0.05",
        "--sgd_momentum", str(momentum),
        "--vf_coef", "4.0",
        "--cg_damping", "0.03",
        "--fisher_kernel", "exact",
        "--fisher_kernel_normalization", "none",
        "--no_karzmarz",
    ]
    if method == "dual255p1":
        cmd += [
            "--actor_curvature_subsample", "255",
            "--critic_curvature_subsample", "255",
            "--independent_energyfree_anchors",
        ]
    elif method == "curv256":
        cmd += ["--curvature_subsample", "256"]
    elif method == "full":
        cmd += ["--curvature_subsample", "0"]
    else:
        raise ValueError(f"unknown method: {method}")
    return cmd


def start(cmd, cwd, stdout_path, stderr_path):
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not prior_pythonpath else f"{cwd}:{prior_pythonpath}"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env)


def require_log_value(stdout, key, expected):
    pattern = re.compile(rf"\|\s*{key}\s*\|\s*{expected}(?:\.0+)?\s*\|")
    if not pattern.search(stdout):
        raise RuntimeError(f"smoke missing {key}={expected}")


def validate(stdout_path, stderr_path, command_path, method, momentum):
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    combined = stdout + "\n" + stderr
    if any(marker in combined for marker in BAD_MARKERS):
        raise RuntimeError("error marker found")
    values = [float(match.group(1)) for match in FINITE_RE.finditer(stdout)]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("missing or non-finite KL/rho diagnostics")
    command_tokens = shlex.split(command_path.read_text())

    def require_arg(flag, expected=None):
        if flag not in command_tokens:
            raise RuntimeError(f"smoke command missing {flag}")
        if expected is not None:
            index = command_tokens.index(flag)
            if index + 1 >= len(command_tokens) or command_tokens[index + 1] != str(expected):
                raise RuntimeError(f"smoke command mismatch {flag}={expected}")

    require_arg("--no_karzmarz")
    require_arg("--fisher_kernel", "exact")
    require_arg("--fisher_kernel_normalization", "none")
    require_arg("--cg_damping", "0.03")
    require_arg("--vf_coef", "4.0")
    require_arg("--sgd_momentum", momentum)

    if method == "dual255p1":
        require_log_value(stdout, "actor_anchor_samples", 255)
        require_log_value(stdout, "critic_anchor_samples", 255)
        # The dual Energy-free trainer reports the 255 sampled anchors and
        # the learned free rho separately rather than as 256 curvature rows.
        require_log_value(stdout, "shared_curvature_indices", 0)
        require_arg("--actor_curvature_subsample", "255")
        require_arg("--critic_curvature_subsample", "255")
        require_arg("--independent_energyfree_anchors")
    elif method == "curv256":
        require_arg("--curvature_subsample", "256")
        if "joint_vf_mode=rhs_only vf_coef=4.0" not in stdout:
            raise RuntimeError("missing joint RHS-only VF provenance")
    else:
        require_arg("--curvature_subsample", "0")
        if "joint_vf_mode=rhs_only vf_coef=4.0" not in stdout:
            raise RuntimeError("missing joint RHS-only VF provenance")


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
    parser.add_argument("--trainer-root")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    trainer_root = Path(args.trainer_root).resolve() if args.trainer_root else root
    method, momentum, env = decode_task(args.task_id)
    task = root / method / f"momentum_{momentum}" / env
    task.mkdir(parents=True, exist_ok=True)
    (task / "status").write_text("PREFLIGHT\n")

    smoke = task / "preflight"
    smoke.mkdir(exist_ok=True)
    smoke_cmd = command(args.python, trainer_root, method, momentum, env, 0, 81920)
    (smoke / "command.txt").write_text(" ".join(smoke_cmd) + "\n")
    process = start(smoke_cmd, repo, smoke / "stdout", smoke / "stderr")
    rc = process.wait()
    (smoke / "rc").write_text(f"{rc}\n")
    if rc != 0:
        (task / "status").write_text("PREFLIGHT_FAILED\n")
        raise RuntimeError(f"preflight failed rc={rc}")
    validate(smoke / "stdout", smoke / "stderr", smoke / "command.txt", method, momentum)
    (smoke / "status").write_text("FINISHED\n")
    if args.smoke_only:
        (task / "status").write_text("SMOKE_FINISHED\n")
        return
    (task / "status").write_text("RUNNING\n")

    # The full shared joint kernel is 2048x2048, so retain memory headroom by
    # running its seeds serially. Reduced systems can safely share the A100.
    groups = [[0], [1]] if method == "full" else [[0, 1]]
    for seed_group in groups:
        processes = []
        for seed in seed_group:
            seed_dir = task / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True)
            cmd = command(args.python, trainer_root, method, momentum, env, seed, 10_000_000)
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
