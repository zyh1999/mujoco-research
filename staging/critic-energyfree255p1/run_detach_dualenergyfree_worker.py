#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ENV_SHARDS = [
    ["ant", "hopper", "humanoid", "swimmer"],
    ["halfcheetah", "humanoidstandup", "walker2d"],
]
SEEDS = [0, 1, 2, 3, 4]
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
FINITE_RE = re.compile(
    r"\|\s*(?:kl|actor_free_rho_value|critic_free_rho_value)\s*\|\s*([-+0-9.eE]+)\s*\|"
)


def command(python, trainer, env, seed, timesteps):
    return [
        python,
        "-u",
        trainer,
        "--config",
        "rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml",
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
        "255",
        "--critic_curvature_subsample",
        "255",
        "--actor_subsample_free_rho",
        "anchor",
        "--critic_subsample_free_rho",
        "anchor",
        "--critic_update",
        "gn",
    ]


def run_one(cmd, cwd, stdout_path, stderr_path):
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not prior_pythonpath else f"{cwd}:{prior_pythonpath}"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env)


def validate_smoke(stdout_path, stderr_path):
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    combined = stdout + "\n" + stderr
    bad = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "LinAlgError")
    if any(token in combined for token in bad):
        raise RuntimeError("smoke error marker found")
    required = {
        "actor_free_rho": "1",
        "critic_anchor_samples": "255",
        "critic_free_rho": "1",
        "critic_independent_anchors": "1",
    }
    for key, expected in required.items():
        pattern = re.compile(rf"\|\s*{key}\s*\|\s*{expected}(?:\.0+)?\s*\|")
        if not pattern.search(stdout):
            raise RuntimeError(f"smoke missing {key}={expected}")
    values = [float(match.group(1)) for match in FINITE_RE.finditer(stdout)]
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("smoke KL/rho diagnostics are missing or non-finite")


def parse_last10(path):
    values = [float(match.group(1)) for match in EPREW_RE.finditer(path.read_text(errors="replace"))]
    if not values:
        raise RuntimeError(f"no eprewmean values in {path}")
    return sum(values[-10:]) / len(values[-10:]), len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--worker-index", type=int, required=True, choices=(0, 1))
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--max-concurrent-seeds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    trainer = str(root / "train_detach_dualenergyfree255p1.py")
    worker = root / f"worker{args.worker_index}"
    worker.mkdir(parents=True, exist_ok=True)
    (worker / "status").write_text("PREFLIGHT\n")

    smoke = root / "preflight" / f"worker{args.worker_index}"
    smoke.mkdir(parents=True, exist_ok=True)
    (smoke / "status").write_text("RUNNING\n")
    smoke_cmd = command(args.python, trainer, "halfcheetah", args.worker_index, 81920)
    (smoke / "command.txt").write_text(" ".join(smoke_cmd) + "\n")
    process = run_one(smoke_cmd, repo, smoke / "stdout", smoke / "stderr")
    rc = process.wait()
    (smoke / "rc").write_text(f"{rc}\n")
    if rc != 0:
        (smoke / "status").write_text("FAILED\n")
        raise RuntimeError(f"preflight failed rc={rc}")
    validate_smoke(smoke / "stdout", smoke / "stderr")
    (smoke / "status").write_text("FINISHED\n")
    (worker / "status").write_text("RUNNING\n")

    rows = []
    for env in ENV_SHARDS[args.worker_index]:
        env_dir = root / env
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "status").write_text("RUNNING\n")
        failures = []
        width = max(1, min(args.max_concurrent_seeds, len(SEEDS)))
        for offset in range(0, len(SEEDS), width):
            processes = []
            for seed in SEEDS[offset:offset + width]:
                cmd = command(args.python, trainer, env, seed, 10_000_000)
                (env_dir / f"seed{seed}.command.txt").write_text(" ".join(cmd) + "\n")
                process = run_one(
                    cmd,
                    repo,
                    env_dir / f"seed{seed}.stdout",
                    env_dir / f"seed{seed}.stderr",
                )
                (env_dir / f"seed{seed}.pid").write_text(f"{process.pid}\n")
                processes.append((seed, process))
            for seed, process in processes:
                rc = process.wait()
                (env_dir / f"seed{seed}.rc").write_text(f"{rc}\n")
                if rc != 0:
                    failures.append((seed, rc))
            if failures:
                break
        if failures:
            (env_dir / "status").write_text("FAILED\n")
            raise RuntimeError(f"{env} failed: {failures}")

        for seed in SEEDS:
            mean, count = parse_last10(env_dir / f"seed{seed}.stdout")
            rows.append((env, seed, mean, count))
        (env_dir / "status").write_text("FINISHED\n")

    with (worker / "seed_metrics.tsv").open("w") as handle:
        handle.write("env\tseed\tmean_last10_at10m\tn_eprew_values\n")
        for env, seed, mean, count in rows:
            handle.write(f"{env}\t{seed}\t{mean:.9g}\t{count}\n")
    (worker / "status").write_text("FINISHED\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(repr(error), file=sys.stderr)
        raise
