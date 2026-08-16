#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


ENVS = (
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
)
SEEDS = range(5)
NUM_SHARDS = 3
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
BAD_RE = re.compile(
    r"CUDA out of memory|out of memory|Traceback \(most recent call last\)|"
    r"LinAlgError|\bnan\b|\binf\b",
    re.IGNORECASE,
)


def tasks_for_shard(shard):
    tasks = [(env, seed) for env in ENVS for seed in SEEDS]
    return tasks[shard::NUM_SHARDS]


def command(python, repo, env, seed, timesteps):
    return [
        python,
        "-u",
        str(repo / "train_detach_jointcritic.py"),
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
        "64",
        "--critic_curvature_subsample",
        "64",
        "--critic_update",
        "gn",
        "--critic_optimizer",
        "sgd",
        "--no-actor_subsample_full_batch_gradient",
    ]


def successful(task_dir):
    rc = task_dir / "rc"
    return rc.exists() and rc.read_text().strip() == "0"


def parse_last10(path):
    values = []
    if not path.exists():
        return float("nan"), 0
    with path.open(errors="replace") as handle:
        for line in handle:
            match = EPREW_RE.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        return float("nan"), 0
    return sum(values[-10:]) / min(10, len(values)), len(values)


def scan_bad(task_dir):
    text = ""
    for name in ("stdout", "stderr"):
        path = task_dir / name
        if path.exists():
            text += path.read_text(errors="replace")
    return BAD_RE.search(text)


def run_preflight(args, run_root, repo):
    task_dir = run_root / "preflight" / f"shard{args.shard}"
    task_dir.mkdir(parents=True, exist_ok=True)
    if successful(task_dir) and not scan_bad(task_dir):
        return
    cmd = command(args.python, repo, "halfcheetah", 9640 + args.shard, 8192)
    (task_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    (task_dir / "status").write_text("RUNNING\n")
    with (task_dir / "stdout").open("w") as stdout, (task_dir / "stderr").open("w") as stderr:
        result = subprocess.run(cmd, cwd=repo, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    (task_dir / "rc").write_text(f"{result.returncode}\n")
    if result.returncode != 0 or scan_bad(task_dir):
        (task_dir / "status").write_text("FAILED\n")
        raise RuntimeError(f"preflight failed on shard {args.shard}")
    (task_dir / "status").write_text("FINISHED\n")


def launch(args, repo, run_root, env, seed):
    task_dir = run_root / env / f"seed{seed}"
    task_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(args.python, repo, env, seed, args.timesteps)
    (task_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    (task_dir / "status").write_text("RUNNING\n")
    stdout = (task_dir / "stdout").open("w")
    stderr = (task_dir / "stderr").open("w")
    process = subprocess.Popen(cmd, cwd=repo, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    (task_dir / "pid").write_text(f"{process.pid}\n")
    return process, task_dir, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--shard", required=True, type=int, choices=range(NUM_SHARDS))
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    shard_root = run_root / f"shard{args.shard}"
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "status").write_text("PREFLIGHT\n")
    run_preflight(args, run_root, repo)
    (shard_root / "status").write_text("RUNNING\n")

    pending = [task for task in tasks_for_shard(args.shard) if not successful(run_root / task[0] / f"seed{task[1]}")]
    active = {}
    failures = []
    completed = []

    while pending or active:
        while pending and len(active) < args.max_parallel:
            env, seed = pending.pop(0)
            process, task_dir, stdout, stderr = launch(args, repo, run_root, env, seed)
            active[process.pid] = (process, env, seed, task_dir, stdout, stderr)
            (shard_root / "events.log").open("a").write(
                f"{time.time():.3f}\tlaunch\t{process.pid}\t{env}\t{seed}\n"
            )

        finished = []
        for pid, item in active.items():
            process, env, seed, task_dir, stdout, stderr = item
            rc = process.poll()
            if rc is None:
                continue
            stdout.close()
            stderr.close()
            (task_dir / "rc").write_text(f"{rc}\n")
            bad = scan_bad(task_dir)
            if rc == 0 and bad is None:
                (task_dir / "status").write_text("FINISHED\n")
                completed.append((env, seed))
            else:
                (task_dir / "status").write_text("FAILED\n")
                failures.append((env, seed, rc, bad.group(0) if bad else ""))
            (shard_root / "events.log").open("a").write(
                f"{time.time():.3f}\tfinish\t{pid}\t{env}\t{seed}\t{rc}\n"
            )
            finished.append(pid)
        for pid in finished:
            del active[pid]
        if active and not finished:
            time.sleep(20)

    rows = []
    for env, seed in tasks_for_shard(args.shard):
        task_dir = run_root / env / f"seed{seed}"
        value, count = parse_last10(task_dir / "stdout")
        rc_path = task_dir / "rc"
        rc = rc_path.read_text().strip() if rc_path.exists() else "missing"
        rows.append((env, seed, rc, value, count))
    with (shard_root / "seed_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("env", "seed", "rc", "last10_eprewmean", "n_eprew_values"))
        writer.writerows(rows)

    if failures:
        with (shard_root / "failures.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("env", "seed", "rc", "marker"))
            writer.writerows(failures)
        (shard_root / "status").write_text("PARTIAL_FAILED\n")
        raise SystemExit(1)
    (shard_root / "status").write_text("FINISHED\n")


if __name__ == "__main__":
    main()
