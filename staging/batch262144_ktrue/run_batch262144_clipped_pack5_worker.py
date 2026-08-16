#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import time

from run_batch262144_clipped_worker import (
    ENVS,
    METHODS,
    TOTAL_TIMESTEPS_ARG,
    command_for,
    gpu_memory_used_mib,
    has_bad_output,
    parse_last10,
    successful,
)


# One array element runs all five seeds for one method/environment pair.
PACKS = [(method, env) for env in ENVS for method in METHODS]
SEEDS = range(5)


def task_repo(method, rat_repo):
    return rat_repo


def output_is_healthy(task_dir):
    stdout_path = task_dir / "stdout"
    stderr_path = task_dir / "stderr"
    if not stdout_path.exists() or not stderr_path.exists():
        return False
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    return not has_bad_output(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--task-index", required=True, type=int, choices=range(len(PACKS)))
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    rat_repo = Path(args.repo).resolve()
    method, env = PACKS[args.task_index]
    pack_root = run_root / "packs" / f"task{args.task_index:02d}_{method}_{env}"
    pack_root.mkdir(parents=True, exist_ok=True)
    (pack_root / "status").write_text("STARTING\n")
    (pack_root / "method_env.txt").write_text(f"{method}\t{env}\n")

    processes = {}
    handles = {}
    metric_rows = []

    for seed in SEEDS:
        task_dir = run_root / method / env / f"seed{seed}"
        task_dir.mkdir(parents=True, exist_ok=True)
        if successful(task_dir) and output_is_healthy(task_dir):
            value, count = parse_last10(task_dir / "stdout")
            metric_rows.append((method, env, seed, value, count))
            continue

        command = command_for(
            method,
            args.python,
            rat_repo,
            env,
            seed,
            TOTAL_TIMESTEPS_ARG,
        )
        (task_dir / "command.txt").write_text(shlex.join(command) + "\n")
        (task_dir / "status").write_text("RUNNING\n")
        stdout_handle = (task_dir / "stdout").open("w")
        stderr_handle = (task_dir / "stderr").open("w")
        process = subprocess.Popen(
            command,
            cwd=task_repo(method, rat_repo),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        processes[seed] = (process, task_dir)
        handles[seed] = (stdout_handle, stderr_handle)
        (task_dir / "pid").write_text(f"{process.pid}\n")
        time.sleep(1)

    (pack_root / "status").write_text("RUNNING\n")
    peak_mib = max(gpu_memory_used_mib(), 0)
    while any(process.poll() is None for process, _ in processes.values()):
        peak_mib = max(peak_mib, gpu_memory_used_mib())
        time.sleep(2)
    peak_mib = max(peak_mib, gpu_memory_used_mib())
    (pack_root / "peak_gpu_memory_mib.txt").write_text(f"{peak_mib}\n")

    failures = []
    for seed, (process, task_dir) in processes.items():
        return_code = process.wait()
        stdout_handle, stderr_handle = handles[seed]
        stdout_handle.close()
        stderr_handle.close()
        (task_dir / "rc").write_text(f"{return_code}\n")
        healthy = return_code == 0 and output_is_healthy(task_dir)
        (task_dir / "status").write_text("FINISHED\n" if healthy else "FAILED\n")
        if not healthy:
            failures.append((seed, return_code))
            continue
        value, count = parse_last10(task_dir / "stdout")
        metric_rows.append((method, env, seed, value, count))

    metric_rows.sort(key=lambda row: row[2])
    with (pack_root / "seed_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["method", "env", "seed", "mean_last10", "n_eprew_values"])
        writer.writerows(metric_rows)

    if failures:
        (pack_root / "failure.txt").write_text(
            "\n".join(f"seed={seed} rc={return_code}" for seed, return_code in failures)
            + "\n"
        )
        (pack_root / "status").write_text("FAILED\n")
        raise RuntimeError(f"pack failed: method={method} env={env} failures={failures}")

    if len(metric_rows) != 5:
        (pack_root / "status").write_text("FAILED\n")
        raise RuntimeError(
            f"pack incomplete: method={method} env={env} metrics={len(metric_rows)}"
        )

    (pack_root / "status").write_text("FINISHED\n")


if __name__ == "__main__":
    main()
