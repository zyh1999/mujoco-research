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


ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 130_809_855
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
FINITE_RE = re.compile(
    r"\|\s*(?:kl|actor_free_rho_value|actor_fvp_quadratic_pre|actor_clip_coef)"
    r"\s*\|\s*([-+0-9.eE]+)\s*\|"
)
BAD_RE = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|LinAlgError|\bnan\b", re.I)


def command(python, trainer, config, env, seed, device, timesteps):
    return [
        python,
        "-u",
        str(trainer),
        "--config",
        config,
        "--env_name",
        env,
        "--seed",
        str(seed),
        "--device",
        str(device),
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
        "256",
        "--actor_subsample_free_rho",
        "anchor",
        "--critic_update",
        "gn",
        "--critic_optimizer",
        "sgd",
        "--no-actor_subsample_full_batch_gradient",
    ]


def healthy(task_dir):
    rc = task_dir / "rc"
    stdout = task_dir / "stdout"
    stderr = task_dir / "stderr"
    if not (rc.exists() and stdout.exists() and stderr.exists()):
        return False
    text = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
    return rc.read_text().strip() == "0" and BAD_RE.search(text) is None


def last10(path):
    values = [float(m.group(1)) for m in EPREW_RE.finditer(path.read_text(errors="replace"))]
    if not values:
        raise RuntimeError(f"no eprewmean in {path}")
    return sum(values[-10:]) / len(values[-10:]), len(values)


def validate_smoke(task_dir, clip_mode):
    if not healthy(task_dir):
        raise RuntimeError("smoke failed or emitted an error marker")
    stdout = (task_dir / "stdout").read_text(errors="replace")
    expected_clip = "fvp_fisher" if clip_mode == "fvp" else "l2"
    marker = (
        "LARGEBS_ENERGYFREE255P1_CONFIG "
        "actor_anchor_samples=255 actor_free_rho_mode=anchor actor_dof=256 "
        "critic_curvature_samples=256 critic_update=gn "
        f"clip_mode={expected_clip} damping=0.03 kaczmarz=False normalization=none "
        "rollout=262144 minibatches=4 epochs=4"
    )
    if marker not in stdout:
        raise RuntimeError("smoke missing strict large-batch 255+1 configuration marker")
    if not re.search(rf"Actor clip mode:\s*{expected_clip}\b", stdout):
        raise RuntimeError(f"smoke missing clip mode {expected_clip}")
    values = [float(m.group(1)) for m in FINITE_RE.finditer(stdout)]
    if values and not all(math.isfinite(value) for value in values):
        raise RuntimeError("smoke has missing or non-finite KL/rho/clip diagnostics")


def run_process(cmd, repo, task_dir):
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    (task_dir / "status").write_text("RUNNING\n")
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not prior else f"{repo}:{prior}"
    stdout = (task_dir / "stdout").open("w")
    stderr = (task_dir / "stderr").open("w")
    process = subprocess.Popen(cmd, cwd=repo, stdout=stdout, stderr=stderr, env=env)
    (task_dir / "pid").write_text(f"{process.pid}\n")
    return process, stdout, stderr


def finish_process(process, stdout, stderr, task_dir):
    rc = process.wait()
    stdout.close()
    stderr.close()
    (task_dir / "rc").write_text(f"{rc}\n")
    ok = healthy(task_dir)
    (task_dir / "status").write_text("FINISHED\n" if ok else "FAILED\n")
    return ok, rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--variant", choices=("fvp", "l2"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    repo = Path(args.repo).resolve()
    worker_key = f"{args.variant}_{'_'.join(args.envs)}"
    status_path = root / ("status" if args.envs == ENVS else f"status_{worker_key}")
    trainer = repo / "train_detach_energyfree255p1_criticggn256_batch262144.py"
    config = (
        "rat_mlp_detjc_energyfree255p1_criticggn256_batch262144_fvpclip05.yaml"
        if args.variant == "fvp"
        else "rat_mlp_detjc_energyfree255p1_criticggn256_batch262144_l2clip05.yaml"
    )
    root.mkdir(parents=True, exist_ok=True)
    status_path.write_text("PREFLIGHT\n")

    smoke = root / "preflight" / worker_key
    if not healthy(smoke):
        cmd = command(args.python, trainer, config, "halfcheetah", 4, args.device, 1)
        process, stdout, stderr = run_process(cmd, repo, smoke)
        ok, rc = finish_process(process, stdout, stderr, smoke)
        if not ok:
            raise RuntimeError(f"smoke failed rc={rc}")
    validate_smoke(smoke, args.variant)
    if args.smoke_only:
        status_path.write_text("SMOKE_FINISHED\n")
        return

    status_path.write_text("RUNNING\n")
    rows = []
    failures = []
    width = max(1, args.max_concurrent)
    for env_name in args.envs:
        for offset in range(0, len(args.seeds), width):
            active = []
            for seed in args.seeds[offset:offset + width]:
                task_dir = root / args.variant / env_name / f"seed{seed}"
                if healthy(task_dir):
                    value, count = last10(task_dir / "stdout")
                    rows.append((args.variant, env_name, seed, value, count))
                    continue
                cmd = command(
                    args.python,
                    trainer,
                    config,
                    env_name,
                    seed,
                    args.device,
                    TOTAL_TIMESTEPS,
                )
                process, stdout, stderr = run_process(cmd, repo, task_dir)
                active.append((seed, task_dir, process, stdout, stderr))
                time.sleep(1)
            for seed, task_dir, process, stdout, stderr in active:
                ok, rc = finish_process(process, stdout, stderr, task_dir)
                if not ok:
                    failures.append((env_name, seed, rc))
                    continue
                value, count = last10(task_dir / "stdout")
                rows.append((args.variant, env_name, seed, value, count))
        if failures:
            break

    rows.sort(key=lambda row: (ENVS.index(row[1]), row[2]))
    with (root / f"seed_metrics_{worker_key}.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["variant", "env", "seed", "mean_last10", "n_eprew_values"])
        writer.writerows(rows)
    if failures:
        (root / f"failures_{worker_key}.tsv").write_text(
            "\n".join(f"{env}\t{seed}\t{rc}" for env, seed, rc in failures) + "\n"
        )
        status_path.write_text("FAILED\n")
        raise RuntimeError(f"formal failures: {failures}")
    status_path.write_text("FINISHED\n")


if __name__ == "__main__":
    main()
