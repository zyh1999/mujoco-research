#!/usr/bin/env python3
"""Run one matched Full-EF/Full-GGN momentum cell with auditable artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ENVS = ("ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d")
EPREW_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")
FINITE_RE = re.compile(r"\|\s*(?:kl|v_grad_norm|v_step_norm|grad_norm)\s*\|\s*([-+0-9.eE]+)\s*\|")
BAD_RE = re.compile(r"out of memory|\bnan\b|\binf\b|traceback|linalgerror|assertionerror", re.I)


def command(args: argparse.Namespace, seed: int, timesteps: int) -> list[str]:
    return [
        args.python, "-u", str(args.trainer),
        "--config", args.config,
        "--env_name", args.env,
        "--seed", str(seed),
        "--device", "0",
        "--timesteps_per_proc", str(timesteps),
        "--pi_epochs", "4",
        "--lr_pi", "0.05",
        "--sgd_momentum", str(args.momentum),
        "--cg_damping", "0.03",
        "--fisher_kernel", "exact",
        "--fisher_kernel_normalization", "none",
        "--critic_update", "gn",
        "--critic_optimizer", "sgd",
        "--actor_curvature_subsample", "0",
        "--critic_curvature_subsample", "0",
        "--actor_subsample_free_rho", "none",
        "--critic_subsample_free_rho", "none",
    ]


def validate(seed_dir: Path, momentum: float, require_endpoint: bool) -> tuple[float | None, int]:
    stdout = (seed_dir / "stdout.log").read_text(errors="replace")
    stderr = (seed_dir / "stderr.log").read_text(errors="replace")
    combined = stdout + "\n" + stderr
    if BAD_RE.search(combined):
        raise RuntimeError("error marker in stdout/stderr")
    expected = f"Independent SGD momentum: actor={momentum} critic={momentum}"
    if expected not in stdout:
        raise RuntimeError(f"missing runtime momentum telemetry: {expected}")
    finite = [float(m.group(1)) for m in FINITE_RE.finditer(stdout)]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise RuntimeError("missing or non-finite training telemetry")
    rewards = [float(m.group(1)) for m in EPREW_RE.finditer(stdout)]
    if require_endpoint and len(rewards) < 10:
        raise RuntimeError("fewer than ten reward reports")
    mean = sum(rewards[-10:]) / min(10, len(rewards)) if rewards else None
    return mean, len(rewards)


def run_seed(args: argparse.Namespace, seed: int, timesteps: int, seed_dir: Path, require_endpoint: bool) -> tuple[float | None, int]:
    seed_dir.mkdir(parents=True, exist_ok=False)
    seed_tmp = seed_dir / "tmp"
    seed_tmp.mkdir()
    cmd = command(args, seed, timesteps)
    (seed_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    (seed_dir / "status").write_text("RUNNING\n")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONPATH": str(args.repo),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
        "TMPDIR": str(seed_tmp),
    })
    with (seed_dir / "stdout.log").open("w") as stdout, (seed_dir / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(cmd, cwd=args.repo, stdout=stdout, stderr=stderr, env=env)
        (seed_dir / "pid").write_text(f"{process.pid}\n")
        rc = process.wait()
    (seed_dir / "rc").write_text(f"{rc}\n")
    if rc:
        (seed_dir / "status").write_text("FAILED\n")
        raise RuntimeError(f"trainer rc={rc}")
    result = validate(seed_dir, args.momentum, require_endpoint)
    (seed_dir / "status").write_text("FINISHED\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--env", choices=ENVS, required=True)
    parser.add_argument("--momentum", type=float, choices=(0.7, 0.8), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--parallel-seeds", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        cell = args.run_root / "preflight" / args.env / f"momentum_{args.momentum}"
        base_seed = 970 if args.momentum == 0.7 else 980
        seed_specs = [(base_seed, 81_920)]
        if args.parallel_seeds:
            seed_specs.append((base_seed + 1, 81_920))
    else:
        cell = args.run_root / f"momentum_{args.momentum}" / args.env
        seed_specs = [(seed, args.timesteps) for seed in args.seeds]
    if cell.exists():
        print(f"refusing collision: {cell}", file=sys.stderr)
        return 3
    cell.mkdir(parents=True)
    (cell / "status").write_text("RUNNING\n")
    (cell / "run_info.txt").write_text(
        f"identity=no_shared_largebatch_mlp_fullEF_fullGGN_momentum0708\n"
        f"environment={args.env}\nmomentum={args.momentum}\nseeds={','.join(str(x[0]) for x in seed_specs)}\n"
        f"timesteps={seed_specs[0][1]}\ndamping=0.03\nnormalization=none\nkaczmarz=false\n"
        f"post_grad=parameter_l2_clip\nmax_grad_norm=0.5\nactor_rows=1024\ncritic_rows=1024\n"
    )
    rows = []
    try:
        if args.parallel_seeds and len(seed_specs) > 1:
            with ThreadPoolExecutor(max_workers=len(seed_specs)) as executor:
                futures = [
                    executor.submit(
                        run_seed,
                        args,
                        seed,
                        timesteps,
                        cell / f"seed{seed}",
                        not args.preflight,
                    )
                    for seed, timesteps in seed_specs
                ]
                for (seed, _), future in zip(seed_specs, futures):
                    mean, count = future.result()
                    rows.append((seed, mean, count))
        else:
            for seed, timesteps in seed_specs:
                mean, count = run_seed(args, seed, timesteps, cell / f"seed{seed}", not args.preflight)
                rows.append((seed, mean, count))
    except Exception as exc:
        (cell / "status").write_text("FAILED\n")
        (cell / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    with (cell / "metrics.tsv").open("w") as handle:
        handle.write("env\tmomentum\tseed\tmean_last10\tn_reports\n")
        for seed, mean, count in rows:
            handle.write(f"{args.env}\t{args.momentum}\t{seed}\t{mean if mean is not None else 'NA'}\t{count}\n")
    (cell / "status").write_text("FINISHED\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
