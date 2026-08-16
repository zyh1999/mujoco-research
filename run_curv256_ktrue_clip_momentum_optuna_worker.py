#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import optuna


ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
MODES = ["l2", "fvp_fisher"]
RADII = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
# Kaczmarz reads the actor SGD momentum buffer. The near-zero baseline stays at
# 1e-6 instead of literal zero so Kaczmarz=true remains a meaningful trial.
MOMENTA = [1.0e-6, 0.5, 0.9]
KACZMARZ_CHOICES = [False, True]
TARGET_STEPS = 130_809_855.0
LOG_RE = re.compile(r"^Logging to\s+(\S+)\s*$")
ERROR_RE = re.compile(r"OOM|out of memory|NaN|Traceback|LinAlgError", re.I)


def free_mb(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def wait_for_memory(gpu, minimum):
    while free_mb(gpu) < minimum:
        time.sleep(60)


def find_log(stdout, repo):
    for line in stdout.read_text(errors="replace").splitlines():
        match = LOG_RE.match(line.strip())
        if match:
            path = Path(match.group(1))
            return path if path.is_absolute() else repo / path
    raise RuntimeError(f"missing Logging to marker in {stdout}")


def endpoint(progress):
    values = []
    with progress.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = float(row.get("misc/total_timesteps", row.get("total_timesteps", "nan")))
            reward = float(row.get("eprewmean", "nan"))
            if math.isfinite(step) and math.isfinite(reward) and step <= TARGET_STEPS:
                values.append((step, reward))
    values.sort()
    if len(values) < 10 or values[-1][0] < 130_000_000:
        raise RuntimeError(f"incomplete progress: {progress} last={values[-1][0] if values else None}")
    return sum(value for _, value in values[-10:]) / 10.0


def enqueue_initial_trials(study):
    if study.trials:
        return
    for mode, radius, momentum, kaczmarz in [
        ("l2", 0.5, 1.0e-6, False),
        ("fvp_fisher", 0.5, 1.0e-6, False),
        ("l2", 0.5, 0.5, False),
        ("fvp_fisher", 0.5, 0.5, False),
        ("l2", 0.5, 0.5, True),
        ("fvp_fisher", 0.5, 0.5, True),
        ("l2", 0.5, 0.9, True),
        ("fvp_fisher", 1.0, 0.5, True),
    ]:
        study.enqueue_trial({
            "clip_mode": mode,
            "clip_radius": radius,
            "momentum": momentum,
            "kaczmarz": kaczmarz,
        })


def run_trial(trial, args, env):
    mode = trial.suggest_categorical("clip_mode", MODES)
    radius = trial.suggest_categorical("clip_radius", RADII)
    momentum = trial.suggest_categorical("momentum", MOMENTA)
    kaczmarz = trial.suggest_categorical("kaczmarz", KACZMARZ_CHOICES)

    # Both radius=.5, near-zero-momentum baselines already belong to the
    # formal five-seed matrix and must not be duplicated by this sweep.
    if kaczmarz and radius == 0.5 and momentum == 1.0e-6:
        trial.set_user_attr("external_baseline", "formal_5seed_batch262144_ktrue")
        raise optuna.TrialPruned("existing formal baseline")

    trial_dir = args.root / env / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "clip_mode": mode,
        "clip_radius": radius,
        "actor_sgd_momentum": momentum,
        "critic_sgd_momentum": momentum,
        "damping": 0.03,
        "kaczmarz": kaczmarz,
        "actor_curvature_rows": 256,
        "critic_curvature_rows": 256,
        "rollout": 262144,
        "seeds": [0, 1],
    }
    (trial_dir / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True))

    wait_for_memory(args.gpu, args.min_free_mb)
    processes = []
    handles = []
    for seed in (0, 1):
        stdout = trial_dir / f"seed{seed}.stdout"
        stderr = trial_dir / f"seed{seed}.stderr"
        out_handle = stdout.open("w")
        err_handle = stderr.open("w")
        command = [
            str(args.python), "-u", str(args.trainer),
            "--config", args.config,
            "--env_name", env,
            "--seed", str(seed),
            "--device", "0",
            "--timesteps_per_proc", str(int(TARGET_STEPS)),
            "--pi_epochs", "4",
            "--lr_pi", "0.05",
            "--cg_damping", "0.03",
            "--fisher_kernel", "exact",
            "--fisher_kernel_normalization", "none",
            "--actor_curvature_subsample", "256",
            "--critic_curvature_subsample", "256",
            "--critic_update", "gn",
            "--critic_optimizer", "sgd",
            "--no-actor_subsample_full_batch_gradient",
            "--actor_clip_mode", mode,
            "--max_grad_norm", str(radius),
            "--actor_sgd_momentum", str(momentum),
            "--critic_sgd_momentum", str(momentum),
            "--is_karzmarz" if kaczmarz else "--no-is_karzmarz",
        ]
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        (trial_dir / f"seed{seed}.command.txt").write_text(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            stdout=out_handle,
            stderr=err_handle,
            env=env_vars,
        )
        (trial_dir / f"seed{seed}.pid").write_text(f"{process.pid}\n")
        processes.append((seed, process, stdout, stderr))
        handles.extend([out_handle, err_handle])

    return_codes = {}
    for seed, process, _, _ in processes:
        return_codes[seed] = process.wait()
        (trial_dir / f"seed{seed}.rc").write_text(f"{return_codes[seed]}\n")
    for handle in handles:
        handle.close()
    if any(return_codes.values()):
        raise RuntimeError(f"nonzero return codes: {return_codes}")

    seed_values = []
    log_dirs = []
    for seed, _, stdout, stderr in processes:
        combined = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
        if ERROR_RE.search(combined):
            raise RuntimeError(f"error marker in seed{seed}")
        log_dir = find_log(stdout, args.repo)
        log_dirs.append(str(log_dir))
        seed_values.append(endpoint(log_dir / "progress.csv"))
    trial.set_user_attr("seed_values", seed_values)
    trial.set_user_attr("log_dirs", log_dirs)
    trial.set_user_attr("gpu", args.gpu)
    objective = sum(seed_values) / len(seed_values)
    (trial_dir / "objective.json").write_text(
        json.dumps({"seed_values": seed_values, "mean": objective, "log_dirs": log_dirs}, indent=2)
    )
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--envs", nargs="+", choices=ENVS, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--new-trials", type=int, default=8)
    parser.add_argument("--min-free-mb", type=int, default=7000)
    parser.add_argument("--storage-kind", choices=["sqlite", "journal"], default="sqlite")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for env in args.envs:
        env_root = args.root / env
        env_root.mkdir(parents=True, exist_ok=True)
        if args.storage_kind == "journal":
            storage = optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(str(env_root / "study.journal"))
            )
        else:
            storage = f"sqlite:///{env_root / 'study.db'}"
        study = optuna.create_study(
            study_name=f"curv256_ggn256_kopt_batch262144_{env}_2seed",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=20260805 + ENVS.index(env), n_startup_trials=5),
            storage=storage,
            load_if_exists=True,
        )
        enqueue_initial_trials(study)
        completed = sum(
            1 for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        )
        remaining = max(0, args.new_trials - completed)
        if remaining:
            study.optimize(
                lambda trial: run_trial(trial, args, env),
                n_trials=remaining,
                catch=(RuntimeError,),
            )
        complete_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if complete_trials:
            (env_root / "best.json").write_text(json.dumps({
                "number": study.best_trial.number,
                "value": study.best_value,
                "params": study.best_params,
                "user_attrs": study.best_trial.user_attrs,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
