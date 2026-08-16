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
from optuna.distributions import CategoricalDistribution


ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
MODES = ["l2", "fvp_fisher", "fvp_fisher_sqrt"]
RADII = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
MOMENTA = [0.0, 0.5, 0.9]
LOG_RE = re.compile(r"^Logging to\s+(\S+)\s*$")
ERROR_RE = re.compile(r"OOM|out of memory|NaN|Traceback|LinAlgError", re.I)


def free_mb(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True)
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


def endpoint(progress, target=10_000_000.0):
    values = []
    with progress.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = float(row.get("misc/total_timesteps", row.get("total_timesteps", "nan")))
            reward = float(row.get("eprewmean", "nan"))
            if math.isfinite(step) and math.isfinite(reward) and step <= target:
                values.append((step, reward))
    values.sort()
    if len(values) < 10 or values[-1][0] < 9_900_000:
        raise RuntimeError(f"incomplete progress: {progress} last={values[-1][0] if values else None}")
    return sum(value for _, value in values[-10:]) / 10.0


def import_anchors(study, env, anchors):
    if study.trials:
        return
    distributions = {
        "clip_mode": CategoricalDistribution(MODES),
        "clip_radius": CategoricalDistribution(RADII),
        "momentum": CategoricalDistribution(MOMENTA),
    }
    for name, anchor in anchors.items():
        values = anchor["values"][env]
        trial = optuna.trial.create_trial(
            params=anchor["params"],
            distributions=distributions,
            value=sum(values) / len(values),
            user_attrs={"anchor": name, "provenance": anchor["provenance"], "seed_values": values},
        )
        study.add_trial(trial)
    for mode, momentum in [
        ("l2", 0.5),
        ("fvp_fisher", 0.5),
        ("fvp_fisher", 0.9),
        ("fvp_fisher_sqrt", 0.5),
        ("fvp_fisher_sqrt", 0.9),
    ]:
        study.enqueue_trial({"clip_mode": mode, "clip_radius": 0.5, "momentum": momentum})


def run_trial(trial, env, gpu, repo, root, trainer, config, min_free_mb):
    mode = trial.suggest_categorical("clip_mode", MODES)
    radius = trial.suggest_categorical("clip_radius", RADII)
    momentum = trial.suggest_categorical("momentum", MOMENTA)
    trial_dir = root / env / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "params.json").write_text(json.dumps({
        "clip_mode": mode,
        "clip_radius": radius,
        "momentum": momentum,
        "damping": 0.03,
        "seeds": [0, 1],
    }, indent=2, sort_keys=True))

    wait_for_memory(gpu, min_free_mb)
    processes = []
    files = []
    for seed in (0, 1):
        stdout = trial_dir / f"seed{seed}.stdout"
        stderr = trial_dir / f"seed{seed}.stderr"
        out_handle = stdout.open("w")
        err_handle = stderr.open("w")
        command = [
            "/home/yihe/.venv/bin/python", "-u", str(trainer),
            "--config", config,
            "--env_name", env,
            "--seed", str(seed),
            "--device", "0",
            "--timesteps_per_proc", "10000000",
            "--pi_epochs", "4",
            "--lr_pi", "0.05",
            "--cg_damping", "0.03",
            "--fisher_kernel", "exact",
            "--fisher_kernel_normalization", "none",
            "--actor_curvature_subsample", "0",
            "--critic_curvature_subsample", "0",
            "--critic_update", "gn",
            "--critic_optimizer", "sgd",
            "--no-actor_subsample_full_batch_gradient",
            "--actor_clip_mode", mode,
            "--max_grad_norm", str(radius),
            "--actor_sgd_momentum", str(momentum),
            "--critic_sgd_momentum", str(momentum),
        ]
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(command, cwd=repo, stdout=out_handle, stderr=err_handle, env=env_vars)
        (trial_dir / f"seed{seed}.pid").write_text(f"{process.pid}\n")
        processes.append((seed, process, stdout, stderr))
        files.extend([out_handle, err_handle])

    return_codes = {}
    for seed, process, _, _ in processes:
        return_codes[seed] = process.wait()
        (trial_dir / f"seed{seed}.rc").write_text(f"{return_codes[seed]}\n")
    for handle in files:
        handle.close()
    if any(return_codes.values()):
        raise RuntimeError(f"nonzero return codes: {return_codes}")

    seed_values = []
    log_dirs = []
    for seed, _, stdout, stderr in processes:
        combined = stdout.read_text(errors="replace") + "\n" + stderr.read_text(errors="replace")
        if ERROR_RE.search(combined):
            raise RuntimeError(f"error marker in seed{seed}")
        log_dir = find_log(stdout, repo)
        log_dirs.append(str(log_dir))
        seed_values.append(endpoint(log_dir / "progress.csv"))
    trial.set_user_attr("seed_values", seed_values)
    trial.set_user_attr("log_dirs", log_dirs)
    trial.set_user_attr("gpu", gpu)
    (trial_dir / "objective.json").write_text(json.dumps({
        "seed_values": seed_values,
        "mean": sum(seed_values) / len(seed_values),
        "log_dirs": log_dirs,
    }, indent=2))
    return sum(seed_values) / len(seed_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--envs", nargs="+", choices=ENVS, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--new-trials", type=int, default=8)
    parser.add_argument("--min-free-mb", type=int, default=8000)
    args = parser.parse_args()
    anchors = json.loads((args.root / "anchor_objectives.json").read_text())
    trainer = args.repo / "train_detach_jointcritic_clip_momentum_optuna.py"
    for env in args.envs:
        env_root = args.root / env
        env_root.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=f"fullEF_fullGGN_{env}_d003_2seed",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=20260805 + ENVS.index(env), n_startup_trials=4),
            storage=f"sqlite:///{env_root / 'study.db'}",
            load_if_exists=True,
        )
        import_anchors(study, env, anchors)
        completed_new = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and "anchor" not in t.user_attrs)
        remaining = max(0, args.new_trials - completed_new)
        if remaining:
            study.optimize(
                lambda trial: run_trial(
                    trial, env, args.gpu, args.repo, args.root, trainer,
                    "rat_mlp_detjc_full_ef_ggn_optuna.yaml", args.min_free_mb,
                ),
                n_trials=remaining,
                catch=(RuntimeError,),
            )
        (env_root / "best.json").write_text(json.dumps({
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
            "user_attrs": study.best_trial.user_attrs,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
