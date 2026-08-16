#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--clips", default="025,05,1,2")
parser.add_argument("--seeds", default="0,1,2")
args = parser.parse_args()

ROOT = args.root
CLIPS = tuple(args.clips.split(","))
SEEDS = tuple(int(seed) for seed in args.seeds.split(","))
ENVS = (
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
)
FIELD = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")


def read_series(path):
    rewards = []
    steps = []
    if not path.exists():
        return rewards, steps
    with path.open(errors="replace") as handle:
        for raw in handle:
            match = FIELD.match(raw.strip())
            if not match:
                continue
            key, value = match.groups()
            try:
                number = float(value)
            except ValueError:
                continue
            if key.strip() == "eprewmean" and math.isfinite(number):
                rewards.append(number)
            elif key.strip() == "misc/total_timesteps" and math.isfinite(number):
                steps.append(number)
    return rewards, steps


print("clip\tenv\trc0\tfailed\tactive_or_pending\tsteps_min\tsteps_max\tendpoint_last10_mean\tsem")
for clip in CLIPS:
    for env in ENVS:
        endpoints = []
        latest_steps = []
        rc0 = 0
        failed = 0
        for seed in SEEDS:
            base = ROOT / f"clip{clip}" / env / f"seed{seed}"
            rc_path = base.with_suffix(".rc")
            rc = rc_path.read_text().strip() if rc_path.exists() else ""
            rewards, steps = read_series(base.with_suffix(".stdout"))
            if steps:
                latest_steps.append(steps[-1])
            if rc == "0":
                rc0 += 1
                if rewards:
                    endpoints.append(float(np.mean(rewards[-10:])))
            elif rc and rc not in {"RUNNING", "RUNNING_RETRY"}:
                failed += 1
        active_or_pending = len(SEEDS) - rc0 - failed
        lo = min(latest_steps) if latest_steps else float("nan")
        hi = max(latest_steps) if latest_steps else float("nan")
        mean = float(np.mean(endpoints)) if endpoints else float("nan")
        sem = (
            float(np.std(endpoints, ddof=1) / math.sqrt(len(endpoints)))
            if len(endpoints) > 1
            else float("nan")
        )
        print(
            f"{clip}\t{env}\t{rc0}\t{failed}\t{active_or_pending}\t"
            f"{lo:.0f}\t{hi:.0f}\t{mean:.3f}\t{sem:.3f}"
        )
