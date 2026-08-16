#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re


ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
STEP_RE = re.compile(r"\|\s*(?:misc/)?total_timesteps\s*\|\s*([-+0-9.eE]+)\s*\|")
REWARD_RE = re.compile(r"\|\s*eprewmean\s*\|\s*([-+0-9.eE]+)\s*\|")


def curve(path):
    text = path.read_text(errors="replace")
    steps = [float(value) for value in STEP_RE.findall(text)]
    rewards = [float(value) for value in REWARD_RE.findall(text)]
    if len(steps) != len(rewards) or not steps:
        raise RuntimeError(f"unmatched or empty curve {path}: steps={len(steps)} rewards={len(rewards)}")
    return [[round(step), reward] for step, reward in zip(steps, rewards)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--momentum-root", type=Path, required=True)
    parser.add_argument("--ppo3-root", type=Path, required=True)
    parser.add_argument("--ppo2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {"meta": {"architecture": "no-shared", "horizon": 10_000_000}, "environments": {}}
    for env in ENVS:
        methods = {}
        methods["PPO fixed lr=3e-4"] = [
            curve(args.ppo3_root / "runs" / "detach" / env / f"seed{seed}.stdout")
            for seed in range(5)
        ]
        methods["PPO fixed lr=2e-4"] = [
            curve(args.ppo2_root / env / f"seed{seed}.stdout")
            for seed in range(5)
        ]
        for method in ("dual255p1", "curv256", "full"):
            for momentum in ("0.5", "0.9"):
                label = f"{method} m={momentum}"
                methods[label] = [
                    curve(args.momentum_root / method / f"momentum_{momentum}" / env / f"seed{seed}" / "stdout")
                    for seed in range(2)
                ]
        output["environments"][env] = methods
    args.output.write_text(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
