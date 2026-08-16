#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


LOG_RE = re.compile(r"^Logging to\s+(\S+)\s*$")


def find_log(stdout: Path, repo: Path) -> Path:
    for line in stdout.read_text(errors="replace").splitlines():
        match = LOG_RE.match(line.strip())
        if match:
            path = Path(match.group(1))
            return path if path.is_absolute() else repo / path
    raise RuntimeError(f"missing Logging to marker: {stdout}")


def endpoint(progress: Path, target: float = 10_000_000.0) -> float:
    values = []
    with progress.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = float(row.get("misc/total_timesteps", row.get("total_timesteps", "nan")))
            reward = float(row.get("eprewmean", "nan"))
            if math.isfinite(step) and math.isfinite(reward) and step <= target:
                values.append((step, reward))
    if not values:
        raise RuntimeError(f"no finite points at target: {progress}")
    values.sort()
    return sum(value for _, value in values[-10:]) / min(10, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    result = {}
    for env in ("ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"):
        result[env] = {}
        for seed in (0, 1):
            stdout = args.root / args.prefix / env / f"seed{seed}.stdout"
            log_dir = find_log(stdout, args.repo)
            result[env][str(seed)] = endpoint(log_dir / "progress.csv")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
