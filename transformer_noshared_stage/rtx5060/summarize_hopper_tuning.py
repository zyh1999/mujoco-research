#!/usr/bin/env python3
"""Field-name based summary for the direct Hopper tuning campaign."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    print("tag\tstatus\tsteps\tlast10_reward\tlast_reward\tkl\tlr_pi\tfisher_pre\tfisher_post\tl2_pre\tl2_post")
    for tag_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        statuses = list(tag_dir.glob("**/status"))
        progress_files = list(tag_dir.glob("**/progress.csv"))
        status = statuses[0].read_text().strip() if statuses else "PENDING"
        if not progress_files:
            print(f"{tag_dir.name}\t{status}\t-")
            continue

        with progress_files[0].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"{tag_dir.name}\t{status}\tempty")
            continue

        last = rows[-1]
        window = rows[-args.window :]
        reward = sum(number(row, "eprewmean") for row in window) / len(window)
        values = [
            tag_dir.name,
            status,
            f"{number(last, 'misc/total_timesteps'):.0f}",
            f"{reward:.2f}",
            f"{number(last, 'eprewmean'):.2f}",
            f"{number(last, 'kl'):.5f}",
            f"{number(last, 'lr_pi'):.6f}",
            f"{number(last, 'actor_fisher_norm_pre'):.5f}",
            f"{number(last, 'actor_fisher_norm_post'):.5f}",
            f"{number(last, 'actor_l2_norm_pre'):.5f}",
            f"{number(last, 'actor_l2_norm_post'):.5f}",
        ]
        print("\t".join(values))


if __name__ == "__main__":
    main()
