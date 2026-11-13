"""Compare fixed drivers and the centreline ghost on the same fresh simulator starts."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from race import ROOT, Driver, episode, save_json


def compare(track, parent, candidate, seeds, workers):
    names = ("centreline", "parent", "challenger")
    drivers = (Driver.constant(), parent, candidate)
    with Pool(workers) as pool:
        results = pool.map(
            episode,
            [
                (str(track), driver.vector(), seed)
                for driver in drivers
                for seed in seeds
            ],
            chunksize=1,
        )
    summary = {"seeds": list(seeds)}
    groups = {}
    for index, name in enumerate(names):
        group = results[index * len(seeds) : (index + 1) * len(seeds)]
        groups[name] = group
        times = [r["time"] for r in group if r["finished"]]
        summary[name] = {
            "finishes": len(times),
            "median_clean_s": float(np.median(times)) if times else None,
            "attempts": [
                {
                    "seed": seed,
                    "time": row["time"],
                    "progress": row["progress"],
                    "reason": row["reason"],
                }
                for seed, row in zip(seeds, group)
            ],
        }
    for name in ("centreline", "parent"):
        gains = [
            left["time"] - right["time"]
            for left, right in zip(groups[name], groups["challenger"])
            if left["finished"] and right["finished"]
        ]
        summary[f"challenger_vs_{name}"] = {
            "paired_finishes": len(gains),
            "wins": sum(g > 0 for g in gains),
            "median_seconds_saved": round(float(np.median(gains)), 5)
            if gains
            else None,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument(
        "--track", type=Path, default=ROOT / "data" / "reinvent2018.json"
    )
    parser.add_argument("--first-seed", type=int, default=8200)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.workers, args.starts) < 1 or args.first_seed < 0:
        parser.error(
            "Use at least one worker and start, with a non-negative first seed"
        )

    def read_driver(path):
        return Driver(**json.loads(path.read_text())["driver"])

    summary = compare(
        args.track,
        read_driver(args.parent),
        read_driver(args.challenger),
        range(args.first_seed, args.first_seed + args.starts),
        args.workers,
    )
    save_json(args.output, summary)
    for key in ("centreline", "parent", "challenger"):
        print(
            key,
            summary[key]["finishes"],
            "/",
            args.starts,
            "clean, median seconds",
            summary[key]["median_clean_s"],
        )
    print("Challenger versus parent", summary["challenger_vs_parent"])
