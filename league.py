"""Run three independent policy searches concurrently within a shared CPU budget."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from race import ROOT, Driver, save_json
from refine import assess, quality

STRATEGIES = {"pace": 12, "corner": 12, "stability": 6}


def train_one(root, name, args, workers, search_seed):
    folder = root / name
    folder.mkdir()
    command = [
        sys.executable,
        "-u",
        str(ROOT / "race.py"),
        "train",
        "--start-from",
        str(root / "parent.json"),
        "--track",
        str(args.track.resolve()),
        "--generations",
        str(args.generations),
        "--sectors",
        str(STRATEGIES[name]),
        "--workers",
        str(workers),
        "--strategy",
        name,
        "--search-seed",
        str(search_seed),
        "--policy",
        str(folder / "driver.json"),
        "--output",
        str(folder / "race.json"),
    ]
    env = os.environ | {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    print(
        f"{name}: {workers} workers, {STRATEGIES[name]} sectors, search seed {search_seed}",
        flush=True,
    )
    evaluate = [
        sys.executable,
        "-u",
        str(ROOT / "race.py"),
        "evaluate",
        "--policy",
        str(folder / "driver.json"),
        "--track",
        str(args.track.resolve()),
        "--workers",
        str(workers),
        "--report",
        str(folder / "evaluation.json"),
    ]
    with (folder / "training.log").open("w") as log:
        for invocation in (command, evaluate):
            process = subprocess.run(
                invocation, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
            )
            if process.returncode:
                raise RuntimeError(
                    f"{name} exited with {process.returncode}; inspect {folder / 'training.log'}"
                )
    print(f"{name}: training complete", flush=True)
    return name


def select(root, track, workers):
    names = ["parent", *STRATEGIES]
    paths = [
        root / "parent.json",
        *[root / name / "driver.json" for name in STRATEGIES],
    ]
    vectors = [
        Driver(**json.loads(path.read_text())["driver"]).vector() for path in paths
    ]
    seeds = [1, *range(10600, 10647)]
    with Pool(workers) as pool:
        groups = assess(pool, str(track.resolve()), vectors, seeds)
    # Seed 1 is the demo, then all other starts equally govern selection.
    values = [
        quality(group) if group[0]["finished"] else (-1, -95.0) for group in groups
    ]
    best = max(range(len(values)), key=values.__getitem__)
    summary = {}
    for name, group in zip(names, groups):
        times = [r["time"] for r in group if r["finished"]]
        summary[name] = {
            "finishes": len(times),
            "median_clean_s": float(np.median(times)) if times else None,
            "attempts": [
                {"seed": s, "time": r["time"], "reason": r["reason"]}
                for r, s in zip(group, seeds)
            ],
        }
    shutil.copyfile(paths[best], root / "winner.json")
    save_json(
        root / "selection.json",
        {
            "winner": names[best],
            "winner_sha256": hashlib.sha256(
                (root / "winner.json").read_bytes()
            ).hexdigest(),
            "seeds": seeds,
            "results": summary,
        },
    )
    print(f"Winner: {names[best]} ({values[best][0]}/{len(seeds)} clean)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train"])
    parser.add_argument("--start-from", type=Path, required=True)
    parser.add_argument(
        "--track", type=Path, default=ROOT / "data" / "reinvent2018.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Total physics workers shared by concurrent searches (1–18)",
    )
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=20261113)
    args = parser.parse_args()
    if not 1 <= args.workers <= 18 or args.generations < 1 or args.search_seed < 0:
        parser.error(
            "Use 1–18 workers, positive generations and a non-negative search seed"
        )
    if not args.track.is_file():
        parser.error("Could not read track")
    parent = args.start_from.read_bytes()
    Driver(**json.loads(parent)["driver"])
    root = args.output.resolve()
    if root.exists():
        parser.error(
            f"{root} already exists; choose a fresh directory to protect earlier runs"
        )
    root.mkdir(parents=True)
    (root / "parent.json").write_bytes(parent)
    active = min(args.workers, len(STRATEGIES))
    per, remainder = divmod(args.workers, active)
    failures = []
    with ThreadPoolExecutor(max_workers=active) as pool:
        futures = [
            pool.submit(
                train_one, root, name, args, per + (i < remainder), args.search_seed + i
            )
            for i, name in enumerate(STRATEGIES)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except (OSError, RuntimeError) as error:
                failures.append(str(error))
                print(error, file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(
            "Some searches failed; successful driver files and individual logs are preserved"
        )
    select(root, args.track, args.workers)


if __name__ == "__main__":
    main()
