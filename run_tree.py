"""Save a driver's results next to its parent and build the GitHub run tree."""

import argparse
import itertools
import json
import math
import re
import tempfile
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "lineage"
ARTIFACTS = ("driver.json", "race.json", "evaluation.json")
FIRST = "jackdh-first-lap"
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
TRACK = ROOT / "data" / "reinvent2018.json"


def run_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Use a run ID with 1–64 lowercase letters, digits or hyphens.")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def load(path):
    require(
        path.is_file() and not path.is_symlink(), f"Expected a regular file at {path}"
    )
    return json.loads(path.read_bytes())


def filesums(folder):
    return {
        name: sha256((folder / name).read_bytes()).hexdigest() for name in ARTIFACTS
    }


def assess(folder):
    driver = load(folder / "driver.json")["driver"]
    speeds, offsets = driver["speed"], driver["offset"]
    require(
        isinstance(speeds, list)
        and isinstance(offsets, list)
        and len(speeds) == len(offsets)
        and len(speeds) > 0,
        "Driver speeds and offsets need the same number of sectors.",
    )
    require(
        all(
            finite(n)
            for n in [*speeds, *offsets, driver["lookahead"], driver["turn_gain"]]
        ),
        "Driver settings must be finite numbers.",
    )
    race = load(folder / "race.json")
    require(race["track"] == load(TRACK), "This tree uses the re:Invent 2018 course.")
    require(type(race["seed"]) is int, "Replay needs an integer start seed.")
    for key in ("baseline", "challenger"):
        lap = race[key]
        require(
            finite(lap["time"]) and lap["time"] > 0 and type(lap["finished"]) is bool,
            f"Invalid replay result for {key}.",
        )
        frames = lap["frames"]
        require(
            isinstance(frames, list)
            and len(frames) >= 2
            and all(
                isinstance(row, list) and len(row) == 22 and all(map(finite, row))
                for row in frames
            ),
            f"Invalid replay frames for {key}.",
        )
        require(
            frames[0][0] == 0
            and all(a[0] < b[0] for a, b in itertools.pairwise(frames))
            and math.isclose(frames[-1][0], lap["time"], abs_tol=1e-5),
            f"Replay frames do not match the race clock for {key}.",
        )
    evaluation = load(folder / "evaluation.json")
    seeds = {}
    for key in ("baseline", "challenger"):
        item = evaluation[key]
        attempts = item["attempts"]
        require(
            isinstance(attempts, list) and attempts,
            "An evaluation must include timed attempts.",
        )
        require(
            all(
                type(a["seed"]) is int
                and finite(a["time"])
                and a["time"] > 0
                and isinstance(a["reason"], str)
                and a["reason"]
                for a in attempts
            ),
            f"Invalid evaluation attempt for {key}.",
        )
        seeds[key] = {a["seed"] for a in attempts}
        require(
            len(seeds[key]) == len(attempts), f"Duplicate evaluation seed for {key}."
        )
        times = [a["time"] for a in attempts if a["reason"] == "finished"]
        recorded = item["median_clean_s"]
        valid_median = (
            finite(recorded) and math.isclose(median(times), recorded, abs_tol=1e-5)
            if times
            else recorded is None
        )
        require(
            item["finishes"] == len(times) and valid_median,
            f"Finish count or median differs from individual attempts for {key}.",
        )
    require(
        seeds["baseline"] == seeds["challenger"], "Evaluation starts must be paired."
    )
    return race["challenger"], evaluation["challenger"], race["seed"]


def records(archive):
    require(
        archive.is_dir() and not archive.is_symlink(),
        f"Missing run folder at {archive}",
    )
    entries = {}
    for folder in sorted(archive.iterdir()):
        if folder.name.startswith(".") or folder.name in ("index.md", "index.json"):
            continue
        require(
            folder.is_dir() and not folder.is_symlink(), f"Unexpected entry at {folder}"
        )
        ident = run_id(folder.name)
        card = load(folder / "run.json")
        require(
            card["version"] == 1 and card["id"] == ident,
            f"Invalid run card at {folder}",
        )
        parent = card["parent"]
        if ident == FIRST:
            require(parent is None, f"{FIRST} must be the root.")
        else:
            run_id(parent)
        lap, ev, seed = assess(folder)
        require(
            card["files"] == filesums(folder), f"Recorded files were changed in {ident}"
        )
        entries[ident] = (parent, lap, ev, seed)
    require(FIRST in entries, f"Missing first run {FIRST}")
    for ident, (parent, *_) in entries.items():
        require(
            parent is None or parent in entries, f"Unknown parent {parent} for {ident}."
        )
    return entries


def index_text(entries):
    lines = [
        "# Run tree",
        "",
        "Each driver sits under the driver it started from. The lap is the",
        "recorded replay. Finishes and the median cover the evaluation starts listed",
        "in each run’s results. [Add your run](../docs/runs.md).",
        "",
    ]
    children = defaultdict(list)
    for ident, (parent, *_) in entries.items():
        children[parent].append(ident)
    seen = set()
    stack = [(FIRST, 0)]
    while stack:
        ident, depth = stack.pop()
        require(ident not in seen, f"Cycle at {ident}")
        seen.add(ident)
        _, lap, ev, seed = entries[ident]
        time = f"{lap['time']:.2f} s" if lap["finished"] else "DNF"
        typical = f", median {ev['median_clean_s']:.2f} s" if ev["finishes"] else ""
        lines.append(
            f"{'  ' * depth}- **[{ident}]({ident}/driver.json)** · seed {seed} lap {time} · "
            f"{ev['finishes']}/{len(ev['attempts'])} finishes{typical} · "
            f"[Replay JSON]({ident}/race.json) · [Results]({ident}/evaluation.json)"
        )
        stack.extend(
            (child, depth + 1) for child in sorted(children[ident], reverse=True)
        )
    require(
        len(seen) == len(entries),
        "Some runs form a cycle disconnected from the first run.",
    )
    return "\n".join(lines) + "\n"


def indexes(entries):
    catalog = [{"id": ident, "parent": entries[ident][0]} for ident in sorted(entries)]
    return {
        "index.md": index_text(entries),
        "index.json": json.dumps(catalog, indent=2) + "\n",
    }


def replace_text(target, content):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=".index-", delete=False
    ) as temp:
        temp.write(content)
    path = Path(temp.name)
    path.chmod(0o644)
    path.replace(target)


def write_index(archive):
    for name, content in indexes(records(archive)).items():
        replace_text(archive / name, content)


def add(archive, ident, parent, paths):
    run_id(ident)
    run_id(parent)
    require(
        not (archive / ident).exists(), f"Run {ident} already exists. Choose a new ID."
    )
    require(parent in records(archive), f"Unknown parent {parent}.")
    with tempfile.TemporaryDirectory(prefix=".new-run-", dir=archive) as directory:
        stage = Path(directory) / ident
        stage.mkdir()
        for name, source in zip(ARTIFACTS, paths, strict=True):
            require(
                source.is_file() and not source.is_symlink(),
                f"Expected a regular file at {source}",
            )
            (stage / name).write_bytes(source.read_bytes())
        assess(stage)
        card = {"version": 1, "id": ident, "parent": parent, "files": filesums(stage)}
        (stage / "run.json").write_text(
            json.dumps(card, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(archive / ident)
    write_index(archive)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    new = actions.add_parser(
        "add", help="archive a checkpoint and its matching replay and evaluation"
    )
    new.add_argument("id")
    new.add_argument("--parent", required=True)
    for name in ("driver", "replay", "evaluation"):
        new.add_argument("--" + name, type=Path, required=True)
    actions.add_parser(
        "check", help="check the run cards, hashes, results, ancestry and GitHub index"
    )
    actions.add_parser("index", help="rebuild the GitHub index after merging branches")
    args = cli.parse_args()
    try:
        if args.action == "add":
            add(
                ARCHIVE,
                args.id,
                args.parent,
                [args.driver, args.replay, args.evaluation],
            )
        elif args.action == "index":
            write_index(ARCHIVE)
        else:
            for name, content in indexes(records(ARCHIVE)).items():
                require(
                    (ARCHIVE / name).read_text(encoding="utf-8") == content,
                    f"{name} is out of date. Run python run_tree.py index.",
                )
    except (OSError, TypeError, KeyError, ValueError, IndexError) as exc:
        cli.exit(1, f"Run tree: {exc}\n")
    print("Run tree is up to date.")


if __name__ == "__main__":
    main()
