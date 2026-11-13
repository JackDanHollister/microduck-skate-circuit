"""Save a driver's results next to its parent and build the GitHub run tree."""

import argparse
import itertools
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "lineage"
ARTIFACTS = ("driver.json", "race.json", "evaluation.json")
FIRST = "jackdh-first-lap"
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
TRACK = ROOT / "data" / "reinvent2018.json"
LEAGUE_STRATEGIES = ("pace", "corner", "stability")
SCORE_SEEDS = range(1001, 1017)
INDEX_FILES = ("index.md", "index.json", "scoreboard.md")


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
    by_seed = [
        {a["seed"]: a for a in evaluation[key]["attempts"]}
        for key in ("baseline", "challenger")
    ]
    paired = [
        (a["time"], by_seed[1][seed]["time"])
        for seed, a in by_seed[0].items()
        if a["reason"] == by_seed[1][seed]["reason"] == "finished"
    ]
    require(
        evaluation["paired_clean_laps"] == len(paired)
        and evaluation["challenger_wins"] == sum(c < b for b, c in paired),
        "Paired finishes or wins do not match the per-start results.",
    )
    return race["challenger"], evaluation["challenger"], race["seed"]


def validate_record(folder, ident):
    card = load(folder / "run.json")
    require(
        card["version"] == 1 and card["id"] == ident,
        f"Invalid run card at {folder}",
    )
    source = card["source"]
    require(
        all(
            isinstance(source[k], str) and re.fullmatch(r"[0-9a-f]{64}", source[k])
            for k in ("simulator_sha256", "track_sha256")
        )
        and (
            source["revision"] is None
            or (
                isinstance(source["revision"], str)
                and re.fullmatch(r"[0-9a-f]{40,64}", source["revision"])
            )
        ),
        f"Invalid source revision or file hashes for {ident}.",
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
    return parent, lap, ev, seed


def records(archive):
    require(
        archive.is_dir() and not archive.is_symlink(),
        f"Missing run folder at {archive}",
    )
    entries = {}
    for folder in sorted(archive.iterdir()):
        if folder.name.startswith(".") or folder.name in INDEX_FILES:
            continue
        require(
            folder.is_dir() and not folder.is_symlink(), f"Unexpected entry at {folder}"
        )
        ident = run_id(folder.name)
        entries[ident] = validate_record(folder, ident)
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
        "in each run’s results. [Scoreboard](scoreboard.md) · [Add your run](../docs/runs.md).",
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


def milliseconds(seconds):
    return int((Decimal(str(seconds)) * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def standings(entries):
    scored = []
    for ident, (parent, _, evaluation, _) in entries.items():
        attempts = evaluation["attempts"]
        if sorted(a["seed"] for a in attempts) != list(SCORE_SEEDS):
            continue
        times = [a["time"] for a in attempts if a["reason"] == "finished"]
        scored.append({
            "id": ident, "parent": parent, "finishes": len(times),
            "median_ms": milliseconds(median(times)) if times else None,
            "best_ms": milliseconds(min(times)) if times else None,
        })
    scored.sort(key=lambda row: (-row["finishes"], row["median_ms"] or 0, row["id"]))
    previous, place = None, None
    for position, row in enumerate(scored, 1):
        key = (row["finishes"], row["median_ms"])
        if key != previous:
            place = position if row["finishes"] else None
        row["place"] = place
        previous = key
    return scored


def scoreboard_text(rows):
    lines = [
        "# Course driver scoreboard", "",
        "I run each course-trained driver with Pollen’s roller from a standing start on the same "
        "16 starts (seeds 1001–1016). More finishes rank first, then the quicker median finished lap. "
        "Matching scores to the millisecond share a place. A driver needs one finish to place.", "",
        "[Watch the 3D replays](https://jackdanhollister.github.io/microduck-skate-circuit/scoreboard.html)"
        " · [Enter a racer](../docs/runs.md#enter-the-shared-scoreboard) · [Run tree](index.md)", "",
        "| Place | Runner | Parent | Finishes | Median lap | Best lap |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        ident, parent = row["id"], row["parent"]
        ancestor = f"[{parent}]({parent}/driver.json)" if parent else "—"
        times = ["—" if row[k] is None else f"{row[k] / 1000:.3f} s" for k in ("median_ms", "best_ms")]
        lines.append(f"| {row['place'] or '—'} | [{ident}]({ident}/evaluation.json) | {ancestor} | "
                     f"{row['finishes']}/{len(SCORE_SEEDS)} | {times[0]} | {times[1]} |")
    if not rows:
        lines.extend(["", "No runs have a standard 16-start evaluation yet."])
    return "\n".join(lines) + "\n"


def indexes(entries):
    rows = standings(entries)
    scores = {row["id"]: {k: v for k, v in row.items() if k not in ("id", "parent")} for row in rows}
    catalog = [{"id": ident, "parent": entries[ident][0], "score": scores.get(ident)} for ident in sorted(entries)]
    return {
        "index.md": index_text(entries),
        "index.json": json.dumps(catalog, indent=2) + "\n",
        "scoreboard.md": scoreboard_text(rows),
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


def verify_scores(archive, workers):
    """Re-score archived course drivers using the checked-out simulator and stock roller."""
    from multiprocessing import Pool
    from race import Driver, episode

    entries = records(archive)
    jobs, expected = [], []
    for row in standings(entries):
        ident = row["id"]
        settings = Driver(**load(archive / ident / "driver.json")["driver"]).vector()
        for attempt in entries[ident][2]["attempts"]:
            jobs.append((str(TRACK), settings, attempt["seed"]))
            expected.append((ident, attempt))
    require(workers > 0, "Use a positive worker count.")
    with Pool(workers) as pool:
        for (ident, saved), actual in zip(expected, pool.imap(episode, jobs), strict=True):
            require(actual["reason"] == saved["reason"] and abs(actual["time"] - saved["time"]) <= 0.00001,
                    f"{ident} differs on seed {saved['seed']}: saved {saved['reason']} at {saved['time']} s, "
                    f"simulated {actual['reason']} at {actual['time']} s.")
    print(f"Reproduced {len(jobs)} scored attempts.")


def provenance():
    """Identify unchanged committed code, including when run from a downloaded ZIP."""
    simulator_hash = sha256((ROOT / "race.py").read_bytes()).hexdigest()
    track_hash = sha256(TRACK.read_bytes()).hexdigest()
    source = {
        "simulator_sha256": simulator_hash,
        "track_sha256": track_hash,
        "revision": None,
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        for name, expected in (
            ("race.py", simulator_hash),
            ("data/reinvent2018.json", track_hash),
        ):
            data = subprocess.check_output(
                ["git", "show", f"{revision}:{name}"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            if sha256(data).hexdigest() != expected:
                return source
        source["revision"] = revision
    except (OSError, subprocess.CalledProcessError):
        pass
    return source


def stage_record(stage, ident, parent, paths):
    for name, source in zip(ARTIFACTS, paths, strict=True):
        require(
            source.is_file() and not source.is_symlink(),
            f"Expected a regular file at {source}",
        )
        (stage / name).write_bytes(source.read_bytes())
    assess(stage)
    card = {
        "version": 1,
        "id": ident,
        "parent": parent,
        "source": provenance(),
        "files": filesums(stage),
    }
    (stage / "run.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )


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
        stage_record(stage, ident, parent, paths)
        stage.rename(archive / ident)
    write_index(archive)


def add_league(archive, prefix, parent, directory):
    """Archive all three timed replays, including slower drivers or a DNF."""
    run_id(prefix)
    run_id(parent)
    ids = set(records(archive))
    require(parent in ids, f"Unknown parent {parent}.")
    original = directory / "parent.json"
    require(original.is_file() and not original.is_symlink() and
            original.read_bytes() == (archive / parent / "driver.json").read_bytes(),
            "The league must have started from the checkpoint of the specified parent.")
    with tempfile.TemporaryDirectory(prefix=".league-", dir=archive) as tmp:
        staged = []
        for strategy in LEAGUE_STRATEGIES:
            ident = run_id(f"{prefix}-{strategy}")
            child = Path(tmp) / ident
            child.mkdir()
            paths = [directory / strategy / name for name in ARTIFACTS]
            stage_record(child, ident, parent, paths)
            validate_record(child, ident)
            if ident in ids:
                existing = archive / ident
                require(load(existing / "run.json")["parent"] == parent and
                        filesums(existing) == filesums(child),
                        f"Run {ident} already exists with different results. Choose another prefix.")
            else:
                staged.append(child)
        # Validate every sibling before installing any of them. An interrupted
        # run can be retried with matching records already present.
        for child in staged:
            child.rename(archive / child.name)
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
    family = actions.add_parser("add-league", help="archive all three ghosts from a completed league")
    family.add_argument("prefix", help="new run ID prefix shared by the three siblings")
    family.add_argument("--parent", required=True)
    family.add_argument("--from-dir", required=True, type=Path)
    actions.add_parser(
        "check", help="check the run cards, hashes, results, ancestry and GitHub index"
    )
    actions.add_parser("index", help="rebuild the GitHub index after merging branches")
    verify = actions.add_parser("verify-scores", help="simulate the scoreboard starts again from the saved drivers")
    verify.add_argument("--workers", type=int, default=2)
    args = cli.parse_args()
    try:
        if args.action == "add":
            add(
                ARCHIVE,
                args.id,
                args.parent,
                [args.driver, args.replay, args.evaluation],
            )
        elif args.action == "add-league":
            add_league(ARCHIVE, args.prefix, args.parent, args.from_dir)
        elif args.action == "index":
            write_index(ARCHIVE)
        elif args.action == "verify-scores":
            verify_scores(ARCHIVE, args.workers)
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
