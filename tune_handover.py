"""Tune the transition after checkpoint 500's sprint launch on the skate circuit."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from statistics import median

import numpy as np
from arena.constants import JOINT_NAMES

from race import CTRL_DT, ROOT, Driver, course, save_json, simulate
from sprint_transfer import TRACK, controller
from sprint_transfer import driver as duck28

# The selection and final starts are never passed to the optimiser.
TRAIN = range(14100, 14105)
SELECT = range(14200, 14216)
TEST = range(14300, 14364)
EXAMPLE = 182


@dataclass(frozen=True)
class Tune:
    name: str
    speed: float
    turn: float
    turn_limit: float
    blend_s: float
    steer_from_m: float
    change_from_m: float
    change_s: float
    line: str


@lru_cache(maxsize=1)
def corner():
    return Driver(
        **json.loads((ROOT / "lineage/jackdh-corner/driver.json").read_text())["driver"]
    )


class Handover:
    def __init__(self, settings):
        self.settings = settings
        self.line = corner() if settings.line == "corner" else duck28()
        self.steer_ticks = 0
        self.change_ticks = 0
        self.mix = 0.0
        self.at_gate = False

    def command(self, track, along, xy, heading):
        p = self.settings
        # A projection just behind the starting line wraps to the end of the
        # course. Only arm each transition on the opening straight.
        on_start = along < track.length / 2
        if p.name == "abrupt":
            right, _, forward = track.gates[0]
            self.at_gate = self.at_gate or np.dot(xy - right, forward) >= 0
            return (
                self.line.command(track, along, xy, heading)
                if self.at_gate
                else (0.8, 0.0)
            )
        if self.steer_ticks or (on_start and along >= p.steer_from_m):
            self.steer_ticks += 1
        if self.change_ticks or (on_start and along >= p.change_from_m):
            self.change_ticks += 1
        interval = 3 * CTRL_DT  # race.simulate refreshes the driver every third tick
        self.mix = (
            min(1.0, self.change_ticks * interval / p.change_s)
            if p.name == "relay"
            else 0.0
        )
        blend = min(1.0, self.steer_ticks * interval / p.blend_s)
        speed, yaw = self.line.command(track, along, xy, heading)
        target_speed = 0.8 + p.speed * (speed - 0.8)
        target_yaw = max(-p.turn_limit, min(p.turn_limit, p.turn * yaw))
        return 0.8 + blend * (target_speed - 0.8), blend * target_yaw

    def act(self, obs):
        if self.mix == 0:
            return controller("checkpoint-500").act(obs)
        if self.mix == 1:
            return controller("pollen").act(obs)
        sprint = controller("checkpoint-500").act(obs)
        skate = controller("pollen").act(obs)
        return (1 - self.mix) * sprint + self.mix * skate


def candidates():
    rng = np.random.default_rng(20261114)
    rows = []
    for name in ("sprint-only", "relay"):
        for _ in range(48):
            if name == "relay":
                change_from = rng.uniform(0.85, 2.5)
                change_s = rng.uniform(0.08, 1.1)
                speed, turn, limit = 1.0, 1.0, 2.2
            else:
                change_from, change_s = 0.0, 1.0
                speed = rng.uniform(0.0, 1.15)
                turn = rng.uniform(0.08, 1.2)
                limit = rng.uniform(0.12, 1.6)
            rows.append(
                Tune(
                    name,
                    speed,
                    turn,
                    limit,
                    rng.uniform(0.15, 1.9),
                    rng.uniform(1.25, 2.65),
                    change_from,
                    change_s,
                    "corner" if rng.random() < 0.5 else "duck28",
                )
            )
    return rows


def attempt(job):
    settings, seed, record = job
    agent = Handover(settings)
    skater = controller("pollen") if settings.name == "pollen-warm" else agent
    if settings.name == "pollen-standard":
        lap = simulate(TRACK, corner(), seed, record=record)
    else:
        lap = simulate(
            TRACK, agent, seed, record=record, controller=skater, settle_speed=0.8
        )
    lap.pop("commands", None)
    if not record:
        lap.pop("frames", None)
    return lap


def evaluate(pool, options, seeds):
    jobs = [(p, seed, False) for p in options for seed in seeds]
    results = pool.map(attempt, jobs, chunksize=1)
    return [results[i : i + len(seeds)] for i in range(0, len(results), len(seeds))]


def rank(laps):
    return (
        sum(r["finished"] for r in laps),
        sum(len(r["gates"]) for r in laps),
        sum(r["progress"] for r in laps),
        -sum(r["time"] for r in laps if r["finished"]),
    )


def summarise(laps):
    clean = [r["time"] for r in laps if r["finished"]]
    return {
        "starts": len(laps),
        "gate1": sum(bool(r["gates"]) for r in laps),
        "gate2": sum(len(r["gates"]) >= 2 for r in laps),
        "gate3": sum(len(r["gates"]) >= 3 for r in laps),
        "finishes": len(clean),
        "median_clean_s": median(clean) if clean else None,
    }


def run(output, workers):
    options = candidates()
    prior = Tune(
        "abrupt", 1.0, 1.0, 2.2, 0.001, course(TRACK).length / 8, 0.0, 1.0, "duck28"
    )
    with Pool(workers) as pool:
        development = evaluate(pool, options, TRAIN)
        shortlist = []
        for family in ("sprint-only", "relay"):
            indices = [i for i, p in enumerate(options) if p.name == family]
            shortlist.extend(
                sorted(indices, key=lambda i: rank(development[i]), reverse=True)[:5]
            )
        print(
            "Initial search",
            [
                (options[i].name, summarise(development[i]))
                for i in [shortlist[0], shortlist[5]]
            ],
            flush=True,
        )
        selected = evaluate(pool, [options[i] for i in shortlist], SELECT)
        best = {}
        for family in ("sprint-only", "relay"):
            top = max(
                (j for j, i in enumerate(shortlist) if options[i].name == family),
                key=lambda j: rank(selected[j]),
            )
            best[family] = options[shortlist[top]]
            print("Selection", family, summarise(selected[top]), flush=True)
        result = {
            "search_seed": 20261114,
            "training_seeds": list(TRAIN),
            "selection_seeds": list(SELECT),
            "held_out_seeds": list(TEST),
            "candidates": [
                {"settings": asdict(p), "summary": summarise(l)}
                for p, l in zip(options, development)
            ],
            "shortlist": [
                {"candidate": i, "summary": summarise(l)}
                for i, l in zip(shortlist, selected)
            ],
            "frozen": {k: asdict(v) for k, v in best.items()},
        }
        labels = {
            "abrupt": "Checkpoint 500 · abrupt steering",
            "sprint-only": "Checkpoint 500 · gradual steering",
            "relay": "Checkpoint 500 → Pollen roller",
            "pollen-standard": "Pollen · standing start",
            "pollen-warm": "Pollen · same rolling start",
        }
        methods = {
            "abrupt": prior,
            **best,
            "pollen-standard": replace(best["relay"], name="pollen-standard"),
            "pollen-warm": replace(best["relay"], name="pollen-warm"),
        }
        held_out = evaluate(pool, list(methods.values()), TEST)
        field = []
        result["held_out"] = {}
        for (name, p), laps in zip(methods.items(), held_out):
            result["held_out"][name] = dict(**summarise(laps), attempts=laps)
            example = attempt((p, EXAMPLE, True))
            print(
                "Held out",
                name,
                summarise(laps),
                "example",
                example["reason"],
                example["time"],
                len(example["gates"]),
                flush=True,
            )
            field.append({"id": name, "label": labels[name], "run": example})
        save_json(
            output / "race.json",
            {
                "title": "Checkpoint 500 into the bends",
                "detail": "Three driving setups and two Pollen controls",
                "seed": EXAMPLE,
                "track": course(TRACK).source,
                "joints": list(JOINT_NAMES),
                "field": field,
            },
        )
        save_json(output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "handover")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.output, args.workers)
