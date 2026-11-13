"""Run a Microduck time trial or train the racing line and speed in MuJoCo."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from arena import runner
from arena.constants import CTRL_DT, JOINT_NAMES
from arena.course import compile_stage
from arena.manifest import load_event
from arena.policy import load_policy
from arena.timeline import Start
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "pollen-robotics/microduck-policies"
MODEL_REV = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"
MODEL_HASH = "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd"
GATES = 8
SECTORS = 6
MAX_TIME = 95.0
TRACK_WINDOW = np.arange(-3, 4)
WHEEL_WINDOW = np.arange(-2, 3)
WHEEL_IDS = np.arange(4)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def wrapped(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Circuit:
    def __init__(self, filename):
        self.source = json.loads(Path(filename).read_text())
        self.centre = np.asarray(self.source["centre"], dtype=float)
        self.left = np.asarray(self.source["left"], dtype=float)
        self.right = np.asarray(self.source["right"], dtype=float)
        if (
            any(
                a.shape != (len(self.centre), 2) or not np.isfinite(a).all()
                for a in (self.centre, self.left, self.right)
            )
            or len(self.centre) < GATES
        ):
            raise ValueError("Track needs aligned, finite centre/left/right XY points")
        self.vectors = np.roll(self.centre, -1, axis=0) - self.centre
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        if self.lengths.min() < 1e-6:
            raise ValueError("Repeated centre points in track")
        self.s = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = self.s[-1]
        self.unit = self.vectors / self.lengths[:, None]
        self.normals = np.column_stack((-self.unit[:, 1], self.unit[:, 0]))
        self.left_width = np.linalg.norm(self.left - self.centre, axis=1)
        self.right_width = np.linalg.norm(self.right - self.centre, axis=1)
        self.gates = [
            self.cross_section(self.length * i / GATES) for i in range(1, GATES + 1)
        ]

    def at(self, s):
        s = float(s % self.length)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        return i, (s - self.s[i]) / self.lengths[i]

    def cross_section(self, s):
        i, frac = self.at(s)
        j = (i + 1) % len(self.centre)
        left = self.left[i] + frac * (self.left[j] - self.left[i])
        right = self.right[i] + frac * (self.right[j] - self.right[i])
        across = left - right
        forward = np.array([across[1], -across[0]]) / np.linalg.norm(across)
        if forward @ self.unit[i] < 0:
            forward = -forward
        return right, across, forward

    def point(self, s, offset):
        i, frac = self.at(s)
        return self.centre[i] + frac * self.vectors[i] + self.normals[i] * offset

    def project(self, point, previous):
        indices = (previous + TRACK_WINDOW) % len(self.centre)
        diff = point - self.centre[indices]
        along = np.clip((diff * self.unit[indices]).sum(1), 0.0, self.lengths[indices])
        err = diff - along[:, None] * self.unit[indices]
        best = np.argmin((err * err).sum(1))
        i = int(indices[best])
        return i, self.s[i] + along[best]


@lru_cache(maxsize=2)
def course(filename):
    return Circuit(filename)


@lru_cache(maxsize=1)
def roller():
    override = os.environ.get("MICRODUCK_ROLLER_ONNX")
    path = Path(
        override or hf_hub_download(MODEL_REPO, "roller.onnx", revision=MODEL_REV)
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_HASH:
        raise ValueError("Roller ONNX differs from the pinned Pollen checkpoint")
    policy = load_policy(path)

    class Skater:
        def act(self, obs):
            # Pollen's policy uses a 0.8 motor scale; Arena stores the scaled previous action.
            x = obs.copy()
            x[34:48] /= np.float32(0.8)
            output = policy.session.run(
                [policy._output_name], {policy._input_name: x[None, :]}
            )
            return output[0][0] * np.float32(0.8)

    return Skater()


@dataclass
class Driver:
    speed: list
    offset: list
    lookahead: float
    turn_gain: float

    @classmethod
    def constant(cls, speed=0.37, lookahead=0.6, gain=0.8):
        return cls([speed] * SECTORS, [0.0] * SECTORS, lookahead, gain)

    def __post_init__(self):
        if len(self.speed) != len(self.offset) or len(self.speed) not in (6, 12):
            raise ValueError(
                "Driver needs matching speed and offset lists for 6 or 12 sectors"
            )
        if not np.isfinite(self.vector()).all() or self.lookahead <= 0:
            raise ValueError(
                "Driver values must be finite and lookahead must be positive"
            )

    def resample(self, sectors):
        """Keep the same line and speeds while giving the learner more control points."""
        sample = [i / sectors for i in range(sectors)]
        speed = [self.interpolate(self.speed, along, 1.0) for along in sample]
        offset = [self.interpolate(self.offset, along, 1.0) for along in sample]
        return Driver(speed, offset, self.lookahead, self.turn_gain)

    def vector(self):
        return np.r_[self.speed, self.offset, self.lookahead, self.turn_gain]

    @classmethod
    def from_vector(cls, v):
        sectors = (len(v) - 2) // 2
        if 2 * sectors + 2 != len(v):
            raise ValueError("Driver parameter vector has the wrong length")
        return cls(
            v[:sectors].tolist(),
            v[sectors : 2 * sectors].tolist(),
            float(v[-2]),
            float(v[-1]),
        )

    def interpolate(self, array, s, length):
        phase = (s % length) * len(array) / length
        i = int(phase)
        return array[i] + (phase - i) * (array[(i + 1) % len(array)] - array[i])

    def command(self, track, s, xy, heading):
        ahead = s + self.lookahead
        offset = self.interpolate(self.offset, ahead, track.length)
        aim = track.point(ahead, offset) - xy
        bearing = wrapped(math.atan2(aim[1], aim[0]) - heading)
        speed = self.interpolate(self.speed, s, track.length)
        yaw = np.clip(self.turn_gain * bearing, -2.2, 2.2)
        return speed, yaw


def crossing(previous, current, gate):
    right, across, forward = gate
    u = np.dot(previous - right, forward)
    v = np.dot(current - right, forward)
    if not u < 0 <= v:
        return None
    fraction = -u / (v - u)
    where = previous + fraction * (current - previous)
    width_fraction = np.dot(where - right, across) / np.dot(across, across)
    return float(fraction) if 0 <= width_fraction <= 1 else None


@lru_cache(maxsize=2)
def setup(filename):
    track = course(filename)
    heading = math.degrees(math.atan2(*track.unit[0, ::-1]))
    start = Start(
        x=float(track.centre[0, 0]), y=float(track.centre[0, 1]), yaw_deg=heading
    )
    event = load_event("got-a-talent")
    model = compile_stage(event, [("rollers", start)])
    wheels = [
        i
        for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name.startswith("tire")
    ]
    if len(wheels) != 4:
        raise ValueError("Expected four wheel collision shapes")
    return event, model, start, wheels


def simulate(filename, driver, seed, record=False):
    track = course(filename)
    event, model, start, wheels = setup(filename)
    command = np.zeros(13)
    sim, _, _ = runner._start(
        event, model, [roller()], [start.yaw_deg], int(seed), [command]
    )
    data = sim.data
    trunk = sim.ducks[0].adr.trunk
    xy = data.xpos[trunk, :2].copy()
    segment, s = track.project(xy, 0)
    frames = []
    samples = []
    times = []
    reason = "time limit"
    furthest = 0.0
    elapsed = 0.0
    frame = np.empty((1, 21), np.float32)
    previous_frame = np.empty_like(frame)

    def record_frame(t):
        sim.record(frame)
        frames.append([round(t, 5), *np.round(frame[0].astype(float), 6).tolist()])

    if record:
        record_frame(0.0)
    for tick in range(round(MAX_TIME / CTRL_DT)):
        rot = data.xmat[trunk].reshape(3, 3)
        if tick % 3 == 0:
            heading = math.atan2(rot[1, 0], rot[0, 0])
            command[0], command[2] = driver.command(track, s, xy, heading)
        prev = xy.copy()
        if record and len(times) == GATES - 1:
            sim.record(previous_frame)
        sim.step([command])
        xy = data.xpos[trunk, :2].copy()
        segment, s = track.project(xy, segment)
        elapsed = (tick + 1) * CTRL_DT
        if record and (tick + 1) % 2 == 0:
            record_frame(elapsed)
        # Wheel centres must all lie inside the surveyed edges. Project each
        # onto its nearest local segment (the four wheels can straddle a vertex).
        ids = (segment + WHEEL_WINDOW) % len(track.centre)
        rel = data.geom_xpos[wheels, :2][:, None, :] - track.centre[ids]
        along = np.clip((rel * track.unit[ids]).sum(2), 0.0, track.lengths[ids])
        diff = rel - along[:, :, None] * track.unit[ids]
        nearest = np.argmin((diff * diff).sum(2), axis=1)
        wi = WHEEL_IDS
        si = ids[nearest]
        lateral = (diff[wi, nearest] * track.normals[si]).sum(1)
        frac = along[wi, nearest] / track.lengths[si]
        sj = (si + 1) % len(track.centre)
        lw = track.left_width[si] + frac * (track.left_width[sj] - track.left_width[si])
        rw = track.right_width[si] + frac * (
            track.right_width[sj] - track.right_width[si]
        )
        if np.any((lateral > lw) | (lateral < -rw)):
            reason = "off track"
            break
        if data.xmat[trunk, 8] < 0.5 or data.xpos[trunk, 2] < 0.065:
            reason = "fall"
            break
        if len(times) and s < 0.2 * track.length and len(times) >= GATES - 1:
            progress = track.length + s
        elif not len(times) and s > 0.8 * track.length:
            progress = s - track.length
        else:
            progress = s
        furthest = max(furthest, min(progress / track.length, (len(times) + 1) / GATES))
        hit = crossing(prev, xy, track.gates[len(times)])
        if hit is not None:
            gate_time = (tick + hit) * CTRL_DT
            times.append(round(gate_time, 5))
            if len(times) == GATES:
                elapsed = gate_time
                reason = "finished"
                furthest = 1.0
                break
        if tick % 50 == 0 and record:
            samples.append(
                [
                    round(elapsed, 3),
                    round(float(s), 4),
                    round(float(command[0]), 4),
                    round(float(command[2]), 4),
                ]
            )
    if record:
        # End the replay exactly on the interpolated finish, without a 20 ms overshoot.
        if reason == "finished":
            sim.record(frame)
            if frames and frames[-1][0] >= elapsed - 0.00001:
                frames.pop()
            start_pose = previous_frame[0].astype(float)
            end_pose = frame[0].astype(float)
            if start_pose[3:7] @ end_pose[3:7] < 0:
                end_pose[3:7] *= -1
            interpolated = start_pose + hit * (end_pose - start_pose)
            interpolated[3:7] /= np.linalg.norm(interpolated[3:7])
            frames.append([round(elapsed, 5), *np.round(interpolated, 6).tolist()])
        elif frames[-1][0] < elapsed:
            record_frame(elapsed)
    return {
        "finished": reason == "finished",
        "time": round(elapsed, 5),
        "progress": round(float(furthest), 5),
        "gates": times,
        "reason": reason,
        "frames": frames,
        "commands": samples,
    }


def score(result):
    if result["finished"]:
        return 100.0 - result["time"] * 0.5
    return 25.0 * result["progress"] - result["time"] * 0.02


def episode(job):
    filename, v, seed = job
    return simulate(filename, Driver.from_vector(v), seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "replay", "baseline", "evaluate"])
    parser.add_argument(
        "--track", type=Path, default=ROOT / "data" / "reinvent2018.json"
    )
    parser.add_argument("--policy", type=Path, default=ROOT / "driver.json")
    parser.add_argument("--output", type=Path, default=ROOT / "replays" / "race.json")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "replays" / "evaluation.json"
    )
    parser.add_argument(
        "--start-from", type=Path, help="Train a descendant of this saved driver"
    )
    parser.add_argument(
        "--sectors",
        type=int,
        choices=[6, 12],
        help="Number of speed and line control points",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        help="Random seed for driver search, independently of the replay start",
    )
    parser.add_argument(
        "--strategy",
        choices=["balanced", "pace", "corner", "stability"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--generations", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.mode != "train" and (
        args.start_from is not None
        or args.sectors is not None
        or args.search_seed is not None
        or args.strategy != "balanced"
    ):
        parser.error("Driver search options require train mode")
    if args.search_seed is not None and args.search_seed < 0:
        parser.error("--search-seed must be non-negative")
    filename = str(args.track.resolve())
    base = Driver.constant()
    if args.mode == "baseline":
        print(
            {
                k: v
                for k, v in simulate(filename, base, args.seed).items()
                if k not in ("frames", "commands")
            }
        )
        return
    if args.mode == "train":
        refinement = bool(
            args.start_from or args.sectors or args.strategy != "balanced"
        )
        seed = (
            args.search_seed
            if args.search_seed is not None
            else (21113 if refinement else 20261113)
        )
        if refinement:
            initial = (
                Driver(**json.loads(args.start_from.read_text())["driver"])
                if args.start_from
                else base
            )
            from refine import optimize

            best, history = optimize(
                filename,
                initial,
                args.workers,
                args.generations,
                args.sectors or len(initial.speed),
                seed,
                args.strategy,
            )
        else:
            best, history = train(filename, base, args.workers, args.generations, seed)
        save_json(
            args.policy,
            {
                "driver": asdict(best),
                "history": history,
                "search_seed": seed,
                "strategy": args.strategy,
            },
        )
    else:
        best = Driver(**json.loads(args.policy.read_text())["driver"])
    if args.mode == "evaluate":
        seeds = list(range(1001, 1017))
        with Pool(args.workers) as pool:
            results = pool.map(
                episode,
                [
                    (filename, driver.vector(), s)
                    for driver in (base, best)
                    for s in seeds
                ],
                chunksize=1,
            )
        summary = {}
        for index, name in enumerate(("baseline", "challenger")):
            attempts = results[index * len(seeds) : (index + 1) * len(seeds)]
            times = [r["time"] for r in attempts if r["finished"]]
            summary[name] = {
                "finishes": len(times),
                "median_clean_s": float(np.median(times)) if times else None,
                "attempts": [
                    {
                        "seed": s,
                        "time": r["time"],
                        "progress": r["progress"],
                        "reason": r["reason"],
                    }
                    for s, r in zip(seeds, attempts)
                ],
            }
        paired = [
            (b["time"], c["time"])
            for b, c in zip(results[: len(seeds)], results[len(seeds) :])
            if b["finished"] and c["finished"]
        ]
        summary["paired_clean_laps"] = len(paired)
        summary["challenger_wins"] = sum(c < b for b, c in paired)
        save_json(args.report, summary)
        print(
            "Evaluation:",
            [
                (k, summary[k]["finishes"], summary[k]["median_clean_s"])
                for k in ("baseline", "challenger")
            ],
        )
        return
    race = {
        "track": course(filename).source,
        "joints": list(JOINT_NAMES),
        "seed": args.seed,
        "baseline": simulate(filename, base, args.seed, True),
        "challenger": simulate(filename, best, args.seed, True),
    }
    save_json(args.output, race)
    for key in ["baseline", "challenger"]:
        print(key, race[key]["time"], race[key]["reason"], flush=True)


def train(filename, base, workers, generations, search_seed=20261113):
    if workers < 1 or generations < 1:
        raise ValueError("Workers and generations must be positive")
    rng = np.random.default_rng(search_seed)
    mu = Driver.constant(0.40, 0.55, 1.35).vector()
    sd = np.r_[np.full(SECTORS, 0.10), np.full(SECTORS, 0.075), 0.065, 0.27]
    floor = np.r_[np.full(SECTORS, 0.03), np.full(SECTORS, 0.02), 0.02, 0.07]
    bounds = (
        np.r_[np.full(SECTORS, 0.12), np.full(SECTORS, -0.19), 0.16, 0.3],
        np.r_[np.full(SECTORS, 1.2), np.full(SECTORS, 0.19), 0.85, 3.7],
    )
    best = base.vector()
    history = []
    best_validation = float("-inf")
    validation_seeds = [1, 2, 4, 6, 8, 10]
    with Pool(workers) as pool:
        for g in range(generations):
            pop = np.clip(rng.normal(mu, sd, size=(16, len(mu))), *bounds)
            pop[0], pop[1] = best, mu
            seeds = rng.choice(np.arange(11, 31), 3, replace=False).tolist()
            out = pool.map(
                episode,
                [(filename, v, seed) for v in pop for seed in seeds],
                chunksize=1,
            )
            returns = [
                np.mean([score(o) for o in out[i : i + len(seeds)]])
                for i in range(0, len(out), len(seeds))
            ]
            ranking = np.argsort(returns)
            finalists = ranking[-4:][::-1]
            valid = pool.map(
                episode,
                [
                    (filename, pop[i], seed)
                    for i in finalists
                    for seed in validation_seeds
                ],
                chunksize=1,
            )
            for j, i in enumerate(finalists):
                group = valid[
                    j * len(validation_seeds) : (j + 1) * len(validation_seeds)
                ]
                merit = np.mean([score(o) for o in group])
                if all(o["finished"] for o in group) and merit > best_validation:
                    best_validation = float(merit)
                    best = pop[i].copy()
            ix = ranking[-1]
            elite = pop[ranking[-4:]]
            mu = 0.2 * mu + 0.8 * elite.mean(0)
            sd = np.maximum(floor, 0.7 * sd + 0.3 * elite.std(0))
            sample = out[ix * len(seeds) : (ix + 1) * len(seeds)]
            history.append(
                {
                    "generation": g + 1,
                    "seeds": seeds,
                    "reward": float(returns[ix]),
                    "times": [o["time"] for o in sample],
                    "finished": [o["finished"] for o in sample],
                    "best_validation_reward": round(best_validation, 5)
                    if np.isfinite(best_validation)
                    else None,
                }
            )
            print("generation", g + 1, history[-1], flush=True)
    return Driver.from_vector(best), history


if __name__ == "__main__":
    main()
