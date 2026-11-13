"""Race three pinned skating controllers with Duck 28's steering held fixed."""
import argparse
from functools import lru_cache
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
from statistics import median

from arena.constants import JOINT_NAMES
from arena.evaluate import evaluate_at, seed_row
from arena.manifest import load_event
from arena.policy import load_policy
from huggingface_hub import hf_hub_download

from race import ROOT, Driver, course, roller, save_json, simulate

TRACK = str(ROOT / 'data' / 'reinvent2018.json')
SETTINGS = ROOT / 'replays' / 'sprint-transfer' / 'duck-28.json'
VIEW_SEED = 182
CONTROLLERS = {
    'pollen': 'Pollen roller',
    'checkpoint-500': 'JackDH · checkpoint 500',
    'quicksilver-v2': 'HlibPlakhtii · Quicksilver v2',
}
# Pinned to the downloadable policies scored in these Arena entries on 13 Nov 2026.
ENTRIES = {
    'checkpoint-500': {
        'repo': 'JackDH/microduck-roller-sprint-20261110-best',
        'revision': '43e9b57825fbfe08cf4b510368b643ff9ae47d93',
        'sha256': '0b724f5cebe7f50815f6dd9631009e47c4b0a607f81b22d0f4048ffd6d5a77d6',
        'run': '01M72WA65TCS7STJFS7QAX', 'command_vx': 0.8,
        'official_times_s': [2.995, 2.980, 3.010, 3.010, 2.970],
    },
    'quicksilver-v2': {
        'repo': 'HlibPlakhtii/microduck-roller-quicksilver-v2',
        'revision': '8dc1408c8e856ed4d5b6f773e305fe1302e01899',
        'sha256': '6d8aacdac8094c094dae34607815d461e9b4bad3584f395f99fde294c5682108',
        'run': '01M6487YAJEV5NF6J5ZJ5W', 'command_vx': 0.7,
        'official_times_s': [0.890, 0.880, 0.885, 0.895, 0.880],
    },
}


@lru_cache(maxsize=3)
def controller(name):
    if name == 'pollen':
        return roller()
    entry = ENTRIES[name]
    path = Path(hf_hub_download(entry['repo'], 'policy.onnx', revision=entry['revision']))
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'{name} differs from its published weights')
    # Both competition exports take raw Arena observations and return motor targets.
    return load_policy(path)


def check_sprints():
    event = load_event('roller-sprint-2m')
    checks = {}
    for name, entry in ENTRIES.items():
        result = evaluate_at(event, controller(name), entry['command_vx'])
        per_seed = [seed_row(row) for row in result.per_seed]
        observed = [(row['seed'], row['status'], round(row['time_s'], 6) if row['time_s'] is not None else None)
                    for row in per_seed]
        expected = [(seed, 'finish', t) for seed, t in enumerate(entry['official_times_s'], 1)]
        if observed != expected:
            raise ValueError(f'{name} sprint reproduction differs: {observed}')
        checks[name] = dict(**entry, mean_s=result.score, local_per_seed=per_seed)
        print(f'{name}: reproduced five official starts, {result.score:.3f} s', flush=True)
    return checks


@lru_cache(maxsize=1)
def driver():
    return Driver(**json.loads(SETTINGS.read_text())['driver'])


def episode(job):
    name, seed = job
    result = simulate(TRACK, driver(), seed, record=(seed == VIEW_SEED), controller=controller(name))
    return name, seed, result


def compare(output, workers):
    official = check_sprints()
    seeds = range(13100, 13164)
    with Pool(workers) as pool:
        results = pool.map(episode, [(name, seed) for name in CONTROLLERS for seed in (VIEW_SEED, *seeds)], chunksize=1)
    summary = {}
    field = []
    for name, label in CONTROLLERS.items():
        rows = {seed: lap for key, seed, lap in results if key == name}
        example = rows.pop(VIEW_SEED)
        if name == 'pollen' and example['time'] != json.loads(SETTINGS.read_text())['pollen_time_seed_182']:
            raise ValueError('Pollen control did not reproduce the archived Duck 28 lap')
        field.append(dict(id=name, label=label, run=example))
        attempts = [dict(seed=seed, finished=lap['finished'], time=lap['time'], reason=lap['reason'],
                         progress=lap['progress'], gates=lap['gates']) for seed, lap in rows.items()]
        clean = [a['time'] for a in attempts if a['finished']]
        summary[name] = dict(finishes=len(clean), starts=len(attempts),
                             median_clean_s=median(clean) if clean else None, attempts=attempts)
        print(f'{label}: seed {VIEW_SEED}: {example["reason"]} at {example["time"]:.3f} s; '
              f'{len(clean)}/{len(seeds)} other starts, median clean {summary[name]["median_clean_s"]}', flush=True)
    save_json(output / 'race.json', dict(track=course(TRACK).source, joints=list(JOINT_NAMES), seed=VIEW_SEED, field=field))
    save_json(output / 'results.json', dict(example_seed=VIEW_SEED, checked_sprints=official, results=summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'sprint-transfer')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    compare(args.output, args.workers)
