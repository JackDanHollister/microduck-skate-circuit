"""Reproduce the original population search and archive its actual training laps for viewing.

Run from the repository with the simulation dependencies installed. Each group stores the
30 candidate attempts from one training generation and one starting seed. Selection,
qualification and benchmark rollouts are not training candidates and are excluded.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from arena.constants import JOINT_NAMES

import race
import refine
from league import STRATEGIES

SCALE = 4096
HZ = 12.5
ROOT = Path(__file__).resolve().parent
TRACK = ROOT / 'data' / 'reinvent2018.json'
PARENT = ROOT / 'lineage' / 'jackdh-round-two' / 'driver.json'
SELECTED = ROOT / 'lineage'
VERSION = 1


def packed_episode(job):
    """Sample timestamps 0, 0.08, ... and include the exact final frame exactly once."""
    track, settings, seed = job
    lap = race.simulate(track, race.Driver.from_vector(settings), seed, record=True)
    rows = np.asarray(lap.pop('frames'), dtype=np.float64)
    lap.pop('commands', None)
    assert rows.shape[1] == 22 and rows[0, 0] == 0 and rows[-1, 0] == lap['time']
    # The instrumented simulator records at 25 Hz and interpolates the finish.
    regular = rows[:-1:2]
    # Some DNF records have the same rounded final timestamp twice. Keep only
    # the final measured pose at that instant instead of duplicating a video frame.
    regular = regular[regular[:, 0] < lap['time']]
    expected = np.arange(len(regular), dtype=float) / HZ
    np.testing.assert_allclose(regular[:, 0], expected, rtol=0, atol=.000006)
    assert len(regular) == 0 or regular[-1, 0] < lap['time']
    picked = np.concatenate((regular[:, 1:], rows[-1:, 1:]))
    integers = np.rint(picked * SCALE)
    if not np.isfinite(integers).all() or np.any((integers < -32768) | (integers > 32767)):
        raise ValueError('Pose outside int16 precision/range; cannot pack this episode')
    packed = integers.astype('<i2')
    if np.max(np.abs(packed.astype(np.float64) / SCALE - picked)) > .5 / SCALE + 1e-12:
        raise ValueError('Pose failed quantisation round trip')
    lap['frames'] = len(packed)
    return lap, packed.tobytes()


def write_binary(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(data)
    os.replace(temp.name, path)
    path.chmod(0o644)
    return hashlib.sha256(data).hexdigest()


def capture_strategy(strategy, destination, workers, limit):
    checkpoint = json.loads((SELECTED / f'jackdh-{strategy}' / 'driver.json').read_text())
    parent = race.Driver(**json.loads(PARENT.read_text())['driver'])
    generations = checkpoint['history'][:limit]
    records = []
    original_assess = refine.assess
    previous_group = 0

    def record(pool, track, policies, seeds):
        nonlocal previous_group
        # The production trainer makes exactly one population evaluation with 30
        # policies in each generation. Verify the expected seeds before recording.
        if len(policies) != 30:
            return original_assess(pool, track, policies, seeds)
        item = generations[previous_group]
        seed_list = [int(s) for s in seeds]
        assert seed_list == item['training_seeds']
        previous_group += 1
        generation = previous_group
        results = list(pool.imap(packed_episode, (
            (track, v, int(seed)) for v in policies for seed in seeds), chunksize=1))
        n = len(seed_list)
        for j, seed in enumerate(seed_list):
            rows = results[j::n]
            group = f'{strategy}-g{generation:02}-s{seed}'
            name = f'{group}.bin'
            blob = b''.join(data for _, data in rows)
            digest = write_binary(destination / name, blob)
            attempts = [dict(candidate=i, **lap) for i,(lap,_) in enumerate(rows, 1)]
            assert len(attempts) == 30 and len(blob) == 42 * sum(a['frames'] for a in attempts)
            records.append(dict(id=group, strategy=strategy, generation=generation, seed=seed,
                                file=name, attempts=attempts, sha256=digest))
        # Compare saved search scores and dynamics against the original training
        # via the optimizer's own elite selection and validation, not new tuning.
        return [[lap for lap,_ in results[i:i+n]] for i in range(0, len(results), n)]

    def verify_history(pool, track, history, *args, **kwargs):
        assert len(history) == len(generations)
        for old, replayed in zip(generations, history):
            for key in ('checkpoint', 'generation', 'training_seeds', 'selection_finishes',
                        'selection_count', 'selection_weighted_time_s'):
                if replayed[key] != old[key]:
                    raise ValueError(f'{strategy} generation {old["generation"]} does not reproduce {key}')
        return np.asarray(history[-1]['checkpoint']), {'archived_candidate_laps': True}

    with patch.object(refine, 'assess', record), patch.object(refine, 'final_selection', verify_history):
        refine.optimize(str(TRACK), parent, workers, len(generations), STRATEGIES[strategy],
                        checkpoint['search_seed'], strategy)
    assert len(records) == sum(len(g['training_seeds']) for g in generations)
    (destination / f'.{strategy}.json').write_text(json.dumps(records))


def build(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f'Choose a new output directory; {output} exists.')
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.training-replays-') as temporary:
        stage = Path(temporary)
        strategies = list(STRATEGIES)
        concurrent = min(args.workers, len(strategies))
        n = args.workers // concurrent
        env = os.environ | {'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        def start(strategy):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--capture', strategy,
                   '--output', str(stage), '--workers', str(n), '--generations', str(args.generations)]
            subprocess.run(cmd, check=True, env=env)
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            list(executor.map(start, strategies))
        groups = []
        for strategy in strategies:
            path = stage / f'.{strategy}.json'
            groups.extend(json.loads(path.read_text()))
            path.unlink()
        seeds = sorted({g['seed'] for g in groups})
        base = race.Driver.constant().vector()
        with Pool(args.workers) as pool:
            baselines = []
            for seed, (meta, data) in zip(seeds, pool.imap(packed_episode,
                    [(str(TRACK), base, seed) for seed in seeds],chunksize=1)):
                name = f'baseline-s{seed}.bin'
                digest = write_binary(stage / name, data)
                baselines.append(dict(seed=seed,file=name,sha256=digest,**meta))
        total = sum(len(g['attempts']) for g in groups)
        assert total == 300 * args.generations
        report = dict(version=VERSION, source='reconstructed', generations=args.generations,
                      totalAttempts=total, frameHz=HZ, poseScale=SCALE,
                      track=json.loads(TRACK.read_text()), joints=list(JOINT_NAMES),
                      groups=groups, baselines=baselines)
        (stage / 'index.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')
        print(f'{len(groups)} groups, {total} attempts, {len(baselines)} reference laps, '
              f'{sum(f.stat().st_size for f in stage.iterdir())/1e6:.1f} MB',flush=True)
        # Rename the complete, independently verified directory as a single operation.
        stage.chmod(0o755)
        stage.rename(output)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=min(os.cpu_count() or 1, 12))
    p.add_argument('--generations',type=int,default=10)
    p.add_argument('--output',type=Path,default=ROOT/'training-replays')
    p.add_argument('--capture',choices=list(STRATEGIES),help='Internal per-strategy worker')
    args = p.parse_args()
    if not 1 <= args.generations <= 10 or not 1 <= args.workers <= 18:
        p.error('Use 1–10 archived generations and 1–18 total workers.')
    if args.capture:
        capture_strategy(args.capture,args.output,args.workers,args.generations)
    else:
        build(args)
