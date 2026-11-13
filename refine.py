"""Warm-started policy search, selected across independent perturbed simulator starts."""

from multiprocessing import Pool

import numpy as np

from race import Driver, episode


def assess(pool, track, policies, seeds):
    jobs = [(track, policy, int(seed)) for policy in policies for seed in seeds]
    rows = pool.map(episode, jobs, chunksize=1)
    return [rows[i : i + len(seeds)] for i in range(0, len(rows), len(seeds))]


def quality(attempts):
    """Finish count first, then elapsed time with some weight on the slow tail."""
    finishes = [r["time"] for r in attempts if r["finished"]]
    seconds = (
        0.7 * np.mean(finishes) + 0.3 * np.quantile(finishes, 0.8) if finishes else 95.0
    )
    return len(finishes), -float(seconds)


def optimize(track, parent, workers, generations, sectors):
    if generations < 1 or workers < 1:
        raise ValueError("Workers and generations must be positive")
    rng = np.random.default_rng(21113)
    mean = parent.resample(sectors).vector()
    spread = np.r_[np.full(sectors, 0.060), np.full(sectors, 0.045), 0.045, 0.17]
    minimum = spread * 0.42
    bounds = (
        np.r_[np.full(sectors, 0.16), np.full(sectors, -0.23), 0.28, 0.4],
        np.r_[np.full(sectors, 1.05), np.full(sectors, 0.23), 0.83, 3.0],
    )
    leader = mean.copy()
    history = []
    # Training, regular selection and final (external) evaluation always use different starts.
    select = np.arange(440, 452)
    with Pool(workers) as pool:
        leader_quality = quality(assess(pool, track, [leader], select)[0])
        for gen in range(generations):
            n = 30
            noise = rng.normal(size=(n, len(mean)))
            # Correlated perturbations cover faster straights as well as individual corners.
            noise[:, :sectors] += rng.normal(0.14, 0.64, size=(n, 1))
            noise[:, sectors : 2 * sectors] += rng.normal(0, 0.45, size=(n, 1))
            pop = np.clip(mean + spread * noise, *bounds)
            pop[0] = leader
            pop[1] = mean
            # Stay close to a validated leader as the wider population explores.
            for i in range(2, 9):
                pop[i] = np.clip(leader + spread * noise[i] * 0.55, *bounds)
            seeds = rng.choice(np.arange(160, 260), 3, replace=False)
            groups = assess(pool, track, pop, seeds)
            ranks = sorted(range(n), key=lambda i: quality(groups[i]), reverse=True)
            elite = pop[ranks[:6]]
            mean = 0.25 * mean + 0.75 * elite.mean(0)
            spread = np.maximum(minimum, 0.83 * spread + 0.17 * elite.std(0))
            finalists = [pop[ranks[0]], pop[ranks[1]], mean]
            # Simple successive halving avoids spending validation episodes on unstable policies.
            screening = assess(pool, track, finalists, select[:4])
            keep = sorted(range(3), key=lambda i: quality(screening[i]), reverse=True)[
                :2
            ]
            confirmation = assess(pool, track, [finalists[i] for i in keep], select[4:])
            for i, extra in zip(keep, confirmation):
                value = quality(screening[i] + extra)
                if value > leader_quality:
                    leader, leader_quality = finalists[i].copy(), value
            item = {
                "generation": gen + 1,
                "training_seeds": seeds.tolist(),
                "selection_finishes": leader_quality[0],
                "selection_count": len(select),
                "selection_weighted_time_s": round(-leader_quality[1], 5),
                "checkpoint": leader.tolist(),
            }
            history.append(item)
            print(
                f"generation {gen + 1}: {leader_quality[0]}/{len(select)} selection starts, "
                f"weighted clean time {-leader_quality[1]:.3f} s",
                flush=True,
            )
        leader, selection = final_selection(pool, track, history)
        history[-1]["final_selection"] = selection
    return Driver.from_vector(leader), history


def final_selection(pool, track, history):
    """Screen local variations, then freeze one driver on independent selection races."""
    rng = np.random.default_rng(818419)
    leaders = np.unique(np.array([row["checkpoint"] for row in history]), axis=0)
    leader = np.asarray(history[-1]["checkpoint"])
    sectors = (len(leader) - 2) // 2
    small = np.r_[np.full(sectors, 0.043), np.full(sectors, 0.03), 0.026, 0.115]
    lower = np.r_[np.full(sectors, 0.16), np.full(sectors, -0.23), 0.28, 0.4]
    upper = np.r_[np.full(sectors, 1.05), np.full(sectors, 0.23), 0.83, 3.0]
    candidates = [leader.copy(), *leaders]
    # Long-run variations explore adjacent sectors together, local variations
    # change individual corners. Neither modifies the physics or contact checks.
    for i in range(96):
        base = leader if i % 3 else leaders[rng.integers(len(leaders))]
        perturbation = rng.normal(size=len(leader))
        perturbation[:sectors] += rng.normal(0.25, 0.7)
        scale = (0.35, 0.8, 1.4)[i % 3]
        candidates.append(np.clip(base + small * scale * perturbation, lower, upper))
    trial_seeds = [601, 602, 603, 604]
    rows = assess(pool, track, candidates, trial_seeds)
    ranking = sorted(
        range(len(candidates)), key=lambda i: quality(rows[i]), reverse=True
    )
    # Carry the incumbent and earlier 12-seed winners to the wider selection.
    shortlist = np.unique(
        np.vstack([leaders, np.asarray(candidates)[ranking[:14]]]), axis=0
    )
    selection_seeds = [1, *range(650, 673)]
    rows = assess(pool, track, shortlist, selection_seeds)
    ranking = sorted(
        range(len(shortlist)), key=lambda i: quality(rows[i]), reverse=True
    )
    finalists = ranking[:5]
    last_seeds = list(range(700, 748))
    extra = assess(pool, track, shortlist[finalists], last_seeds)
    checks = [rows[ix] + more for ix, more in zip(finalists, extra)]
    i = max(
        range(len(finalists)),
        key=lambda j: (
            quality(checks[j]) if rows[finalists[j]][0]["finished"] else (-1, -95)
        ),
    )
    selected = checks[i]
    result = {
        "screened": len(candidates),
        "shortlisted": len(shortlist),
        "finalists": len(finalists),
        "selection_seeds": selection_seeds + last_seeds,
        "selection_finishes": sum(r["finished"] for r in selected),
        "median_clean_s": float(
            np.median([r["time"] for r in selected if r["finished"]])
        ),
    }
    print(
        "Final selection",
        result["selection_finishes"],
        "/",
        len(selected),
        "clean, median",
        round(result["median_clean_s"], 3),
        flush=True,
    )
    return shortlist[finalists[i]].copy(), result
