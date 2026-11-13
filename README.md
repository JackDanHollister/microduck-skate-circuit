# Microduck Skate Circuit

I wanted to see if my Microduck could find a faster way around a racing circuit on skates. I put it on the original AWS DeepRacer re:Invent 2018 course and gave it a centreline ghost to chase. The ghost and the challenger both use Pollen’s roller policy to move their legs. The challenger learns where to skate and how fast to take each part of the lap.

**[Watch the race](https://jackdanhollister.github.io/microduck-skate-circuit/)**. Press **Play** on the overhead map to see the speed trail, or switch to **Trackside** to watch the white duck race the blue ghost. The viewer runs in a browser on a phone or laptop.

[![A white Microduck skating into a bend with the blue centreline ghost on its inside](replays/trackside.jpg)](https://jackdanhollister.github.io/microduck-skate-circuit/)

## The first race

On the featured start, the duck skated a clean **27.54 s** lap. The ghost took **29.83 s**. That’s 2.29 seconds off a 17.71 m lap.

I tried 16 more starts that the trainer hadn’t used. The challenger beat the ghost on all 15 laps where both finished. On the other start the challenger fell. The ghost finished all 16. Median times for completed laps were 27.95 s for the challenger and 29.55 s for the ghost. These are simulated results on this circuit. [All 16 results](replays/evaluation.json)

## Race from a fork

Choose **Fork** on GitHub, then **Actions → Train a racer** in your fork. Pick a parent driver and a new run ID. The job gives you a replay to watch and can save the child on a branch in your fork. [Browse the run tree](lineage/index.md) · [How to record and share a run](docs/runs.md)

## Try your own duck

The [race viewer](https://jackdanhollister.github.io/microduck-skate-circuit/) opens straight away in a browser. To run the simulation or train a driver, I use a Linux x86-64 machine with Git, Python 3.12 and a CPU. You can run it on a laptop. The first run needs internet access to download Pollen's roller policy; the download is public and is cached for later runs.

### 1. Get the project

If you plan to share your own runs, [fork this repo](https://github.com/JackDanHollister/microduck-skate-circuit/fork) first. On your fork, use **Code → HTTPS** and use that URL in the clone command below. To try my version as it is:

```bash
git clone https://github.com/JackDanHollister/microduck-skate-circuit.git
cd microduck-skate-circuit
```

### 2. Set up Python

From that folder, create and activate a virtual environment, then install the simulator.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3.12` is missing or your system Python cannot create a virtual environment, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv venv --python 3.12 --seed .venv` in place of the first line. It can download Python 3.12 and includes pip. When you return in a new terminal, change into the repo and run `source .venv/bin/activate` again.

### 3. Run a lap and watch it

```bash
python race.py replay --seed 1 --output runs/my-lap.json
```

The terminal prints a time and finish status for both the **baseline** (the ghost) and the **challenger** (the saved driver). Choose **Open replay** in [the viewer](https://jackdanhollister.github.io/microduck-skate-circuit/) and select `runs/my-lap.json` to watch on the overhead map. The file opens locally in your browser. If a driver falls, the replay shows where the attempt ended.

You can also serve the viewer from your clone. Run `python -m http.server 8000 --bind 127.0.0.1` in the repo root, open [localhost:8000](http://localhost:8000), then use **Open replay**. Press Ctrl+C in the terminal to stop the server.

### 4. Train a driver

Pick a parent in the [run tree](lineage/index.md). Here's how to start from my first published driver, write your results under `runs/` and compare the new driver with the centreline ghost on fresh starts:

```bash
python race.py train --start-from lineage/jackdh-first-lap/driver.json \
  --workers 4 --generations 6 --policy runs/my-driver.json --output runs/my-race.json
python race.py evaluate --policy runs/my-driver.json --workers 4 --report runs/my-results.json
```

`my-driver.json` contains the settings you trained, `my-race.json` opens in the viewer, and `my-results.json` records the evaluation. Use `--workers 1` on a small machine or if you want to leave cores free. To keep the parent and child together on GitHub, follow [Share a run from your fork](docs/runs.md).

## How the lap works

I split the track into six sectors. A driver chooses a forward speed and an offset from the centreline at each one. It blends those values between sectors, looks ahead to a point on the chosen line, and asks the roller policy for a forward speed and turn rate. The driver reads the duck's position and heading from the simulator.

I used cross-entropy search over complete episodes to train those 12 sector settings, the lookahead distance and the turn response. Six generations tried 16 drivers each. A clean finish earns a time-based reward. A failed lap earns partial credit for progress. Eight gates must be crossed in order, and the simulation checks wheel-centre boundaries and falls. The white duck and blue ghost are separate runs from the same start, played against the same clock. [Simulation and training code](race.py)

## Make a trackside film

The overhead view works with every replay. To make a film from a new replay on a machine with an EGL graphics driver:

```bash
python -m pip install -r requirements-render.txt
MUJOCO_GL=egl python render_race.py runs/my-race.json runs/my-race.mp4
```

## Credits

This is a community Microduck experiment. The physics come from [Pollen’s Microduck Arena](https://github.com/pollen-robotics/microduck-arena/releases/tag/v0.7.0). The leg movements come from [Pollen’s roller policy](https://huggingface.co/pollen-robotics/microduck-policies/tree/1b56c396825c052a4e26e95cf2b8d8298af9e9b4). I converted the AWS route coordinates to JSON and used those points for the timing gates, track edges and replay. [Track source and notices](data/README.md)
