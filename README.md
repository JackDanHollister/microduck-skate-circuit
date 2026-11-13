# Microduck Skate Circuit

I wanted to see if my Microduck could find a faster way around a racing circuit on skates. I put it on the original AWS DeepRacer re:Invent 2018 course and gave it a centreline ghost to chase. The ghost and the challenger both use Pollen’s roller policy to move their legs. The challenger learns where to skate and how fast to take each part of the lap.

**[Watch the race](https://jackdanhollister.github.io/microduck-skate-circuit/)**. Press **Play** on the overhead map to see the speed trail, or switch to **Trackside** to watch the white duck race the blue ghost. The viewer runs in a browser on a phone or laptop.

[![A white Microduck skating into a bend with the blue centreline ghost on its inside](replays/trackside.jpg)](https://jackdanhollister.github.io/microduck-skate-circuit/)

## The first race

On the featured start, the duck skated a clean **27.54 s** lap. The ghost took **29.83 s**. That’s 2.29 seconds off a 17.71 m lap.

I tried 16 more starts that the trainer hadn’t used. The challenger beat the ghost on all 15 laps where both finished. On the other start the challenger fell. The ghost finished all 16. Median times for completed laps were 27.95 s for the challenger and 29.55 s for the ghost. These are simulated results on this circuit. [All 16 results](replays/evaluation.json)

## Try your own duck

I tested the simulator on Linux with Python 3.12. The physics and training run on the CPU. The first run fetches the public roller model from Hugging Face and checks its checksum.

```bash
git clone https://github.com/JackDanHollister/microduck-skate-circuit.git
cd microduck-skate-circuit
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python race.py replay --seed 1 --output runs/my-lap.json
```

Open **[the viewer](https://jackdanhollister.github.io/microduck-skate-circuit/)** and choose **Open replay** to watch your JSON. Your file opens locally in the browser. To run the viewer from the clone, use `python -m http.server 8000` and open `http://localhost:8000`.

To train another driver and compare it with the same ghost:

```bash
python race.py train --workers 4 --generations 6 --policy runs/my-driver.json --output runs/my-race.json
python race.py evaluate --policy runs/my-driver.json --workers 4 --report runs/my-results.json
```

You can also copy [my driver settings](driver.json), change the six speed or line-offset values, and pass the copy to `python race.py replay --policy runs/my-driver.json --output runs/my-race.json`. The offsets are in metres, positive to the left of the centreline. Try changing one sector and watching where the ducks pull apart.

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
