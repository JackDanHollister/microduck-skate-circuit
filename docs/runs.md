# Fork a driver and keep its run

I keep each driver with the one it came from, the replay you can watch and every start I used to evaluate it. Pick any driver in the [run tree](../lineage/index.md) and you can start your own branch of the experiment. A GitHub fork gives you your own copy of the whole tree.

## Repeat the record lap

My best single clean lap is **23.37083 s**, using start seed **32905**. The [saved driver](../replays/new-time-20261115/driver.json) includes the 24-sector speed and line settings, lookahead and turn gain. Pollen's pinned roller policy handles the skating and downloads automatically when you first run the simulator.

After the [usual installation](../README.md#2-set-up-python), run this from the repository root:

```bash
python race.py replay --policy replays/new-time-20261115/driver.json \
  --seed 32905 --output runs/repeat-23-37.json
```

The challenger should finish in **23.37083 s**. The [verification file](../replays/new-time-20261115/verification.json) has all eight gate times and the driver, replay and track checksums. The original [solo 3D replay](../replays/new-time-20261115/race.json) opens through **Open replays** in the [viewer](https://jackdanhollister.github.io/microduck-skate-circuit/). The [film](https://jackdanhollister.github.io/microduck-skate-circuit/replays/new-time-20261115/lap.mp4) shows that same recorded lap.

To make your own film with just the trained duck, install the [rendering extras](../README.md#make-a-trackside-film), then run:

```bash
MUJOCO_GL=egl python render_race.py runs/repeat-23-37.json runs/repeat-23-37.mp4 --solo
```

## Enter the shared scoreboard

1. **Fork and train.** Pick a parent from the [scoreboard](../lineage/scoreboard.md). In your fork, open **Actions → Train a racer → Run workflow**. Give your runner a unique ID (for example `yourname-late-braker`) and tick **Save each run on a new branch**. The workflow trains, scores and records it.
2. **Submit the saved branch.** From the job summary, open the run branch and create a pull request to `JackDanHollister/microduck-skate-circuit` on `main`. Include the run folder and the generated files in `lineage/`. Say what you trained and which parent you started from.
3. **See where it places.** The pull request check reruns the scored drivers in the simulator. Once the run is merged, it appears on the main [live scoreboard](https://jackdanhollister.github.io/microduck-skate-circuit/scoreboard.html). The board on your fork shows the runs merged into your own Pages branch.

The course-driver board uses the standard `driver.json`, Pollen’s roller, a standing start and seeds 1001–1016. A fall, timeout or departure from the track uses up that start. Most finishes wins, then the lowest median across finished laps. Scores matching to the millisecond share a place. The fastest evaluation lap is shown for interest. **Watch** opens the runner’s separately recorded replay. Rolling starts, custom skating weights and 64-start experiments have their own comparisons in the viewer.

## Run it on GitHub

Open **Actions → Train a racer → Run workflow** on your own fork. If Actions are disabled on a newly forked repository, enable them on its Actions tab first. Enter a parent from the [run tree](../lineage/index.md) and a new ID such as `yourname-wide-exit`. **Independent searches** defaults to **one**. The generations setting applies to each search.

Choose **three** to run three separate searches from the same parent using GitHub’s runners. The jobs use different strategies and fixed search seeds. They train concurrently when three runners are available, and take roughly three times the total runner minutes of one search.

| Child ID suffix | Strategy | Track sectors | Search seed |
| --- | --- | --- | --- |
| `-a` | Pace | 12 | 20261113 |
| `-b` | Corner | 12 | 20261114 |
| `-c` | Stability | 6 | 20261115 |

The single search keeps your ID as entered and uses the balanced strategy with seed 20261113 and the parent’s sector count. After the jobs finish, open the workflow summary. Each successful child has its own `race-record-<child-id>` artifact containing the driver, lap replay and results on the same evaluation starts. Compare finish counts and median clean times before picking one. Download the artifact, unzip it and choose its `race.json` in **Open replays**.

Tick **Save each run on a new branch** to keep the results in your fork. Each child gets its own branch and a link to its tree in the save job summary. You can merge just the one you want, send a pull request upstream, or share all three. If branch saving is disabled by your fork’s Actions permissions, download the artifacts, copy each one’s four files into `lineage/<child-id>/`, run `python run_tree.py index` and `python run_tree.py check`, then commit those run folders and the two indexes and the scoreboard.

## Start from a saved driver

1. Click **Fork** on the [GitHub project](https://github.com/JackDanHollister/microduck-skate-circuit/fork). On **your fork**, click **Code** and clone the URL it shows. If you've already cloned your fork, use that checkout.
2. Follow the [installation instructions](../README.md) in the clone and activate the Python environment.
3. Create a branch for your run. Here’s an example using my first driver. Change the run ID to one starting with your GitHub name, so someone else can contribute a run with a similar name.

```bash
git switch -c my-wide-exit
parent=jackdh-first-lap
run_id=yourname-wide-exit

python race.py train --start-from "lineage/$parent/driver.json" \
  --policy "runs/$run_id/driver.json" --output "runs/$run_id/race.json" --workers 4
python race.py evaluate --policy "runs/$run_id/driver.json" \
  --report "runs/$run_id/evaluation.json" --workers 4
```

The driver, replay and evaluation should all be from the same experiment. If you change the driver, regenerate its replay and evaluation. To watch it before you save it, open the [viewer](https://jackdanhollister.github.io/microduck-skate-circuit/) and choose **Open replays**, then select your `runs/$run_id/race.json`.

## Save all three ghosts from a local league

If you ran three local trainers from the same archived checkpoint, save the three siblings together. Use the actual parent ID you passed to the league and a new prefix:

```bash
python run_tree.py add-league yourname-morning --parent jackdh-round-two --from-dir runs/league
python run_tree.py check
git add lineage/
git commit -m "feat(runs): save my three morning racers"
git push -u origin HEAD
```

Each child keeps its checkpoint, full lap replay and evaluation. Merge your branch into the branch served by your fork’s Pages site to watch them together. The `add-league` command checks the saved parent and records every driver, including any that fell.

## Add the child to your tree

```bash
python run_tree.py add "$run_id" --parent "$parent" \
  --driver "runs/$run_id/driver.json" \
  --replay "runs/$run_id/race.json" \
  --evaluation "runs/$run_id/evaluation.json"
python run_tree.py check

git add -- "lineage/$run_id" lineage/index.md lineage/index.json lineage/scoreboard.md
git commit -m "feat(runs): record my wide exit driver"
git push -u origin HEAD
```

Open the branch on your fork. The new driver sits under its parent in `lineage/index.md`. Share that page directly, merge the branch into your fork’s default branch, or open a pull request to the original project to add it to the shared tree. In the pull request, say what you changed and include the number of clean finishes as well as the lap time. If you changed the simulator or trainer, include the code so the run can be repeated.

## Watch or check a saved run

The [saved-driver view](https://jackdanhollister.github.io/microduck-skate-circuit/?view=all) shows the archived drivers together in 3D. Choose **Map** for their lines from above, or **Film** for the saved trackside recording. Each numbered duck is a separate simulated lap. Compare compatible runs from your own fork by selecting their JSON files together in **Open replays**.

The default viewer opens on [training attempts](https://jackdanhollister.github.io/microduck-skate-circuit/?view=training). Those groups hold the population candidates from the three original searches. A forked driver that you archive joins **Saved drivers**. Its checkpoint, replay and evaluation sit under its parent in your run tree.

The GitHub tree links to the checkpoint, replay and individual results. You can download any replay JSON and open it in the browser viewer. On a Pages site that includes the run, add `?run=yourname-wide-exit` to the viewer URL to go straight to it. A fork can host its own viewer by opening **Settings → Pages → Deploy from a branch** and selecting the default branch and `/ (root)`. Use the URL GitHub gives your fork.

Each run folder is a snapshot and the ID is permanent. Use a new ID for another attempt. Each run card records the simulator and course hashes and, when they match the checkout, its Git commit. If you change either, commit the changes before recording the run to include that revision. The check command reads the local files, checks their hashes and the saved results, and verifies that every parent exists and every driver belongs to the same tree. To repeat a result independently, run `race.py replay` and `race.py evaluate` on the saved driver. Use its recorded replay seed and evaluation seeds for comparisons.

If two branches add runs at the same time, keep both run folders on merge. To resolve a conflict in the generated indexes, run `python run_tree.py index`, then `python run_tree.py check` and stage the two indexes and the scoreboard. An ID collision needs a fresh child ID.
