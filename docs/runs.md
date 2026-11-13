# Fork a driver and keep its run

I keep each driver with the one it came from, the replay you can watch and every start I used to evaluate it. Pick any driver in the [run tree](../lineage/index.md) and you can start your own branch of the experiment. A GitHub fork gives you your own copy of the whole tree.

## Run it on GitHub

On your own fork, open **Actions → Train a racer → Run workflow**. GitHub may first show an **Enable workflows** button on a new fork. Enter a parent ID from the tree and a new ID for your duck. One generation is a quick first try. Pick **Save the run on a new branch** if you want to keep it in your fork’s history.

Open the completed workflow run to download the **race-record** artifact. Unzip it and choose its `race.json` in the viewer. If you selected the branch option, the **save-to-fork** job also links to the new branch. Merge that branch into your fork’s `main` to keep it in your tree, or open a pull request to share it upstream. If your fork restricts writing with Actions, the training job still provides the downloadable artifact.

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

The driver, replay and evaluation should all be from the same experiment. If you change the driver, regenerate its replay and evaluation. To watch it before you save it, open the [viewer](https://jackdanhollister.github.io/microduck-skate-circuit/) and choose **Open replay**, then select your `runs/$run_id/race.json`.

## Add the child to your tree

```bash
python run_tree.py add "$run_id" --parent "$parent" \
  --driver "runs/$run_id/driver.json" \
  --replay "runs/$run_id/race.json" \
  --evaluation "runs/$run_id/evaluation.json"
python run_tree.py check

git add -- "lineage/$run_id" lineage/index.md lineage/index.json
git commit -m "feat(runs): record my wide exit driver"
git push -u origin HEAD
```

Open the branch on your fork. The new driver sits under its parent in `lineage/index.md`. Share that page directly, merge the branch into your fork’s default branch, or open a pull request to the original project to add it to the shared tree. In the pull request, say what you changed and include the number of clean finishes as well as the lap time. If you changed the simulator or trainer, include the code so the run can be repeated.

## Watch or check a saved run

The GitHub tree links to the checkpoint, replay and individual results. You can download any replay JSON and open it in the browser viewer. On a Pages site that includes the run, add `?run=yourname-wide-exit` to the viewer URL to go straight to it. A fork can host its own viewer by opening **Settings → Pages → Deploy from a branch** and selecting the default branch and `/ (root)`. Use the URL GitHub gives your fork.

Each run folder is a snapshot and the ID is permanent. Use a new ID for another attempt. The check command reads the local files, checks their hashes and the saved results, and verifies that every parent exists and every driver belongs to the same tree. To repeat a result independently, run `race.py replay` and `race.py evaluate` on the saved driver. Use its recorded replay seed and evaluation seeds for comparisons.

If two branches add runs at the same time, keep both run folders on merge. To resolve a conflict in the generated indexes, run `python run_tree.py index`, then `python run_tree.py check` and stage the two index files. An ID collision needs a fresh child ID.
