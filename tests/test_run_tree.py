import json
import shutil
import tempfile
import unittest
from pathlib import Path

import run_tree as tree


class RunTreeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.archive = Path(temp.name) / "lineage"
        self.root_folder = self.archive / tree.FIRST
        shutil.copytree(tree.ARCHIVE / tree.FIRST, self.root_folder)
        self.artifacts = [self.root_folder / name for name in tree.ARTIFACTS]

    def test_branch_order_and_reused_id(self):
        tree.add(self.archive, "skater-wide", tree.FIRST, self.artifacts)
        tree.add(self.archive, "skater-fast", tree.FIRST, self.artifacts)
        tree.add(self.archive, "skater-faster", "skater-fast", self.artifacts)
        catalog = json.loads((self.archive / "index.json").read_text())
        self.assertEqual(
            {r["id"]: r["parent"] for r in catalog},
            {
                tree.FIRST: None,
                "skater-wide": tree.FIRST,
                "skater-fast": tree.FIRST,
                "skater-faster": "skater-fast",
            },
        )
        page = (self.archive / "index.md").read_text()
        self.assertIn("\n    - **[skater-faster]", page)
        with self.assertRaisesRegex(ValueError, "already exists"):
            tree.add(self.archive, "skater-wide", "skater-fast", self.artifacts)
        with self.assertRaisesRegex(ValueError, "Unknown parent"):
            tree.add(self.archive, "missing-parent", "unknown", self.artifacts)
        with self.assertRaisesRegex(ValueError, "run ID"):
            tree.add(self.archive, "../escape", tree.FIRST, self.artifacts)

    def test_hash_and_detached_cycle(self):
        tree.add(self.archive, "skater-wide", tree.FIRST, self.artifacts)
        child = self.archive / "skater-wide"
        with (child / "driver.json").open("a") as target:
            target.write(" ")
        with self.assertRaisesRegex(ValueError, "Recorded files were changed"):
            tree.records(self.archive)
        shutil.copyfile(self.root_folder / "driver.json", child / "driver.json")
        card = json.loads((child / "run.json").read_text())
        card["parent"] = "skater-wide"
        (child / "run.json").write_text(json.dumps(card))
        with self.assertRaisesRegex(ValueError, "cycle"):
            tree.write_index(self.archive)

    def test_corrupt_median_does_not_create_partial_entry(self):
        old = json.loads(self.artifacts[-1].read_text())
        old["challenger"]["median_clean_s"] = 0.1
        bad = self.archive.parent / "bad-eval.json"
        bad.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "median"):
            tree.add(
                self.archive, "skater-invalid", tree.FIRST, [*self.artifacts[:2], bad]
            )
        self.assertFalse((self.archive / "skater-invalid").exists())
        self.assertEqual(list(self.archive.iterdir()), [self.root_folder])


if __name__ == "__main__":
    unittest.main()
