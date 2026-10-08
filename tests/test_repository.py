"""Integration contracts: unchanged source, portable CLI and lesson execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from mlfirst.registry import TITLES, run_lesson

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_original_listings_match_manifest(self):
        manifest = json.loads((ROOT / "book_code/source_manifest.json").read_text())
        for name, metadata in manifest["files"].items():
            with self.subTest(file=name):
                data = (ROOT / "book_code" / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])

    def test_original_programs_pass_assertions(self):
        for path in sorted((ROOT / "book_code").glob("*.py")):
            if path.name.startswith("__"):
                continue
            with self.subTest(file=path.name):
                run = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stderr)

    def test_all_lessons_are_finite_json_and_repeatable(self):
        for number in TITLES:
            with self.subTest(lesson=number):
                first = json.dumps(run_lesson(number, seed=42), sort_keys=True, allow_nan=False)
                second = json.dumps(run_lesson(number, seed=42), sort_keys=True, allow_nan=False)
                self.assertEqual(first, second)

    def test_cli_valid_json_and_invalid_seed(self):
        run = subprocess.run([sys.executable, "-m", "mlfirst", "--lesson", "3"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["lessons"]["03"]["result"]["prediction"], [12, 6, 21])
        bad = subprocess.run([sys.executable, "-m", "mlfirst", "--lesson", "3", "--seed", "-1"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("nonnegative", bad.stderr)

    def test_each_guide_and_notebook_exists(self):
        for number in TITLES:
            with self.subTest(lesson=number):
                self.assertEqual(len(list((ROOT / "lessons").glob(f"{number:02d}_*.md"))), 1)
                notebooks = list((ROOT / "notebooks").glob(f"{number:02d}_*.ipynb"))
                self.assertEqual(len(notebooks), 1)
                notebook = json.loads(notebooks[0].read_text())
                self.assertEqual(notebook["nbformat"], 4)
                self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
