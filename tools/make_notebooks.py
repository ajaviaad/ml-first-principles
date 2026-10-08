"""Regenerate lightweight lesson notebooks from the canonical Markdown guides.

Run from any working directory: python tools/make_notebooks.py
Only uses the standard library. Outputs have no stored execution results.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, identifier):
    value = {"cell_type": kind, "id": identifier, "metadata": {},
             "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def main():
    output = ROOT / "notebooks"
    output.mkdir(exist_ok=True)
    for guide in sorted((ROOT / "lessons").glob("[0-9][0-9]_*.md")):
        number = int(guide.name[:2])
        module = "foundations" if number <= 6 else "classical" if number <= 15 else "neural" if number <= 27 else "decisions"
        source = guide.read_text(encoding="utf-8")
        # Files such as 02_data_and_experiments.md live under lessons, while
        # links starting ../ remain valid from the sibling notebooks directory.
        source = re.sub(r"\]\((?!https?://|\.\./|#)([^)]+\.md(?:#[^)]*)?)\)", r"](../lessons/\1)", source)
        bootstrap = '''from pathlib import Path
import sys

# Locate this checkout from its root or from the notebooks directory.
root = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "mlfirst" / "registry.py").is_file()), None)
if root is None:
    raise RuntimeError("Open this notebook from inside the extracted repository.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from mlfirst.registry import run_lesson
'''
        experiment = f'''import json
seed = 42  # Change this, rerun, and distinguish sampling effects from fixed arithmetic.
result = run_lesson({number}, seed=seed)
print(json.dumps(result, indent=2, allow_nan=False))
'''
        inspect_code = f'''import inspect
from mlfirst.{module} import lesson_{number:02d}
print(inspect.getsource(lesson_{number:02d}))
'''
        cells = [cell("markdown", source, "lesson-guide"),
                 cell("markdown", "## Run the experiment\n\nRun the next cells in order. Predict the key result first.", "run-instructions"),
                 cell("code", bootstrap, "environment"), cell("code", experiment, "experiment"),
                 cell("markdown", "## Inspect and change\n\nRead the implementation. Follow its calls into the source file when a helper does the interesting work. Change one mechanism at a time in a copied function or in your own cell.", "inspect-instructions"),
                 cell("code", inspect_code, "implementation"),
                 cell("markdown", "## Your experiment record\n\n- My prediction:\n- The one change I made:\n- What happened:\n- Which assumption explains it:\n- What the experiment does not establish:\n\nWork the guide's exercises before expanding its solutions.", "learning-record"),
                 cell("code", "# Write your variation or exercise solution here.\n", "your-work")]
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12.14"}}, "nbformat": 4, "nbformat_minor": 5}
        (output / (guide.stem + ".ipynb")).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {len(list(output.glob('*.ipynb')))} notebooks.")


if __name__ == "__main__":
    main()
