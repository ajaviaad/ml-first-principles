# Changing the experiments

Start with a small deterministic example and write down its expected answer.
Prefer a hand calculation, a finite-difference derivative, exhaustive enumeration
or an invariance over comparing a function to a restatement of its own code.

The `book_code/` listings are preserved reference artifacts. Put corrections or
extensions in `mlfirst/`, describe the difference, and keep the original hashes.
When changing a lesson, update its explanation and exercises as well as the
function. Regenerate notebooks with `python tools/make_notebooks.py`.

Core code must run with Python 3.11+ and NumPy 2.3.5, with no network activity,
GPU, dataset download or trained weights. Keep data synthetic and avoid global
random-state mutation. Public functions should document shapes, normalization,
seed behavior and limits. Lesson entry points take `seed=42` and return strict
JSON-compatible dictionaries; use no NaN or Infinity in their output.

Before sharing a change:

```bash
python -m unittest discover -s tests -v
python -m mlfirst --all --output results/all_lessons.json
```

Rerun notebook cells after changing imports or APIs. Examples are deliberately
small; explain when a readable implementation has poor scaling or omits parts
of a larger algorithm. Do not convert a mechanistic demonstration into a claim
of production performance.
