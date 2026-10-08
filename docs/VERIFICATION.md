# Verification record

Prepared on 8 October 2026 with **Python 3.12.14** and **NumPy 2.3.5** on macOS
arm64. NumPy is pinned to the version used by the supplied book. Python 3.11+
is the declared minimum; other interpreter and operating-system combinations
were not separately tested in this session.

The repository includes tests against hand calculations, finite-difference
gradients, exhaustive enumeration and invariance properties. Integration tests
run the original programs, compare their source hashes, execute every added
lesson, check strict JSON serialization and repeatability, and exercise the CLI.
The exact final test count and release checks are recorded in
[`verification.json`](../examples/verification.json).

## What the checks establish

- All five original Python listings execute with assertions enabled. Their
  extracted contents match `book_code/source_manifest.json` byte for byte.
- All 32 added experiments execute with default seed 42 and serialize as strict
  JSON without NaN or Infinity. Repeated runs in this environment match.
- Independent mathematical checks cover gradients, loss normalization, stable
  probability calculations, neighborhood behavior, tree split boundaries,
  ensemble residual updates, PCA reconstruction, itemset support, HMM inference,
  attention masks, graph permutations, latent-variable gradients, diffusion
  moments, Bellman backups and conformal quantiles.
- Tests verify that calibration and test labels do not refit selected model
  coefficients, and that preprocessing and explanation functions do not silently
  mutate fitted statistics or input data.
- Notebook JSON and the ordered code-cell sequences are checked from the
  notebooks directory. These checks exercise notebook content without requiring
  the optional browser interface.
- Local Markdown links, archive integrity and extraction into a separate folder
  are checked. The package build and an installed-package import/CLI smoke check
  are also verified.

These are numerical and behavior checks for small teaching examples, not evidence
of production robustness or generalization on real data. Several lessons compute
a mechanism or forward pass rather than train a complete model. Their guides
and `scope` output fields state those limits.

## Reproduce the verification

From the extracted repository root after installing requirements:

```bash
python -m unittest discover -s tests -v
python -m mlfirst --all --seed 42 --output results/all_lessons.json
```

Compare results with [`examples/all_lessons_seed42.json`](../examples/all_lessons_seed42.json).
Do not require every printed low-order digit to match across numerical backends.
The original listings have their own fixed seeds and can be run individually
using the README commands. `examples/book_programs/` contains their captured
standard output, including the final investigation's JSON.

When changing implementations, rerun relevant mathematical checks. Regenerate
notebooks after guide edits with `python tools/make_notebooks.py`. Keep the book
listings unchanged; documented numerical refinements belong in the teaching
modules.
