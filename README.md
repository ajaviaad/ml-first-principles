# Machine Learning from First Principles

**Understand machine learning by implementing, measuring, and changing small experiments.**

This standalone learning toolkit combines readable Python and NumPy code with
32 guided experiments, worked explanations, exercises, solutions, and interactive
notebooks. Explore how algorithms behave, check the mathematics behind them, and
investigate what changes when an assumption no longer holds.

Use it for independent study, teaching, or refreshing the foundations behind more
complex systems. Readers of *Machine Learning from First Principles* can also use
the experiments while studying the book. The guides explain what you need without
requiring the book.

## Install from this repository

### 1. Clone the repository

Copy this repository's HTTPS or SSH clone URL from its **Code** menu. Replace
`REPOSITORY_URL` below with that URL:

```sh
git clone REPOSITORY_URL ml-first-principles
cd ml-first-principles
```

You need Git and Python. The package declares **Python 3.11 or newer**; use
**Python 3.12** to follow the documented verification environment. All setup
commands below run from the repository root, where `pyproject.toml` is located.

### 2. Create an environment and install

**macOS / Linux**

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

**Windows PowerShell**

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

If PowerShell blocks activation, use
`.\.venv\Scripts\python.exe -m pip install .` directly. Then replace `python`
with `.\.venv\Scripts\python.exe` in the module commands below. No
execution-policy change is needed.

Installation uses the checked-out repository and installs its pinned numerical
dependency, **NumPy 2.3.5**. A PyPI release of this project is not required.
Initial installation may download build dependencies and NumPy. Core experiments
then run locally without a GPU, account, external dataset, downloaded model
weights, or network connection.

### 3. Run your first experiment

With the environment active:

```sh
python -m mlfirst --list
python -m mlfirst --lesson 1
```

The first command lists the available experiments. The second runs a small
learning problem and prints JSON containing the Python and NumPy versions,
requested seed, lesson title, and results.

Installation also provides a shorter terminal command:

```sh
mlfirst --lesson 1
```

Keep the clone for its lesson guides, notebooks, tests, and captured examples.
Package installation makes the Python modules and CLI available in the environment;
it does not place the repository's learning materials in your current directory.

## What you can explore

| Area | Topics and experiments |
| --- | --- |
| Foundations | Learning problems, experimental design, array shapes, probability, optimization, generalization, and metrics |
| Predictive methods | Regression, classification, neighbors, separating surfaces, decision trees, and ensembles |
| Unsupervised and probabilistic methods | Clustering, dimensionality reduction, association rules, and graphical models |
| Neural representations | Backpropagation, convolution, recurrent calculations, attention, and graph operations |
| Generative and multimodal mechanisms | Autoencoders, representation learning, transfer, language modeling, adversarial objectives, diffusion, and modality alignment |
| Decisions and evaluation | Sequential decisions, value learning, policy learning, monitoring, and an end-to-end investigation |

These are small educational experiments. Several demonstrate an individual
calculation or forward pass rather than training a complete architecture. For
example, the language-model experiment uses a smoothed word bigram model; it does
not train a modern LLM. Each guide, and relevant `scope` fields in experiment
output, explains what its implementation covers.

## Choose a learning path

**Starting out:** read the [Python and NumPy primer](docs/GETTING_STARTED.md),
then begin with [Learning problems and evidence](lessons/01_learning_problems.md).
Work through lessons 1–6 to practice data splits, shapes, gradients, uncertainty,
and measurement.

**Building breadth:** use lessons 7–15 to compare predictive and unsupervised
methods. Continue with lessons 16–27 to inspect learned representations and
generative calculations, then lessons 28–32 for decisions and evaluation.

**Already experienced:** choose a mechanism, inspect its implementation, predict
an outcome, and change one assumption. Use the [study projects](docs/PROJECTS.md)
for longer investigations and the tests to check numerical and behavioral
properties. The [glossary](docs/GLOSSARY.md) provides quick definitions.

Each lesson includes a question, essential mathematics, a worked example,
interpretation of the output, limitations, and exercises with hints and solutions.

## Run and compare experiments

```sh
python -m mlfirst --lesson 6
python -m mlfirst --lesson 16 --seed 7
python -m mlfirst --all --seed 42 --output results/all_lessons_seed42.json
```

| Option | Behavior |
| --- | --- |
| `--list` | Print the 32 available experiment names |
| `--lesson N` | Run one experiment, where `N` is 1–32 |
| `--all` | Run every experiment |
| `--seed N` | Request a nonnegative seed; default is 42 |
| `--output PATH` | Save experiment JSON as well as printing it |

Choose one of `--list`, `--lesson`, or `--all` per invocation. Use
`python -m mlfirst --help` for command help.

The output option creates missing parent directories and **replaces an existing
file at the selected path**. Use distinct filenames when preserving comparisons.
Seeded examples support repeatable investigations, but some calculations are
deterministic and ignore the seed. The preserved reference programs retain their
own fixed seeds.

Compare mathematical properties and documented tolerances across environments,
rather than expecting every printed digit to match. Hardware and numerical
libraries can affect low-order results. See the
[captured examples](examples/README.md) and [verification guide](docs/VERIFICATION.md).

## Use the code in Python

```python
import numpy as np

from mlfirst.foundations import Standardizer, binary_metrics
from mlfirst.registry import run_lesson

# Fit preprocessing on training data, then reuse the fitted statistics.
scaler = Standardizer().fit([[1, 7], [3, 7]])
transformed = scaler.transform([[5, 7]])
np.testing.assert_allclose(transformed, [[3.0, 0.0]])
print(transformed)

# Evaluate probabilities against binary labels.
metrics = binary_metrics([0, 1, 1], [0.1, 0.7, 0.8])
print(metrics)

# Run the attention experiment programmatically.
attention = run_lesson(19, seed=42)
print(attention)
```

Implementations are grouped in `mlfirst/foundations.py`, `classical.py`,
`neural.py`, and `decisions.py`. These modules expose the calculations used by the
guides and command-line runner so you can inspect and adapt them.

## Optional notebooks

There are 32 notebooks in `notebooks/`, each presenting a lesson and calling the
same implementation used by the CLI. Notebooks include editable seeds and
code-inspection cells.

From the repository root with the environment active:

```sh
python -m pip install -r requirements-notebooks.txt
python -m notebook
```

Open a notebook and select **Run All** using a kernel with the same installed
dependencies. Notebooks locate the repository when started from either its root
or its `notebooks/` directory. Restart the kernel after changing imported module
code. The notebooks contain no saved execution output; captured reference results
are available in `examples/`.

Jupyter is optional. The command-line experiments and tests do not require it.

## Reference programs

The `book_code/` directory preserves five reference listings with source hashes
and provenance. They provide additional complete examples in classical learning,
neural learning, reinforcement learning, and an end-to-end investigation.

Run them from the repository root:

```sh
python book_code/chapter03_array_example.py
python book_code/companion_classical.py
python book_code/companion_deep.py
python book_code/companion_rl.py
python book_code/companion_investigation.py
```

The array example is intentionally silent when its assertions pass. Assertions
are part of these programs' checks; run with normal Python, without `-O`.
Their original filenames and contents are retained for traceability. See
[provenance and reuse](NOTICE.md) for their origin and the distinction between
preserved listings and the added teaching implementations.

## Develop and verify

Use editable installation when changing the source:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests check numerical calculations, gradients, invariance properties, input
handling, reproducibility, reference-program hashes, notebook code cells, and
CLI behavior. See [verification](docs/VERIFICATION.md) for the recorded environment
and the checks actually performed. Those checks support the small teaching
examples; they do not establish performance or generalization on real datasets.

After editing lesson guides, regenerate their notebooks with:

```sh
python tools/make_notebooks.py
```

Commit or save notebook edits before regeneration because generated notebooks may
be replaced. See [contributing](CONTRIBUTING.md) for the development workflow.

## Repository guide

```text
mlfirst/                  Teaching functions, experiment registry, and CLI
lessons/                  Explanations, worked examples, exercises, and solutions
notebooks/                Interactive versions of the lesson guides
book_code/                Preserved reference programs and source provenance
tests/                    Numerical, behavioral, and integration checks
examples/                 Captured outputs and verification records
docs/                     Primer, glossary, projects, and verification guide
tools/                    Notebook-generation utility
pyproject.toml            Package configuration and CLI entry point
requirements.txt          Pinned core dependency
requirements-notebooks.txt Optional notebook dependencies
```

All generated datasets are synthetic. The repository includes neither private
records nor the book manuscript.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `mlfirst` is not found | Activate the environment used for installation, or use its Python executable with `-m mlfirst`. |
| `No module named mlfirst` | Run `python -m pip install .` from the repository root using the interpreter that will run the experiments. |
| `No module named numpy` | Install the project or run `python -m pip install -r requirements.txt` in the same environment. |
| Installation fails with a different Python version | Use Python 3.12 to match the documented environment; the declared minimum is 3.11. |
| A notebook cannot import the modules | Start it from the clone, check its kernel environment, and restart the kernel after source changes. |
| Results differ slightly | Check versions, seeds, mathematical properties, and tolerances before comparing final digits. |
| An assertion fails | Keep assertions enabled, inspect the relevant calculation, and run its tests before changing expected values. |

## Provenance and reuse

See [NOTICE.md](NOTICE.md) for source provenance and the repository's existing
reuse information. Dependency packages retain their own licenses.

Current package version: **1.0.0**.
