# Machine Learning from First Principles

A self-contained course companion with readable Python code, small experiments,
worked explanations, exercises and solutions. Use it alongside the supplied
*Machine Learning from First Principles* (first edition, 2026), or learn directly
from the lesson guides. You do not need the book to run or understand the lessons.

The repository includes **every Python listing from the book**: four complete
appendix programs and the Chapter 3 array example. It also adds **32 guided
experiments**, one for each chapter, including toy convolution, recurrence,
generative modeling and multimodal examples beyond the original appendices.
These are educational mechanisms, not full implementations of every named
architecture or production system discussed in the book.

## Start here

Use Python **3.11–3.13**; Python **3.12** is the verified version. All core
experiments use only NumPy and the standard library. No GPU, accounts, datasets,
model weights or network connection are needed after installation.

Extract the ZIP, open a terminal inside `ml-first-principles`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mlfirst --lesson 1
```

On Windows PowerShell, use `py -3.12 -m venv .venv`, then
`.venv\Scripts\Activate.ps1`. If activation is unavailable, invoke
`.venv\Scripts\python.exe` directly in place of `python`; on macOS/Linux the
equivalent is `.venv/bin/python`. These commands run from the repository root;
installing this repository itself is optional.

New to Python? Begin with [the short Python and NumPy primer](docs/GETTING_STARTED.md),
then open [the first lesson](lessons/01_learning_problems.md).

## Learn by predicting, running and changing

Each file in `lessons/` explains the question, essential mathematics, a tiny
worked example, the experiment's output, and its limits. It ends with exercises,
hints and expandable solutions. First predict the result, then run the code,
then change one assumption and explain the difference.

- **Build a foundation:** work through lessons 1–6 in order. Practice shapes,
  gradients, splits, uncertainty and measurement before choosing complex models.
- **Learn predictive methods:** continue with lessons 7–15 and compare simple
  rules, ensembles and unsupervised structure on constructed data.
- **Explore learned representations:** lessons 16–27 expose neural, sequence,
  graph, generative and multimodal calculations using small arrays.
- **Study decisions and complete investigations:** lessons 28–32 connect rewards,
  learned behavior, monitoring and evaluation under distribution shift.

The [study projects](docs/PROJECTS.md) turn these lessons into longer investigations.
Use the [glossary](docs/GLOSSARY.md) whenever a term is unfamiliar.

## Run experiments

```bash
python -m mlfirst --list
python -m mlfirst --lesson 6
python -m mlfirst --lesson 16 --seed 7
python -m mlfirst --all --output results/all_lessons.json
python -m unittest discover -s tests -v
```

The CLI prints JSON with environment versions, the requested seed and results.
Some examples are deterministic calculations, so changing the seed will not
change them. The original book programs retain their original fixed seeds.
Assertions in those programs are part of their checks: do not run with `-O`.

To reproduce the book programs exactly:

```bash
python book_code/chapter03_array_example.py
python book_code/companion_classical.py
python book_code/companion_deep.py
python book_code/companion_rl.py
python book_code/companion_investigation.py
```

The array example is intentionally silent when its assertions pass. The RL
appendix can run without NumPy. The other listings require NumPy.

Use the code interactively as well:

```python
from mlfirst.foundations import Standardizer, binary_metrics
from mlfirst.registry import run_lesson

scaler = Standardizer().fit([[1, 7], [3, 7]])
print(scaler.transform([[5, 7]]))  # [[3.0, 0.0]]
print(binary_metrics([0, 1, 1], [0.1, 0.7, 0.8]))
print(run_lesson(19, seed=42))
```

Optional package installation: `python -m pip install -e .` makes `mlfirst`
available as a terminal command. The source checkout remains the place to read
guides and notebooks; the wheel contains the importable code.

## Optional notebooks

There is a notebook for each lesson in `notebooks/`. Each contains the full
lesson, a runnable experiment, editable seed and code-inspection cells. These
are an alternative interface to the same tested implementations.

```bash
python -m pip install -r requirements-notebooks.txt
python -m notebook
```

Open a notebook and use **Run All**. The notebook locates this repository from
either its root or the `notebooks/` folder. Restart the kernel after editing
module source so imported code is refreshed. Notebooks are distributed without
stored execution output; the checked example outputs live in `examples/`.

## What's in the repository

```text
book_code/       Original Python listings, source hashes and provenance
mlfirst/         Reusable teaching functions and command-line runner
lessons/         Standalone explanations, experiments and worked exercises
notebooks/       Interactive versions of the lesson guides
tests/           Numerical, behavioral, reproducibility and integration checks
examples/        Captured results from the verified environment
docs/            Getting started, glossary, projects and verification notes
requirements.txt Pinned core numerical dependency
```

All generated datasets are synthetic and created in memory. No private records
or original book document are included. See [provenance and reuse](NOTICE.md)
for the distinction between unchanged book code and the added learning material.
See [verification](docs/VERIFICATION.md) for what was actually run.

## Troubleshooting

- **`No module named mlfirst`:** open your terminal in the extracted repository
  directory or install the package with `python -m pip install -e .`.
- **`No module named numpy`:** use `python -m pip install -r requirements.txt`
  in the same environment that runs the scripts or notebook kernel.
- **Installation fails on an older Python:** use Python 3.12. The minimum is
  3.11; newer untested interpreter versions may need different NumPy wheels.
- **Numbers differ slightly:** compare mathematical properties and tolerances,
  not the last printed digit. Hardware and numerical libraries can change them.
- **An assertion fails:** check versions, keep assertions enabled, and run the
  relevant test file before changing the expected result.

For more language and array practice, use the official
[Python tutorial](https://docs.python.org/3/tutorial/),
[virtual environment guide](https://docs.python.org/3/library/venv.html),
[NumPy quickstart](https://numpy.org/doc/2.3/user/quickstart.html) and
[Jupyter installation guide](https://docs.jupyter.org/en/latest/install/notebook-classic.html).
