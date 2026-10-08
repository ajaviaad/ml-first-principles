# Python and NumPy in one sitting

This primer prepares you to read the experiments. You need arithmetic, a terminal
and Python; you do not need prior experience with machine learning frameworks.
Follow the environment setup in the [README](../README.md) first. Run all commands
from the repository root.

## Run your first calculation

Start an interactive Python prompt by typing `python`. Enter these lines one at
a time; `#` introduces a comment. Use `exit()` to return to the terminal.

```python
loaves = [70, 90, 110, 130]
mean = sum(loaves) / len(loaves)
print(mean)  # 100.0

def absolute_error(prediction, target):
    return abs(prediction - target)

print(absolute_error(100, 82))  # 18
```

Indentation defines the function body. A function receives arguments, performs a
calculation and returns a result. A list stores several values. `len` counts
them. A `for` loop repeats work:

```python
for target in loaves:
    print(absolute_error(mean, target))
```

A dictionary gives values names: `{"mae": 6.5, "samples": 20}`. The runner prints
dictionaries as JSON. Braces group named fields; square brackets hold lists.
Numbers such as `1e-6` mean one millionth. JSON `null` means no defined value,
not zero; for example, precision is undefined when there are no positive predictions.

## Arrays have shapes

```python
import numpy as np

X = np.array([[2., 3.], [0., 1.], [4., 2.]])
w = np.array([4., -1.])
print(X.shape)       # (3, 2): three examples, two features
print(X[0])          # first row: [2., 3.]
print(X[:, 1])       # second column: [3., 1., 2.]
print(X @ w + 7)     # [12., 6., 21.]
print(X.mean(axis=0))  # one mean for each feature
```

Python indexing starts at zero. `:` selects all entries along an axis. `axis=0`
reduces down the rows, leaving one value per column; `axis=1` reduces across
columns. `@` performs matrix multiplication and `*` performs elementwise
multiplication. Always write down the expected shape before an operation.

```python
y = np.array([12., 6., 21.])
prediction = X @ w + 7
assert prediction.shape == y.shape
print(np.mean((prediction - y) ** 2))  # 0.0
```

An assertion stops the program when an expected property fails. A passing check
supports that particular property, not every possible use of a function.

## Use an experiment as a laboratory

```python
from mlfirst.registry import run_lesson
result = run_lesson(1, seed=42)
print(result["weekday_model"])
```

The seed controls pseudo-random generation. Reusing it helps compare changes on
the same constructed inputs. Also repeat across seeds: conclusions tied to one
lucky sample are weak. Examples involving exact arithmetic may ignore the seed.

To inspect an implementation:

```python
import inspect
from mlfirst.foundations import lesson_01
print(inspect.getsource(lesson_01))
```

Edit a copy of a function or a notebook cell. Make one change at a time, first
writing a prediction about the outcome. Preserve an unmodified reference run.
Restart notebook kernels when module edits are not reflected in imported code.

## Read an error from the bottom

The last traceback line describes the exception. `ModuleNotFoundError` often
means a dependency is missing or the working directory is wrong. `ValueError`
often describes invalid shapes or parameters. `AssertionError` means a checked
claim failed. Start at the final line, then inspect the referenced calculation.
Avoid deleting the check simply to make execution finish.

Now open [Learning problems and evidence](../lessons/01_learning_problems.md).
For a longer language introduction, use the official
[Python tutorial](https://docs.python.org/3/tutorial/). For array practice, use
the [NumPy quickstart](https://numpy.org/doc/2.3/user/quickstart.html).
