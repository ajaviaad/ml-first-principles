"""Command line entry point; run from the repository root or after installation."""
import argparse
import json
import platform
from pathlib import Path
from .registry import TITLES, run_lesson


def main():
    parser = argparse.ArgumentParser(description="Small offline experiments for learning ML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lesson", type=int, choices=range(1, 33), metavar="1..32")
    group.add_argument("--all", action="store_true", help="run all 32 lessons")
    group.add_argument("--list", action="store_true", help="list available experiments")
    parser.add_argument("--seed", type=int, default=42, help="nonnegative experiment seed (default: 42)")
    parser.add_argument("--output", type=Path, help="also save the result as JSON")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.list:
        for number, title in TITLES.items():
            print(f"{number:02d}  {title}")
        return
    try:
        import numpy as np
    except ImportError:
        parser.exit(1, "NumPy is missing. Run: python -m pip install -r requirements.txt\n")
    numbers = list(TITLES) if args.all else [args.lesson]
    result = {"environment": {"python": platform.python_version(), "numpy": np.__version__},
              "seed": args.seed,
              "lessons": {f"{n:02d}": {"title": TITLES[n], "result": run_lesson(n, args.seed)} for n in numbers}}
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
