"""Reproduce the manuscript analyses from the public source data."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuairand", type=Path, help="KuaiRand-1K data directory")
    parser.add_argument("--kuairec", type=Path, help="KuaiRec 2.0 data directory")
    parser.add_argument("--output", type=Path, default=Path("output"), help="result directory")
    args = parser.parse_args()
    studies = [("Study1", args.kuairand), ("Study2", args.kuairec)]
    if not any(data for _, data in studies):
        parser.error("provide --kuairand, --kuairec, or both")
    for _, data in studies:
        if data is not None and not data.is_dir():
            parser.error(f"data directory does not exist: {data}")

    package = Path(__file__).resolve().parent
    for study, data in studies:
        if data is None:
            continue
        output = args.output.resolve() / study.lower()
        print(f"Running {study}. Results will be written to {output}", flush=True)
        subprocess.run(
            [sys.executable, str(package / study / "run.py"),
             "--data", str(data.resolve()), "--output", str(output)],
            check=True,
        )


if __name__ == "__main__":
    main()
