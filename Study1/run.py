"""Rebuild Study 1 from the public KuaiRand-1K CSV files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

INPUT_FILES = (
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv",
    "log_random_4_22_to_5_08_1k.csv",
    "video_features_basic_1k.csv",
)

SCRIPTS = (
    "01_build_panel.py",
    "02_daily_models.py",
    "03_parallel_trends.py",
    "04_event_models.py",
    "s1_insertion_quality.py",
    "05_causal_forest_shap.py",
    "06_baseline_heterogeneity.py",
    "07_descriptive_tables.py",
    "10_daily_rate_robustness.py",
    "08_make_empirical_figures.py",
    "09_make_manuscript_diagrams.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the four KuaiRand-1K CSVs.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated Study 1 results.")
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()

    missing = [name for name in INPUT_FILES if not (data / name).is_file()]
    if missing:
        parser.error(f"Missing files in {data}: {', '.join(missing)}")

    (output / "panels").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(KUAIRAND_DATA_DIR=str(data), S1_RESULT_DIR=str(output),
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    scripts = Path(__file__).resolve().parent / "scripts"
    for name in SCRIPTS:
        print(f"Study 1: {name}", flush=True)
        subprocess.run([sys.executable, str(scripts / name)], env=env, check=True)


if __name__ == "__main__":
    main()
