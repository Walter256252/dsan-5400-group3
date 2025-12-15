"""
run_preprocessing.py

Entry point for the full preprocessing pipeline:

1. Merge all raw .jsonl files → data/raw/biographies_raw.csv
2. Add pronoun-based gender columns to biographies_raw.csv
3. Clean text in chunks and create data/processed/biographies_clean.csv

This is just a thin wrapper around
dsan_5400_group3.preprocessing.preprocessor.Preprocessor.
"""

import logging
from pathlib import Path
from dsan_5400_group3.preprocessing.preprocessing_runner import Preprocessor


def setup_logging():
    """Configure root logger for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()

    # project_root = repo root
    project_root = Path(__file__).resolve().parents[1]

    logging.info("Launching full preprocessing pipeline")
    logging.info(f"Project root: {project_root}")

    pre = Preprocessor(project_root=project_root)
    pre.run_all()

    logging.info("Preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()
