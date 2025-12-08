"""
run_preprocessing.py

Entry point for the full preprocessing pipeline:

1. Merge all raw .jsonl files → data/raw/biographies_raw.csv
2. Add pronoun-based gender columns to biographies_raw.csv
3. Clean text in chunks and create data/processed/biographies_clean.csv
   (+ train/val/test splits inside the cleaner pipeline, if you keep that step)

This is just a thin wrapper around
dsan_5400_group3.preprocessing.preprocessor.Preprocessor.
"""

from pathlib import Path
from dsan_5400_group3.preprocessing.preprocessing_runner import Preprocessor


def main() -> None:
    # project_root = repo root
    project_root = Path(__file__).resolve().parents[1]

    pre = Preprocessor(project_root=project_root)
    pre.run_all()


if __name__ == "__main__":
    main()