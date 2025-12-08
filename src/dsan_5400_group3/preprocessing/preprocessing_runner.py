"""
preprocessor.py

High-level Preprocessor that orchestrates the full preprocessing pipeline:

1. Merge all raw .jsonl scraped files into biographies_raw.csv
2. Add pronoun-based gender metadata to biographies_raw.csv
3. Run chunked text cleaning → biographies_clean.csv
4. Perform train/val/test split from cleaned dataset

This file does NOT change any logic from the underlying modules.
It only coordinates the sequence of preprocessing steps.
"""

from pathlib import Path

from dsan_5400_group3.preprocessing.loader import RawJSONLLoader
from dsan_5400_group3.preprocessing.pronoun_annotator import PronounGenderAnnotator
from dsan_5400_group3.preprocessing.cleaner import ChunkedCleanerPipeline


class Preprocessor:
    """
    Full preprocessing orchestrator.

    Usage:
        pre = Preprocessor(project_root="/path/to/project")
        pre.run_all()
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)

        # Directories
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed"

        # Files
        self.raw_csv = self.raw_dir / "biographies_raw.csv"
        self.cleaned_csv = self.processed_dir / "biographies_clean.csv"

    # Merge .jsonl into biographies_raw.csv
    def merge_jsonl(self):
        print("\nMerging .jsonl files into biographies_raw.csv ...")

        loader = RawJSONLLoader(self.raw_dir)
        loader.save_csv(self.raw_csv)

        print(f"✓ Saved merged raw CSV to {self.raw_csv}")

    # Pronoun-based gender annotation
    def annotate_pronoun_gender(self):
        print("\nAdding pronoun-based gender metadata ...")

        annotator = PronounGenderAnnotator(self.raw_csv)
        annotator.run()

        print("✓ Pronoun-based gender annotation complete.")

    # Chunked cleaning and NLP preprocessing
    def clean_text(self):
        print("\nCleaning text and generating biographies_clean.csv ...")

        pipeline = ChunkedCleanerPipeline(
            raw_csv=self.raw_csv,
            processed_dir=self.processed_dir
        )
        pipeline.run()

        print(f"✓ Cleaning complete. Output saved to {self.cleaned_csv}")

    # Run all steps
    def run_all(self):
        """
        Run the complete preprocessing pipeline in order.
        """
        print("\n========== DSAN5400 FULL PREPROCESSING PIPELINE ==========")

        self.merge_jsonl()
        self.annotate_pronoun_gender()
        self.clean_text()

        print("\nDone! Preprocessing pipeline successfully completed.")
        print("==========================================================\n")
