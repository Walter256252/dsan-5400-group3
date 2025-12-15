"""
pronoun_annotator.py

Chunk-based pronoun counting and gender assignment for raw Wikipedia
biographies. 
"""

from __future__ import annotations
from pathlib import Path
import os
import pandas as pd


MALE_PRONOUNS = ["he", "him", "his", "himself"]
FEMALE_PRONOUNS = ["she", "her", "hers", "herself"]


def add_pronoun_gender_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Add male/female pronoun counts and majority-vote gender to a chunk.

    100% identical logic to the original add_pronoun_gender.py.
    """
    if "text" not in chunk.columns:
        raise ValueError("Expected a 'text' column in biographies_raw.csv")

    text = chunk["text"].fillna("").astype(str).str.lower()

    # male counts
    male_counts = 0
    for p in MALE_PRONOUNS:
        male_counts = male_counts + text.str.count(rf"\b{p}\b")

    # female counts
    female_counts = 0
    for p in FEMALE_PRONOUNS:
        female_counts = female_counts + text.str.count(rf"\b{p}\b")

    chunk["male_pronoun_count"] = male_counts.astype("int64")
    chunk["female_pronoun_count"] = female_counts.astype("int64")

    # majority rule
    def decide_gender(m: int, f: int) -> str:
        if m > f:
            return "male"
        elif f > m:
            return "female"
        else:
            return "unknown"

    chunk["gender"] = [
        decide_gender(int(m), int(f))
        for m, f in zip(chunk["male_pronoun_count"], chunk["female_pronoun_count"])
    ]

    return chunk


class PronounGenderAnnotator:
    """
    Chunked annotator for adding gender features to the raw CSV.
    After processing, the original raw CSV is atomically replaced.
    """

    def __init__(self, raw_csv: Path | str):
        self.raw_csv = Path(raw_csv)
        self.raw_dir = self.raw_csv.parent
        self.temp_csv = self.raw_dir / "biographies_raw_with_gender.tmp.csv"

    def run(self, chunksize: int = 2000):
        print(f"Reading raw data (chunked) from: {self.raw_csv}")

        # Remove temp file from previous runs
        if self.temp_csv.exists():
            self.temp_csv.unlink()

        chunks = pd.read_csv(self.raw_csv, chunksize=chunksize)

        total_rows = 0
        first_chunk = True

        for chunk_idx, chunk in enumerate(chunks, start=1):
            # process chunk
            chunk = add_pronoun_gender_chunk(chunk)

            # append to temporary CSV
            mode = "w" if first_chunk else "a"
            header = first_chunk
            chunk.to_csv(self.temp_csv, mode=mode, header=header, index=False)

            total_rows += len(chunk)
            first_chunk = False
            print(f"Processed rows so far: {total_rows}")

        print(f"\nFinished pronoun processing. Total rows: {total_rows}")
        print(f"Temporary file written to: {self.temp_csv}")

        # atomic replacement
        os.replace(self.temp_csv, self.raw_csv)

        print(f"Updated raw CSV with gender column: {self.raw_csv}")
        print("Done.")
