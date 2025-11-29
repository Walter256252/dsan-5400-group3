"""
add_pronoun_gender.py

Goal:
- Read the raw biographies dataset in chunks:
      data/raw/biographies_raw.csv
- Count gendered English pronouns in `text`:
      male:   he, him, his, himself
      female: she, her, hers, herself
- Add these columns:
      * male_pronoun_count
      * female_pronoun_count
      * gender    (male / female / unknown, based on majority)
- Write to a temporary CSV and then replace the original
  biographies_raw.csv, so later cleaning scripts automatically
  see the new `gender` column.

This version is CHUNKED to avoid "killed" due to memory.
"""

from __future__ import annotations

from pathlib import Path
import os
import pandas as pd

# Male and female pronouns we want to count
MALE_PRONOUNS = ["he", "him", "his", "himself"]
FEMALE_PRONOUNS = ["she", "her", "hers", "herself"]


def add_pronoun_gender_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Given a chunk with a `text` column, add:
      - male_pronoun_count
      - female_pronoun_count
      - gender  (male / female / unknown)
    and return the same chunk with extra columns.
    """

    if "text" not in chunk.columns:
        raise ValueError("Expected a 'text' column in biographies_raw.csv")

    # Make sure text is lowercased string
    text = chunk["text"].fillna("").astype(str).str.lower()

    # Count male pronouns
    male_counts = 0
    for p in MALE_PRONOUNS:
        # \b to match whole words
        male_counts = male_counts + text.str.count(rf"\b{p}\b")

    # Count female pronouns
    female_counts = 0
    for p in FEMALE_PRONOUNS:
        female_counts = female_counts + text.str.count(rf"\b{p}\b")

    chunk["male_pronoun_count"] = male_counts.astype("int64")
    chunk["female_pronoun_count"] = female_counts.astype("int64")

    # Decide gender for each row using majority rule
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


def main() -> None:
    # Project root = parent of scripts/
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_csv = raw_dir / "biographies_raw.csv"

    print(f"Reading raw data (chunked) from: {raw_csv}")

    # Temporary output file
    temp_csv = raw_dir / "biographies_raw_with_gender.tmp.csv"

    # If temp file exists from previous run, remove it
    if temp_csv.exists():
        temp_csv.unlink()

    chunks = pd.read_csv(raw_csv, chunksize=2000)

    total_rows = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(chunks, start=1):
        # Add pronoun-based gender information for this chunk
        chunk = add_pronoun_gender_chunk(chunk)

        # Append to temp CSV
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(temp_csv, mode=mode, header=header, index=False)

        total_rows += len(chunk)
        first_chunk = False
        print(f"Processed rows so far: {total_rows}")

    print(f"\nFinished processing pronouns. Total rows: {total_rows}")
    print(f"Temporary file with gender written to: {temp_csv}")

    # Replace original file with the temp file (atomic replace)
    os.replace(temp_csv, raw_csv)
    print(f"Replaced original raw CSV with new file including gender: {raw_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
