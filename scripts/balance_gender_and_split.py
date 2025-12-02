"""
Balance biographies by gender in a memory-safe way (chunked + reservoir sampling)
and create train/val/test splits.

Steps:
- Load data/processed/biographies_clean.csv in chunks
- Keep rows with gender in {"male", "female"}
- For each gender, use reservoir sampling to get up to 100,000 rows
  (uniform random sample across the whole file, without loading it fully)
- Combine and shuffle to get a balanced dataset
- Save the balanced dataset to:
    * data/processed/biographies_gender_balanced.csv
- Split the balanced dataset into train/val/test (80/10/10):
    * data/processed/train_gender_balanced.csv
    * data/processed/val_gender_balanced.csv
    * data/processed/test_gender_balanced.csv
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


def reservoir_update(
    row: Dict,
    reservoir: List[Dict],
    seen_count: int,
    target_n: int,
    rng: random.Random,
) -> int:
    """
    Update a reservoir sample with one new row.

    - row: dict representing one row
    - reservoir: current reservoir list (max size = target_n)
    - seen_count: how many rows of this group have been seen so far
    - target_n: desired reservoir size
    - rng: random.Random instance for reproducibility

    Returns the updated seen_count.
    """
    seen_count += 1

    if len(reservoir) < target_n:
        # Still filling the reservoir
        reservoir.append(row)
    else:
        # Reservoir sampling: replace with probability target_n / seen_count
        j = rng.randint(0, seen_count - 1)
        if j < target_n:
            reservoir[j] = row

    return seen_count


def main() -> None:
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    clean_csv = processed_dir / "biographies_clean.csv"
    balanced_csv = processed_dir / "biographies_gender_balanced.csv"

    train_csv = processed_dir / "train_gender_balanced.csv"
    val_csv = processed_dir / "val_gender_balanced.csv"
    test_csv = processed_dir / "test_gender_balanced.csv"

    print(f"Loading cleaned data (chunked) from: {clean_csv}")

    # Parameters
    target_per_gender = 100_000
    chunksize = 20_000  # You can lower this if memory is still tight

    # Reservoirs for each gender
    reservoir_male: List[Dict] = []
    reservoir_female: List[Dict] = []

    # Counters: how many rows have we seen for each gender
    male_seen = 0
    female_seen = 0

    # Reproducible RNG
    rng = random.Random(42)

    # Read in chunks
    total_rows = 0
    for chunk_idx, chunk in enumerate(
        pd.read_csv(clean_csv, chunksize=chunksize)
    ):
        total_rows += len(chunk)
        print(f"Processing chunk {chunk_idx + 1}, rows so far: {total_rows}")

        # Ensure gender column exists and keep only male/female
        if "gender" not in chunk.columns:
            raise ValueError(
                "Column 'gender' not found in biographies_clean.csv. "
                "Make sure you added gender before running this script."
            )

        chunk = chunk[chunk["gender"].isin(["male", "female"])]

        # Convert to list of dicts for easy per-row handling
        records = chunk.to_dict(orient="records")

        # Update reservoirs
        for row in records:
            g = row.get("gender", None)
            if g == "male":
                male_seen = reservoir_update(
                    row=row,
                    reservoir=reservoir_male,
                    seen_count=male_seen,
                    target_n=target_per_gender,
                    rng=rng,
                )
            elif g == "female":
                female_seen = reservoir_update(
                    row=row,
                    reservoir=reservoir_female,
                    seen_count=female_seen,
                    target_n=target_per_gender,
                    rng=rng,
                )

    print("Finished streaming through the file.")
    print(f"Total rows seen: {total_rows}")
    print(f"Total male rows seen: {male_seen}")
    print(f"Total female rows seen: {female_seen}")

    print(f"Male reservoir size: {len(reservoir_male)} (target {target_per_gender})")
    print(f"Female reservoir size: {len(reservoir_female)} (target {target_per_gender})")

    if len(reservoir_male) < target_per_gender:
        print(
            f"Warning: only {len(reservoir_male)} male rows in reservoir "
            f"(less than target {target_per_gender})."
        )
    if len(reservoir_female) < target_per_gender:
        print(
            f"Warning: only {len(reservoir_female)} female rows in reservoir "
            f"(less than target {target_per_gender})."
        )

    # Convert reservoirs to DataFrames
    df_male_sample = pd.DataFrame(reservoir_male)
    df_female_sample = pd.DataFrame(reservoir_female)

    print("Sampled male shape:", df_male_sample.shape)
    print("Sampled female shape:", df_female_sample.shape)

    # Combine and shuffle balanced dataset
    df_balanced = pd.concat([df_male_sample, df_female_sample], axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=123).reset_index(drop=True)

    print("Balanced dataset shape:", df_balanced.shape)
    print("Balanced gender counts:")
    print(df_balanced["gender"].value_counts())

    # Save full balanced dataset
    balanced_csv.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(balanced_csv, index=False)
    print(f"Balanced dataset saved to: {balanced_csv}")

    # Train/Val/Test split (80/10/10) on balanced data
    print("Creating train/val/test splits (80/10/10) on balanced data...")

    df_shuffled = df_balanced.sample(frac=1.0, random_state=999).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val :].copy()

    print("Train size:", len(df_train))
    print("Val size:  ", len(df_val))
    print("Test size: ", len(df_test))

    # Save splits
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"Train saved to: {train_csv}")
    print(f"Val saved to:   {val_csv}")
    print(f"Test saved to:  {test_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
