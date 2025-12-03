#!/usr/bin/env python
"""
run_analysis.py

Analysis of sentiment-enriched Wikipedia biographies.

Features:
- Optional sampling (for quick tests).
- Computes:
    * Sentiment summary by gender (VADER + TextBlob).
    * RoBERTa label distribution by gender.
    * Pronoun-based stats by gender (if available).
- Saves summary tables to CSV under results/sentiment/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "biographies_with_sentiment.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sentiment patterns in Wikipedia biographies."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path to sentiment-enriched CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for saving summary tables (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="If set, only use the first N rows (for testing). If omitted, use full dataset.",
    )
    return parser.parse_args()


def load_data(path: Path, sample_n: Optional[int]) -> pd.DataFrame:
    print(">>> run_analysis.py is running <<<")
    print(f"[INFO] Loading from: {path}")

    if sample_n is not None:
        print(f"[INFO] Using sample-n = {sample_n}")
        df = pd.read_csv(path, nrows=sample_n)
    else:
        print("[INFO] Using FULL dataset")
        df = pd.read_csv(path)

    print(f"[INFO] Loaded shape: {df.shape}")

    # Map RoBERTa labels to strings 
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    if "roberta_label" in df.columns:
        df["roberta_label_str"] = df["roberta_label"].map(label_map).fillna(
            df["roberta_label"]
        )

    # Simple pronoun balance feature
    if {"male_pronoun_count", "female_pronoun_count"} <= set(df.columns):
        df["pronoun_male_minus_female"] = (
            df["male_pronoun_count"].fillna(0) - df["female_pronoun_count"].fillna(0)
        )

    print("\n[INFO] First 3 rows:")
    print(df.head(3))

    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = load_data(input_path, sample_n=args.sample_n)

    # Sentiment summary by gender 
    sentiment_cols = [
        c
        for c in [
            "vader_neg",
            "vader_neu",
            "vader_pos",
            "vader_compound",
            "textblob_polarity",
            "textblob_subjectivity",
        ]
        if c in df.columns
    ]

    if "gender" not in df.columns:
        print("\n[WARN] No 'gender' column found; cannot group by gender.")
    else:
        print("\n[INFO] Sentiment summary by gender:")
        sent_summary = df.groupby("gender")[sentiment_cols].agg(["mean", "std", "count"])
        print(sent_summary)

        out_sent = output_dir / "sentiment_summary_by_gender.csv"
        sent_summary.to_csv(out_sent)
        print(f"[INFO] Saved sentiment summary to: {out_sent}")

    # ---- 2) RoBERTa label distribution by gender ----
    if "gender" in df.columns and "roberta_label_str" in df.columns:
        print("\n[INFO] RoBERTa label proportions by gender:")
        roberta_dist = pd.crosstab(
            df["gender"], df["roberta_label_str"], normalize="index"
        )
        print(roberta_dist)

        out_rob = output_dir / "roberta_distribution_by_gender.csv"
        roberta_dist.to_csv(out_rob)
        print(f"[INFO] Saved RoBERTa distribution to: {out_rob}")
    else:
        print("\n[WARN] Missing 'gender' or 'roberta_label_str'; skipping RoBERTa dist.")

    # Pronoun stats by gender 
    if "gender" in df.columns and "pronoun_male_minus_female" in df.columns:
        print("\n[INFO] Pronoun stats by gender:")
        pronoun_cols = [
            c
            for c in [
                "male_pronoun_count",
                "female_pronoun_count",
                "pronoun_male_minus_female",
            ]
            if c in df.columns
        ]
        pronoun_summary = df.groupby("gender")[pronoun_cols].agg(
            ["mean", "std", "count"]
        )
        print(pronoun_summary)

        out_pron = output_dir / "pronoun_stats_by_gender.csv"
        pronoun_summary.to_csv(out_pron)
        print(f"[INFO] Saved pronoun stats to: {out_pron}")
    else:
        print("\n[WARN] Pronoun columns not found; skipping pronoun stats.")

    print("\n>>> run_analysis.py finished successfully <<<")


if __name__ == "__main__":
    main()
