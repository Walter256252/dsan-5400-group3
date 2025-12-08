"""
analysis_runner.py

Unified analysis pipeline combining:
    - Descriptive sentiment summaries
    - Pronoun statistics
    - RoBERTa label summaries
    - Statistical tests (t-test, U-test, KS-test, chi-square)

This file DEPRECATING the old run_analysis.py and run_stats.py,
and merges both into a single coherent analysis stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from dsan_5400_group3.analysis.summaries import (
    compute_sentiment_summary,
    compute_roberta_distribution,
    compute_pronoun_summary,
)
from dsan_5400_group3.analysis.stats import (
    run_all_numeric_tests,
    run_chi_square_test,
)


DEFAULT_INPUT = Path("data/processed/biographies_with_sentiment.csv")
DEFAULT_OUTPUT_DIR = Path("results/sentiment")


class AnalysisRunner:
    def __init__(
        self,
        input_csv: Path = DEFAULT_INPUT,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        sample_n: Optional[int] = None,
        gender_a: str = "male",
        gender_b: str = "female",
    ):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.sample_n = sample_n
        self.gender_a = gender_a
        self.gender_b = gender_b

    # Load Data
    def load_data(self) -> pd.DataFrame:
        print(f"[INFO] Loading input CSV: {self.input_csv}")

        if self.sample_n:
            df = pd.read_csv(self.input_csv, nrows=self.sample_n)
            print(f"[INFO] Using first {self.sample_n} rows")
        else:
            df = pd.read_csv(self.input_csv)

        print(f"[INFO] Loaded shape: {df.shape}")

        # Map RoBERTa labels to string
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        if "roberta_label" in df.columns:
            df["roberta_label_str"] = df["roberta_label"].map(label_map).fillna(
                df["roberta_label"]
            )

        return df


    # Run analysis pipeline
    def run(self):
        df = self.load_data()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Descriptive summaries

        print("\n[INFO] Computing sentiment summaries by gender...")
        sent_summary = compute_sentiment_summary(df)
        sent_summary.to_csv(self.output_dir / "sentiment_summary_by_gender.csv")

        print("\n[INFO] Computing RoBERTa distribution by gender...")
        rob = compute_roberta_distribution(df)
        rob.to_csv(self.output_dir / "roberta_distribution_by_gender.csv")

        print("\n[INFO] Computing pronoun usage summary...")
        pron = compute_pronoun_summary(df)
        pron.to_csv(self.output_dir / "pronoun_stats_by_gender.csv")

        # Statisical tests

        print("\n[INFO] Running numeric variable statistical tests...")
        numeric_results = run_all_numeric_tests(
            df, self.gender_a, self.gender_b
        )
        numeric_results.to_csv(
            self.output_dir / f"stats_continuous_{self.gender_a}_vs_{self.gender_b}.csv"
        )

        print("\n[INFO] Running chi-square test on RoBERTa label distribution...")
        chi = run_chi_square_test(df, self.gender_a, self.gender_b)
        chi.to_csv(
            self.output_dir
            / f"stats_chi2_roberta_{self.gender_a}_vs_{self.gender_b}.csv",
            index=False,
        )

        print("\n>>> Completed full analysis pipeline <<<")