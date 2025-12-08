"""
plots.py

Visualization utilities for sentiment analysis.
Generates:
    - Sentiment metric boxplots by gender
    - RoBERTa label distribution bar plot
    - Pronoun count distributions
    - Compound sentiment distribution

All plots saved under results/sentiment/plots/.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PlotGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Helper for saving plots
    def _save(self, name: str):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved plot: {path}")

    # Boxplots for sentiment metrics
    def plot_sentiment_boxplots(self, df: pd.DataFrame):
        sentiment_cols = [
            "vader_compound",
            "vader_pos",
            "vader_neg",
            "textblob_polarity",
            "textblob_subjectivity",
        ]

        for col in sentiment_cols:
            if col not in df.columns:
                continue

            plt.figure(figsize=(7, 5))
            sns.boxplot(data=df, x="gender", y=col, palette="Set2")
            plt.title(f"{col} by Gender")
            self._save(f"boxplot_{col}")

    # RoBERTa label distribution
    def plot_roberta_distribution(self, df: pd.DataFrame):
        if "roberta_label_str" not in df.columns:
            print("[WARN] Cannot plot RoBERTa distribution (column missing).")
            return

        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="roberta_label_str", hue="gender", palette="Set3")
        plt.title("RoBERTa Sentiment Distribution by Gender")
        self._save("roberta_label_distribution")

    # Pronoun count distributions
    def plot_pronoun_distributions(self, df: pd.DataFrame):
        cols = ["male_pronoun_count", "female_pronoun_count"]
        for col in cols:
            if col not in df.columns:
                continue

            plt.figure(figsize=(7, 5))
            sns.histplot(df, x=col, hue="gender", bins=30, kde=True, palette="Set1")
            plt.title(f"{col} Distribution by Gender")
            self._save(f"hist_{col}")

    # Compound sentiment KDE
    def plot_compound_kde(self, df: pd.DataFrame):
        if "vader_compound" not in df.columns:
            return

        plt.figure(figsize=(7, 5))
        sns.kdeplot(
            data=df,
            x="vader_compound",
            hue="gender",
            common_norm=False,
            fill=True,
            alpha=0.4,
        )
        plt.title("VADER Compound Sentiment (Density by Gender)")
        self._save("kde_vader_compound")

    # Generate all plots
    def generate_all(self, df: pd.DataFrame):
        print("\n[INFO] Generating all plots...")

        self.plot_sentiment_boxplots(df)
        self.plot_roberta_distribution(df)
        self.plot_pronoun_distributions(df)
        self.plot_compound_kde(df)

        print("[INFO] All plots generated.")
