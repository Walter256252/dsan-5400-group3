"""
summaries.py

Descriptive summaries (means, proportions) for sentiment, pronouns,
and RoBERTa label distributions.
"""

import pandas as pd


def compute_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
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
        return pd.DataFrame()

    return df.groupby("gender")[sentiment_cols].agg(["mean", "std", "count"])


def compute_roberta_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "gender" not in df.columns or "roberta_label_str" not in df.columns:
        return pd.DataFrame()

    return pd.crosstab(df["gender"], df["roberta_label_str"], normalize="index")


def compute_pronoun_summary(df: pd.DataFrame) -> pd.DataFrame:
    pron_cols = [
        c
        for c in [
            "male_pronoun_count",
            "female_pronoun_count",
            "pronoun_male_minus_female",
        ]
        if c in df.columns
    ]

    if "gender" not in df.columns or not pron_cols:
        return pd.DataFrame()

    return df.groupby("gender")[pron_cols].agg(["mean", "std", "count"])
