"""
vader.py

Provides lexicon-based sentiment analysis using VADER.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_vader_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    print("[INFO] Running VADER sentiment...")
    vader_scores = df[text_col].fillna("").astype(str).apply(analyzer.polarity_scores)
    vader_df = pd.DataFrame(list(vader_scores))

    for col in vader_df.columns:
        df[f"vader_{col}"] = vader_df[col]

    return df
