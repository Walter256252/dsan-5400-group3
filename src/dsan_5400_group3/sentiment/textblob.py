"""
textblob_sentiment.py

Computes TextBlob polarity and subjectivity.
Logic copied directly from run_sentiment.py.
"""

import re
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob


def _clean_text(text):
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def add_textblob_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    print("[INFO] Running TextBlob sentiment...")

    polarities = []
    subjectivities = []

    for txt in tqdm(df[text_col].fillna("").astype(str),
                    desc="TextBlob", ncols=80):

        cleaned = _clean_text(txt)
        blob = TextBlob(cleaned)

        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)

    df["textblob_polarity"] = polarities
    df["textblob_subjectivity"] = subjectivities

    return df
