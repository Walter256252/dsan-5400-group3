#!/usr/bin/env python
"""
run_sentiment.py

Sentiment analysis on cleaned Wikipedia biographies.

Pipeline:
- Load `data/processed/biographies_clean.csv`
  - Or optionally download it from Google Drive via --drive-url
- Optional: subset to a fixed-size sample (--sample-n) or fraction (--sample-frac) for speed
- Use the `text_clean` column as input text
- Compute:
    * VADER sentiment (lexicon-based)
    * TextBlob sentiment (polarity + subjectivity)
    * RoBERTa sentiment via `cardiffnlp/twitter-roberta-base-sentiment`
- Save enriched dataset to:
    * data/processed/biographies_with_sentiment.csv (by default)

Usage examples
--------------

From project root, full dataset:

    poetry run python scripts/run_sentiment.py

20k sample (for reproducibility / speed):

    poetry run python scripts/run_sentiment.py \
        --sample-n 20000 \
        --output-path data/processed/biographies_with_sentiment_sample20k.csv

Or with Google Drive download (overwrites local biographies_clean.csv):

    poetry run python scripts/run_sentiment.py \
        --drive-url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, List

import gdown
import numpy as np
import pandas as pd
import torch
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Paths and constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT_CSV = DATA_PROCESSED_DIR / "biographies_clean.csv"
DEFAULT_OUTPUT_CSV = DATA_PROCESSED_DIR / "biographies_with_sentiment.csv"

TEXT_COLUMN_NAME = "text_clean"  # from clean_data.py
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


# Google Drive function

def download_from_drive(drive_url: str, out_path: Path) -> Path:
    """
    Download a file from Google Drive using the *view* link you provided, e.g.:

        https://drive.google.com/file/d/1P77lUspCFPhhdVV_GmH6s-fWFCcxMcW-/view?usp=sharing

    We convert it to a direct download URL:

        https://drive.google.com/uc?id=1P77lUspCFPhhdVV_GmH6s-fWFCcxMcW-
    """

    # Extract file id from /file/d/<ID>/
    m = re.search(r"/file/d/([^/]+)/", drive_url)
    if not m:
        raise ValueError(
            f"Could not extract file id from Google Drive URL: {drive_url}\n"
            "Expected something like 'https://drive.google.com/file/d/<FILE_ID>/view?...'"
        )
    file_id = m.group(1)
    direct_url = f"https://drive.google.com/uc?id={file_id}"

    # Always overwrite so we don't keep an old HTML login page
    if out_path.exists():
        print(f"[INFO] Removing existing file at {out_path}")
        out_path.unlink()

    print(f"[INFO] Downloading from Google Drive (file id={file_id}) to {out_path} ...")
    gdown.download(direct_url, str(out_path), quiet=False)
    print("[INFO] Download complete.")
    return out_path


# Lexicon-based sentiment: VADER + TextBlob

def add_vader_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    print("[INFO] Running VADER sentiment...")
    vader_scores = df[text_col].fillna("").astype(str).apply(analyzer.polarity_scores)
    vader_df = pd.DataFrame(list(vader_scores))

    for col in vader_df.columns:
        df[f"vader_{col}"] = vader_df[col]

    print("[INFO] Added VADER columns: vader_neg, vader_neu, vader_pos, vader_compound")
    return df


def _clean_text_for_textblob(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_textblob_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    print("[INFO] Running TextBlob sentiment...")

    polarities: List[float] = []
    subjectivities: List[float] = []

    for txt in tqdm(df[text_col].fillna("").astype(str),
                    desc="TextBlob", ncols=80):
        cleaned = _clean_text_for_textblob(txt)
        blob = TextBlob(cleaned)
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)

    df["textblob_polarity"] = polarities
    df["textblob_subjectivity"] = subjectivities

    print("[INFO] Added TextBlob columns: textblob_polarity, textblob_subjectivity")
    return df


# Transformer-based sentiment: CardiffNLP RoBERTa

def add_roberta_sentiment(
    df: pd.DataFrame,
    text_col: str,
    batch_size: int = 16,
    max_length: int = 256,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading transformer model '{MODEL_NAME}' on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    label_map = model.config.id2label
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)

    roberta_labels: List[str] = []
    roberta_scores: List[float] = []

    print("[INFO] Running RoBERTa sentiment inference...")
    for i in tqdm(range(0, n, batch_size), desc="RoBERTa", ncols=80):
        batch_texts = texts[i: i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        max_probs, pred_ids = torch.max(probs, dim=-1)

        for pid, p in zip(pred_ids, max_probs):
            label = label_map[int(pid)]
            roberta_labels.append(label)
            roberta_scores.append(float(p))

    df["roberta_label"] = roberta_labels
    df["roberta_confidence"] = roberta_scores

    print("[INFO] Added RoBERTa columns: roberta_label, roberta_confidence")
    return df


# Argument parsing & main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis on cleaned Wikipedia biographies."
    )

    parser.add_argument(
        "--drive-url",
        type=str,
        default=None,
        help=(
            "Optional: Google Drive *file* URL for biographies_clean.csv. "
            "If provided, the file will be downloaded to data/processed/biographies_clean.csv "
            "before running sentiment."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help=f"Path to input cleaned CSV (default: {DEFAULT_INPUT_CSV}).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=(
            "Where to save output CSV with sentiment scores "
            f"(default: {DEFAULT_OUTPUT_CSV})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for RoBERTa sentiment model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max tokenized length for RoBERTa inputs.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help=(
            "Optional: use only the first N rows of the cleaned dataset "
            "for a faster run (e.g., 20000 for a 20k sample)."
        ),
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help=(
            "Optional: use a random fraction of the cleaned dataset "
            "(e.g., 0.1 for 10%). Cannot be used together with --sample-n."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_n is not None and args.sample_frac is not None:
        raise ValueError("Use either --sample-n OR --sample-frac, not both.")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Make sure processed dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull the cleaned CSV from Google Drive
    if args.drive_url:
        print(f"[INFO] Using Google Drive URL: {args.drive_url}")
        download_from_drive(args.drive_url, input_path)

    # Load data
    print(f"[INFO] Loading data from {input_path} ...")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run your cleaning pipeline first (clean_data.py) or provide --drive-url."
        )

    df = pd.read_csv(input_path)
    print(f"[INFO] Full data shape: {df.shape}")

    # Apply sampling
    if args.sample_n is not None:
        df = df.head(args.sample_n)
        print(f"[INFO] Using first {args.sample_n} rows. Sample shape: {df.shape}")
    elif args.sample_frac is not None:
        df = df.sample(frac=args.sample_frac, random_state=42)
        print(
            f"[INFO] Using random {args.sample_frac:.2%} sample. "
            f"Sample shape: {df.shape}"
        )

    # Ensure text_clean exists
    if TEXT_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Expected text column '{TEXT_COLUMN_NAME}' not found in input CSV.\n"
            f"Available columns: {list(df.columns)}\n"
            "Make sure you're running this on data/processed/biographies_clean.csv "
            "produced by clean_data.py, or adjust TEXT_COLUMN_NAME."
        )

    text_col = TEXT_COLUMN_NAME
    print(f"[INFO] Using '{text_col}' as the text column.")

    # Run sentiment computation
    df = add_vader_sentiment(df, text_col)
    df = add_textblob_sentiment(df, text_col)
    df = add_roberta_sentiment(
        df,
        text_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Save
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved sentiment-enriched data to: {output_path}")


if __name__ == "__main__":
    main()
