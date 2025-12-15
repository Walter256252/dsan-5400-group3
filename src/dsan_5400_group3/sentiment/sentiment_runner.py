"""
sentiment_runner.py

High-level orchestrator for sentiment analysis on Wikipedia biographies.

This module calls three independent sentiment components:
    - VADER (lexicon-based)
    - TextBlob (polarity & subjectivity)
    - RoBERTa (CardiffNLP transformer sentiment)

"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import gdown
import pandas as pd

from dsan_5400_group3.sentiment.vader import add_vader_sentiment
from dsan_5400_group3.sentiment.textblob import add_textblob_sentiment
from dsan_5400_group3.sentiment.roberta import RobertaSentiment


logger = logging.getLogger(__name__)


class SentimentRunner:
    """
    Orchestrates the full sentiment pipeline:
        1. Load cleaned biography CSV
        2. Optional: download from Google Drive
        3. Optional: sample by N or fraction
        4. Apply VADER, TextBlob, and RoBERTa sentiment
        5. Save enriched CSV
    """

    TEXT_COLUMN = "text_clean"

    def __init__(
        self,
        input_csv: Path,
        output_csv: Path,
        sample_n: Optional[int] = None,
        sample_frac: Optional[float] = None,
        batch_size: int = 16,
        max_length: int = 256,
        drive_url: Optional[str] = None,
    ):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        self.sample_n = sample_n
        self.sample_frac = sample_frac
        self.drive_url = drive_url
        self.batch_size = batch_size
        self.max_length = max_length

    # Google Drive download
    @staticmethod
    def download_from_drive(drive_url: str, out_path: Path):
        """
        Converts a Google Drive *view* URL into a direct download link.
        """
        match = re.search(r"/file/d/([^/]+)/", drive_url)
        if not match:
            raise ValueError(
                f"Invalid Google Drive URL format:\n{drive_url}\n"
                "Expected something like: https://drive.google.com/file/d/<ID>/view"
            )

        file_id = match.group(1)
        direct = f"https://drive.google.com/uc?id={file_id}"

        if out_path.exists():
            out_path.unlink()

        logger.info(f"Downloading cleaned CSV from Google Drive → {out_path}")
        gdown.download(direct, str(out_path), quiet=False)
        logger.info("Download complete")

    # Load + optional sampling
    def load_data(self) -> pd.DataFrame:
        if self.drive_url:
            self.download_from_drive(self.drive_url, self.input_csv)

        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {self.input_csv}\n"
                "Run preprocessing pipeline first."
            )

        logger.info(f"Loading dataset from {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        logger.info(f"Full dataset shape: {df.shape}")

        # Apply sampling if needed
        if self.sample_n:
            df = df.head(self.sample_n)
            logger.info(f"Using first {self.sample_n} rows → {df.shape}")

        elif self.sample_frac:
            df = df.sample(frac=self.sample_frac, random_state=42)
            logger.info(f"Using random {self.sample_frac:.1%} sample → {df.shape}")

        return df

    # Run the entire sentiment pipeline
    def run(self):
        logger.info("Starting sentiment analysis pipeline")

        df = self.load_data()

        # Validate input text column
        if self.TEXT_COLUMN not in df.columns:
            raise ValueError(
                f"Expected column '{self.TEXT_COLUMN}' not found.\n"
                f"Available: {list(df.columns)}\n"
                "Make sure this is the cleaned CSV from preprocessing."
            )

        text_col = self.TEXT_COLUMN

        # VADER
        logger.info("Running VADER sentiment analysis")
        df = add_vader_sentiment(df, text_col)

        # TextBlob
        logger.info("Running TextBlob sentiment analysis")
        df = add_textblob_sentiment(df, text_col)

        # RoBERTa
        logger.info("Running RoBERTa transformer sentiment analysis")
        roberta = RobertaSentiment(
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        df = roberta.add_roberta_sentiment(df, text_col)

        # Save output
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)

        logger.info(f"Sentiment-enriched CSV saved to {self.output_csv}")
        logger.info("Sentiment analysis pipeline completed successfully")

        return df
