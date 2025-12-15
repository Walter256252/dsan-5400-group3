"""
run_sentiment.py

Entrypoint script for the sentiment analysis pipeline.

This script is a thin wrapper around:
    dsan_5400_group3.sentiment.sentiment_runner.SentimentRunner

Only argument parsing and logging configuration happen here.
All sentiment logic lives in SentimentRunner.
"""

import argparse
import logging
from pathlib import Path
from dsan_5400_group3.sentiment.sentiment_runner import SentimentRunner


def setup_logging():
    """Configure root logger for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/biographies_clean.csv")
    p.add_argument("--output", default="data/processed/biographies_with_sentiment.csv")
    p.add_argument("--sample-n", type=int)
    p.add_argument("--sample-frac", type=float)
    p.add_argument("--drive-url")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info("Launching sentiment analysis pipeline")
    logging.info(f"Input CSV: {args.input}")
    logging.info(f"Output CSV: {args.output}")

    if args.sample_n is not None:
        logging.info(f"Sampling mode: first {args.sample_n} rows")
    elif args.sample_frac is not None:
        logging.info(f"Sampling mode: random {args.sample_frac:.1%} of dataset")

    if args.drive_url:
        logging.info("Input CSV will be downloaded from Google Drive")

    runner = SentimentRunner(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        sample_n=args.sample_n,
        sample_frac=args.sample_frac,
        drive_url=args.drive_url,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    runner.run()

    logging.info("Sentiment analysis pipeline completed successfully")


if __name__ == "__main__":
    main()
