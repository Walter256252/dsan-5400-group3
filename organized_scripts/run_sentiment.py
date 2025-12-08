"""
run_sentiment.py

Entrypoint script for the sentiment analysis pipeline.

This script is a thin wrapper around:
    dsan_5400_group3.sentiment.sentiment_runner.SentimentRunner

It provides a CLI for running the sentiment workflow:
    INPUT → VADER → TextBlob → RoBERTa → OUTPUT

Arguments
---------
--input:
    Path to the cleaned biographies CSV (default: data/processed/biographies_clean.csv)

--output:
    Path where the sentiment-enriched CSV should be written.

--sample-n:
    Use only the first N rows of the dataset (deterministic sampling).
    Useful for debugging or fast prototype runs.

--sample-frac:
    Use a random fraction of the dataset (e.g., 0.1 for 10%).
    Cannot be used together with --sample-n.

--drive-url:
    A Google Drive *file link*. If provided, the script downloads the input CSV
    before running sentiment analysis. Helpful when data is large or shared via Drive.

--batch-size:
    Batch size for RoBERTa inference. Larger = faster but needs more VRAM.

--max-length:
    Maximum tokenized sequence length for RoBERTa. Default = 256 (balanced speed/accuracy).

Only argument parsing happens here. The actual work is done in SentimentRunner.
"""

import argparse
from pathlib import Path
from dsan_5400_group3.sentiment.sentiment_runner import SentimentRunner

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


if __name__ == "__main__":
    args = parse_args()

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
