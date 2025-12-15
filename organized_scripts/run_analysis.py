"""
run_analysis.py

Unified analysis entrypoint for:
    - sentiment descriptive summaries
    - RoBERTa label proportions
    - pronoun usage summaries
    - statistical tests (t-test, U-test, KS-test, chi-square)
    - optional sampling
    - optional gender-pair selection

This script only parses arguments and constructs AnalysisRunner.
"""

import argparse
import logging
from pathlib import Path
from dsan_5400_group3.analysis.analysis_runner import AnalysisRunner


def setup_logging():
    """Configure root logger for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/biographies_with_sentiment.csv")
    p.add_argument("--output-dir", default="results/sentiment")
    p.add_argument("--sample-n", type=int)
    p.add_argument("--gender-a", default="male")
    p.add_argument("--gender-b", default="female")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info("Launching analysis pipeline")
    logging.info(f"Input CSV: {args.input}")
    logging.info(f"Output directory: {args.output_dir}")

    if args.sample_n is not None:
        logging.info(f"Sampling mode: first {args.sample_n} rows")

    logging.info(f"Gender comparison: {args.gender_a} vs {args.gender_b}")

    runner = AnalysisRunner(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir),
        sample_n=args.sample_n,
        gender_a=args.gender_a,
        gender_b=args.gender_b,
    )

    runner.run()

    logging.info("Analysis pipeline completed successfully")


if __name__ == "__main__":
    main()
