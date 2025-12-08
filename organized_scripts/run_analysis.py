"""
run_analysis.py

Unified analysis entrypoint for:
    - sentiment descriptive summaries
    - RoBERTa label proportions
    - pronoun usage summaries
    - statistical tests (t-test, U-test, KS-test, chi-square)
    - optional sampling
    - optional gender-pair selection
    - optional plot generation

Arguments
---------
--input:
    Path to the sentiment-enriched CSV produced by run_sentiment.py.

--output-dir:
    Directory for saving summaries, test results, and optionally plots.

--sample-n:
    Use only the first N rows.
    Useful for debugging robustness of tests or reducing compute time.

--gender-a / --gender-b:
    The two gender groups being compared in statistical tests.
    Defaults: "male" vs "female".

This script only parses arguments and constructs AnalysisRunner.
"""

import argparse
from pathlib import Path
from dsan_5400_group3.analysis.analysis_runner import AnalysisRunner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/biographies_with_sentiment.csv")
    p.add_argument("--output-dir", default="results/sentiment")
    p.add_argument("--sample-n", type=int)
    p.add_argument("--gender-a", default="male")
    p.add_argument("--gender-b", default="female")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runner = AnalysisRunner(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir),
        sample_n=args.sample_n,
        gender_a=args.gender_a,
        gender_b=args.gender_b,
    )

    runner.run()
