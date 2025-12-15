#!/usr/bin/env python
"""
run_scraping.py

Entrypoint for the Wikipedia scraping pipeline.

This script wraps the WikiScraper orchestrator, providing a simple CLI:
    URL_LIST → multiprocessing HTML fetch → JSONL outputs

Only coordination happens here; scraping logic lives in scraping_runner.py.
"""

import argparse
import logging
from dsan_5400_group3.scraping.scraping_runner import WikiScraper


def setup_logging():
    """Configure root logger for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Wikipedia Scraper")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    logging.info("Launching Wikipedia scraping pipeline")
    logging.info(f"Input URL file: {args.input}")
    logging.info(f"Workers: {args.workers}")
    logging.info(f"Output directory: {args.outdir}")

    scraper = WikiScraper(
        workers=args.workers,
        outdir=args.outdir,
    )
    scraper.run(args.input)

    logging.info("Scraping pipeline completed successfully")


if __name__ == "__main__":
    main()
