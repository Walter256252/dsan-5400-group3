#!/usr/bin/env python
"""
run_scraping.py

Entrypoint for the Wikipedia scraping pipeline.

This script wraps the WikiScraper orchestrator, providing a simple CLI:
    URL_LIST → multiprocessing HTML fetch → JSONL outputs

Arguments
---------
--input:
    Path to a text file containing Wikipedia biography URLs.
    Expected format: one URL per line; each should include ?curid=...

--workers:
    Number of parallel processes to use for scraping.
    More workers = faster scraping, but higher load on Wikipedia servers.

--outdir:
    Directory in which worker output files (part_0.jsonl, part_1.jsonl, ...)
    will be written.

Only coordination happens here; scraping logic lives in scraping_runner.py.
"""

import argparse
from dsan_5400_group3.scraping.scraping_runner import WikiScraper

def main():
    parser = argparse.ArgumentParser(description="Wikipedia Scraper")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    scraper = WikiScraper(
        workers=args.workers,
        outdir=args.outdir,
    )
    scraper.run(args.input)

if __name__ == "__main__":
    main()
