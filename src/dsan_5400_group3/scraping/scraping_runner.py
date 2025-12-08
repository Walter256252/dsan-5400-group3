"""
scraper.py

Provides the high-level WikiScraper class that orchestrates the entire
Wikipedia scraping pipeline, including loading URLs, extracting page IDs,
splitting workloads, and coordinating multiprocessing workers.
"""

import os
import time
import json
from multiprocessing import Process
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter

from .utils import extract_pageid_from_url, retry_strategy
from .fetcher import fetch_html_page
from .splitter import split_into_chunks


class WikiScraper:
    """
    Orchestrates the scraping pipeline:
    - Load URLs
    - Extract pageids
    - Split workloads
    - Spawn multiprocessing worker processes
    """
    def __init__(
        self,
        workers=5,
        max_requests_per_hour=100000,
        outdir="output",
        user_agent_prefix="DSAN5400-wikiscraper"
    ):
        self.workers = workers
        self.max_requests_per_hour = max_requests_per_hour
        self.seconds_per_request = 3600 / max_requests_per_hour
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        self.user_agent_prefix = user_agent_prefix

    def load_urls(self, path):
        """Reads input URLs & extracts pageids."""
        with open(path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        pageids = [extract_pageid_from_url(u) for u in urls]
        return [pid for pid in pageids if pid is not None]

    def _make_session(self, worker_id):
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        session.headers.update({
            "User-Agent": f"{self.user_agent_prefix}/worker-{worker_id}"
        })
        return session

    def _worker(self, worker_id, pageids):
        """Worker logic (same as original worker_process)."""

        session = self._make_session(worker_id)
        worker_limit = self.max_requests_per_hour / self.workers
        sleep_per_request = self.seconds_per_request

        # stagger to avoid bursts
        time.sleep(worker_id * 0.1)

        print(f"[Worker {worker_id}] Starting {len(pageids)} pages...")
        print(f"[Worker {worker_id}] Hourly limit: {worker_limit:.1f}")

        requests_made = 0
        hour_start = time.time()
        output_path = os.path.join(self.outdir, f"part_{worker_id}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f_out:

            for pid in tqdm(pageids, desc=f"Worker {worker_id}", position=worker_id):

                # enforce per-worker rate limit
                if requests_made >= worker_limit:
                    elapsed = time.time() - hour_start
                    wait = max(0, 3600 - elapsed)
                    print(f"\n[Worker {worker_id}] Limit reached — sleeping {wait:.1f}s")
                    time.sleep(wait)
                    hour_start = time.time()
                    requests_made = 0

                # scrape page
                result = fetch_html_page(session, pid)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

                requests_made += 1
                time.sleep(sleep_per_request)

        print(f"[Worker {worker_id}] DONE.")

    def run(self, input_path):
        """Main entry point for the scraper."""
        pageids = self.load_urls(input_path)

        print(f"Loaded {len(pageids)} valid page IDs.")
        print(f"Using {self.workers} workers.")

        chunks = split_into_chunks(pageids, self.workers)
        processes = []

        for worker_id, chunk in enumerate(chunks):
            p = Process(target=self._worker, args=(worker_id, chunk))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All workers finished.")