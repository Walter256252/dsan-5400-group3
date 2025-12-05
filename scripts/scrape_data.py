"""
Summary of this script

* Reads a list of Wikipedia URLs (bibiography_urls.txt) containing ?curid=###.
* Extracts the numeric pageid.
* Splits the list into N workers (multiprocessing).
* Each worker:
    * Scrapes the article HTML page.
    * Extracts the article title and body text with BeautifulSoup.
    * Saves each result as one JSONL line.
    * Enforces a rate limit so system don't overload.
* Writes results into separate output JSONL files

Example usage:
python scrape_data.py \
    --input ../biography_urls.txt \
    --workers 5 \
    --outdir ../output
"""

import time
import json
import argparse
from multiprocessing import Process
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry

# Total allowed request rate across all workers
MAX_REQUESTS_PER_HOUR = 100000
SECONDS_PER_REQUEST = 3600 / MAX_REQUESTS_PER_HOUR  

# Retry strategy for requests
retry_strategy = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)


def extract_pageid_from_url(url: str):
    """Extract pageid from ?curid=XXXX format."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "curid" in qs:
        return qs["curid"][0].strip()
    return None


def fetch_html_page(session, pageid: str):
    """Scrape the wiki HTML page and extract text."""
    # Construct URL
    url = f"https://en.wikipedia.org/wiki/index.php?curid={pageid}"

    try:
        # Fetch page
        response = session.get(url, timeout=20)
        response.raise_for_status()
    except Exception as e:
        # Return empty JSON object
        return {
            "pageid": pageid,
            "title": None,
            "missing": True,
            "text": None,
            "error": str(e)
        }

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("h1", id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Extract main article content
    content = soup.find("div", id="mw-content-text")
    if content:
        text = content.get_text(separator="\n").strip()
    else:
        text = ""

    # Build JSON object
    return {
        "pageid": pageid,
        "title": title,
        "missing": False if text else True,
        "text": text
    }

# This runs in parallel for each chunk of page IDs
def worker_process(worker_id, pageids, output_path, total_workers):
    """
    Worker logic:
    - Rate limit enforced: MAX_REQUESTS_PER_HOUR / total_workers
    - Slight staggering to avoid synchronized bursts
    """
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    session.headers.update({
        "User-Agent": f"DSAN5400-wikiscraper/worker-{worker_id}"
    })

    worker_limit = MAX_REQUESTS_PER_HOUR / total_workers
    sleep_per_request = SECONDS_PER_REQUEST

    # Stagger start
    time.sleep(worker_id * 0.1)

    print(f"[Worker {worker_id}] Starting with {len(pageids)} pages...")
    print(f"[Worker {worker_id}] Hourly limit: {worker_limit:.1f} requests/hr")

    requests_made = 0
    hour_start = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        # Loop over page IDs
        for pid in tqdm(pageids, desc=f"Worker {worker_id}", position=worker_id):

            # Enforce hourly limit per worker
            if requests_made >= worker_limit:
                elapsed = time.time() - hour_start
                wait = max(0, 3600 - elapsed)
                print(f"\n[Worker {worker_id}] Limit reached — sleeping {wait:.1f}s")
                time.sleep(wait)
                hour_start = time.time()
                requests_made = 0

            # Scrape page
            result = fetch_html_page(session, pid)
            # Save each result to a JSONL line
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            requests_made += 1
            time.sleep(sleep_per_request)

    print(f"[Worker {worker_id}] DONE.")


def split_into_chunks(list_of_urls, n_workers):
    """Split list of URLs into n roughly equal chunks."""
    k, m = divmod(len(list_of_urls), n_workers)
    return [
        list_of_urls[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n_workers)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--outdir", default="../output/")
    args = parser.parse_args()

    # Load input file and extract clean URLs
    with open(args.input, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Get page IDs
    pageids = [extract_pageid_from_url(u) for u in urls]
    pageids = [p for p in pageids if p is not None]

    print(f"Loaded {len(pageids)} valid page IDs.")
    print(f"Using {args.workers} workers.")

    # Split into worker chunks
    chunks = split_into_chunks(pageids, args.workers)

    processes = []
    for i, chunk in enumerate(chunks):
        out_path = f"{args.outdir}/part_{i}.jsonl"
        # Create separate Python process
        p = Process(target=worker_process, args=(i, chunk, out_path, args.workers))
        # Start worker process
        p.start()
        processes.append(p)

    for p in processes:
        # Waits for each worker to finish
        p.join()

    print("\nAll workers finished.")


if __name__ == "__main__":
    main()

