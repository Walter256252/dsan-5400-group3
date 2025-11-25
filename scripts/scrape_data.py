import time
import json
import math
import argparse
from multiprocessing import Process
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry


# Global API configuration
API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
MAX_REQUESTS_PER_HOUR = 5000   # total across ALL processes

# Create retry strategy once
retry_strategy = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)


def extract_pageid_from_url(url: str):
    """Extract pageid from ?curid=NNNN Wikipedia URLs."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "curid" in qs:
        return qs["curid"][0].strip()
    return None


def fetch_page_text_parse(session, pageid: str):
    """Use action=parse to fetch HTML → convert to plaintext."""
    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "text",
        "format": "json",
        "formatversion": "2"
    }

    resp = session.get(API_ENDPOINT, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        return {
            "pageid": pageid,
            "title": None,
            "missing": True,
            "text": None,
            "error": data["error"].get("info")
        }

    html = data["parse"]["text"]
    title = data["parse"]["title"]

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n").strip()

    return {
        "pageid": pageid,
        "title": title,
        "missing": False,
        "text": text
    }


def worker_process(worker_id, pageid_list, output_path, api_token, total_workers):
    """
    Runs inside each worker:
    - rate limit = MAX_REQUESTS_PER_HOUR / total_workers
    - stagger workers slightly to avoid sync bursts
    """
    # Each worker has its own HTTP session
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    session.headers.update({
        "Authorization": f"Bearer {api_token}",
        "User-Agent": f"DSAN5400-worker-{worker_id}"
    })

    # Worker-specific request budget
    worker_limit = MAX_REQUESTS_PER_HOUR / total_workers
    SLEEP_PER_REQUEST = 3600.0 / MAX_REQUESTS_PER_HOUR  # global ~0.72 sec

    # Stagger workers: worker 0 starts immediately, worker 1 waits 0.1 sec, etc.
    time.sleep(worker_id * 0.1)

    print(f"[Worker {worker_id}] Starting with {len(pageid_list)} pages…")
    print(f"[Worker {worker_id}] Requests/hour limit: {worker_limit:.1f}")

    requests_made = 0
    hour_start = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for pid in tqdm(pageid_list, desc=f"Worker {worker_id}", position=worker_id):

            # Per-worker rate limit
            if requests_made >= worker_limit:
                elapsed = time.time() - hour_start
                wait_time = max(0, 3600 - elapsed)
                print(f"\n[Worker {worker_id}] Hourly limit reached — sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                hour_start = time.time()
                requests_made = 0

            try:
                result = fetch_page_text_parse(session, pid)
            except Exception as e:
                result = {
                    "pageid": pid,
                    "title": None,
                    "missing": True,
                    "text": None,
                    "error": str(e)
                }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            requests_made += 1

            time.sleep(SLEEP_PER_REQUEST)  # small throttle

    print(f"[Worker {worker_id}] DONE.")


def split_into_chunks(lst, n):
    """Split list into n chunks for worker assignment."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--token", required=True)
    parser.add_argument("--outdir", default="./output/")
    args = parser.parse_args()

    # Load URLs
    with open(args.input, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Extract pageids
    pageids = [extract_pageid_from_url(u) for u in urls]
    pageids = [p for p in pageids if p is not None]

    print(f"Loaded {len(pageids)} valid page IDs.")
    print(f"Using {args.workers} worker processes.")

    # Split into N chunks
    chunks = split_into_chunks(pageids, args.workers)

    procs = []
    for i, chunk in enumerate(chunks):
        out_path = f"{args.outdir}/part_{i}.jsonl"
        p = Process(
            target=worker_process,
            args=(i, chunk, out_path, args.token, args.workers)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\nAll workers finished. Combine output files if needed.")


if __name__ == "__main__":
    main()

# Usage
# python scrape_parallel.py \
#     --input ../biography_urls.txt \
#     --workers 5 \
#     --token YOUR_API_TOKEN \
#     --outdir output
