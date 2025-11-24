import requests
import time
import json
from urllib.parse import urlparse, parse_qs
from pathlib import Path

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Insert your API token here
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIyNTAyMTFiNzQ5Y2U2MzM4OGY0YjE3MDgwMWNhMjAzYiIsImp0aSI6Ijc5NzIxNGMwZGQ5OGEzZWUxOTBjYWVlNjFhNzdiYWExNTg0MmUyY2MwMDFlMmNkYjY0MWZkZWFlM2FjYmUxM2M5OWIzMzc0MTY0MjFmMmE5IiwiaWF0IjoxNzY0MDE2NDA0Ljc2NzEwNCwibmJmIjoxNzY0MDE2NDA0Ljc2NzEwNiwiZXhwIjozMzMyMDkyNTIwNC43NjQ3MTMsInN1YiI6IjgwNzU1ODczIiwiaXNzIjoiaHR0cHM6Ly9tZXRhLndpa2ltZWRpYS5vcmciLCJyYXRlbGltaXQiOnsicmVxdWVzdHNfcGVyX3VuaXQiOjUwMDAsInVuaXQiOiJIT1VSIn0sInNjb3BlcyI6WyJiYXNpYyJdfQ.ucp5uOkdKU0Y9xlkpSsCPSWSow_RviXxWuI1hxL9bYe9p3GD4gQmqVm8PbktQ1x7FDOC9mg29IlWUbXpEMKr2HjQvEa4W4egPc1g2EYZMU_v_RwrkqnL1kIfmqUYytGvQ0C05YV70Z-aW_ZvV79sGPtPH3wM9j8to5ql9fZr4GjK-Z6o0o02xsukhJKFHv7_vIHe4jCy2Zcihb4JKSPGGP2INplBdLyglpdvCFzp0KGsIDudXPox5eLyJvx-_ave6ArGgQaLQ0nGPZfJNPT5ntMtopkXvom7VjZsSdeQo_MYJGOxB6vtYdS8gUKA3l_5bAzbz4KlbrPNBXse3U55BgXI_XZ4Ox3h52d4mZ5Rv5lfnnQE07a6dCkp4825Hy3PNo_S5Gc1cy9lXm_5S5Jk23GhANJ23n1gUFHSHaDy1ST2yzGr5YwyEwwoiw7LQwENFz8gOHdqDJaGkusJ8KoCzbhr-EhNkpDQlb-ZVk13Jv4WZdM6KdWGF80o1RO_Vv5ZaudeDi5ybzt68WfeWYuaOHMS4XhDDaivwOcowE6XtTUWgbxMHe0xUky3ukUFwbyl3FCX0w1d7q1uJA-Fxm7zAlnV6Lx19dxuro26w4PgRoZcH-3-_3FtiQYobW7Ob9FJZZwgRsPI_dUM4_3VM_3I4XZzbYHTYtgt8xhUr8iXb5A"

# Wikipedia API limit for authenticated users = 5000 req/hr
REQUESTS_PER_HOUR = 5000
SECONDS_PER_REQUEST = 3600 / REQUESTS_PER_HOUR  # ≈ 0.72 seconds per request
BATCH_SIZE = 50  # Max allowed by MediaWiki

OUTPUT_FILE = "wikipedia_pages.jsonl"


def extract_pageid_from_url(url: str) -> str:
    """Extract curid=#### from Wikipedia URL."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "curid" in qs:
        return qs["curid"][0]
    else:
        return None


def fetch_batch(page_ids):
    """Fetch up to 50 pages using pageids batching."""
    params = {
        "action": "query",
        "pageids": "|".join(page_ids),
        "prop": "extracts",
        "explaintext": 1,       # get plain text version
        "format": "json"
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "User-Agent": "MyResearchScraper/1.0"
    }

    response = requests.get(API_ENDPOINT, params=params, headers=headers)
    response.raise_for_status()  # crash early on API errors
    return response.json()


def scrape_all(url_list):
    Path(OUTPUT_FILE).unlink(missing_ok=True)

    all_page_ids = [extract_pageid_from_url(url) for url in url_list]
    all_page_ids = [pid for pid in all_page_ids if pid is not None]

    print(f"Total valid page IDs: {len(all_page_ids)}")

    with open(OUTPUT_FILE, "a", encoding="utf8") as f:

        for i in range(0, len(all_page_ids), BATCH_SIZE):
            batch = all_page_ids[i : i + BATCH_SIZE]
            print(f"Fetching batch {i // BATCH_SIZE + 1} / {len(all_page_ids) // BATCH_SIZE + 1} ...")

            try:
                data = fetch_batch(batch)
            except Exception as e:
                print(f"Error fetching batch: {e}")
                time.sleep(2)
                continue

            pages = data.get("query", {}).get("pages", {})

            for pid, page in pages.items():
                out = {
                    "pageid": pid,
                    "title": page.get("title", ""),
                    "missing": "missing" in page,
                    "text": page.get("extract", "")  # FINAL FIX: correct key for plaintext
                }
                f.write(json.dumps(out) + "\n")

            # Respect rate limit (≈1 request/0.72s)
            time.sleep(SECONDS_PER_REQUEST)

    print("Done scraping.")


if __name__ == "__main__":

    with open("../biography_urls.txt") as f:
        urls = [line.strip() for line in f if line.strip()]

    scrape_all(urls)
