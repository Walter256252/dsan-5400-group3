import time
import json
import requests
from urllib.parse import urlparse, parse_qs
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm import tqdm


# CONFIGURATION
API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIyNTAyMTFiNzQ5Y2U2MzM4OGY0YjE3MDgwMWNhMjAzYiIsImp0aSI6Ijc5NzIxNGMwZGQ5OGEzZWUxOTBjYWVlNjFhNzdiYWExNTg0MmUyY2MwMDFlMmNkYjY0MWZkZWFlM2FjYmUxM2M5OWIzMzc0MTY0MjFmMmE5IiwiaWF0IjoxNzY0MDE2NDA0Ljc2NzEwNCwibmJmIjoxNzY0MDE2NDA0Ljc2NzEwNiwiZXhwIjozMzMyMDkyNTIwNC43NjQ3MTMsInN1YiI6IjgwNzU1ODczIiwiaXNzIjoiaHR0cHM6Ly9tZXRhLndpa2ltZWRpYS5vcmciLCJyYXRlbGltaXQiOnsicmVxdWVzdHNfcGVyX3VuaXQiOjUwMDAsInVuaXQiOiJIT1VSIn0sInNjb3BlcyI6WyJiYXNpYyJdfQ.ucp5uOkdKU0Y9xlkpSsCPSWSow_RviXxWuI1hxL9bYe9p3GD4gQmqVm8PbktQ1x7FDOC9mg29IlWUbXpEMKr2HjQvEa4W4egPc1g2EYZMU_v_RwrkqnL1kIfmqUYytGvQ0C05YV70Z-aW_ZvV79sGPtPH3wM9j8to5ql9fZr4GjK-Z6o0o02xsukhJKFHv7_vIHe4jCy2Zcihb4JKSPGGP2INplBdLyglpdvCFzp0KGsIDudXPox5eLyJvx-_ave6ArGgQaLQ0nGPZfJNPT5ntMtopkXvom7VjZsSdeQo_MYJGOxB6vtYdS8gUKA3l_5bAzbz4KlbrPNBXse3U55BgXI_XZ4Ox3h52d4mZ5Rv5lfnnQE07a6dCkp4825Hy3PNo_S5Gc1cy9lXm_5S5Jk23GhANJ23n1gUFHSHaDy1ST2yzGr5YwyEwwoiw7LQwENFz8gOHdqDJaGkusJ8KoCzbhr-EhNkpDQlb-ZVk13Jv4WZdM6KdWGF80o1RO_Vv5ZaudeDi5ybzt68WfeWYuaOHMS4XhDDaivwOcowE6XtTUWgbxMHe0xUky3ukUFwbyl3FCX0w1d7q1uJA-Fxm7zAlnV6Lx19dxuro26w4PgRoZcH-3-_3FtiQYobW7Ob9FJZZwgRsPI_dUM4_3VM_3I4XZzbYHTYtgt8xhUr8iXb5A"

MAX_REQUESTS_PER_HOUR = 5000   # Wikipedia personal API token limit

# Retry for transient errors
retry_strategy = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
session.headers.update({
    "Authorization": f"Bearer {API_TOKEN}",
    "User-Agent": "DSAN5400 (University project, contact if issues)"
})


# Helpers
def extract_pageid_from_url(url: str):
    """
    Extract page ID from URLs of the form:
       https://en.wikipedia.org/wiki/index.php?curid=1264090
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "curid" in qs:
        return qs["curid"][0].strip()
    return None  # Unexpected format — caller handles this


def fetch_page_text_parse(pageid: str):
    """
    Fetch page content using action=parse (most reliable).
    Returns dict containing:
    {
        "pageid": ...,
        "title": ...,
        "missing": True/False,
        "text": ...
    }
    """
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

    # API error example: {"error": {...}}
    if "error" in data:
        return {
            "pageid": pageid,
            "title": None,
            "missing": True,
            "text": None,
            "error": data["error"].get("info", "Unknown error")
        }

    # Extract HTML
    html = data["parse"]["text"]
    title = data["parse"]["title"]

    # Convert HTML → plain text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n").strip()

    return {
        "pageid": pageid,
        "title": title,
        "missing": False,
        "text": text
    }



# Main scraping process
def scrape_wikipedia(url_list, output_path="../wikipedia_output.jsonl"):
    requests_made = 0
    hour_start = time.time()

    # Extract pageids
    pageids = []
    for u in url_list:
        pid = extract_pageid_from_url(u)
        if pid:
            pageids.append(pid)
        else:
            print(f"Warning: Could not extract pageid from URL: {u}")

    print(f"Loaded {len(pageids)} valid page IDs.")

    with open(output_path, "w", encoding="utf-8") as f_out:

        for pageid in tqdm(pageids, desc="Scraping pages"):

            # Rate limit enforcement: max 5000 requests per hour
            if requests_made >= MAX_REQUESTS_PER_HOUR:
                elapsed = time.time() - hour_start
                wait_time = max(0, 3600 - elapsed)
                print(f"\nRate limit reached — sleeping {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                hour_start = time.time()
                requests_made = 0

            # Retry logic is handled by session mount
            try:
                result = fetch_page_text_parse(pageid)

            except Exception as e:
                # On error, log this page as failed
                result = {
                    "pageid": pageid,
                    "title": None,
                    "missing": True,
                    "text": None,
                    "error": str(e)
                }

            # Write JSONL line
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            requests_made += 1

            # Light sleep (optional — helps avoid token-based throttling)
            time.sleep(0.05)

    print("Done scraping!")

if __name__ == "__main__":
    with open("../biography_urls.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    scrape_wikipedia(urls)


# Usage in CLI (run in scripts folder)
# python download_data.py