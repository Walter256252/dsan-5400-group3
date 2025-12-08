"""
utils.py

Utility functions and shared objects for the scraping pipeline, including
URL pageid extraction and the retry strategy used by all HTTP requests.
"""

from urllib.parse import urlparse, parse_qs
from requests.adapters import Retry

# Same retry strategy as original code
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
