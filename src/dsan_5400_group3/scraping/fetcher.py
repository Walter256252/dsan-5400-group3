"""
fetcher.py

Contains low-level HTML fetching and parsing logic for individual Wikipedia
pages. Given a page ID and a configured requests session, it returns the
extracted title and article text.
"""

from bs4 import BeautifulSoup

def fetch_html_page(session, pageid: str):
    """Scrape the wiki HTML page and extract text."""
    url = f"https://en.wikipedia.org/wiki/index.php?curid={pageid}"

    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
    except Exception as e:
        return {
            "pageid": pageid,
            "title": None,
            "missing": True,
            "text": None,
            "error": str(e)
        }

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("h1", id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Extract main article content
    content = soup.find("div", id="mw-content-text")
    text = content.get_text(separator="\n").strip() if content else ""

    return {
        "pageid": pageid,
        "title": title,
        "missing": not bool(text),
        "text": text
    }