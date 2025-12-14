"""
Tests for scraping module.

Tests cover:
- URL parsing and pageid extraction
- HTML fetching (with mocks)
- Text extraction from BeautifulSoup
- Error handling
- Retry strategy
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from dsan_5400_group3.scraping.utils import (
    extract_pageid_from_url,
    retry_strategy
)
from dsan_5400_group3.scraping.fetcher import fetch_html_page


class TestURLUtils:
    """Test URL utility functions."""
    
    def test_extract_pageid_from_url_basic(self):
        """Test extracting pageid from standard Wikipedia URL."""
        url = "https://en.wikipedia.org/wiki/Albert_Einstein?curid=736"
        pageid = extract_pageid_from_url(url)
        assert pageid == "736"
    
    def test_extract_pageid_from_url_with_title(self):
        """Test extraction with article title in URL."""
        url = "https://en.wikipedia.org/wiki/Marie_Curie?curid=20094"
        pageid = extract_pageid_from_url(url)
        assert pageid == "20094"
    
    def test_extract_pageid_from_url_no_curid(self):
        """Test handling URL without curid parameter."""
        url = "https://en.wikipedia.org/wiki/Albert_Einstein"
        pageid = extract_pageid_from_url(url)
        assert pageid is None
    
    def test_extract_pageid_from_url_multiple_params(self):
        """Test extraction when URL has multiple query parameters."""
        url = "https://en.wikipedia.org/wiki/Test?foo=bar&curid=123&baz=qux"
        pageid = extract_pageid_from_url(url)
        assert pageid == "123"
    
    def test_extract_pageid_from_url_with_whitespace(self):
        """Test that whitespace is stripped from pageid."""
        url = "https://en.wikipedia.org/wiki/Test?curid= 456 "
        pageid = extract_pageid_from_url(url)
        assert pageid == "456"
    
    def test_extract_pageid_from_url_empty(self):
        """Test handling empty URL."""
        url = ""
        pageid = extract_pageid_from_url(url)
        assert pageid is None
    
    def test_extract_pageid_from_url_invalid(self):
        """Test handling invalid URL."""
        url = "not a valid url"
        pageid = extract_pageid_from_url(url)
        assert pageid is None


class TestRetryStrategy:
    """Test retry configuration."""
    
    def test_retry_strategy_exists(self):
        """Test that retry_strategy is configured."""
        assert retry_strategy is not None
        assert retry_strategy.total == 5
        assert retry_strategy.backoff_factor == 1.5
    
    def test_retry_strategy_status_codes(self):
        """Test that retry strategy handles correct status codes."""
        expected_codes = [429, 500, 502, 503, 504]
        assert retry_strategy.status_forcelist == expected_codes
    
    def test_retry_strategy_allowed_methods(self):
        """Test that retry strategy allows GET requests."""
        assert "GET" in retry_strategy.allowed_methods


class TestWikiFetcher:
    """Test Wikipedia HTML fetching functionality."""
    
    @pytest.fixture
    def mock_successful_response(self):
        """Create a mock successful HTTP response."""
        mock = Mock()
        mock.status_code = 200
        mock.text = """
        <html>
        <head><title>Albert Einstein</title></head>
        <body>
            <h1 id="firstHeading">Albert Einstein</h1>
            <div id="mw-content-text">
                <p>Albert Einstein was a German-born theoretical physicist.</p>
                <p>He developed the theory of relativity.</p>
                <p>Einstein received the Nobel Prize in Physics in 1921.</p>
            </div>
        </body>
        </html>
        """
        mock.raise_for_status = Mock()
        return mock
    
    @pytest.fixture
    def mock_session(self, mock_successful_response):
        """Create a mock requests session."""
        session = Mock()
        session.get = Mock(return_value=mock_successful_response)
        return session
    
    def test_fetch_html_page_success(self, mock_session):
        """Test successful page fetch."""
        result = fetch_html_page(mock_session, pageid="736")
        
        assert result is not None
        assert result["pageid"] == "736"
        assert result["title"] == "Albert Einstein"
        assert result["missing"] is False
        assert "Albert Einstein" in result["text"]
        assert "theoretical physicist" in result["text"]
        
        # Should not have error field on success
        assert "error" not in result or result["error"] is None
    
    def test_fetch_html_page_constructs_correct_url(self, mock_session):
        """Test that correct URL is constructed."""
        fetch_html_page(mock_session, pageid="12345")
        
        # Verify the URL passed to session.get
        call_args = mock_session.get.call_args
        url = call_args[0][0]
        
        assert "curid=12345" in url
        assert "en.wikipedia.org" in url
    
    def test_fetch_html_page_timeout_parameter(self, mock_session):
        """Test that timeout parameter is passed."""
        fetch_html_page(mock_session, pageid="123")
        
        call_kwargs = mock_session.get.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 20
    
    def test_fetch_html_page_http_error(self):
        """Test handling of HTTP errors (404, 500, etc)."""
        session = Mock()
        session.get.side_effect = Exception("404 Not Found")
        
        result = fetch_html_page(session, pageid="999999")
        
        assert result["pageid"] == "999999"
        assert result["missing"] is True
        assert result["title"] is None
        assert result["text"] is None
        assert "error" in result
        assert "404" in result["error"]
    
    def test_fetch_html_page_timeout(self):
        """Test handling of timeout errors."""
        session = Mock()
        session.get.side_effect = Exception("Request timed out")
        
        result = fetch_html_page(session, pageid="123")
        
        assert result["missing"] is True
        assert "error" in result
        assert "timed out" in result["error"].lower()
    
    def test_fetch_html_page_connection_error(self):
        """Test handling of connection errors."""
        session = Mock()
        session.get.side_effect = Exception("Connection refused")
        
        result = fetch_html_page(session, pageid="456")
        
        assert result["missing"] is True
        assert "error" in result
    
    def test_fetch_html_page_missing_title(self):
        """Test handling when page has no title."""
        session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><div id='mw-content-text'>Content</div></body></html>"
        mock_response.raise_for_status = Mock()
        session.get.return_value = mock_response
        
        result = fetch_html_page(session, pageid="789")
        
        assert result["title"] is None
        assert result["text"] == "Content"
    
    def test_fetch_html_page_missing_content(self):
        """Test handling when page has no main content."""
        session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1 id='firstHeading'>Title</h1></body></html>"
        mock_response.raise_for_status = Mock()
        session.get.return_value = mock_response
        
        result = fetch_html_page(session, pageid="101")
        
        assert result["title"] == "Title"
        assert result["text"] == ""
        assert result["missing"] is True  # Empty text counts as missing
    
    def test_fetch_html_page_extracts_text_only(self):
        """Test that only text is extracted, not HTML tags."""
        session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <h1 id="firstHeading">Test</h1>
        <div id="mw-content-text">
            <p>Paragraph <b>with bold</b> and <i>italic</i>.</p>
        </div>
        </html>
        """
        mock_response.raise_for_status = Mock()
        session.get.return_value = mock_response
        
        result = fetch_html_page(session, pageid="202")

        cleaned_result = " ".join(result["text"].split())
        # Should extract text without HTML tags
        assert "Paragraph with bold and italic" in cleaned_result
        assert "<b>" not in result["text"]
        assert "<i>" not in result["text"]
    
    def test_fetch_html_page_multiple_paragraphs(self):
        """Test extraction of multiple paragraphs."""
        session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <h1 id="firstHeading">Article</h1>
        <div id="mw-content-text">
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
            <p>Third paragraph.</p>
        </div>
        </html>
        """
        mock_response.raise_for_status = Mock()
        session.get.return_value = mock_response
        
        result = fetch_html_page(session, pageid="303")
        
        assert "First paragraph" in result["text"]
        assert "Second paragraph" in result["text"]
        assert "Third paragraph" in result["text"]


class TestScrapingIntegration:
    """Integration tests for scraping workflow."""
    
    def test_scrape_multiple_pages(self):
        """Test scraping multiple pages in sequence."""
        session = Mock()
        
        # Create different responses for different pageids
        def mock_get(url, timeout):
            if "curid=1" in url:
                response = Mock()
                response.status_code = 200
                response.text = "<html><h1 id='firstHeading'>Page 1</h1><div id='mw-content-text'>Content 1</div></html>"
                response.raise_for_status = Mock()
                return response
            elif "curid=2" in url:
                response = Mock()
                response.status_code = 200
                response.text = "<html><h1 id='firstHeading'>Page 2</h1><div id='mw-content-text'>Content 2</div></html>"
                response.raise_for_status = Mock()
                return response
            else:
                raise Exception("Not found")
        
        session.get = mock_get
        
        # Fetch multiple pages
        results = [
            fetch_html_page(session, "1"),
            fetch_html_page(session, "2")
        ]
        
        assert len(results) == 2
        assert results[0]["title"] == "Page 1"
        assert results[1]["title"] == "Page 2"
        assert results[0]["text"] == "Content 1"
        assert results[1]["text"] == "Content 2"
    
    def test_url_to_pageid_to_fetch_pipeline(self):
        """Test complete pipeline from URL to fetched content."""
        # Step 1: Extract pageid from URL
        url = "https://en.wikipedia.org/wiki/Test_Article?curid=12345"
        pageid = extract_pageid_from_url(url)
        
        assert pageid == "12345"
        
        # Step 2: Fetch page with that pageid
        session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <h1 id="firstHeading">Test Article</h1>
        <div id="mw-content-text">This is the article content.</div>
        </html>
        """
        mock_response.raise_for_status = Mock()
        session.get.return_value = mock_response
        
        result = fetch_html_page(session, pageid)
        
        assert result["pageid"] == "12345"
        assert result["title"] == "Test Article"
        assert "article content" in result["text"]