"""
Shared HTTP client for WazeCargo scrapers.
==========================================
Provides consistent request handling with retries, rate limiting, and logging.
"""

import requests
import time
import logging

logger = logging.getLogger(__name__)

# Default headers to mimic a browser
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Rate limiting
DEFAULT_DELAY = 2  # seconds between requests


class HttpClient:
    """HTTP client with retry logic and rate limiting."""
    
    def __init__(self, headers=None, delay=DEFAULT_DELAY):
        self.session = requests.Session()
        self.session.headers.update(headers or DEFAULT_HEADERS)
        self.delay = delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure minimum delay between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch(self, url, retries=3, timeout=30):
        """
        Fetch a URL with retry logic.
        
        Args:
            url: URL to fetch
            retries: Number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Response text if successful, None otherwise
        """
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{retries})")
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response.text
            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.HTTPError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                if response.status_code == 403:
                    logger.error("Access forbidden. May be blocked.")
                    return None
                if response.status_code == 404:
                    logger.error("Page not found.")
                    return None
            except requests.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            
            if attempt < retries - 1:
                wait_time = self.delay * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"All {retries} attempts failed for {url}")
        return None
    
    def fetch_json(self, url, retries=3, timeout=30):
        """
        Fetch a URL and parse as JSON.
        
        Returns:
            Parsed JSON if successful, None otherwise
        """
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                logger.info(f"Fetching JSON {url} (attempt {attempt + 1}/{retries})")
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            except ValueError as e:
                logger.warning(f"JSON parse failed on attempt {attempt + 1}: {e}")
            
            if attempt < retries - 1:
                time.sleep(self.delay * (attempt + 1))
        
        return None


# Convenience function for simple usage
def fetch_page(url, retries=3, delay=2):
    """
    Simple function to fetch a page.
    
    For repeated requests, use HttpClient class instead.
    """
    client = HttpClient(delay=delay)
    return client.fetch(url, retries=retries)
