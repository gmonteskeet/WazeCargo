"""
Scraper Template for WazeCargo
==============================
This template shows the structure needed for scrapers to work with GitHub Actions.

Usage:
    python scrapers/template_scraper.py

Output:
    Saves JSON file to data/raw/{source_name}/YYYY-MM-DD_HH-MM.json
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
import time
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_NAME = "template"  # Change this: directemar, san_antonio, valparaiso, vesselfinder
BASE_URL = "https://example.com"  # Change this to actual URL
OUTPUT_DIR = f"data/raw/{SOURCE_NAME}"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# SCRAPER CLASS
# =============================================================================

class Scraper:
    """Base scraper with retry logic and error handling."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
    
    def fetch_page(self, url, retries=3, delay=2):
        """Fetch a page with retry logic."""
        for attempt in range(retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{retries})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {retries} attempts failed for {url}")
                    raise
        return None
    
    def parse_data(self, html_content):
        """
        Parse the HTML and extract data.
        
        Override this method for each specific scraper.
        Returns a list of dictionaries.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Example: Extract table rows
        # Customize this for each website
        data = []
        
        # Example parsing logic (replace with actual logic):
        # table = soup.find("table", {"class": "data-table"})
        # if table:
        #     rows = table.find_all("tr")[1:]  # Skip header
        #     for row in rows:
        #         cells = row.find_all("td")
        #         if len(cells) >= 3:
        #             data.append({
        #                 "vessel_name": cells[0].get_text(strip=True),
        #                 "eta": cells[1].get_text(strip=True),
        #                 "port": cells[2].get_text(strip=True),
        #             })
        
        # Placeholder data for template
        data.append({
            "message": "Replace this with actual parsing logic",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return data
    
    def scrape(self):
        """Main scrape method. Returns parsed data."""
        response = self.fetch_page(BASE_URL)
        if response:
            return self.parse_data(response.text)
        return []


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")


def save_data(data):
    """Save data to JSON file with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    filename = f"{OUTPUT_DIR}/{timestamp}.json"
    
    output = {
        "source": SOURCE_NAME,
        "scrape_timestamp": datetime.utcnow().isoformat() + "Z",
        "record_count": len(data),
        "data": data
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} records to {filename}")
    return filename


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    logger.info(f"Starting {SOURCE_NAME} scraper")
    
    try:
        # Setup
        ensure_output_dir()
        
        # Scrape
        scraper = Scraper()
        data = scraper.scrape()
        
        # Save
        if data:
            save_data(data)
            logger.info(f"Scraping completed successfully: {len(data)} records")
        else:
            logger.warning("No data scraped")
            # Still save empty result to track that scraper ran
            save_data([])
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        # Don't raise, so GitHub Actions continues with other scrapers
        # If you want the workflow to fail, uncomment the next line:
        # raise


if __name__ == "__main__":
    main()
