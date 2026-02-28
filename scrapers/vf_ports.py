"""
VesselFinder Port Scraper
=========================
Scrapes port status (expected ships, in port, arrivals, departures).
Runs automatically via GitHub Actions every 15 minutes.

Usage:
    python scrapers/ports/vesselfinder_ports.py
    
Output:
    data/raw/vesselfinder/ports_YYYY-MM-DD_HH-MM.json
"""

import sys
import os
import re
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.http_client import HttpClient

# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_NAME = "vesselfinder"
OUTPUT_DIR = "data/raw/vesselfinder"

# Chilean ports to monitor
PORTS = {
    "CLSAI001": "San Antonio",
    "CLVAP001": "Valparaíso",
}

BASE_URL = "https://www.vesselfinder.com/ports"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PARSER FUNCTIONS
# =============================================================================

def parse_port_summary(soup):
    """Extract expected ships and ships in port counts."""
    summary = {
        "expected_ships": None,
        "ships_in_port": None
    }
    
    pei_div = soup.find("div", class_="pei")
    if pei_div:
        text = pei_div.get_text()
        
        expected_match = re.search(r"Expected ships:\s*(\d+)", text)
        in_port_match = re.search(r"Ships in port:\s*(\d+)", text)
        
        if expected_match:
            summary["expected_ships"] = int(expected_match.group(1))
        if in_port_match:
            summary["ships_in_port"] = int(in_port_match.group(1))
    
    return summary


def parse_vessel_row(row):
    """Parse a single vessel row from the table."""
    cells = row.find_all("td")
    if len(cells) < 2:
        return None
    
    vessel = {}
    
    # First column: time (ETA, arrival, departure, or last report)
    vessel["time"] = cells[0].get_text(strip=True)
    
    # Second column: vessel info
    vessel_cell = cells[1]
    
    # Vessel name
    named_title = vessel_cell.find("div", class_="named-title")
    if named_title:
        vessel["vessel_name"] = named_title.get_text(strip=True)
    
    # Vessel type
    named_subtitle = vessel_cell.find("div", class_="named-subtitle")
    if named_subtitle:
        vessel["vessel_type"] = named_subtitle.get_text(strip=True)
    
    # IMO from link
    vessel_link = vessel_cell.find("a", class_="named-item")
    if vessel_link and vessel_link.get("href"):
        href = vessel_link.get("href")
        imo_match = re.search(r"/details/(\d+)", href)
        if imo_match:
            vessel["imo"] = imo_match.group(1)
    
    # Flag from image title
    flag_div = vessel_cell.find("div", class_="m-flag-small")
    if flag_div and flag_div.get("title"):
        vessel["flag"] = flag_div.get("title")
    
    # Additional columns
    for cell in cells:
        classes = cell.get("class", [])
        text = cell.get_text(strip=True)
        
        if "col-y" in classes and text and text != "-":
            try:
                vessel["built"] = int(text)
            except ValueError:
                pass
        
        if "col-gt" in classes and text and text != "-":
            try:
                vessel["gt"] = int(text.replace(",", ""))
            except ValueError:
                pass
        
        if "col-dwt" in classes and text and text != "-":
            try:
                vessel["dwt"] = int(text.replace(",", ""))
            except ValueError:
                pass
        
        if "col-sizes" in classes and text and text != "-":
            vessel["size"] = text
    
    return vessel if vessel.get("vessel_name") else None


def parse_vessel_table(soup, table_id):
    """Parse a vessel table (expected, arrivals, departures, in-port)."""
    vessels = []
    
    section = soup.find("section", id=table_id)
    if not section:
        return vessels
    
    table = section.find("table")
    if not table:
        return vessels
    
    tbody = table.find("tbody")
    if not tbody:
        return vessels
    
    for row in tbody.find_all("tr"):
        vessel = parse_vessel_row(row)
        if vessel:
            vessels.append(vessel)
    
    return vessels


# =============================================================================
# SCRAPER CLASS
# =============================================================================

class VesselFinderPortScraper:
    """Scrapes port status from VesselFinder."""
    
    def __init__(self):
        self.client = HttpClient(delay=2)
    
    def scrape_port(self, port_code, port_name):
        """Scrape all data for a single port."""
        url = f"{BASE_URL}/{port_code}"
        html = self.client.fetch(url)
        
        if not html:
            logger.error(f"Failed to fetch {port_name}")
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        data = {
            "source": SOURCE_NAME,
            "port_code": port_code.replace("001", ""),  # CLSAI001 -> CLSAI
            "port_name": port_name,
            "scrape_timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": parse_port_summary(soup),
            "expected_vessels": parse_vessel_table(soup, "expected"),
            "vessels_in_port": parse_vessel_table(soup, "in-port"),
            "recent_arrivals": parse_vessel_table(soup, "arrivals"),
            "recent_departures": parse_vessel_table(soup, "departures"),
        }
        
        return data
    
    def scrape_all_ports(self):
        """Scrape all configured ports."""
        all_data = []
        
        for port_code, port_name in PORTS.items():
            logger.info(f"Scraping {port_name} ({port_code})")
            
            try:
                data = self.scrape_port(port_code, port_name)
                if data:
                    all_data.append(data)
                    summary = data["summary"]
                    logger.info(
                        f"  Expected: {summary['expected_ships']}, "
                        f"In Port: {summary['ships_in_port']}"
                    )
            except Exception as e:
                logger.error(f"Failed to scrape {port_name}: {e}")
        
        return all_data


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_data(data):
    """Save scraped data to JSON file."""
    ensure_output_dir()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    filename = f"{OUTPUT_DIR}/ports_{timestamp}.json"
    
    output = {
        "scrape_type": "port_status",
        "scrape_timestamp": datetime.utcnow().isoformat() + "Z",
        "ports_scraped": len(data),
        "data": data
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved to {filename}")
    return filename


def print_summary(data):
    """Print summary of scraped data."""
    print("\n" + "=" * 60)
    print("PORT STATUS SUMMARY")
    print("=" * 60)
    
    for port in data:
        summary = port["summary"]
        print(f"\n{port['port_name']} ({port['port_code']})")
        print(f"  Expected ships: {summary['expected_ships']}")
        print(f"  Ships in port:  {summary['ships_in_port']}")
        print(f"  Vessels tracked:")
        print(f"    - Expected:   {len(port['expected_vessels'])}")
        print(f"    - In port:    {len(port['vessels_in_port'])}")
        print(f"    - Arrivals:   {len(port['recent_arrivals'])}")
        print(f"    - Departures: {len(port['recent_departures'])}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("Starting VesselFinder port scraper")
    
    try:
        scraper = VesselFinderPortScraper()
        data = scraper.scrape_all_ports()
        
        if data:
            save_data(data)
            print_summary(data)
            logger.info("Scraping completed successfully")
        else:
            logger.warning("No data collected")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
