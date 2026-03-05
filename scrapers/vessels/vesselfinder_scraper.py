"""
VesselFinder Vessel Scraper
===========================
On-demand vessel lookup by name or IMO number.
Used when a user wants to track their specific shipment.

Usage:
    # By vessel name
    python scrapers/vessels/vesselfinder_vessel.py --name "COSCO SHIPPING SEINE"
    
    # By IMO number
    python scrapers/vessels/vesselfinder_vessel.py --imo 9731949
    
    # Search local data only (no web request)
    python scrapers/vessels/vesselfinder_vessel.py --name "COSCO" --local-only
    
Output:
    data/raw/vesselfinder/vessel_{IMO}_{timestamp}.json
"""

import sys
import os
import re
import json
import logging
import argparse
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
PORT_DATA_DIR = "data/raw/vesselfinder"

BASE_URL = "https://www.vesselfinder.com"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# LOCAL DATA SEARCH
# =============================================================================

def search_local_data(query, data_dir=PORT_DATA_DIR):
    """
    Search for vessel in locally stored port data.
    Faster than making a new request if we recently scraped.
    
    Args:
        query: Vessel name or IMO to search for
        data_dir: Directory containing port JSON files
        
    Returns:
        List of matching vessels with port context
    """
    results = []
    
    if not os.path.exists(data_dir):
        logger.info(f"Local data directory not found: {data_dir}")
        return results
    
    # Get most recent JSON files
    json_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("ports_") and f.endswith(".json")],
        reverse=True
    )[:10]  # Check last 10 scrapes
    
    if not json_files:
        logger.info("No local port data files found")
        return results
    
    query_lower = query.lower()
    
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            
            port_list = file_data.get("data", [])
            
            for port_data in port_list:
                for list_name in ["expected_vessels", "vessels_in_port", "recent_arrivals", "recent_departures"]:
                    vessels = port_data.get(list_name, [])
                    
                    for vessel in vessels:
                        vessel_name = vessel.get("vessel_name", "").lower()
                        vessel_imo = vessel.get("imo", "")
                        
                        # Match by name (partial) or IMO (exact)
                        if query_lower in vessel_name or query == vessel_imo:
                            result = {
                                **vessel,
                                "found_in": list_name,
                                "port_code": port_data.get("port_code"),
                                "port_name": port_data.get("port_name"),
                                "data_timestamp": port_data.get("scrape_timestamp"),
                                "source_file": filename
                            }
                            
                            # Avoid duplicates
                            if not any(r.get("imo") == result.get("imo") and r.get("found_in") == result.get("found_in") for r in results):
                                results.append(result)
        
        except Exception as e:
            logger.debug(f"Error reading {filename}: {e}")
    
    return results


# =============================================================================
# WEB SCRAPER
# =============================================================================

class VesselFinderVesselScraper:
    """Scrapes individual vessel details from VesselFinder."""
    
    def __init__(self):
        self.client = HttpClient(delay=2)
    
    def search_vessel(self, query):
        """
        Search for a vessel by name or IMO.
        
        Args:
            query: Vessel name or IMO number
            
        Returns:
            Vessel data dict if found, None otherwise
        """
        # Check if query looks like an IMO number
        if query.isdigit() and len(query) == 7:
            return self.fetch_by_imo(query)
        else:
            return self.search_by_name(query)
    
    def fetch_by_imo(self, imo):
        """Fetch vessel details by IMO number."""
        url = f"{BASE_URL}/vessels/details/{imo}"
        html = self.client.fetch(url)
        
        if not html:
            logger.error(f"Failed to fetch vessel {imo}")
            return None
        
        return self.parse_vessel_page(html, imo)
    
    def search_by_name(self, name):
        """
        Search for vessel by name.
        
        Note: VesselFinder search may use JavaScript.
        This tries the vessels list page first.
        """
        # URL encode the name
        encoded_name = name.replace(" ", "+")
        url = f"{BASE_URL}/vessels?name={encoded_name}"
        
        html = self.client.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for vessel links in results
        # Try different possible selectors
        selectors = [
            ("a", {"class": "ship-link"}),
            ("a", {"class": "named-item"}),
            ("tr", {"class": "ship"}),
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                # Find the link with IMO
                link = element if tag == "a" else element.find("a")
                if link:
                    href = link.get("href", "")
                    imo_match = re.search(r"/(\d{7})", href)
                    if imo_match:
                        imo = imo_match.group(1)
                        logger.info(f"Found vessel with IMO {imo}")
                        return self.fetch_by_imo(imo)
        
        logger.warning(f"Vessel '{name}' not found in search results")
        return None
    
    def parse_vessel_page(self, html, imo):
        """Parse vessel details page."""
        soup = BeautifulSoup(html, "html.parser")
        
        vessel = {
            "source": SOURCE_NAME,
            "imo": imo,
            "scrape_timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        # Vessel name from h1
        h1 = soup.find("h1")
        if h1:
            vessel["vessel_name"] = h1.get_text(strip=True)
        
        # Look for data in various possible structures
        # VesselFinder uses different layouts, so we try multiple approaches
        
        # Approach 1: Look for table rows with labels
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    self._extract_field(vessel, label, value)
        
        # Approach 2: Look for labeled divs/spans
        for div in soup.find_all("div", class_=re.compile(r"(info|detail|param)")):
            label_elem = div.find(["span", "strong", "label"], class_=re.compile(r"label|name|key"))
            value_elem = div.find(["span", "strong"], class_=re.compile(r"value|data"))
            
            if label_elem and value_elem:
                label = label_elem.get_text(strip=True).lower()
                value = value_elem.get_text(strip=True)
                self._extract_field(vessel, label, value)
        
        # Approach 3: Look for specific known elements
        dest_elem = soup.find(string=re.compile(r"Destination", re.I))
        if dest_elem:
            parent = dest_elem.find_parent()
            if parent:
                value = parent.get_text().replace("Destination", "").strip(": \n")
                if value:
                    vessel["destination"] = value
        
        return vessel
    
    def _extract_field(self, vessel, label, value):
        """Extract a field based on its label."""
        if not value or value == "-":
            return
        
        label = label.lower().strip(": ")
        
        field_map = {
            "destination": ["destination", "dest", "destino"],
            "eta": ["eta", "estimated arrival"],
            "status": ["status", "navigation status", "nav status"],
            "speed": ["speed", "velocity"],
            "course": ["course", "heading", "direction"],
            "position": ["position", "coordinates", "lat/lon"],
            "flag": ["flag", "country"],
            "vessel_type": ["type", "ship type", "vessel type"],
            "mmsi": ["mmsi"],
            "callsign": ["call sign", "callsign"],
            "length": ["length", "loa"],
            "beam": ["beam", "width"],
            "draught": ["draught", "draft"],
            "dwt": ["dwt", "deadweight"],
            "gt": ["gt", "gross tonnage"],
            "built": ["built", "year built", "year"],
        }
        
        for field, keywords in field_map.items():
            if any(kw in label for kw in keywords):
                vessel[field] = value
                return


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_vessel_data(data):
    """Save vessel lookup to JSON file."""
    ensure_output_dir()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    imo = data.get("imo", "unknown")
    filename = f"{OUTPUT_DIR}/vessel_{imo}_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved to {filename}")
    return filename


def print_results(results, source="local"):
    """Print search results."""
    print(f"\n{'=' * 60}")
    print(f"VESSEL SEARCH RESULTS ({source})")
    print("=" * 60)
    
    if isinstance(results, list):
        print(f"Found {len(results)} match(es)\n")
        for i, vessel in enumerate(results, 1):
            print(f"[{i}] {vessel.get('vessel_name', 'Unknown')}")
            print(f"    IMO: {vessel.get('imo', 'N/A')}")
            print(f"    Type: {vessel.get('vessel_type', 'N/A')}")
            if vessel.get("port_name"):
                print(f"    Port: {vessel.get('port_name')} ({vessel.get('found_in', 'N/A')})")
            if vessel.get("time"):
                print(f"    Time: {vessel.get('time')}")
            if vessel.get("destination"):
                print(f"    Destination: {vessel.get('destination')}")
            if vessel.get("eta"):
                print(f"    ETA: {vessel.get('eta')}")
            print()
    else:
        # Single vessel dict
        vessel = results
        print(f"Vessel: {vessel.get('vessel_name', 'Unknown')}")
        print(f"IMO: {vessel.get('imo', 'N/A')}")
        for key, value in vessel.items():
            if key not in ["source", "scrape_timestamp", "vessel_name", "imo"]:
                print(f"{key}: {value}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VesselFinder Vessel Lookup for WazeCargo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --name "COSCO SHIPPING SEINE"
    %(prog)s --imo 9731949
    %(prog)s --name "MAERSK" --local-only
        """
    )
    parser.add_argument("--name", type=str, help="Search by vessel name")
    parser.add_argument("--imo", type=str, help="Search by IMO number")
    parser.add_argument("--local-only", action="store_true", 
                       help="Only search in local data (no web request)")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    
    args = parser.parse_args()
    
    if not args.name and not args.imo:
        parser.print_help()
        print("\nError: Please provide --name or --imo")
        sys.exit(1)
    
    query = args.imo if args.imo else args.name
    logger.info(f"Searching for vessel: {query}")
    
    # Step 1: Search local data
    local_results = search_local_data(query)
    
    if local_results:
        print_results(local_results, source="local data")
        
        if args.local_only:
            return
        
        print("\nSearching web for more details...")
    
    # Step 2: Search web (unless local-only)
    if not args.local_only:
        scraper = VesselFinderVesselScraper()
        web_result = scraper.search_vessel(query)
        
        if web_result:
            print_results(web_result, source="web")
            
            if args.save:
                save_vessel_data(web_result)
        else:
            if not local_results:
                logger.warning(f"Vessel not found: {query}")
                sys.exit(1)


if __name__ == "__main__":
    main()
