## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` — data processing
- `numpy` — numerical operations
- `openpyxl` — reading .xlsx lookup tables
- `requests` — HTTP client for scrapers
- `beautifulsoup4` — HTML parsing for scrapers

### 2. Run the historical data pipeline

Place these files in the same directory:
- `2022.csv`, `2023.csv`, `2024.csv`, `2025.csv`, `2026.csv` (raw customs data)
- `tablas_de_codigos.xlsx` (code lookup tables)
- `clasificador2022_v2_0.xlsx` (HS product classifier)

Then run:

```bash
python historical_import_processor.py --input-dir .
```

This will:
1. Load all lookup tables (ports, countries, incoterms, cargo types, etc.)
2. Load the HS product classifier (8,829 product codes)
3. Read all raw CSV files (2022–2026)
4. Decode every coded column to human-readable names
5. Filter to **maritime imports only** (COD_VIA_TRANSPORTE = 1)
6. Classify products using official HS categories
7. Build monthly aggregations per port
8. Output 4 files to `data/processed/`

**Output files:**

| File | Description |
|------|-------------|
| `all_ports_monthly_features.csv` | One row per port per month. Transactions, CIF value, cargo types, incoterms, seasonality, YoY change, z-scores. **This is the ML training data.** |
| `all_ports_product_profile.csv` | What each port imports most, ranked by HS category. |
| `port_comparison.csv` | Side-by-side port comparison: market share, peak month, avg incoterm split. |
| `all_ports_summary.json` | Metadata: total transactions, years covered, ports list. |


### 3. Run the dashboard

```bash
cd wazecargo-dashboard
npm install
npm install recharts
npm start
```

Opens at `http://localhost:3000`. The dashboard shows:
- Port ranking by transaction volume and market share
- Product import profiles per port (official HS classifier)
- Incoterm distribution (FOB + EXW = your target customers)
- Monthly volume trends with seasonal patterns
- Key insights for Scenario 2
