# WazeCargo

**Port congestion prediction platform for Chilean importers.**

WazeCargo forecasts vessel traffic and congestion risk at Chilean ports using 20+ years of customs data (2000-2025), real-time vessel tracking, and maritime weather conditions. Built for importers who need to plan around port delays.

## Architecture

```
GitHub Actions (scrape every 4h)
        │
        ▼
   S3 Raw Layer (CSV/JSON)
        │
   AWS Glue (ETL)
        │
        ▼
   RDS PostgreSQL
   ┌──────────────────────────┐
   │  structured.*            │  ← raw imports/exports + 13 lookup tables
   │  maritime.*              │  ← filtered maritime-only (clean tables)
   │  ml.*                    │  ← model outputs, forecasts, CV metrics
   │  waze_cargo.*            │  ← weather, operability, adjusted congestion
   └──────────────────────────┘
        │
        ├──→ ML Pipelines (congestion + commodity forecasting)
        │         │
        │         ▼
        │    ml.port_forecast_2026 / ml.commodity_forecast_2026
        │
        └──→ Dashboard (React + Leaflet + Recharts)
```

## Data Sources

| Source | What | Method |
|--------|------|--------|
| **DIRECTEMAR** (Chilean Maritime Authority) | Import/export customs records, 2000-2025 | CSV bulk download (~20M records) |
| **VesselFinder** | Real-time port status: ships in port, expected arrivals/departures | Web scraping every 4 hours |
| **Open-Meteo Marine API** | Wave height, swell, wind speed for 15 Chilean ports | API (hourly forecasts) |

## Repository Structure

```
WazeCargo/
├── scrapers/                  # Data collection
│   ├── ports/                 #   VesselFinder port scraper (GitHub Actions)
│   ├── vessels/               #   VesselFinder vessel lookup (on-demand)
│   ├── utils/                 #   Shared HTTP client
│   └── template_scraper.py    #   Scraper template for new sources
│
├── pipeline/                  # AWS Glue ETL (S3 → RDS)
│   ├── 00_loading_to_raw.py   #   CSV upload to S3 raw layer
│   ├── 01_raw_to_staging.py   #   S3 raw → S3 staging (Glue/Spark)
│   └── 02_staging_to_structured.py  # Staging → RDS structured schema
│
├── modeling/                  # ML forecasting
│   ├── 03_rebuild_clean_maritime.sh  # Build maritime.clean_* tables from structured
│   ├── 04_ml_congestion.py    #   Port congestion forecast (hybrid ensemble)
│   ├── 05_ml_commodity.py     #   Commodity-level forecast (HS2 x port x direction)
│   ├── wz_ml_utils.py         #   Shared ML utilities (features, CV, scoring)
│   ├── notebooks/             #   Executed Jupyter notebooks (EDA, model comparison)
│   └── INDEX.md               #   Full reference of scripts, tables, and results
│
├── dashboard/                 # Dashboard v1 (Streamlit, deprecated)
│   └── wazecargo-dashboard/   #   React-based dashboard (production build)
│
├── dashboard-2.0/             # Dashboard v2 (branch: add-dashboard-2.0)
│   └── src/                   #   React 19 + Leaflet maps + Recharts
│                              #   Congestion + weather fused port intelligence UI
│
├── db_sql_scripts/            # Database setup
│   └── deprecated/            #   Original RDS schema DDL, DuckDB migration, SQL pipelines
│
├── eda_databrew/              # AWS DataBrew EDA screenshots (imports/exports)
│
├── docs/                      # Documentation
│   ├── architecture.md        #   System architecture overview
│   ├── er_mermaid.md          #   Full ER diagram (Mermaid) — all 4 schemas
│   └── er_ddl_reference.md    #   DDL reference for all tables
│
├── .github/workflows/
│   └── scrape.yml             #   Automated port data collection (every 4 hours)
│
├── requirements.txt           #   Python dependencies
├── CONTRIBUTING.md            #   Branch workflow and team contribution guide
└── .env.example               #   Environment variable template
```

## ML Models

WazeCargo evaluates 7 models per port-direction combination using walk-forward cross-validation (4-fold, COVID-aware weighting):

| Model | Volume-Weighted MAPE |
|-------|---------------------|
| **LightGBM** | **12.27%** |
| ElasticNet | 12.63% |
| XGBoost | 12.63% |
| Random Forest | 12.69% |
| Ridge | 12.86% |
| Lasso | 14.73% |
| Baseline (Seasonal Naive) | 14.84% |
| Prophet | 21.33% |

**Production strategy:** Hybrid ensemble -- Baseline Seasonal Naive for big ports (>= 500 ships/month avg), best ML model per port for smaller ports. Final wMAPE: 4.03%.

### Features

- Lagged shipment counts (1-month, 12-month)
- Rolling 12-month mean (past-only, no leakage)
- Year-over-year growth
- Seasonal indicators
- COVID shock flags (2020-2022 down-weighted: 0.1, 0.2, 0.4)
- Commodity diversity, cargo type mix, origin country diversity

## Database Schemas

The RDS PostgreSQL database (`waze_cargo`) is organized into 4 schemas:

| Schema | Purpose | Key Tables |
|--------|---------|------------|
| `structured` | Raw customs data + lookups | `all_imports`, `all_exports`, 13 `lkp_*` tables |
| `maritime` | Filtered maritime-only records | `clean_maritime_imports`, `clean_maritime_exports` |
| `ml` | Model outputs and forecasts | `port_forecast_2026`, `commodity_forecast_2026`, `port_model_selection` |
| `waze_cargo` | Weather and operability | `port_weather_hourly/daily/monthly`, `port_risk_config`, `port_congestion_weather_adjusted` |

Full ER diagram: [`docs/er_mermaid.md`](docs/er_mermaid.md)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Scraping | Python, BeautifulSoup, Selenium, GitHub Actions |
| Storage | AWS S3 (raw/staging), RDS PostgreSQL |
| ETL | AWS Glue (PySpark), boto3 |
| ML | LightGBM, XGBoost, scikit-learn, Prophet |
| Dashboard | React 19, Leaflet, Recharts |
| Infrastructure | AWS (S3, Glue, RDS, Secrets Manager) |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for dashboard)
- AWS credentials configured (for pipeline/modeling scripts)
- Access to the RDS instance

### Installation

```bash
git clone https://github.com/gmonteskeet/WazeCargo.git
cd WazeCargo
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials

# For ML pipelines and database access:
export RDS_HOST=<your-rds-host>
export RDS_PORT=5432
export RDS_USER=<your-rds-user>
export RDS_PASSWORD=<your-rds-password>
export RDS_DBNAME=waze_cargo
```

### Run the ML Pipeline

```bash
cd modeling

# 1. Rebuild clean maritime tables (if source data changed)
bash 03_rebuild_clean_maritime.sh

# 2. Port congestion forecasting
python 04_ml_congestion.py

# 3. Commodity-level forecasting
python 05_ml_commodity.py
```

### Run the Dashboard (v2)

```bash
git checkout add-dashboard-2.0
cd dashboard-2.0
npm install
npm start
# Opens at http://localhost:3000
```

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production -- stable code |
| `add-dashboard-2.0` | Dashboard v2: congestion + weather fused port intelligence UI |
| `feature/dashboard-v2` | Earlier dashboard v2 prototype |

## Team

| Role | Responsibility |
|------|----------------|
| Tech Lead | Architecture, ML pipelines, deployment, PR reviews |
| Junior 1 | DIRECTEMAR scraper, Valparaiso scraper, alert system |
| Junior 2 | San Antonio scraper, dashboard, training dataset |
| Junior 3 | VesselFinder scraper, EDA, baseline model |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch workflow, naming conventions, and code standards.

## License

Private repository. All rights reserved.
