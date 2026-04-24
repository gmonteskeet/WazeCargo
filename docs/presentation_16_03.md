# Waze Cargo
## Big Data & ML Pipeline for Chilean Maritime Trade Intelligence
### Technical Presentation · March 18, 2026

---

## Slide 1: Infrastructure & Architecture

### From Raw Customs Files to Cloud-Native Intelligence

---

### End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                │
├──────────────────────────────────────────────────────────────┤
│  Chilean Customs (SNA)                                       │
│  ingresos_YYYY.csv  ·  salidas_YYYY.csv  ·  2002–2026       │
│  tablas_de_codigos.xlsx  ·  harmonized-system.csv            │
├──────────────────────────────────────────────────────────────┤
│  VesselFinder (In Development)                               │
│  Port congestion data  ·  GitHub Actions every 4h            │
│  Expected vessels  ·  Ships in port  ·  Arrivals/Departures │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  LOCAL  (Python 3.13 · boto3 · venv)                        │
│  upload_raw.py  →  S3 raw layer                             │
│  Cross-platform (Windows/Mac/Linux)                          │
│  --force flag for incomplete years (2026)                    │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  AWS S3  ·  wazecargo-263704545424-eu-north-1-an            │
│                                                              │
│  raw/                                                        │
│    ├── ingresos/year=YYYY/*.csv                             │
│    └── salidas/year=YYYY/*.csv                              │
│                        │                                     │
│                        ▼  AWS Glue Job: raw_to_staging       │
│                           - Schema validation                │
│                           - ML-critical column checks        │
│                           - SNS alerts on rejection          │
│                                                              │
│  staging/                                                    │
│    ├── ingresos/year=YYYY/*.parquet                         │
│    └── salidas/year=YYYY/*.parquet                          │
│                                                              │
│  failed/  (rejected rows with NULL in critical columns)     │
│  databrew-profiles/  (EDA profiling outputs)                │
│  jars/  (postgresql-42.6.0.jar JDBC driver)                 │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼  AWS Glue Job: staging_to_postgresql
                           - Reads from Secrets Manager
                           - Replace by year (DELETE + INSERT)
                           - Creates indexes automatically
┌──────────────────────────────────────────────────────────────┐
│  AWS RDS PostgreSQL 15  ·  eu-north-1  ·  waze_cargo DB     │
│                                                              │
│  structured.ingresos     (all years, clean data)            │
│  structured.salidas      (all years, clean data)            │
│  ml.*                    (features, predictions, forecasts) │
│  analytics.*             (aggregated views for dashboards)  │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼  SQL queries
┌──────────────────────────────────────────────────────────────┐
│  BACKEND  ·  FastAPI  ·  AWS EC2  (In Development)          │
│                                                              │
│  API Endpoints:                                              │
│  GET /ports/{port_code}/congestion?date=YYYY-MM-DD          │
│  GET /vessels/{vessel_name}/risk                            │
│  GET /calendar/{port_code}?month=YYYY-MM                    │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼  HTTP / REST
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND  ·  React (.jsx)  ·  AWS EC2  (In Development)    │
│                                                              │
│  Features:                                                   │
│  - Port congestion calendar (color-coded: green/yellow/red) │
│  - Vessel search → delay risk prediction                    │
│  - Port + date selection → congestion level                 │
└──────────────────────────────────────────────────────────────┘
```

---

### Technology Choices

| Component | Tool | Why |
|---|---|---|
| Local upload | Python + boto3 | Cross-platform, full control, integrates with AWS CLI credentials |
| Cloud storage | AWS S3 | Scalable, cost-effective, native Glue integration, partitioned by year |
| EDA / Profiling | AWS DataBrew (eda_databrew/) | Visual profiling, schema inference, data quality statistics |
| ETL | AWS Glue (PySpark) | Serverless, scalable, native S3/RDS connectors, handles large files |
| Data validation | Custom Glue logic | ML-critical column checks, row-level split (valid/invalid) |
| Alerting | AWS SNS | Real-time notifications on data quality issues |
| Cloud DB | AWS RDS PostgreSQL 15 | Managed, scalable, standard SQL, native SSL, ACID compliant |
| Secrets | AWS Secrets Manager | Secure credential storage, no hardcoded passwords |
| ML | LightGBM + XGBoost | State-of-art for tabular time series, native NaN handling, SHAP built-in |
| Real-time data | VesselFinder + GitHub Actions | Port congestion scraping every 4h (in development) |
| Backend API | FastAPI (Python) | Async, fast, auto-generated docs, matches ML stack |
| Frontend | React (.jsx) | Modern, component-based, reusable UI elements |
| Hosting | AWS EC2 | Full control, scalable, same region as RDS |
| Versioning | GitHub | Full reproducibility — schema, ETL, ML, scrapers all committed |

---

### Cybersecurity

- **Encryption in transit** → `sslmode=require` on every PostgreSQL connection
- **Encryption at rest** → AWS RDS and S3 storage encryption enabled
- **Secrets Management** → AWS Secrets Manager for database credentials (no hardcoded passwords)
- **Network isolation** → RDS in private VPC; Security Group restricts port 5432 to specific IPs
- **IAM least privilege** → Glue role scoped to specific S3 paths and Secrets Manager ARN
- **Data quality firewall** → Invalid rows rejected to `/failed/` before reaching production DB
- **Audit trail** → SNS alerts on every data quality rejection with row counts
- **Raw data separation** → Source CSVs in S3 raw/; only validated data reaches PostgreSQL

---

### Scalability

- **Storage** → S3 unlimited; RDS 500 GB gp3, auto-expandable
- **Compute** → Glue serverless (scales automatically); RDS `db.t3.large` upgradeable to `db.r6g.2xlarge`
- **Partitioning** → S3 partitioned by `year=YYYY` for efficient Glue reads
- **Incremental updates** → `upload_raw.py --year 2026 --force` for partial year updates
- **Read replicas** → Add RDS read replica for BI dashboards without touching production
- **API scaling** → EC2 Auto Scaling Group or migrate to ECS/Lambda when needed
- **Frontend scaling** → Can migrate React to S3 + CloudFront for global CDN
- **Real-time pipeline** → GitHub Actions + VesselFinder scrapers scale independently

---

## Slide 2: ETL Pipeline

### 20 Years · 19.9 Million Records · $225 Billion in Trade

---

### 4-Layer Architecture

```
LAYER 0 — LOCAL
CSV files on local machine
upload_raw.py validates path and uploads to S3

LAYER 1 — RAW (S3)
All columns · No transformation · Original CSVs
s3://bucket/raw/ingresos/year=2024/ingresos2024.csv
s3://bucket/raw/salidas/year=2024/salidas_2024.csv

LAYER 2 — STAGING (S3)
Glue Job: raw_to_staging
- Validates ML-critical columns (PERIODO, MES, COD_PUERTO, CIF_US, etc.)
- Valid rows → staging/*.parquet
- Invalid rows → failed/*.parquet
- SNS alert sent with rejection statistics

LAYER 3 — STRUCTURED (PostgreSQL)
Glue Job: staging_to_postgresql
- Reads credentials from Secrets Manager
- DELETE + INSERT by year (idempotent)
- Creates indexes on year column
- Tables: structured.ingresos, structured.salidas
```

---

### Pipeline Scripts

| Script | Location | Purpose |
|---|---|---|
| `upload_raw.py` | Local | Upload CSVs to S3 raw layer |
| `raw_to_staging.py` | AWS Glue | Validate, split valid/invalid, write Parquet |
| `staging_to_postgresql.py` | AWS Glue | Load Parquet to PostgreSQL structured schema |

---

### Data Quality Validation

**ML-Critical Columns (must not be NULL):**

| ingresos | salidas |
|---|---|
| PERIODO | PERIODO |
| MES | MES |
| COD_PUERTO_DESEMBARQUE | COD_PUERTO_EMBARQUE |
| COD_VIA_TRANSPORTE | COD_VIA_TRANSPORTE |
| ITEM_SA | ITEM_SA |
| CIF_US | FOB_US_DUSLEG |
| COD_TIPO_OPERACION | PESO_BRUTO_KG |

**Rows with NULL in any critical column → rejected to `/failed/` with SNS alert**

---

### Dataset at a Glance

| | Imports | Exports |
|---|---|---|
| Records | 15,452,762 | 4,456,056 |
| Period | 2002–2026 | 2002–2026 |
| Trade value | $86.6B CIF | $138.9B FOB |
| Origin / dest countries | 229 | 238 |
| Chilean ports | 78 | 86 |
| HS2 product chapters | 100 | 100 |

---

### Real-Time Data Pipeline (In Development)

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions  ·  Scheduled every 4 hours                 │
│  scrapers/ports/vesselfinder_ports.py                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  VesselFinder Port Data                                     │
│  - Expected vessels (next 30 days)                          │
│  - Ships currently in port                                  │
│  - Recent arrivals / departures                             │
│  - Vessel details (IMO, GT, DWT, type)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  data/raw/vesselfinder/ports_YYYY-MM-DD_HH-MM.json         │
│  Accumulates over time for ML training                      │
│  Features: ships_in_port, expected_arrivals, congestion    │
└─────────────────────────────────────────────────────────────┘
```

**Ports monitored:** San Antonio (CLSAI), Valparaíso (CLVAP)

**Purpose:** Real-time congestion prediction for importers

---

### 5 Key Engineering Fixes

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 1 | HS code mismatch | Chilean tariff = 8 digits; HS standard = 6 | `SUBSTR(LPAD(item_sa,8,'0'),1,6)` |
| 2 | Excel footer rows | Every lookup sheet ends with "Ir a Listado de Tablas" | String filter on pk column after read |
| 3 | Port key type error | `cod_puerto` DOUBLE in lookup vs INTEGER in fact | `TRY_CAST(cod_puerto AS INTEGER)` |
| 4 | Row loss on filter | WHERE applied before JOINs completed | Moved filter after all JOINs |
| 5 | 754K NULL rows rejected | `periodo`/`mes` legitimately NULL in source | Row-level validation with failed/ output |

---

## Slide 3: Machine Learning — Model Selection

### Why We Chose LightGBM for 2026 Port Forecasting

---

### Problem Framing

**Target:** Predict monthly shipment count per Chilean maritime port-direction pair
**Scope:** 78 port-direction pairs · 2005–2025 training · 2026 forecast
**Challenge:** Time series with structural breaks (COVID), 20× scale difference between ports, strong seasonal patterns per commodity

---

### All Models Tested — Benchmark Table

| Model | Avg MAE | Avg RMSE | Avg MAPE | Avg R² | Status |
|---|---|---|---|---|---|
| **Seasonal Naive** | 5,898 | 9,968 | 61,623% | −0.362 | ✗ COVID lags destroy it |
| **Linear Regression** | N/A | N/A | >500% | <0 | ✗ Collinear features, no temporal structure |
| **SARIMA** | ~3,200 | ~6,100 | ~280% | 0.31 | ✗ 78 separate models, no multivariate |
| **Prophet** | ~2,100 | ~4,400 | ~190% | 0.61 | ~ Good trend, misses cargo mix |
| **XGBoost** | 1,292 | 3,034 | **115%** | 0.877 | ✓ Strong, best MAPE |
| **LightGBM** | **1,106** | **2,674** | ~130%* | **0.909** | ✓✓ Best R², fastest, chosen |

*MAPE inflated by near-zero off-season months — R² is the reliable metric for sparse time series

---

### Why LightGBM Was Chosen

**R² 0.909 — Best explained variance across all 78 ports**
Captures non-linear interactions between lag features, cargo mix, and seasonality that linear models miss entirely.

**Native NaN handling**
Structural NaNs in 2024 (incomplete year) and COVID-era lag features are handled internally — no imputation noise introduced.

**Sample weights for COVID**
The `sample_weight` parameter lets us assign lower influence to 2020 (0.1), 2021 (0.2), 2022 (0.4) during training. No other tested model supported this as cleanly for time series.

**SHAP interpretability**
Built-in SHAP values explain why each port gets a specific prediction — essential for a routing recommendation system that needs to justify its outputs.

**Speed**
78 pairs × 3 CV folds × 500 boosting rounds: LightGBM finishes in **64 seconds**. XGBoost needed **3.5 minutes**.

---

### Top Features by SHAP Importance

| Rank | Feature | SHAP Score | What It Captures |
|---|---|---|---|
| 1 | `lag_12_clean` | 0.891 | Same month last year (COVID-corrected) |
| 2 | `rolling_12_mean` | 0.847 | 12-month trend baseline |
| 3 | `commodity_diversity` | 0.617 | How specialised the port is |
| 4 | `yoy_growth_clean` | 0.612 | Clean year-over-year momentum |
| 5 | `lag_1` | 0.583 | Previous month carry-over |
| 6 | `month_sin` / `month_cos` | 0.421 | Seasonal encoding |
| 7 | `pct_container` | 0.334 | Cargo mix signal |
| 8 | `is_covid_rebound` | 0.287 | Model learned to discount 2021 spike |

---

## Slide 4: COVID Problem & Walk-Forward Validation

### The Hardest Data Science Challenge in the Project

---

### The Contamination Chain

```
Year  Shipments     YoY      lag_12 in next year    Problem
────  ──────────    ──────   ────────────────────   ──────────────────────────
2019  981,004       +5.1%    → used in 2020         ✓ Clean
2020  874,830      −10.8%    → used in 2021         ✗ COVID shock
2021  1,018,185   +16.4%    → used in 2022         ✗ Rebound off low base
2022  966,713      −5.1%    → used in 2023         ✗ lag_12 = inflated 2021
2024  179,622     −80.2%    → used in 2025         ✗ Incomplete data (no FOB)
2025  1,810,179  +907.8%    → used in 2026         ✗ lag_12 = broken 2024
```

**If trained naively:** model "learns" +16% growth is normal because lag_12 was artificially low. Then predicts wrong direction in 2022. Then extrapolates the 2025 spike into perpetuity.

---

### 7 Fixes Applied

| Fix | Type | Description |
|---|---|---|
| `lag_12_clean` | Feature | Replaces COVID-tainted lag_12 using pre-COVID 2015–2019 trend estimate |
| `yoy_growth_clean` | Feature | Replaces 2020–2022 YoY with clean-year median per port+month |
| `is_covid_shock` | Flag | Binary: year == 2020 |
| `is_covid_rebound` | Flag | Binary: year == 2021 |
| `is_covid_aftershock` | Flag | Binary: year == 2022 |
| `lag_12_is_covid` | Flag | Binary: the lag_12 value itself references a COVID year |
| **Sample weights** | Training | 2020→0.1 · 2021→0.2 · 2022→0.4 · 2024→0.0 (excluded) |

---

### Walk-Forward Cross Validation

```
Standard CV is INVALID for time series — it leaks future into past.
COVID years as test targets are INVALID — they measure anomaly, not skill.

CLEAN FOLDS ONLY:
───────────────────────────────────────────────────────────────
Fold 1   Train 2005–2018  →  Test 2019    ✓ pre-COVID
         (skip 2020–2022 as test years)   ✗ contaminated
Fold 2   Train 2005–2019  →  Test 2023    ✓ post-COVID recovery
         (skip 2024)                      ✗ incomplete
Fold 3   Train 2005–2023  →  Test 2025    ✓ most recent clean year
───────────────────────────────────────────────────────────────
Final    Train 2005–2025  →  Forecast 2026  ← production output
```

Each fold trains on everything before the test year — no shuffling, no random splits. The model never sees the future during training.

---

## Slide 5: Results & 2026 Forecasts

### What the Pipeline Produces

---

### Output Tables on AWS RDS

| Table | Rows | Contents |
|---|---|---|
| `structured.ingresos` | 15.5M | Clean import records (2002-2026) |
| `structured.salidas` | 4.5M | Clean export records (2002-2026) |
| `ml.forecast_2026` | 768 | Port congestion · 64 pairs × 12 months |
| `ml.model_evaluation` | 534 | MAE · RMSE · MAPE · R² per model+port+fold |
| `ml.feature_importance` | 5,604 | SHAP values per feature per port |
| `ml.covid_diagnostics` | 64 | COVID drop % · rebound % per port |
| `ml.commodity_forecast_2026` | 5,784 | HS2+port · FOB + ships + weight |

---

### 2026 Port Forecast — Top 8 by Predicted Volume

| Port | Direction | Pred. Ships 2026 | Avg Monthly Value | Congestion |
|---|---|---|---|---|
| San Antonio | Import | 111,490 / year | $45.4M/month | Low (0.10) |
| Valparaíso | Import | 66,280 / year | $19.8M/month | Low (0.14) |
| Puerto Angamos | Import | 14,461 / year | $37.1M/month | **Moderate (0.33)** |
| San Antonio | Export | 13,818 / year | $25.4M/month | Low (0.13) |
| Valparaíso | Export | 11,476 / year | $30.3M/month | Low (0.11) |
| Coronel | Import | 10,531 / year | $12.9M/month | **Moderate (0.30)** |
| Lirquén | Import | 10,429 / year | $10.2M/month | **Moderate (0.24)** |
| San Vicente | Import | 10,286 / year | $12.5M/month | **Moderate (0.21)** |

---

### MVP User Interface (In Development)

**Port Congestion Calendar**
```
┌─────────────────────────────────────────────────────────────┐
│  San Antonio - March 2026                                   │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ Sun │ Mon │ Tue │ Wed │ Thu │ Fri │ Sat │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │  🟢 │  🟢 │  🟡 │  🟡 │  🟢 │     │
│     │  1  │  2  │  3  │  4  │  5  │  6  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  🟢 │  🟢 │  🟡 │  🔴 │  🔴 │  🟡 │  🟢 │
│  7  │  8  │  9  │ 10  │ 11  │ 12  │ 13  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘

🟢 Low Risk    🟡 Medium Risk    🔴 High Risk
```

**Vessel Delay Risk**
```
┌─────────────────────────────────────────────────────────────┐
│  🔍 Search: COSCO SHIPPING SEINE                           │
├─────────────────────────────────────────────────────────────┤
│  Vessel: COSCO SHIPPING SEINE                              │
│  Destination: San Antonio                                   │
│  Scheduled ETA: March 15, 2026                             │
│                                                             │
│  ⚠️  DELAY RISK: HIGH                                      │
│                                                             │
│  Port Status:                                               │
│  - Ships in port: 18                                        │
│  - Expected arrivals: 12                                    │
│  - 40% above normal capacity                                │
│                                                             │
│  Recommendation: Notify client of potential 3-5 day delay  │
└─────────────────────────────────────────────────────────────┘
```

---

### GitHub Repository Structure

```
WazeCargo/
├── .github/
│   └── workflows/
│       └── scrape.yml                 ← GitHub Actions (every 4h)
├── scrapers/
│   ├── ports/
│   │   └── vesselfinder_ports.py      ← Port congestion scraper
│   ├── vessels/
│   │   └── vesselfinder_vessel.py     ← Vessel lookup (on-demand)
│   └── utils/
│       └── http_client.py             ← Shared HTTP logic
├── pipeline/
│   ├── upload_raw.py                  ← Local → S3 raw
│   ├── raw_to_staging.py              ← Glue: S3 raw → S3 staging
│   └── staging_to_postgresql.py       ← Glue: S3 staging → PostgreSQL
├── ml/
│   ├── train_congestion.py            ← Port congestion model
│   └── train_commodity.py             ← Commodity forecast model
├── backend/                           ← (In Development)
│   └── main.py                        ← FastAPI endpoints
├── frontend/                          ← (In Development)
│   └── src/
│       └── App.jsx                    ← React components
├── data/
│   └── raw/
│       └── vesselfinder/              ← Scraped JSON files
└── requirements.txt
```

---

### Tech Stack Summary

| Layer | Technology | Status |
|---|---|---|
| Data Sources | Chilean Customs CSV, VesselFinder | ✅ Active |
| Local Tools | Python 3.13, boto3, DuckDB | ✅ Active |
| Cloud Storage | AWS S3 (partitioned by year) | ✅ Active |
| EDA | AWS DataBrew (eda_databrew/) | ✅ Active |
| ETL | AWS Glue (PySpark) | ✅ Active |
| Validation | Custom Glue + SNS alerts | ✅ Active |
| Database | AWS RDS PostgreSQL 15 | ✅ Active |
| Secrets | AWS Secrets Manager | ✅ Active |
| ML Models | LightGBM, XGBoost, SHAP | ✅ Active |
| Real-time Data | GitHub Actions + VesselFinder | 🔄 In Development |
| Backend API | FastAPI on EC2 | 🔄 In Development |
| Frontend | React on EC2 | 🔄 In Development |
| Version Control | GitHub | ✅ Active |
