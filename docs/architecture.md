# WazeCargo Architecture

## Overview

WazeCargo is a port congestion prediction platform for Chilean importers.

## Data Flow

1. **Ingestion:** GitHub Actions scrapers collect data every 15 minutes
2. **Storage:** Raw JSON files stored in data/raw/
3. **Processing:** Pipeline scripts load data into DuckDB
4. **Analytics:** ML model predicts congestion risk
5. **Delivery:** Streamlit dashboard and email alerts

## Tech Stack

| Component | Technology |
|-----------|------------|
| Scraper Automation | GitHub Actions |
| Data Storage | JSON files, DuckDB |
| ML Framework | XGBoost, scikit-learn |
| Dashboard | Streamlit Cloud |
| Alerts | Gmail SMTP |

## Team

| Role | Responsibility |
|------|----------------|
| Tech Lead | Architecture, ML, deployment |
| Junior 1 | DIRECTEMAR scraper, Valparaíso scraper, alerts |
| Junior 2 | San Antonio scraper, dashboard, training data |
| Junior 3 | VesselFinder scraper, EDA, baseline model |
