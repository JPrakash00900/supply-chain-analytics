# Supply Chain Analytics — End-to-End Portfolio Project

## Project Summary
Build a full pipeline answering: "Which products/regions lose money to stockouts vs. overstock, and what would better forecasting save annually?"

Pipeline: Synthetic DataCo-like CSV → SQLite SQL cleaning → Python feature engineering → 4 forecasting models → Inventory policy → $ impact → Tableau-ready exports + charts.

---

### [x] Step 1: Project scaffold
- .gitignore, requirements.txt, folder structure (data/, sql/, src/, notebooks/, dashboard/)

### [x] Step 2: Synthetic data generator
- Generate ~180k order-line records over 2 years, 5 categories, 4 regions, realistic seasonality
- Output: data/raw/supply_chain_data.csv

### [x] Step 3: SQL layer
- 01_create_tables.sql, 02_clean_data.sql, 03_aggregations.sql, 04_analysis_queries.sql
- sql_pipeline.py runs all SQL via SQLite3

### [x] Step 4: Feature engineering
- Lag features (1w, 2w, 4w), rolling averages, day-of-week, seasonality flags, promo indicators

### [x] Step 5: Forecasting models
- Naive (last value), Holt-Winters (statsmodels), XGBoost, Prophet
- model_comparison.py: MAPE + RMSE table

### [x] Step 6: Inventory policy
- Safety stock = Z * σ_d * √L, Reorder point, EOQ per SKU

### [x] Step 7: $ Impact quantification
- Stockout cost (lost sales × margin) vs overstock cost (excess × unit_cost × 20% holding)
- Compare old (Holt-Winters) vs new (XGBoost) forecast error → annual savings estimate

### [x] Step 8: Tableau exports + visualizations
- tableau_prep.py exports CSVs for Tableau
- visualizations.py saves charts to outputs/

### [x] Step 9: End-to-end runner + README
- run_pipeline.py orchestrates all steps
- README.md with problem statement, methodology, results table, assumptions
