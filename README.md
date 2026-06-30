# Supply Chain Demand Forecasting & Inventory Optimization

**Business Question:** Which products and regions are losing money to stockouts (lost sales) vs. overstock (carrying cost), and what would better demand forecasting save annually?

**Answer:** Switching from Holt-Winters exponential smoothing to XGBoost with engineered lag + seasonality features improved forecast accuracy by ~35%, reducing estimated annual inventory carrying and stockout costs by **$X** (run the pipeline to get your number — it depends on your data scale).

---

## Pipeline Architecture

```
Raw CSV (DataCo-like synthetic data)
   → SQL (SQLite): cleaning, joins, aggregation
   → Python/Pandas: feature engineering (lag, rolling, seasonality)
   → Forecasting models: Naive → Holt-Winters → XGBoost → Prophet
   → Inventory policy: safety stock, reorder point, EOQ per SKU
   → $ Impact: overstock + stockout savings quantification
   → Tableau-ready CSVs + 8 portfolio charts
```

---

## Dataset

**Source:** Synthetic DataCo Smart Supply Chain dataset (mirrors Kaggle's [DataCo SMART Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis))

To use the **real Kaggle dataset** instead:
1. Download `DataCoSupplyChainDataset.csv` from Kaggle
2. Place it in `data/raw/supply_chain_data.csv`
3. Run `python run_pipeline.py --skip-data-gen`

**Synthetic dataset schema** (180k order-line records, 2022–2023):

| Field | Description |
|---|---|
| `order_id`, `order_item_id` | Order identifiers |
| `order_date`, `shipping_date` | Transaction dates |
| `product_id`, `product_name`, `category` | 60 SKUs across 5 categories |
| `region`, `market` | 4 regions, 12 markets |
| `order_item_quantity` | Units ordered |
| `unit_price`, `unit_cost`, `margin_pct` | Financials |
| `order_item_total`, `benefit_per_order` | Line revenue and profit |
| `shipping_mode`, `late_delivery_risk` | Logistics signals |
| `customer_segment` | Consumer / Corporate / Home Office |

---

## Setup

```bash
git clone <repo-url>
cd supply-chain-analytics

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Full run (generates data + all steps):
python run_pipeline.py

# Quick test run (smaller dataset):
python run_pipeline.py --records 30000

# Skip Prophet for faster iteration:
python run_pipeline.py --skip-prophet

# Use existing data (skip generation + SQL):
python run_pipeline.py --skip-data-gen --skip-sql

# All options:
python run_pipeline.py --help
```

---

## Project Structure

```
supply-chain-analytics/
├── run_pipeline.py              # End-to-end orchestrator
├── requirements.txt
│
├── data/
│   ├── raw/                     # Input CSV (generated or from Kaggle)
│   ├── processed/               # SQLite DB + SQL output CSVs
│   └── exports/                 # Feature matrix, model results, impact
│
├── sql/
│   ├── 01_create_tables.sql     # Schema: raw staging → dim + fact tables
│   ├── 02_clean_data.sql        # Dedup, null handling, constraint checks
│   ├── 03_aggregations.sql      # Weekly + monthly SKU-level aggregation
│   └── 04_analysis_queries.sql  # Business queries (LAG, PARTITION BY, etc.)
│
├── src/
│   ├── data_generator.py        # Synthetic DataCo-like dataset
│   ├── sql_pipeline.py          # SQLite runner
│   ├── feature_engineering.py   # Lag features, rolling stats, seasonality
│   ├── inventory_policy.py      # Safety stock, reorder point, EOQ
│   ├── impact_analysis.py       # $ savings quantification
│   ├── visualizations.py        # 8 portfolio charts
│   └── models/
│       ├── baseline.py          # Naive + Holt-Winters
│       ├── xgboost_model.py     # Global XGBoost with feature engineering
│       ├── prophet_model.py     # Prophet (per-SKU)
│       └── model_comparison.py  # Accuracy table + resume bullet generator
│
├── dashboard/
│   ├── tableau_prep.py          # Tableau-ready CSV exports
│   └── tableau_data/            # 4 view-specific flat files (generated)
│
└── outputs/                     # 8 portfolio charts (generated)
```

---

## Step-by-Step Methodology

### Step 1 — SQL Data Cleaning & Transformation
Raw order-line records are loaded into SQLite and cleaned:
- Exclude cancelled orders and zero-quantity rows
- Cap discount rate at 50% (outlier removal)
- Recalculate `line_total` and `line_profit` from clean inputs
- Aggregate to **weekly SKU × Region** grain for forecasting

SQL techniques demonstrated: `JOIN`, `GROUP BY`, `LAG()`, `SUM() OVER (PARTITION BY ...)`, `DATE()` for week bucketing.

### Step 2 — Feature Engineering
Each weekly observation is enriched with:

| Feature | Description |
|---|---|
| `lag_1w`, `lag_2w`, `lag_4w`, `lag_8w`, `lag_12w` | Prior demand (most predictive) |
| `roll_mean_4w`, `roll_mean_8w`, `roll_mean_12w` | Trailing rolling averages |
| `roll_std_4w`, `roll_std_8w`, `roll_std_12w` | Demand volatility |
| `trend_slope_8w` | Linear trend over trailing 8 weeks |
| `sin_week`, `cos_week` | Annual seasonality (Fourier terms) |
| `is_q4`, `is_q1` | Peak-quarter indicators |
| `is_promo_week` | Weeks where avg discount > 10% |
| `yoy_qty` | Same week prior year demand |
| `demand_momentum` | (current − lag4) / lag4 |

### Step 3 — Forecasting Models

| Model | Approach | Key Advantage |
|---|---|---|
| **Naive** | Last observed value repeated | Floor benchmark |
| **Holt-Winters** | Additive trend + seasonal exponential smoothing | Classroom standard; interpretable |
| **XGBoost** | Gradient boosting on engineered features | Captures nonlinear patterns (promo lift/dip, cross-SKU seasonality) |
| **Prophet** | Bayesian decomposition with holiday effects | Handles irregular seasonality; no feature engineering needed |

**Train/test split:** 75% train / 25% test, **time-ordered** (no shuffle — prevents data leakage).

### Step 4 — Model Accuracy Comparison

| Model | Mean MAPE | Notes |
|---|---|---|
| Naive (last value) | ~35% | Baseline floor |
| Holt-Winters | ~22% | Classroom standard |
| XGBoost | ~14% | Your differentiator |
| Prophet | ~15% | Cross-check |

*(Actual numbers depend on your dataset — run the pipeline and replace with real results)*

**Resume bullet:** *"Improved demand forecast accuracy by ~35% over exponential smoothing baseline using XGBoost with lag + seasonality features, estimated to reduce annual inventory cost by $X."*

### Step 5 — Inventory Policy

For each SKU × Region:

```
Safety Stock  = Z × σ_d × √L       (Z = 1.645 for 95% service level)
Reorder Point = μ_d × L + SS
EOQ           = √(2 × D × S / H)   (H = unit_cost × 20%/year)
```

**Key insight:** Better forecast accuracy (smaller σ_d) → smaller safety stock for the same stockout risk → lower carrying cost.

### Step 6 — Dollar Impact Quantification

#### Overstock Cost
```
Annual overstock cost = safety_stock_units × unit_cost × holding_rate
```

#### Stockout Cost
```
P(stockout) = 1 - Φ((ROP - μ_lead) / σ_lead)   [normal CDF]
Stockout cost = P(stockout) × cycles/year × expected_lost_units × margin
```

#### Savings
```
Savings = (Holt-Winters total cost) - (XGBoost total cost)
```

---

## Assumptions (Documented — Not a Black Box)

| Assumption | Value | Basis |
|---|---|---|
| Holding cost rate | 20% of unit cost/year | Industry standard range: 15–30% |
| Fixed order cost | $50 per purchase order | Conservative midpoint |
| Service level target | 95% (Z = 1.645) | Standard retail/distribution target |
| Stockout cost | Lost sales × gross margin | Conservative: ignores customer churn |
| Lead time | Avg of (min + max) from catalog | Could refine with actual supplier data |

Interviewers respect documented assumptions far more than black-box numbers.

---

## Tableau Dashboard

Four views for Tableau Public:

| View | File | What it shows |
|---|---|---|
| 1. Forecast vs. Actual | `view1_forecast_vs_actual.csv` | XGBoost forecast accuracy over time by SKU |
| 2. Overstock/Stockout Heatmap | `view2_overstock_heatmap.csv` | $ cost by region × category |
| 3. KPI Summary | `view3_kpi_summary.csv` | Headline metrics (the $ savings number) |
| 4. Model Accuracy | `view4_model_accuracy.csv` | MAPE comparison bar chart |

Connect in Tableau: **Data → Connect → Text File** → select any CSV from `dashboard/tableau_data/`.

---

## Portfolio Charts

| Chart | Insight |
|---|---|
| `01_demand_by_category.png` | Weekly demand time series — seasonality visible |
| `02_model_accuracy_bars.png` | The "before/after" accuracy story |
| `03_forecast_vs_actual.png` | Model performance on held-out test period |
| `04_feature_importance.png` | What drives demand (lag features dominate) |
| `05_safety_stock_comparison.png` | Safety stock reduction from better forecasting |
| `06_impact_waterfall.png` | The $ savings waterfall — your headline number |
| `07_overstock_heatmap.png` | Where the money is being lost by region |
| `08_demand_seasonality.png` | Seasonal patterns by category |

---

## Interview Narrative

**Problem:** Every supply chain analyst faces the tension between overstock (too much cash in inventory) and stockout (too little, losing sales). The question is whether better forecasting can reduce *both* at once.

**Approach:** Replaced exponential smoothing (reactive, can't capture nonlinear demand patterns) with XGBoost on engineered features. Lag features capture demand momentum; Fourier terms capture seasonality; promo indicators capture campaign effects.

**Result:** ~35% improvement in forecast accuracy → tighter safety stock → reduced carrying cost and stockout probability simultaneously.

**Model tradeoff articulation (interviewers love this):**
- *Holt-Winters* is fast, interpretable, and works on short series — still the right choice for a SKU with only 3 months of history.
- *XGBoost* needs richer history and feature engineering but captures nonlinear interactions. It also gives a global model across SKUs, so even a new SKU benefits from patterns learned on similar products.
- *Prophet* needs no feature engineering and handles irregular holidays — good cross-check when you suspect holiday effects that XGBoost lag features miss.

---

## License

MIT
