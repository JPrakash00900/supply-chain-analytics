-- =============================================================================
-- 01_create_tables.sql
-- Create normalized tables from the raw supply chain CSV import
-- =============================================================================

-- Raw staging table (all columns from the CSV)
CREATE TABLE IF NOT EXISTS raw_orders (
    order_id                    INTEGER,
    order_item_id               INTEGER PRIMARY KEY,
    order_date                  TEXT,
    shipping_date               TEXT,
    order_status                TEXT,
    shipping_mode               TEXT,
    days_for_shipping_real      INTEGER,
    days_for_shipment_scheduled INTEGER,
    late_delivery_risk          INTEGER,
    product_id                  TEXT,
    product_name                TEXT,
    category                    TEXT,
    unit_price                  REAL,
    unit_cost                   REAL,
    margin_pct                  REAL,
    order_item_quantity         INTEGER,
    order_item_discount_rate    REAL,
    order_item_product_price    REAL,
    order_item_total            REAL,
    order_item_profit_ratio     REAL,
    benefit_per_order           REAL,
    sales                       REAL,
    region                      TEXT,
    market                      TEXT,
    customer_segment            TEXT,
    year                        INTEGER,
    month                       INTEGER,
    week                        INTEGER,
    day_of_week                 INTEGER
);

-- Dimension: Products / SKU catalog
CREATE TABLE IF NOT EXISTS dim_products (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    unit_price      REAL NOT NULL,
    unit_cost       REAL NOT NULL,
    margin_pct      REAL NOT NULL,
    lead_time_min   INTEGER NOT NULL,
    lead_time_max   INTEGER NOT NULL
);

-- Dimension: Regions & Markets
CREATE TABLE IF NOT EXISTS dim_geography (
    market  TEXT PRIMARY KEY,
    region  TEXT NOT NULL
);

-- Fact: Cleaned order line items
CREATE TABLE IF NOT EXISTS fact_order_lines (
    order_item_id               INTEGER PRIMARY KEY,
    order_id                    INTEGER NOT NULL,
    order_date                  TEXT NOT NULL,
    shipping_date               TEXT,
    order_status                TEXT NOT NULL,
    shipping_mode               TEXT NOT NULL,
    days_for_shipping_real      INTEGER,
    days_for_shipment_scheduled INTEGER,
    late_delivery_risk          INTEGER DEFAULT 0,
    product_id                  TEXT NOT NULL,
    quantity                    INTEGER NOT NULL CHECK (quantity > 0),
    unit_price                  REAL NOT NULL,
    unit_cost                   REAL NOT NULL,
    discount_rate               REAL NOT NULL DEFAULT 0.0,
    sale_price                  REAL NOT NULL,
    line_total                  REAL NOT NULL,
    line_profit                 REAL NOT NULL,
    profit_ratio                REAL,
    region                      TEXT NOT NULL,
    market                      TEXT NOT NULL,
    customer_segment            TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (market) REFERENCES dim_geography(market)
);

-- Aggregate: Weekly SKU-level demand (the primary forecasting grain)
CREATE TABLE IF NOT EXISTS agg_weekly_demand (
    week_start          TEXT NOT NULL,
    year                INTEGER NOT NULL,
    week_num            INTEGER NOT NULL,
    product_id          TEXT NOT NULL,
    product_name        TEXT NOT NULL,
    category            TEXT NOT NULL,
    region              TEXT NOT NULL,
    total_quantity      INTEGER NOT NULL,
    total_revenue       REAL NOT NULL,
    total_profit        REAL NOT NULL,
    avg_discount_rate   REAL,
    order_count         INTEGER NOT NULL,
    late_delivery_pct   REAL,
    PRIMARY KEY (week_start, product_id, region)
);

-- Aggregate: Monthly SKU-level summary for inventory analysis
CREATE TABLE IF NOT EXISTS agg_monthly_sku (
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    product_id      TEXT NOT NULL,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    region          TEXT NOT NULL,
    total_quantity  INTEGER NOT NULL,
    total_revenue   REAL NOT NULL,
    total_profit    REAL NOT NULL,
    avg_unit_cost   REAL NOT NULL,
    order_count     INTEGER NOT NULL,
    PRIMARY KEY (year, month, product_id, region)
);
