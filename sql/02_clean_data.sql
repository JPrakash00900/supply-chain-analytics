-- =============================================================================
-- 02_clean_data.sql
-- Load and clean raw staging data into normalized dimension and fact tables
-- =============================================================================

-- ── Populate dim_products from distinct SKUs in staging ──────────────────────
INSERT OR IGNORE INTO dim_products (
    product_id, product_name, category,
    unit_price, unit_cost, margin_pct,
    lead_time_min, lead_time_max
)
SELECT DISTINCT
    product_id,
    product_name,
    category,
    unit_price,
    unit_cost,
    margin_pct,
    CASE category
        WHEN 'Electronics'     THEN 7
        WHEN 'Office Supplies' THEN 3
        WHEN 'Furniture'       THEN 10
        WHEN 'Clothing'        THEN 5
        WHEN 'Sports'          THEN 4
        ELSE 7
    END AS lead_time_min,
    CASE category
        WHEN 'Electronics'     THEN 21
        WHEN 'Office Supplies' THEN 10
        WHEN 'Furniture'       THEN 30
        WHEN 'Clothing'        THEN 14
        WHEN 'Sports'          THEN 12
        ELSE 14
    END AS lead_time_max
FROM raw_orders
WHERE product_id IS NOT NULL
  AND product_name IS NOT NULL
  AND unit_price > 0
  AND unit_cost > 0;

-- ── Populate dim_geography ────────────────────────────────────────────────────
INSERT OR IGNORE INTO dim_geography (market, region)
SELECT DISTINCT market, region
FROM raw_orders
WHERE market IS NOT NULL AND region IS NOT NULL;

-- ── Populate fact_order_lines (cleaned) ───────────────────────────────────────
-- Rules applied:
--   1. Exclude cancelled orders (keep for analysis table, not forecasting base)
--   2. Exclude rows where quantity <= 0 or price <= 0
--   3. Exclude rows with NULL order_date
--   4. Cap discount_rate at 0.5 (values > 0.5 treated as data entry errors)
--   5. Recalculate line_total and line_profit from clean inputs
INSERT OR IGNORE INTO fact_order_lines (
    order_item_id, order_id, order_date, shipping_date,
    order_status, shipping_mode,
    days_for_shipping_real, days_for_shipment_scheduled, late_delivery_risk,
    product_id, quantity, unit_price, unit_cost,
    discount_rate, sale_price, line_total, line_profit, profit_ratio,
    region, market, customer_segment
)
SELECT
    r.order_item_id,
    r.order_id,
    r.order_date,
    r.shipping_date,
    r.order_status,
    r.shipping_mode,
    r.days_for_shipping_real,
    r.days_for_shipment_scheduled,
    COALESCE(r.late_delivery_risk, 0),
    r.product_id,
    r.order_item_quantity                                              AS quantity,
    r.unit_price,
    r.unit_cost,
    MIN(r.order_item_discount_rate, 0.5)                              AS discount_rate,
    ROUND(r.unit_price * (1 - MIN(r.order_item_discount_rate, 0.5)), 4) AS sale_price,
    ROUND(r.unit_price * (1 - MIN(r.order_item_discount_rate, 0.5))
          * r.order_item_quantity, 2)                                 AS line_total,
    ROUND((r.unit_price * (1 - MIN(r.order_item_discount_rate, 0.5))
           - r.unit_cost) * r.order_item_quantity, 2)                 AS line_profit,
    r.order_item_profit_ratio                                         AS profit_ratio,
    r.region,
    r.market,
    r.customer_segment
FROM raw_orders r
WHERE r.order_date IS NOT NULL
  AND r.order_item_quantity > 0
  AND r.unit_price > 0
  AND r.unit_cost > 0
  AND r.product_id IS NOT NULL
  AND r.order_status != 'Cancelled';
