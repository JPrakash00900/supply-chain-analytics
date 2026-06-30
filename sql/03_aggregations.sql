-- =============================================================================
-- 03_aggregations.sql
-- Build weekly and monthly aggregate tables used by forecasting models
-- =============================================================================

-- ── Weekly SKU × Region demand aggregation ────────────────────────────────────
-- Uses strftime to get ISO week boundaries (Monday = week start)
-- Window: all complete weeks in the fact table date range
INSERT OR REPLACE INTO agg_weekly_demand (
    week_start, year, week_num,
    product_id, product_name, category, region,
    total_quantity, total_revenue, total_profit,
    avg_discount_rate, order_count, late_delivery_pct
)
SELECT
    DATE(f.order_date, 'weekday 0', '-6 days')     AS week_start,
    CAST(strftime('%Y', f.order_date) AS INTEGER)  AS year,
    CAST(strftime('%W', f.order_date) AS INTEGER)  AS week_num,
    f.product_id,
    p.product_name,
    p.category,
    f.region,
    SUM(f.quantity)                                AS total_quantity,
    ROUND(SUM(f.line_total), 2)                    AS total_revenue,
    ROUND(SUM(f.line_profit), 2)                   AS total_profit,
    ROUND(AVG(f.discount_rate), 4)                 AS avg_discount_rate,
    COUNT(DISTINCT f.order_id)                     AS order_count,
    ROUND(AVG(f.late_delivery_risk) * 100, 2)      AS late_delivery_pct
FROM fact_order_lines f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY
    DATE(f.order_date, 'weekday 0', '-6 days'),
    f.product_id,
    f.region;

-- ── Monthly SKU × Region aggregation ─────────────────────────────────────────
INSERT OR REPLACE INTO agg_monthly_sku (
    year, month,
    product_id, product_name, category, region,
    total_quantity, total_revenue, total_profit,
    avg_unit_cost, order_count
)
SELECT
    CAST(strftime('%Y', f.order_date) AS INTEGER)  AS year,
    CAST(strftime('%m', f.order_date) AS INTEGER)  AS month,
    f.product_id,
    p.product_name,
    p.category,
    f.region,
    SUM(f.quantity)                                AS total_quantity,
    ROUND(SUM(f.line_total), 2)                    AS total_revenue,
    ROUND(SUM(f.line_profit), 2)                   AS total_profit,
    ROUND(AVG(f.unit_cost), 4)                     AS avg_unit_cost,
    COUNT(DISTINCT f.order_id)                     AS order_count
FROM fact_order_lines f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY
    CAST(strftime('%Y', f.order_date) AS INTEGER),
    CAST(strftime('%m', f.order_date) AS INTEGER),
    f.product_id,
    f.region;
