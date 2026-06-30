-- =============================================================================
-- 04_analysis_queries.sql
-- Business analysis queries — run interactively or via sql_pipeline.py
-- These answer the core business question:
--   "Which products/regions are losing money to stockouts vs. overstock?"
-- =============================================================================

-- ── Q1: Top 10 SKUs by total revenue ─────────────────────────────────────────
SELECT
    p.product_name,
    p.category,
    SUM(f.line_total)                              AS total_revenue,
    SUM(f.quantity)                                AS total_units_sold,
    ROUND(SUM(f.line_profit) / SUM(f.line_total) * 100, 1) AS avg_margin_pct
FROM fact_order_lines f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY f.product_id
ORDER BY total_revenue DESC
LIMIT 10;

-- ── Q2: Revenue by region and category ───────────────────────────────────────
SELECT
    f.region,
    p.category,
    SUM(f.line_total)   AS total_revenue,
    SUM(f.line_profit)  AS total_profit,
    SUM(f.quantity)     AS total_units
FROM fact_order_lines f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY f.region, p.category
ORDER BY total_revenue DESC;

-- ── Q3: Weekly demand volatility per SKU (used for safety stock sizing) ───────
-- High std_dev / avg ratio = high demand uncertainty = more safety stock needed
SELECT
    w.product_id,
    w.product_name,
    w.category,
    COUNT(*)                                                    AS week_count,
    ROUND(AVG(w.total_quantity), 1)                             AS avg_weekly_demand,
    ROUND(MAX(w.total_quantity), 0)                             AS max_weekly_demand,
    ROUND(MIN(w.total_quantity), 0)                             AS min_weekly_demand,
    -- Coefficient of variation proxy: range / avg (SQLite has no STDDEV)
    ROUND(
        CAST(MAX(w.total_quantity) - MIN(w.total_quantity) AS REAL)
        / NULLIF(AVG(w.total_quantity), 0), 3
    )                                                           AS demand_range_ratio
FROM agg_weekly_demand w
GROUP BY w.product_id
ORDER BY demand_range_ratio DESC;

-- ── Q4: Late delivery rate by shipping mode and region ────────────────────────
SELECT
    f.region,
    f.shipping_mode,
    COUNT(*)                                            AS shipments,
    ROUND(AVG(f.late_delivery_risk) * 100, 1)          AS late_pct,
    ROUND(AVG(f.days_for_shipping_real), 1)             AS avg_actual_days,
    ROUND(AVG(f.days_for_shipment_scheduled), 1)        AS avg_scheduled_days
FROM fact_order_lines f
GROUP BY f.region, f.shipping_mode
ORDER BY late_pct DESC;

-- ── Q5: Profit margin by category and customer segment ───────────────────────
SELECT
    p.category,
    f.customer_segment,
    ROUND(SUM(f.line_profit) / NULLIF(SUM(f.line_total), 0) * 100, 1) AS margin_pct,
    SUM(f.line_profit)   AS total_profit,
    SUM(f.line_total)    AS total_revenue,
    COUNT(*)             AS order_lines
FROM fact_order_lines f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.category, f.customer_segment
ORDER BY total_profit DESC;

-- ── Q6: Month-over-month revenue trend using LAG window function ──────────────
SELECT
    year,
    month,
    ROUND(SUM(total_revenue), 0)                            AS monthly_revenue,
    LAG(ROUND(SUM(total_revenue), 0)) OVER (
        ORDER BY year, month
    )                                                       AS prev_month_revenue,
    ROUND(
        (SUM(total_revenue) - LAG(SUM(total_revenue)) OVER (ORDER BY year, month))
        / NULLIF(LAG(SUM(total_revenue)) OVER (ORDER BY year, month), 0) * 100,
    1)                                                      AS mom_growth_pct
FROM agg_monthly_sku
GROUP BY year, month
ORDER BY year, month;

-- ── Q7: Running 12-week cumulative demand per category ────────────────────────
-- Demonstrates SUM() OVER (PARTITION BY ... ORDER BY ...)
SELECT
    week_start,
    category,
    SUM(total_quantity)                                     AS weekly_units,
    SUM(SUM(total_quantity)) OVER (
        PARTITION BY category
        ORDER BY week_start
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )                                                       AS rolling_12w_units
FROM agg_weekly_demand
GROUP BY week_start, category
ORDER BY category, week_start;

-- ── Q8: Discount rate impact on profit margins ────────────────────────────────
SELECT
    CASE
        WHEN discount_rate = 0.0          THEN '0% (no discount)'
        WHEN discount_rate <= 0.05        THEN '1-5%'
        WHEN discount_rate <= 0.10        THEN '6-10%'
        WHEN discount_rate <= 0.20        THEN '11-20%'
        ELSE '>20%'
    END                                                     AS discount_bucket,
    COUNT(*)                                                AS order_lines,
    ROUND(AVG(profit_ratio) * 100, 1)                       AS avg_profit_margin_pct,
    ROUND(SUM(line_profit), 0)                              AS total_profit,
    ROUND(SUM(line_total), 0)                               AS total_revenue
FROM fact_order_lines
GROUP BY discount_bucket
ORDER BY avg_profit_margin_pct DESC;
