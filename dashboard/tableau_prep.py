"""
Tableau Dashboard Prep

Exports clean, Tableau-ready CSV files for 4 dashboard views:

  View 1 — Forecast vs. Actual (SKU/Category level)
  View 2 — Stockout/Overstock Heatmap (Region × Product)
  View 3 — $ Impact KPI Summary
  View 4 — Model Accuracy Comparison

Each CSV is structured to minimize Tableau joins — one flat file per view.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
EXPORT_DIR = Path("data/exports")
DASHBOARD_DIR = Path("dashboard/tableau_data")


def export_view1_forecast_vs_actual(
    weekly_demand: pd.DataFrame,
    xgb_results: dict = None,
    hw_results_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    View 1: Weekly forecast vs. actual demand by SKU and category.
    Shows how each model performs over time.
    """
    df = weekly_demand[["week_start", "product_id", "product_name", "category",
                         "region", "total_quantity"]].copy()
    df = df.rename(columns={"total_quantity": "actual_demand"})

    if xgb_results is not None and "test_df" in xgb_results:
        test_df = xgb_results["test_df"].copy()
        test_df["xgb_forecast"] = xgb_results["y_pred"]
        df = df.merge(
            test_df[["week_start", "product_id", "region", "xgb_forecast"]],
            on=["week_start", "product_id", "region"],
            how="left",
        )
    else:
        df["xgb_forecast"] = np.nan

    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["category", "product_id", "region", "week_start"])

    return df


def export_view2_overstock_heatmap(
    impact_sku_df: pd.DataFrame,
    weekly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    View 2: Overstock vs. stockout cost heatmap by region × category.
    The $ amounts make this the most compelling view for non-technical stakeholders.
    """
    weekly_demand_agg = (
        weekly_demand.groupby(["category", "region"])
        .agg(
            avg_weekly_demand=("total_quantity", "mean"),
            total_units_sold=("total_quantity", "sum"),
            total_revenue=("total_revenue", "sum"),
        )
        .reset_index()
    )

    if "category" not in impact_sku_df.columns:
        impact_df = impact_sku_df.merge(
            weekly_demand[["product_id", "category"]].drop_duplicates(),
            on="product_id",
            how="left",
        )
    else:
        impact_df = impact_sku_df.copy()

    heatmap = (
        impact_df.groupby(["category", "region"])
        .agg(
            total_overstock_cost_baseline=("overstock_cost_baseline", "sum"),
            total_overstock_cost_improved=("overstock_cost_improved", "sum"),
            total_stockout_cost_baseline=("stockout_cost_baseline", "sum"),
            total_stockout_cost_improved=("stockout_cost_improved", "sum"),
            total_savings=("savings", "sum"),
            n_skus=("savings", "count"),
        )
        .reset_index()
    )

    heatmap = heatmap.merge(weekly_demand_agg, on=["category", "region"], how="left")
    heatmap["savings_per_sku"] = (heatmap["total_savings"] / heatmap["n_skus"]).round(0)

    return heatmap.round(2)


def export_view3_kpi_summary(impact: dict, model_summary: pd.DataFrame) -> pd.DataFrame:
    """
    View 3: KPI card data — the single $ number that stops the hiring manager.
    Structured for Tableau KPI tiles.
    """
    kpis = [
        {"kpi": "Total Annual Savings", "value": impact["total_annual_savings"],
         "format": "currency", "note": f"{impact['savings_pct']:.1f}% cost reduction"},
        {"kpi": "Overstock Cost Savings", "value": impact["overstock_savings"],
         "format": "currency", "note": "Lower safety stock from better forecasting"},
        {"kpi": "Stockout Cost Savings", "value": impact["stockout_savings"],
         "format": "currency", "note": "Fewer lost sales events"},
        {"kpi": "Baseline Annual Inv. Cost", "value": impact["total_cost_baseline"],
         "format": "currency", "note": f"Using {impact['model_baseline']}"},
        {"kpi": "Improved Annual Inv. Cost", "value": impact["total_cost_improved"],
         "format": "currency", "note": f"Using {impact['model_improved']}"},
    ]

    best_mape = model_summary["mean_mape"].min()
    naive_mape = model_summary.loc[model_summary["model"] == "Naive", "mean_mape"].values
    hw_mape = model_summary.loc[model_summary["model"] == "Holt-Winters", "mean_mape"].values

    if len(naive_mape) > 0:
        kpis.append({
            "kpi": "MAPE Improvement vs Naive",
            "value": round((naive_mape[0] - best_mape) / naive_mape[0] * 100, 1),
            "format": "percent",
            "note": f"Naive: {naive_mape[0]:.1f}% → Best: {best_mape:.1f}%",
        })
    if len(hw_mape) > 0:
        kpis.append({
            "kpi": "MAPE Improvement vs Holt-Winters",
            "value": round((hw_mape[0] - best_mape) / hw_mape[0] * 100, 1),
            "format": "percent",
            "note": f"H-W: {hw_mape[0]:.1f}% → Best: {best_mape:.1f}%",
        })

    return pd.DataFrame(kpis)


def export_view4_model_accuracy(model_summary: pd.DataFrame) -> pd.DataFrame:
    """
    View 4: Model accuracy comparison bar chart data.
    """
    df = model_summary[["model", "mean_mape", "median_mape", "mean_rmse",
                          "vs_naive_improvement_pct", "vs_hw_improvement_pct",
                          "n_series"]].copy()

    model_notes = {
        "Naive": "Persistence baseline — last observed value repeated",
        "Holt-Winters": "Exponential smoothing — standard classroom approach",
        "XGBoost": "Gradient boosting on engineered features — best accuracy",
        "Prophet": "Meta's time series model — handles seasonality/holidays",
    }
    df["model_note"] = df["model"].map(model_notes).fillna("")
    df["model_order"] = df["model"].map(
        {"Naive": 1, "Holt-Winters": 2, "XGBoost": 3, "Prophet": 4}
    ).fillna(5)
    df = df.sort_values("model_order")

    return df


def export_all(
    weekly_demand: pd.DataFrame,
    impact: dict,
    model_summary: pd.DataFrame,
    xgb_results: dict = None,
) -> None:
    """Export all 4 Tableau view files."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting Tableau data files...")

    v1 = export_view1_forecast_vs_actual(weekly_demand, xgb_results)
    v1.to_csv(DASHBOARD_DIR / "view1_forecast_vs_actual.csv", index=False)
    print(f"  View 1 (forecast vs actual): {len(v1):,} rows")

    if "sku_breakdown" in impact:
        sku_df = impact["sku_breakdown"].copy()
        if "category" not in sku_df.columns:
            cat_map = weekly_demand[["product_id", "category"]].drop_duplicates()
            sku_df = sku_df.merge(cat_map, on="product_id", how="left")

        v2 = export_view2_overstock_heatmap(sku_df, weekly_demand)
        v2.to_csv(DASHBOARD_DIR / "view2_overstock_heatmap.csv", index=False)
        print(f"  View 2 (heatmap):            {len(v2):,} rows")

    v3 = export_view3_kpi_summary(impact, model_summary)
    v3.to_csv(DASHBOARD_DIR / "view3_kpi_summary.csv", index=False)
    print(f"  View 3 (KPI summary):        {len(v3):,} rows")

    v4 = export_view4_model_accuracy(model_summary)
    v4.to_csv(DASHBOARD_DIR / "view4_model_accuracy.csv", index=False)
    print(f"  View 4 (model accuracy):     {len(v4):,} rows")

    weekly_demand.to_csv(DASHBOARD_DIR / "weekly_demand_full.csv", index=False)
    print(f"  Full weekly demand:          {len(weekly_demand):,} rows")

    print(f"\nAll Tableau files exported to {DASHBOARD_DIR}/")
    print("Open Tableau Public → Connect to Text File → select any CSV above.")
