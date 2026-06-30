"""
Model Comparison — aggregates results from all four models and builds the
"before vs. after" accuracy table for the portfolio README and dashboard.

Output: data/exports/model_comparison.csv
        data/exports/sku_model_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

EXPORT_DIR = Path("data/exports")


def aggregate_model_results(*result_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate per-model result DataFrames and compute weighted-average metrics.

    Each result_df must have columns: model, mape, rmse, [product_id, region, category]
    """
    combined = pd.concat(result_dfs, ignore_index=True)
    return combined


def build_summary_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the high-level model comparison table (mean MAPE and RMSE across all SKUs).

    This is the "headline" table for the README — formatted to show the
    improvement story: Naive → Holt-Winters → XGBoost → Prophet.
    """
    MODEL_ORDER = ["Naive", "Holt-Winters", "XGBoost", "Prophet"]

    summary = (
        combined_df
        .groupby("model")
        .agg(
            mean_mape=("mape", "mean"),
            median_mape=("mape", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            n_series=("mape", "count"),
        )
        .reset_index()
    )

    summary["mean_mape"] = summary["mean_mape"].round(1)
    summary["median_mape"] = summary["median_mape"].round(1)
    summary["mean_rmse"] = summary["mean_rmse"].round(1)
    summary["median_rmse"] = summary["median_rmse"].round(1)

    summary["model_order"] = summary["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    ).fillna(99)
    summary = summary.sort_values("model_order").drop(columns="model_order")

    naive_mape = summary.loc[summary["model"] == "Naive", "mean_mape"].values
    if len(naive_mape) > 0:
        baseline = naive_mape[0]
        summary["vs_naive_improvement_pct"] = (
            (baseline - summary["mean_mape"]) / baseline * 100
        ).round(1)
    else:
        summary["vs_naive_improvement_pct"] = np.nan

    hw_mape = summary.loc[summary["model"] == "Holt-Winters", "mean_mape"].values
    if len(hw_mape) > 0:
        hw_baseline = hw_mape[0]
        summary["vs_hw_improvement_pct"] = (
            (hw_baseline - summary["mean_mape"]) / hw_baseline * 100
        ).round(1)
    else:
        summary["vs_hw_improvement_pct"] = np.nan

    return summary


def build_category_breakdown(combined_df: pd.DataFrame) -> pd.DataFrame:
    """MAPE by model × category — shows where each model is strongest."""
    if "category" not in combined_df.columns:
        return pd.DataFrame()

    return (
        combined_df
        .groupby(["model", "category"])
        .agg(mean_mape=("mape", "mean"), n_series=("mape", "count"))
        .reset_index()
        .round({"mean_mape": 1})
        .sort_values(["category", "mean_mape"])
    )


def print_comparison_table(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print(" MODEL ACCURACY COMPARISON — Supply Chain Demand Forecasting")
    print("=" * 75)
    print(f"{'Model':<20} {'Mean MAPE':>10} {'Median MAPE':>12} {'Mean RMSE':>10} {'vs Naive':>10} {'vs H-W':>8}")
    print("-" * 75)

    for _, row in summary.iterrows():
        vs_naive = f"{row['vs_naive_improvement_pct']:+.1f}%" if not pd.isna(row.get('vs_naive_improvement_pct')) else "—"
        vs_hw = f"{row['vs_hw_improvement_pct']:+.1f}%" if not pd.isna(row.get('vs_hw_improvement_pct')) else "—"
        print(
            f"{row['model']:<20} "
            f"{row['mean_mape']:>9.1f}% "
            f"{row['median_mape']:>11.1f}% "
            f"{row['mean_rmse']:>10.1f} "
            f"{vs_naive:>10} "
            f"{vs_hw:>8}"
        )

    print("=" * 75)

    best_model = summary.loc[summary["mean_mape"].idxmin(), "model"]
    best_mape = summary["mean_mape"].min()
    naive_mape = summary.loc[summary["model"] == "Naive", "mean_mape"].values
    hw_mape = summary.loc[summary["model"] == "Holt-Winters", "mean_mape"].values

    if len(naive_mape) > 0 and len(hw_mape) > 0:
        improvement_vs_naive = (naive_mape[0] - best_mape) / naive_mape[0] * 100
        improvement_vs_hw = (hw_mape[0] - best_mape) / hw_mape[0] * 100
        print(f"\n  Best model: {best_model} ({best_mape:.1f}% MAPE)")
        print(f"  Improvement vs Naive:        {improvement_vs_naive:.0f}%")
        print(f"  Improvement vs Holt-Winters: {improvement_vs_hw:.0f}%")
        print(f"\n  Resume bullet:")
        print(f"  'Improved demand forecast accuracy by {improvement_vs_hw:.0f}% vs. exponential")
        print(f"   smoothing baseline using XGBoost with lag + seasonality features.'")

    print()


def save_results(combined_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(EXPORT_DIR / "sku_model_results.csv", index=False)
    summary.to_csv(EXPORT_DIR / "model_comparison_summary.csv", index=False)
    print(f"Saved results to {EXPORT_DIR}/")
