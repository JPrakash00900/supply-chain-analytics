"""
Visualizations — generates portfolio-quality charts saved to outputs/

Charts produced:
  01_demand_by_category.png     — Weekly demand time series stacked by category
  02_model_accuracy_bars.png    — MAPE comparison across 4 models
  03_forecast_vs_actual.png     — XGBoost forecast vs actual (top 6 SKUs)
  04_feature_importance.png     — XGBoost top-15 feature importance
  05_safety_stock_comparison.png— Safety stock: Holt-Winters vs XGBoost
  06_impact_waterfall.png       — $ savings waterfall chart (the money slide)
  07_overstock_heatmap.png      — Overstock cost by region × category
  08_demand_seasonality.png     — Seasonal decomposition (avg demand by week)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")

PALETTE = {
    "Naive": "#9E9E9E",
    "Holt-Winters": "#FF9800",
    "XGBoost": "#2196F3",
    "Prophet": "#4CAF50",
}

CATEGORY_PALETTE = {
    "Electronics": "#1565C0",
    "Office Supplies": "#2E7D32",
    "Furniture": "#6A1B9A",
    "Clothing": "#AD1457",
    "Sports": "#E65100",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_demand_by_category(weekly_demand: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    pivot = (
        weekly_demand.groupby(["week_start", "category"])["total_quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot.index = pd.to_datetime(pivot.index)

    colors = [CATEGORY_PALETTE.get(c, "#607D8B") for c in pivot.columns]
    pivot.plot(kind="area", stacked=True, ax=ax, color=colors, alpha=0.85, linewidth=0)

    ax.set_title("Weekly Demand by Product Category (2022–2023)")
    ax.set_xlabel("")
    ax.set_ylabel("Units Sold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "01_demand_by_category.png")


def plot_model_accuracy(model_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = model_summary["model"].tolist()
    mapes = model_summary["mean_mape"].tolist()
    rmses = model_summary["mean_rmse"].tolist()
    colors = [PALETTE.get(m, "#607D8B") for m in models]

    ax = axes[0]
    bars = ax.bar(models, mapes, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Mean MAPE by Model (lower = better)")
    ax.set_ylabel("Mean Absolute Percentage Error (%)")
    ax.set_ylim(0, max(mapes) * 1.25)

    ax = axes[1]
    bars = ax.bar(models, rmses, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Mean RMSE by Model (lower = better)")
    ax.set_ylabel("Root Mean Squared Error (units)")
    ax.set_ylim(0, max(rmses) * 1.25)

    fig.suptitle("Forecasting Model Accuracy Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_model_accuracy_bars.png")


def plot_forecast_vs_actual(
    weekly_demand: pd.DataFrame,
    xgb_results: dict,
    n_skus: int = 6,
) -> None:
    if xgb_results is None or "test_df" not in xgb_results:
        print("  Skipping forecast vs actual (no XGBoost results)")
        return

    test_df = xgb_results["test_df"].copy()
    test_df["y_pred"] = xgb_results["y_pred"]
    test_df["y_true"] = xgb_results["y_true"]

    top_skus = (
        weekly_demand.groupby("product_id")["total_quantity"].sum()
        .nlargest(n_skus).index.tolist()
    )

    n_cols = 3
    n_rows = (n_skus + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, sku in enumerate(top_skus[:n_skus]):
        ax = axes[idx]
        sku_hist = weekly_demand[weekly_demand["product_id"] == sku].sort_values("week_start")
        sku_test = test_df[test_df["product_id"] == sku].sort_values("week_start")

        ax.plot(pd.to_datetime(sku_hist["week_start"]), sku_hist["total_quantity"],
                color="#B0BEC5", linewidth=1.2, label="Historical", alpha=0.7)

        if len(sku_test) > 0:
            ax.plot(pd.to_datetime(sku_test["week_start"]), sku_test["y_true"],
                    color="#1565C0", linewidth=1.8, label="Actual")
            ax.plot(pd.to_datetime(sku_test["week_start"]), sku_test["y_pred"],
                    color="#FF5722", linewidth=1.8, linestyle="--", label="XGBoost Forecast")

        product_name = sku_hist["product_name"].iloc[0] if "product_name" in sku_hist.columns else sku
        ax.set_title(f"{product_name[:30]}", fontsize=10)
        ax.set_ylabel("Units")
        ax.tick_params(axis="x", rotation=30)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(top_skus), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("XGBoost Forecast vs. Actual Demand — Top SKUs", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "03_forecast_vs_actual.png")


def plot_feature_importance(xgb_results: dict, top_n: int = 15) -> None:
    if xgb_results is None or "feature_importance" not in xgb_results:
        print("  Skipping feature importance (no XGBoost results)")
        return

    importance = xgb_results["feature_importance"].head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance["feature"][::-1], importance["importance"][::-1],
                   color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_title(f"XGBoost Feature Importance — Top {top_n}")
    ax.set_xlabel("Importance Score")
    ax.axvline(x=importance["importance"].mean(), color="#FF9800", linestyle="--",
               linewidth=1.2, label="Mean importance")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_feature_importance.png")


def plot_safety_stock_comparison(
    policy_baseline: pd.DataFrame,
    policy_improved: pd.DataFrame,
    top_n: int = 20,
) -> None:
    if policy_baseline is None or policy_improved is None:
        return

    merged = policy_baseline[["product_id", "region", "safety_stock_units"]].merge(
        policy_improved[["product_id", "region", "safety_stock_units"]],
        on=["product_id", "region"],
        suffixes=("_baseline", "_improved"),
    )

    if "product_name" in policy_baseline.columns:
        merged = merged.merge(
            policy_baseline[["product_id", "product_name"]].drop_duplicates(),
            on="product_id", how="left"
        )
        merged["label"] = merged["product_name"].str[:25]
    else:
        merged["label"] = merged["product_id"]

    merged = merged.sort_values("safety_stock_units_baseline", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width/2, merged["safety_stock_units_baseline"], width,
           label="Holt-Winters", color=PALETTE["Holt-Winters"], alpha=0.85)
    ax.bar(x + width/2, merged["safety_stock_units_improved"], width,
           label="XGBoost", color=PALETTE["XGBoost"], alpha=0.85)

    ax.set_title(f"Safety Stock Comparison: Holt-Winters vs XGBoost — Top {top_n} SKUs")
    ax.set_ylabel("Safety Stock (units)")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    _save(fig, "05_safety_stock_comparison.png")


def plot_impact_waterfall(impact: dict) -> None:
    categories = [
        "Baseline\nInventory Cost",
        "Overstock\nSavings",
        "Stockout\nSavings",
        "Improved\nInventory Cost",
    ]
    values = [
        impact["total_cost_baseline"],
        -impact["overstock_savings"],
        -impact["stockout_savings"],
        impact["total_cost_improved"],
    ]

    running_totals = [0, impact["total_cost_baseline"],
                      impact["total_cost_baseline"] - impact["overstock_savings"],
                      0]

    colors = ["#1565C0", "#4CAF50", "#4CAF50", "#2196F3"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cat, val, bottom) in enumerate(zip(categories, values, running_totals)):
        if i == 0 or i == 3:
            ax.bar(i, abs(val), bottom=0, color=colors[i], edgecolor="white", linewidth=0.5, width=0.5)
            ax.text(i, abs(val) + abs(val) * 0.02,
                    f"${abs(val):,.0f}", ha="center", fontsize=10, fontweight="bold")
        else:
            ax.bar(i, abs(val), bottom=bottom + val, color=colors[i], edgecolor="white",
                   linewidth=0.5, width=0.5, alpha=0.85)
            ax.text(i, bottom + 0.5 * val + abs(val) * 0.01,
                    f"-${abs(val):,.0f}", ha="center", fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Annual Inventory Cost ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(
        f"Annual Inventory Cost Reduction: ${impact['total_annual_savings']:,.0f} Saved "
        f"({impact['savings_pct']:.1f}%)\n"
        f"Holt-Winters vs. XGBoost Forecast",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "06_impact_waterfall.png")


def plot_overstock_heatmap(heatmap_df: pd.DataFrame) -> None:
    if heatmap_df is None or len(heatmap_df) == 0:
        return

    if "total_overstock_cost_baseline" not in heatmap_df.columns:
        print("  Skipping heatmap (missing columns)")
        return

    pivot = heatmap_df.pivot_table(
        index="category",
        columns="region",
        values="total_overstock_cost_baseline",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot / 1000,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Annual Overstock Cost ($K)"},
        linewidths=0.5,
    )
    ax.set_title("Annual Overstock Cost Heatmap: Category × Region ($K)\n(Baseline: Holt-Winters forecast)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Category")
    fig.tight_layout()
    _save(fig, "07_overstock_heatmap.png")


def plot_demand_seasonality(weekly_demand: pd.DataFrame) -> None:
    df = weekly_demand.copy()
    df["week_of_year"] = pd.to_datetime(df["week_start"]).dt.isocalendar().week.astype(int)

    seasonal = (
        df.groupby(["week_of_year", "category"])["total_quantity"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    for category, grp in seasonal.groupby("category"):
        grp_sorted = grp.sort_values("week_of_year")
        ax.plot(
            grp_sorted["week_of_year"],
            grp_sorted["total_quantity"],
            label=category,
            color=CATEGORY_PALETTE.get(category, "#607D8B"),
            linewidth=2,
        )

    ax.set_title("Average Weekly Demand by Season (All Years Combined)")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Avg Units Sold")
    ax.axvspan(44, 52, alpha=0.08, color="red", label="Q4 Peak (Electronics)")
    ax.axvspan(32, 38, alpha=0.08, color="green", label="Back-to-School (Office)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "08_demand_seasonality.png")


def generate_all_charts(
    weekly_demand: pd.DataFrame,
    model_summary: pd.DataFrame,
    impact: dict,
    xgb_results: dict = None,
    policy_baseline: pd.DataFrame = None,
    policy_improved: pd.DataFrame = None,
    heatmap_df: pd.DataFrame = None,
) -> None:
    print("\nGenerating portfolio charts...")
    plot_demand_by_category(weekly_demand)
    plot_model_accuracy(model_summary)
    plot_forecast_vs_actual(weekly_demand, xgb_results)
    plot_feature_importance(xgb_results)
    plot_safety_stock_comparison(policy_baseline, policy_improved)
    plot_impact_waterfall(impact)
    plot_overstock_heatmap(heatmap_df)
    plot_demand_seasonality(weekly_demand)
    print(f"All charts saved to {OUTPUT_DIR}/")
