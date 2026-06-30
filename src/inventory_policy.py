"""
Inventory Policy Module

Calculates safety stock, reorder point, and Economic Order Quantity (EOQ)
for each SKU × Region combination using the forecast and its error distribution.

Key insight: better forecast accuracy → smaller residual std dev → smaller
safety stock needed → lower carrying cost for the same service level.
This is how improved forecasting translates into $ savings.

Formulas:
  Safety Stock  = Z * σ_d * √L
  Reorder Point = μ_d * L + Safety Stock
  EOQ           = √(2 * D * S / H)

Where:
  Z = service level Z-score (1.28=90%, 1.65=95%, 2.05=98%)
  σ_d = weekly demand std dev (from forecast residuals)
  L = lead time in weeks
  μ_d = average weekly demand (forecast)
  D = annual demand
  S = fixed order cost (assumed)
  H = holding cost per unit per year = unit_cost * holding_rate
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

EXPORT_DIR = Path("data/exports")

SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
}

DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_HOLDING_RATE = 0.20
DEFAULT_ORDER_COST = 50.0
WEEKS_PER_YEAR = 52


def safety_stock(
    demand_std: float,
    lead_time_weeks: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """
    Safety Stock = Z × σ_d × √L

    Parameters
    ----------
    demand_std : float
        Standard deviation of weekly demand (or forecast residuals).
    lead_time_weeks : float
        Average replenishment lead time in weeks.
    service_level : float
        Target in-stock probability (0.90, 0.95, 0.98, 0.99).
    """
    z = SERVICE_LEVEL_Z.get(service_level, 1.645)
    return max(0.0, z * demand_std * np.sqrt(lead_time_weeks))


def reorder_point(
    avg_weekly_demand: float,
    lead_time_weeks: float,
    ss: float,
) -> float:
    """
    Reorder Point = μ_d × L + Safety Stock

    The on-hand quantity at which a replenishment order should be placed.
    """
    return max(0.0, avg_weekly_demand * lead_time_weeks + ss)


def economic_order_quantity(
    annual_demand: float,
    unit_cost: float,
    order_cost: float = DEFAULT_ORDER_COST,
    holding_rate: float = DEFAULT_HOLDING_RATE,
) -> float:
    """
    EOQ = √(2DS/H)  where H = unit_cost × holding_rate

    Minimizes total annual ordering cost + carrying cost.
    """
    h = unit_cost * holding_rate
    if h <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * order_cost / h))


def calculate_inventory_policy(
    sku_stats: pd.DataFrame,
    product_catalog: pd.DataFrame = None,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    holding_rate: float = DEFAULT_HOLDING_RATE,
    order_cost: float = DEFAULT_ORDER_COST,
) -> pd.DataFrame:
    """
    Calculate inventory policy for all SKUs.

    Parameters
    ----------
    sku_stats : pd.DataFrame
        Must contain: product_id, region, avg_weekly_demand (or similar),
                      residual_std (from model evaluation)
    product_catalog : pd.DataFrame
        Must contain: product_id, unit_cost, lead_time_days_min, lead_time_days_max
    """
    df = sku_stats.copy()

    if product_catalog is not None:
        df = df.merge(
            product_catalog[["product_id", "unit_cost", "lead_time_days_min", "lead_time_days_max"]],
            on="product_id",
            how="left",
        )
    else:
        df["unit_cost"] = df.get("unit_cost", 20.0)
        df["lead_time_days_min"] = df.get("lead_time_days_min", 7)
        df["lead_time_days_max"] = df.get("lead_time_days_max", 14)

    df["lead_time_weeks"] = (
        (df["lead_time_days_min"] + df["lead_time_days_max"]) / 2 / 7
    ).clip(lower=0.5)

    demand_col = next(
        (c for c in ["avg_weekly_demand", "mean_weekly_demand", "avg_demand"] if c in df.columns),
        None,
    )
    if demand_col is None:
        raise ValueError("sku_stats must contain avg_weekly_demand or similar column")
    df = df.rename(columns={demand_col: "avg_weekly_demand"})

    std_col = next(
        (c for c in ["residual_std", "demand_std", "weekly_std"] if c in df.columns),
        None,
    )
    if std_col is None:
        df["residual_std"] = df["avg_weekly_demand"] * 0.25
    else:
        df = df.rename(columns={std_col: "residual_std"})

    df["residual_std"] = df["residual_std"].fillna(df["avg_weekly_demand"] * 0.25).clip(lower=0.1)

    df["safety_stock_units"] = df.apply(
        lambda row: safety_stock(
            row["residual_std"], row["lead_time_weeks"], service_level
        ),
        axis=1,
    ).round(1)

    df["reorder_point_units"] = df.apply(
        lambda row: reorder_point(
            row["avg_weekly_demand"], row["lead_time_weeks"], row["safety_stock_units"]
        ),
        axis=1,
    ).round(1)

    df["annual_demand"] = df["avg_weekly_demand"] * WEEKS_PER_YEAR

    df["eoq_units"] = df.apply(
        lambda row: economic_order_quantity(
            row["annual_demand"], row["unit_cost"], order_cost, holding_rate
        ),
        axis=1,
    ).round(1)

    df["annual_holding_cost"] = (
        df["safety_stock_units"] * df["unit_cost"] * holding_rate
    ).round(2)

    df["annual_order_cost"] = (
        (df["annual_demand"] / df["eoq_units"].replace(0, np.nan)) * order_cost
    ).fillna(0).round(2)

    df["total_annual_inventory_cost"] = (
        df["annual_holding_cost"] + df["annual_order_cost"]
    ).round(2)

    df["service_level"] = service_level
    df["holding_rate"] = holding_rate

    return df


def compare_inventory_costs(
    policy_baseline: pd.DataFrame,
    policy_improved: pd.DataFrame,
    model_baseline: str = "Holt-Winters",
    model_improved: str = "XGBoost",
) -> pd.DataFrame:
    """
    Compare annual inventory costs between baseline and improved forecast models.

    The key comparison:
      baseline (Holt-Winters) → larger residual_std → more safety stock → higher cost
      improved (XGBoost)      → smaller residual_std → less safety stock → lower cost
    """
    merged = policy_baseline[["product_id", "region", "safety_stock_units",
                               "annual_holding_cost", "total_annual_inventory_cost",
                               "residual_std"]].merge(
        policy_improved[["product_id", "region", "safety_stock_units",
                          "annual_holding_cost", "total_annual_inventory_cost",
                          "residual_std"]],
        on=["product_id", "region"],
        suffixes=(f"_{model_baseline.lower().replace('-','_')}",
                  f"_{model_improved.lower()}"),
    )

    baseline_col = f"total_annual_inventory_cost_{model_baseline.lower().replace('-','_')}"
    improved_col = f"total_annual_inventory_cost_{model_improved.lower()}"

    if baseline_col not in merged.columns or improved_col not in merged.columns:
        baseline_col = [c for c in merged.columns if "total_annual" in c][0]
        improved_col = [c for c in merged.columns if "total_annual" in c][1]

    merged["annual_savings"] = (merged[baseline_col] - merged[improved_col]).round(2)
    merged["savings_pct"] = (
        merged["annual_savings"] / merged[baseline_col].replace(0, np.nan) * 100
    ).round(1)

    return merged


def print_policy_summary(policy_df: pd.DataFrame, title: str = "Inventory Policy") -> None:
    print(f"\n── {title} ──────────────────────────────────────────────────")
    print(f"  SKUs × Regions: {len(policy_df)}")
    print(f"  Total Safety Stock (units): {policy_df['safety_stock_units'].sum():,.0f}")
    print(f"  Avg Safety Stock per SKU:   {policy_df['safety_stock_units'].mean():,.1f}")
    print(f"  Total Annual Holding Cost:  ${policy_df['annual_holding_cost'].sum():,.0f}")
    print(f"  Total Annual Order Cost:    ${policy_df['annual_order_cost'].sum():,.0f}")
    print(f"  Total Annual Inv. Cost:     ${policy_df['total_annual_inventory_cost'].sum():,.0f}")
    print("─" * 60)
