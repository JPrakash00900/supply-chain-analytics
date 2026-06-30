"""
Feature Engineering — transforms weekly SKU-level demand into model-ready features.

Features created:
  - Lag demand: 1w, 2w, 4w, 8w, 12w
  - Rolling averages: 4w, 8w, 12w (mean and std dev)
  - Rolling min/max: 4w window
  - Trend: linear slope over trailing 8 weeks
  - Calendar: week_of_year, month, quarter, is_weekend proxy
  - Seasonality: sin/cos Fourier terms (annual cycle)
  - YoY growth: same week prior year
  - Promo indicator: weeks where avg_discount_rate > 0.10
  - Late delivery rate (proxy for supply disruption)
  - Demand momentum: (current - lag4) / lag4

Usage:
    from src.feature_engineering import build_feature_matrix
    df_features = build_feature_matrix(weekly_demand_df)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
EXPORT_DIR = Path("data/exports")

LAG_WEEKS = [1, 2, 4, 8, 12]
ROLLING_WINDOWS = [4, 8, 12]
MIN_HISTORY_WEEKS = 14


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "total_quantity",
    group_cols: list = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from the weekly demand DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: week_start, product_id, region, total_quantity,
                      avg_discount_rate, late_delivery_pct (optional)
    target_col : str
        Column to forecast (default: total_quantity)
    group_cols : list
        Columns defining a time series identity (default: product_id + region)

    Returns
    -------
    pd.DataFrame with lag features, rolling stats, and calendar features.
    Rows with insufficient history (< MIN_HISTORY_WEEKS lags) are retained
    but their lag columns will be NaN — handled downstream by each model.
    """
    if group_cols is None:
        group_cols = ["product_id", "region"]

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(group_cols + ["week_start"]).reset_index(drop=True)

    if "late_delivery_pct" not in df.columns:
        df["late_delivery_pct"] = 0.0
    if "avg_discount_rate" not in df.columns:
        df["avg_discount_rate"] = 0.0

    grp = df.groupby(group_cols)

    for lag in LAG_WEEKS:
        df[f"lag_{lag}w"] = grp[target_col].shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = grp[target_col].shift(1)
        rolling = shifted.groupby(df[group_cols].apply(tuple, axis=1)).transform(
            lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean()
        )
        df[f"roll_mean_{window}w"] = grp[target_col].shift(1).transform(
            lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean()
        )
        df[f"roll_std_{window}w"] = grp[target_col].shift(1).transform(
            lambda x: x.rolling(window, min_periods=max(1, window // 2)).std().fillna(0)
        )

    df["roll_min_4w"] = grp[target_col].shift(1).transform(
        lambda x: x.rolling(4, min_periods=2).min()
    )
    df["roll_max_4w"] = grp[target_col].shift(1).transform(
        lambda x: x.rolling(4, min_periods=2).max()
    )

    def _trend_slope(series: pd.Series, window: int = 8) -> pd.Series:
        slopes = []
        for i in range(len(series)):
            start = max(0, i - window)
            window_vals = series.iloc[start:i].values
            if len(window_vals) < 3:
                slopes.append(np.nan)
            else:
                x = np.arange(len(window_vals))
                slope = np.polyfit(x, window_vals, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    df["trend_slope_8w"] = grp[target_col].transform(_trend_slope)

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["year"] = df["week_start"].dt.year

    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_q4"] = (df["quarter"] == 4).astype(int)
    df["is_q1"] = (df["quarter"] == 1).astype(int)

    df["is_promo_week"] = (df["avg_discount_rate"] > 0.10).astype(int)

    df["supply_disruption"] = (df["late_delivery_pct"] > 25).astype(int)

    yoy_key = df[group_cols + ["week_of_year", "year", target_col]].copy()
    yoy_key["prev_year"] = yoy_key["year"] - 1
    yoy_merged = yoy_key.merge(
        yoy_key[group_cols + ["week_of_year", "year", target_col]].rename(
            columns={"year": "prev_year", target_col: "yoy_qty"}
        ),
        on=group_cols + ["week_of_year", "prev_year"],
        how="left",
    )
    df["yoy_qty"] = yoy_merged["yoy_qty"].values
    df["yoy_growth"] = (df[target_col] - df["yoy_qty"]) / df["yoy_qty"].replace(0, np.nan)

    df["demand_momentum"] = (df[target_col] - df["lag_4w"]) / df["lag_4w"].replace(0, np.nan)

    return df


def load_and_build(
    weekly_csv: Path = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Load weekly demand CSV and build full feature matrix."""
    if weekly_csv is None:
        weekly_csv = PROCESSED_DIR / "weekly_demand.csv"

    if not weekly_csv.exists():
        raise FileNotFoundError(
            f"Weekly demand file not found at {weekly_csv}. "
            "Run the SQL pipeline first: python -m src.sql_pipeline"
        )

    print(f"Loading weekly demand from {weekly_csv}...")
    df = pd.read_csv(weekly_csv, parse_dates=["week_start"])

    print(f"  {len(df):,} rows | {df['product_id'].nunique()} SKUs | {df['region'].nunique()} regions")

    print("Building feature matrix...")
    df_feat = build_feature_matrix(df)

    print(f"  Features added: {len(df_feat.columns) - len(df.columns)}")
    print(f"  Feature matrix shape: {df_feat.shape}")

    if save_csv:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = EXPORT_DIR / "feature_matrix.csv"
        df_feat.to_csv(out, index=False)
        print(f"  Saved to {out}")

    return df_feat


def get_feature_columns() -> list:
    """Return list of feature column names used by models (excludes target and IDs)."""
    lag_cols = [f"lag_{w}w" for w in LAG_WEEKS]
    roll_cols = (
        [f"roll_mean_{w}w" for w in ROLLING_WINDOWS]
        + [f"roll_std_{w}w" for w in ROLLING_WINDOWS]
        + ["roll_min_4w", "roll_max_4w"]
    )
    trend_cols = ["trend_slope_8w"]
    calendar_cols = [
        "week_of_year", "month", "quarter", "year",
        "sin_week", "cos_week", "sin_month", "cos_month",
        "is_q4", "is_q1",
    ]
    signal_cols = ["is_promo_week", "supply_disruption", "yoy_qty", "demand_momentum"]
    return lag_cols + roll_cols + trend_cols + calendar_cols + signal_cols


if __name__ == "__main__":
    df = load_and_build()
    print("\nSample feature matrix:")
    print(df[["week_start", "product_id", "region", "total_quantity"] + get_feature_columns()[:6]].head())
