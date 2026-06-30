"""
Prophet Demand Forecasting Model (Meta/Facebook)

Prophet advantages for supply chain:
  - Handles multiple seasonality (weekly + annual) natively
  - Robust to missing data and outliers
  - Interpretable decomposition: trend + seasonality + holidays
  - No feature engineering required — learns patterns directly from dates

Used here as a cross-check against XGBoost. Both should beat Holt-Winters;
if Prophet and XGBoost agree on direction, it strengthens the forecast.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

from src.models.baseline import mape, rmse

US_HOLIDAYS = pd.DataFrame({
    "holiday": [
        "New Year's Day", "MLK Day", "Presidents Day",
        "Memorial Day", "Independence Day", "Labor Day",
        "Thanksgiving", "Black Friday", "Christmas",
    ],
    "ds": pd.to_datetime([
        "2022-01-01", "2022-01-17", "2022-02-21",
        "2022-05-30", "2022-07-04", "2022-09-05",
        "2022-11-24", "2022-11-25", "2022-12-25",
    ]),
    "lower_window": [-1, 0, 0, -1, -1, -1, -2, 0, -3],
    "upper_window": [1, 0, 0, 0, 2, 0, 1, 3, 1],
})

US_HOLIDAYS_2023 = pd.DataFrame({
    "holiday": [
        "New Year's Day", "MLK Day", "Presidents Day",
        "Memorial Day", "Independence Day", "Labor Day",
        "Thanksgiving", "Black Friday", "Christmas",
    ],
    "ds": pd.to_datetime([
        "2023-01-01", "2023-01-16", "2023-02-20",
        "2023-05-29", "2023-07-04", "2023-09-04",
        "2023-11-23", "2023-11-24", "2023-12-25",
    ]),
    "lower_window": [-1, 0, 0, -1, -1, -1, -2, 0, -3],
    "upper_window": [1, 0, 0, 0, 2, 0, 1, 3, 1],
})

ALL_HOLIDAYS = pd.concat([US_HOLIDAYS, US_HOLIDAYS_2023], ignore_index=True)


def run_prophet_single_sku(
    series_df: pd.DataFrame,
    date_col: str = "week_start",
    target_col: str = "total_quantity",
    train_frac: float = 0.75,
    include_holidays: bool = True,
) -> dict:
    """
    Run Prophet on a single SKU × Region time series.

    Parameters
    ----------
    series_df : pd.DataFrame
        Must have date_col and target_col columns, sorted by date.
    Returns
    -------
    dict with y_true, y_pred, mape, rmse, residuals, residual_std, model, forecast
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed. Run: pip install prophet")

    df_prophet = series_df[[date_col, target_col]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)
    df_prophet["y"] = df_prophet["y"].clip(lower=0)

    n = len(df_prophet)
    if n < 16:
        return None

    train_size = max(int(n * train_frac), 12)
    train_df = df_prophet.iloc[:train_size]
    test_df = df_prophet.iloc[train_size:]

    if len(test_df) == 0:
        return None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=5.0,
        holidays=ALL_HOLIDAYS if include_holidays else None,
        interval_width=0.95,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="W")
    forecast = model.predict(future)

    test_forecast = forecast.iloc[train_size:train_size + len(test_df)]
    y_pred = np.clip(test_forecast["yhat"].values, 0, None)
    y_true = test_df["y"].values
    residuals = y_true - y_pred

    return {
        "model": model,
        "model_name": "Prophet",
        "train_size": train_size,
        "test_size": len(test_df),
        "y_true": y_true,
        "y_pred": y_pred,
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "residuals": residuals,
        "residual_std": float(np.std(residuals)),
        "forecast_df": forecast,
    }


def run_prophet_all_skus(
    df: pd.DataFrame,
    group_cols: list = None,
    target_col: str = "total_quantity",
    train_frac: float = 0.75,
    min_weeks: int = 26,
    top_n_skus: int = None,
) -> pd.DataFrame:
    """
    Run Prophet on each SKU × Region series.

    For large datasets, set top_n_skus to limit computation
    (e.g., top 20 SKUs by revenue for the portfolio demo).

    Returns
    -------
    pd.DataFrame with per-series MAPE, RMSE, residual_std
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed.")

    if group_cols is None:
        group_cols = ["product_id", "region"]

    if top_n_skus is not None:
        top_skus = (
            df.groupby("product_id")["total_quantity"].sum()
            .nlargest(top_n_skus).index.tolist()
        )
        df = df[df["product_id"].isin(top_skus)]

    records = []
    groups = list(df.groupby(group_cols))
    total = len(groups)

    for i, (keys, grp) in enumerate(groups):
        if i % 20 == 0:
            print(f"  Prophet: {i}/{total} series...", end="\r")

        grp_sorted = grp.sort_values("week_start")
        if len(grp_sorted) < min_weeks:
            continue

        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        category = grp_sorted["category"].iloc[0] if "category" in grp_sorted.columns else ""

        result = run_prophet_single_sku(grp_sorted, train_frac=train_frac)
        if result is None:
            continue

        records.append({
            **key_dict,
            "category": category,
            "model": "Prophet",
            "mape": result["mape"],
            "rmse": result["rmse"],
            "residual_std": result["residual_std"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
        })

    print(f"  Prophet: {total}/{total} series complete.  ")
    return pd.DataFrame(records)
