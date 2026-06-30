"""
Baseline Forecasting Models — Naive and Holt-Winters (Exponential Smoothing).

These serve as the "before" benchmark that XGBoost and Prophet are measured against.
Holt-Winters is the standard classroom approach; Naive (last observed value) is
the floor every model must beat.

Both models operate on a per-SKU × Region time series.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — ignores zero-demand weeks."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def naive_forecast(series: pd.Series, horizon: int = 1) -> np.ndarray:
    """Last-observed-value naive forecast (persistence model)."""
    preds = np.empty(len(series))
    preds[0] = np.nan
    for i in range(1, len(series)):
        preds[i] = series.iloc[i - 1]
    return preds


def holt_winters_cv(
    series: pd.Series,
    train_frac: float = 0.75,
    seasonal_periods: int = 52,
) -> dict:
    """
    Walk-forward cross-validation of Holt-Winters on a single time series.

    Uses an additive trend + additive seasonal model (best for supply chain
    where demand spikes are additive rather than multiplicative).

    Returns
    -------
    dict with keys: y_true, y_pred, mape, rmse, residuals
    """
    n = len(series)
    train_size = max(int(n * train_frac), seasonal_periods + 4)

    if n < seasonal_periods + 8:
        return _fallback_naive(series, train_size)

    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    if len(test) == 0:
        return _fallback_naive(series, train_size)

    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, remove_bias=False)
        y_pred = fit.forecast(len(test)).values
    except Exception:
        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            y_pred = fit.forecast(len(test)).values
        except Exception:
            return _fallback_naive(series, train_size)

    y_pred = np.clip(y_pred, 0, None)
    y_true = test.values

    residuals = y_true - y_pred

    return {
        "model": "Holt-Winters",
        "train_size": train_size,
        "test_size": len(test),
        "y_true": y_true,
        "y_pred": y_pred,
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "residuals": residuals,
        "residual_std": float(np.std(residuals)),
    }


def _fallback_naive(series: pd.Series, train_size: int) -> dict:
    test = series.iloc[train_size:]
    if len(test) == 0:
        test = series.iloc[-4:]
        train_size = len(series) - 4

    last_train_val = series.iloc[train_size - 1]
    y_pred = np.full(len(test), last_train_val)
    y_true = test.values
    residuals = y_true - y_pred

    return {
        "model": "Holt-Winters (fallback: naive)",
        "train_size": train_size,
        "test_size": len(test),
        "y_true": y_true,
        "y_pred": y_pred,
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "residuals": residuals,
        "residual_std": float(np.std(residuals)),
    }


def run_naive_cv(series: pd.Series, train_frac: float = 0.75) -> dict:
    """Cross-validated naive forecast."""
    n = len(series)
    train_size = max(int(n * train_frac), 4)
    test = series.iloc[train_size:]

    if len(test) == 0:
        test = series.iloc[-4:]
        train_size = len(series) - 4

    last_train_val = series.iloc[train_size - 1]
    y_pred = np.full(len(test), last_train_val)
    y_true = test.values
    residuals = y_true - y_pred

    return {
        "model": "Naive",
        "train_size": train_size,
        "test_size": len(test),
        "y_true": y_true,
        "y_pred": y_pred,
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "residuals": residuals,
        "residual_std": float(np.std(residuals)),
    }


def evaluate_baselines_all_skus(
    df: pd.DataFrame,
    group_cols: list = None,
    target_col: str = "total_quantity",
    train_frac: float = 0.75,
    min_weeks: int = 26,
) -> pd.DataFrame:
    """
    Run both Naive and Holt-Winters on every SKU × Region series.

    Returns
    -------
    pd.DataFrame with columns:
        product_id, region, category, model, mape, rmse, residual_std,
        train_size, test_size
    """
    if group_cols is None:
        group_cols = ["product_id", "region"]

    records = []
    groups = df.groupby(group_cols)

    for keys, grp in groups:
        grp_sorted = grp.sort_values("week_start")
        series = grp_sorted[target_col].reset_index(drop=True)

        if len(series) < min_weeks:
            continue

        if isinstance(keys, str):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        category = grp_sorted["category"].iloc[0] if "category" in grp_sorted.columns else ""

        for model_name, result in [
            ("Naive", run_naive_cv(series, train_frac)),
            ("Holt-Winters", holt_winters_cv(series, train_frac)),
        ]:
            records.append({
                **key_dict,
                "category": category,
                "model": model_name,
                "mape": result["mape"],
                "rmse": result["rmse"],
                "residual_std": result["residual_std"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
            })

    return pd.DataFrame(records)
