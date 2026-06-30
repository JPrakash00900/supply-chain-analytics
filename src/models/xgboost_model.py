"""
XGBoost Demand Forecasting Model

Approach: treat demand forecasting as a supervised regression problem.
Each row is a (SKU, region, week) observation; features are lag values,
rolling statistics, calendar encodings, and promo signals.

Key advantage over Holt-Winters: captures non-linear interactions between
lag demand, seasonality, and promotions — e.g., demand after a promo week
often drops (post-promo dip), which Holt-Winters cannot model.

Train/test split: time-based (no random shuffle) to prevent data leakage.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.feature_engineering import get_feature_columns
from src.models.baseline import mape, rmse


XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["product_id", "category", "region"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


def train_global_xgboost(
    df: pd.DataFrame,
    target_col: str = "total_quantity",
    train_frac: float = 0.75,
    feature_cols: list = None,
) -> dict:
    """
    Train a single global XGBoost model across all SKUs and regions.

    A global model (one model for all series) is preferred when individual
    series are too short for per-SKU models. It learns cross-SKU patterns
    (e.g., Electronics spikes in Q4 across all regions).

    Returns
    -------
    dict with: model, feature_importance, y_true, y_pred, mape, rmse,
               residuals, residual_std, feature_cols
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    if feature_cols is None:
        feature_cols = get_feature_columns()

    df = _encode_categoricals(df)
    extra_cols = [c for c in ["product_id_enc", "category_enc", "region_enc"] if c in df.columns]
    all_feature_cols = [c for c in feature_cols + extra_cols if c in df.columns]

    df = df.sort_values("week_start").reset_index(drop=True)
    df_clean = df.dropna(subset=[c for c in all_feature_cols if c in df.columns] + [target_col])

    split_idx = int(len(df_clean) * train_frac)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    if len(test_df) == 0:
        raise ValueError("No test data after train/test split. Need more data.")

    X_train = train_df[all_feature_cols].fillna(0)
    y_train = train_df[target_col].values
    X_test = test_df[all_feature_cols].fillna(0)
    y_test = test_df[target_col].values

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = np.clip(model.predict(X_test), 0, None)
    residuals = y_test - y_pred

    importance = pd.DataFrame({
        "feature": all_feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "model_name": "XGBoost",
        "feature_cols": all_feature_cols,
        "feature_importance": importance,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "y_true": y_test,
        "y_pred": y_pred,
        "mape": mape(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "residuals": residuals,
        "residual_std": float(np.std(residuals)),
        "test_df": test_df.copy(),
    }


def get_sku_residual_std(result: dict) -> pd.DataFrame:
    """
    Extract per-SKU residual standard deviation from XGBoost global model results.
    Used downstream for inventory safety stock calculations.
    """
    test_df = result["test_df"].copy()
    test_df["residual"] = result["y_true"] - result["y_pred"]
    test_df["abs_error"] = np.abs(test_df["residual"])

    sku_stats = test_df.groupby(["product_id", "region"]).agg(
        residual_std=("residual", "std"),
        mae=("abs_error", "mean"),
        n_weeks=("residual", "count"),
    ).reset_index()

    return sku_stats


def predict_future(
    model,
    last_known_df: pd.DataFrame,
    horizon_weeks: int = 12,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Recursive multi-step forecast using the trained XGBoost model.
    Uses predicted values as lag inputs for future steps.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    forecasts = []
    current_df = last_known_df.copy()

    for step in range(1, horizon_weeks + 1):
        last_row = current_df.iloc[-1:].copy()
        next_week = pd.to_datetime(last_row["week_start"].values[0]) + pd.Timedelta(weeks=1)
        next_row = last_row.copy()
        next_row["week_start"] = next_week
        next_row["week_of_year"] = next_week.isocalendar()[1]
        next_row["month"] = next_week.month
        next_row["quarter"] = (next_week.month - 1) // 3 + 1
        next_row["year"] = next_week.year
        next_row["sin_week"] = np.sin(2 * np.pi * next_row["week_of_year"].values[0] / 52)
        next_row["cos_week"] = np.cos(2 * np.pi * next_row["week_of_year"].values[0] / 52)

        X = next_row[[c for c in feature_cols if c in next_row.columns]].fillna(0)
        if X.empty:
            break

        pred = float(np.clip(model.predict(X), 0, None)[0])
        next_row["total_quantity"] = pred
        forecasts.append({
            "week_start": next_week,
            "forecast_qty": pred,
            "step_ahead": step,
        })
        current_df = pd.concat([current_df, next_row], ignore_index=True)

    return pd.DataFrame(forecasts)
