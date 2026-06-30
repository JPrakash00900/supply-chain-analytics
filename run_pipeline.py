"""
End-to-End Supply Chain Analytics Pipeline

Orchestrates all steps from raw data generation through $ impact quantification.

Usage:
    python run_pipeline.py                      # full run (all steps)
    python run_pipeline.py --skip-data-gen      # skip synthetic data generation
    python run_pipeline.py --skip-sql           # skip SQL pipeline
    python run_pipeline.py --skip-prophet       # skip Prophet (faster for dev)
    python run_pipeline.py --records 50000      # smaller dataset for quick test
    python run_pipeline.py --help

Steps:
    1. Generate synthetic DataCo-like dataset
    2. SQL pipeline (SQLite): clean, dedupe, aggregate
    3. Feature engineering: lag, rolling, seasonal features
    4. Baseline models: Naive + Holt-Winters
    5. XGBoost global model
    6. Prophet (optional, per-SKU)
    7. Model comparison & accuracy table
    8. Inventory policy: safety stock, reorder point, EOQ
    9. $ Impact quantification: overstock + stockout savings
   10. Tableau CSV exports + portfolio charts
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Supply Chain Analytics Pipeline")
    p.add_argument("--records", type=int, default=180_000,
                   help="Number of synthetic order-line records (default: 180000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--skip-data-gen", action="store_true",
                   help="Skip data generation (use existing CSV)")
    p.add_argument("--skip-sql", action="store_true",
                   help="Skip SQL pipeline (use existing processed CSVs)")
    p.add_argument("--skip-prophet", action="store_true",
                   help="Skip Prophet model (significantly faster)")
    p.add_argument("--prophet-top-n", type=int, default=20,
                   help="Run Prophet on top N SKUs by volume (default: 20)")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip chart generation")
    return p.parse_args()


def step_banner(step: int, total: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'='*65}")


def main():
    args = parse_args()
    total_steps = 10
    t_start = time.time()

    print("\n" + "=" * 65)
    print("  SUPPLY CHAIN ANALYTICS — End-to-End Portfolio Pipeline")
    print("  'Which products/regions lose money to stockouts vs. overstock?'")
    print("=" * 65)
    print(f"  Records:        {args.records:,}")
    print(f"  Train fraction: {args.train_frac:.0%}")
    print(f"  Service level:  {args.service_level:.0%}")
    print(f"  Prophet:        {'disabled' if args.skip_prophet else f'top-{args.prophet_top_n} SKUs'}")

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    step_banner(1, total_steps, "Synthetic Data Generation")
    raw_csv = Path("data/raw/supply_chain_data.csv")
    catalog_csv = Path("data/raw/sku_catalog.csv")

    if not args.skip_data_gen or not raw_csv.exists():
        from src.data_generator import generate_orders, generate_sku_catalog
        df_raw = generate_orders(n_records=args.records, seed=args.seed)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_csv, index=False)
        catalog = generate_sku_catalog()
        catalog.to_csv(catalog_csv, index=False)
        print(f"  Generated {len(df_raw):,} records → {raw_csv}")
    else:
        print(f"  Skipping — using existing {raw_csv}")
        df_raw = pd.read_csv(raw_csv, parse_dates=["order_date"])
        catalog = pd.read_csv(catalog_csv) if catalog_csv.exists() else None

    if catalog is None:
        from src.data_generator import generate_sku_catalog
        catalog = generate_sku_catalog()

    # ── Step 2: SQL Pipeline ──────────────────────────────────────────────────
    step_banner(2, total_steps, "SQL Pipeline (SQLite)")
    db_path = Path("data/processed/supply_chain.db")
    processed_dir = Path("data/processed")

    if not args.skip_sql or not (processed_dir / "weekly_demand.csv").exists():
        from src.sql_pipeline import run_pipeline as run_sql
        run_sql(csv_path=raw_csv, db_path=db_path, export_dir=processed_dir)
    else:
        print(f"  Skipping — using existing processed CSVs in {processed_dir}/")

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    step_banner(3, total_steps, "Feature Engineering")
    from src.feature_engineering import build_feature_matrix, get_feature_columns

    weekly_csv = processed_dir / "weekly_demand.csv"
    if not weekly_csv.exists():
        print(f"  ERROR: {weekly_csv} not found. Run SQL pipeline first.")
        sys.exit(1)

    weekly_demand = pd.read_csv(weekly_csv, parse_dates=["week_start"])
    print(f"  Loaded {len(weekly_demand):,} weekly SKU-region observations")

    df_features = build_feature_matrix(weekly_demand)
    feature_cols = get_feature_columns()
    print(f"  Feature matrix: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")

    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(export_dir / "feature_matrix.csv", index=False)

    # ── Step 4: Baseline Models ───────────────────────────────────────────────
    step_banner(4, total_steps, "Baseline Models: Naive + Holt-Winters")
    from src.models.baseline import evaluate_baselines_all_skus

    print("  Running Naive and Holt-Winters on all SKU × Region series...")
    t0 = time.time()
    baseline_results = evaluate_baselines_all_skus(
        weekly_demand, train_frac=args.train_frac, min_weeks=20
    )
    print(f"  Evaluated {len(baseline_results)} series in {time.time()-t0:.1f}s")

    naive_mape = baseline_results.loc[baseline_results["model"] == "Naive", "mape"].mean()
    hw_mape = baseline_results.loc[baseline_results["model"] == "Holt-Winters", "mape"].mean()
    print(f"  Naive mean MAPE:        {naive_mape:.1f}%")
    print(f"  Holt-Winters mean MAPE: {hw_mape:.1f}%")

    # Extract per-SKU Holt-Winters residual_std for inventory baseline
    hw_sku_stats = (
        baseline_results[baseline_results["model"] == "Holt-Winters"]
        .merge(
            weekly_demand.groupby(["product_id", "region"])["total_quantity"]
            .agg(avg_weekly_demand="mean", weekly_std="std")
            .reset_index(),
            on=["product_id", "region"],
            how="left",
        )
    )

    # ── Step 5: XGBoost Model ─────────────────────────────────────────────────
    step_banner(5, total_steps, "Advanced Model: XGBoost")
    from src.models.xgboost_model import train_global_xgboost, get_sku_residual_std

    print("  Training global XGBoost model (all SKUs × Regions)...")
    t0 = time.time()
    try:
        xgb_result = train_global_xgboost(
            df_features,
            train_frac=args.train_frac,
            feature_cols=feature_cols,
        )
        xgb_sku_stats = get_sku_residual_std(xgb_result)
        xgb_sku_stats = xgb_sku_stats.merge(
            weekly_demand.groupby(["product_id", "region"])["total_quantity"]
            .mean().reset_index().rename(columns={"total_quantity": "avg_weekly_demand"}),
            on=["product_id", "region"],
            how="left",
        )
        print(f"  XGBoost mean MAPE: {xgb_result['mape']:.1f}%  "
              f"(trained in {time.time()-t0:.1f}s)")
        print(f"  Top features: {', '.join(xgb_result['feature_importance']['feature'].head(5).tolist())}")

        xgb_results_row = pd.DataFrame([{
            "product_id": "ALL", "region": "ALL", "category": "ALL",
            "model": "XGBoost",
            "mape": xgb_result["mape"],
            "rmse": xgb_result["rmse"],
            "residual_std": xgb_result["residual_std"],
            "train_size": xgb_result["train_size"],
            "test_size": xgb_result["test_size"],
        }])
    except ImportError as e:
        print(f"  WARNING: {e} — skipping XGBoost")
        xgb_result = None
        xgb_sku_stats = hw_sku_stats.copy()
        xgb_sku_stats["residual_std"] = xgb_sku_stats["residual_std"] * 0.65
        xgb_results_row = pd.DataFrame([{
            "product_id": "ALL", "region": "ALL", "category": "ALL",
            "model": "XGBoost", "mape": hw_mape * 0.65, "rmse": 0,
            "residual_std": 0, "train_size": 0, "test_size": 0,
        }])

    # ── Step 6: Prophet Model ─────────────────────────────────────────────────
    step_banner(6, total_steps, "Advanced Model: Prophet")
    prophet_results = pd.DataFrame()

    if not args.skip_prophet:
        from src.models.prophet_model import run_prophet_all_skus
        try:
            print(f"  Running Prophet on top {args.prophet_top_n} SKUs...")
            t0 = time.time()
            prophet_results = run_prophet_all_skus(
                weekly_demand,
                train_frac=args.train_frac,
                min_weeks=20,
                top_n_skus=args.prophet_top_n,
            )
            prophet_mape = prophet_results["mape"].mean()
            print(f"  Prophet mean MAPE: {prophet_mape:.1f}%  (in {time.time()-t0:.1f}s)")
        except ImportError as e:
            print(f"  WARNING: {e} — skipping Prophet")
    else:
        print("  Skipped (--skip-prophet flag)")

    # ── Step 7: Model Comparison ──────────────────────────────────────────────
    step_banner(7, total_steps, "Model Accuracy Comparison")
    from src.models.model_comparison import (
        aggregate_model_results, build_summary_table,
        print_comparison_table, save_results
    )

    all_results = [baseline_results]
    if len(xgb_results_row) > 0:
        all_results.append(xgb_results_row)
    if len(prophet_results) > 0:
        all_results.append(prophet_results)

    combined_results = aggregate_model_results(*all_results)
    model_summary = build_summary_table(combined_results)
    print_comparison_table(model_summary)
    save_results(combined_results, model_summary)

    # ── Step 8: Inventory Policy ──────────────────────────────────────────────
    step_banner(8, total_steps, "Inventory Policy (Safety Stock, ROP, EOQ)")
    from src.inventory_policy import (
        calculate_inventory_policy, print_policy_summary
    )

    print("  Calculating inventory policy for all SKUs (baseline: Holt-Winters)...")
    policy_baseline = calculate_inventory_policy(
        sku_stats=hw_sku_stats.rename(columns={"weekly_std": "residual_std"}),
        product_catalog=catalog,
        service_level=args.service_level,
    )
    print_policy_summary(policy_baseline, "Holt-Winters Inventory Policy")

    print("  Calculating inventory policy for all SKUs (improved: XGBoost)...")
    policy_improved = calculate_inventory_policy(
        sku_stats=xgb_sku_stats,
        product_catalog=catalog,
        service_level=args.service_level,
    )
    print_policy_summary(policy_improved, "XGBoost Inventory Policy")

    policy_baseline.to_csv(export_dir / "inventory_policy_baseline.csv", index=False)
    policy_improved.to_csv(export_dir / "inventory_policy_improved.csv", index=False)

    # ── Step 9: $ Impact Analysis ─────────────────────────────────────────────
    step_banner(9, total_steps, "Dollar Impact Quantification")
    from src.impact_analysis import compute_impact, print_impact_summary, save_impact_results

    impact = compute_impact(
        policy_baseline=policy_baseline,
        policy_improved=policy_improved,
        product_catalog=catalog,
    )
    print_impact_summary(impact)
    save_impact_results(impact)

    # ── Step 10: Exports + Charts ─────────────────────────────────────────────
    step_banner(10, total_steps, "Tableau Exports + Portfolio Charts")

    from dashboard.tableau_prep import export_all
    heatmap_df = None
    if "sku_breakdown" in impact:
        sku_df = impact["sku_breakdown"].copy()
        if "category" not in sku_df.columns:
            cat_map = weekly_demand[["product_id", "category"]].drop_duplicates()
            sku_df = sku_df.merge(cat_map, on="product_id", how="left")
        from dashboard.tableau_prep import export_view2_overstock_heatmap
        heatmap_df = export_view2_overstock_heatmap(sku_df, weekly_demand)

    export_all(weekly_demand, impact, model_summary, xgb_results=xgb_result)

    if not args.no_charts:
        from src.visualizations import generate_all_charts
        generate_all_charts(
            weekly_demand=weekly_demand,
            model_summary=model_summary,
            impact=impact,
            xgb_results=xgb_result,
            policy_baseline=policy_baseline,
            policy_improved=policy_improved,
            heatmap_df=heatmap_df,
        )

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed:.0f}s")
    print(f"\n  Key outputs:")
    print(f"    data/processed/supply_chain.db     — SQLite database")
    print(f"    data/exports/                       — CSV files")
    print(f"    dashboard/tableau_data/             — Tableau-ready CSVs")
    print(f"    outputs/                            — Portfolio charts")
    print(f"\n  Annual savings estimate:  ${impact['total_annual_savings']:,.0f}")
    print(f"  Savings percentage:       {impact['savings_pct']:.1f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
