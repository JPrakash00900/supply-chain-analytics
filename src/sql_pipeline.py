"""
SQL Pipeline — loads raw CSV into SQLite, runs cleaning + aggregation SQL scripts.

Usage:
    python -m src.sql_pipeline                        # default paths
    python -m src.sql_pipeline --csv data/raw/supply_chain_data.csv
    python -m src.sql_pipeline --db data/processed/supply_chain.db --verbose
"""

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/supply_chain_data.csv")
DB_PATH = Path("data/processed/supply_chain.db")
SQL_DIR = Path("sql")

SQL_SCRIPTS = [
    "01_create_tables.sql",
    "02_clean_data.sql",
    "03_aggregations.sql",
]

EXPORT_QUERIES = {
    "weekly_demand": "SELECT * FROM agg_weekly_demand ORDER BY week_start, product_id, region",
    "monthly_sku": "SELECT * FROM agg_monthly_sku ORDER BY year, month, product_id, region",
    "fact_lines": "SELECT * FROM fact_order_lines",
    "dim_products": "SELECT * FROM dim_products",
    "dim_geography": "SELECT * FROM dim_geography",
}


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  {msg}")


def load_csv_to_sqlite(csv_path: Path, db_path: Path, verbose: bool = True) -> None:
    print(f"\n[1/4] Loading {csv_path} → SQLite staging table...")
    t0 = time.time()

    df = pd.read_csv(csv_path, parse_dates=["order_date", "shipping_date"])
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    df["shipping_date"] = df["shipping_date"].dt.strftime("%Y-%m-%d")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    df.to_sql("raw_orders", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    _log(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s", verbose)


def run_sql_scripts(db_path: Path, verbose: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    for script_name in SQL_SCRIPTS:
        script_path = SQL_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"SQL script not found: {script_path}")

        step = SQL_SCRIPTS.index(script_name) + 2
        print(f"\n[{step}/4] Running {script_name}...")
        t0 = time.time()

        sql = script_path.read_text()
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        for stmt in statements:
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            cleaned = "\n".join(lines).strip()
            if not cleaned:
                continue
            try:
                conn.execute(cleaned)
            except sqlite3.Error as e:
                first_line = cleaned.split("\n")[0][:80]
                raise RuntimeError(f"SQL error in {script_name}:\n{e}\nStatement: {first_line}...")

        conn.commit()
        _log(f"Done in {time.time()-t0:.1f}s", verbose)

    conn.close()


def export_to_csv(db_path: Path, export_dir: Path, verbose: bool = True) -> None:
    print(f"\n[4/4] Exporting processed tables to {export_dir}/...")
    export_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    for name, query in EXPORT_QUERIES.items():
        t0 = time.time()
        df = pd.read_sql_query(query, conn)
        out = export_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        _log(f"{name}.csv → {len(df):,} rows ({time.time()-t0:.1f}s)", verbose)

    conn.close()


def print_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tables = {
        "raw_orders": "Raw staging",
        "fact_order_lines": "Cleaned fact",
        "dim_products": "Products",
        "dim_geography": "Geography",
        "agg_weekly_demand": "Weekly demand",
        "agg_monthly_sku": "Monthly SKU",
    }

    print("\n── Database Summary ─────────────────────────────────────────")
    for tbl, label in tables.items():
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {label:<22} ({tbl}): {count:>10,} rows")
        except sqlite3.OperationalError:
            print(f"  {label:<22} ({tbl}): table not found")

    print("─────────────────────────────────────────────────────────────\n")
    conn.close()


def run_analysis_query(db_path: Path, query_name: str) -> pd.DataFrame:
    """Run a named query from 04_analysis_queries.sql by block index."""
    script = (SQL_DIR / "04_analysis_queries.sql").read_text()
    blocks = [b.strip() for b in script.split("-- ── ") if b.strip()]
    conn = sqlite3.connect(db_path)
    results = {}
    for block in blocks:
        lines = block.split("\n")
        label = lines[0].split("──")[0].strip().lower().replace(" ", "_")
        sql_block = "\n".join(l for l in lines[1:] if not l.startswith("--")).strip()
        if sql_block:
            try:
                results[label] = pd.read_sql_query(sql_block, conn)
            except Exception:
                pass
    conn.close()
    return results.get(query_name)


def run_pipeline(
    csv_path: Path = RAW_CSV,
    db_path: Path = DB_PATH,
    export_dir: Path = Path("data/processed"),
    verbose: bool = True,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}. "
            "Run `python -m src.data_generator` first."
        )

    print("=" * 60)
    print(" SQL Pipeline — Supply Chain Analytics")
    print("=" * 60)

    load_csv_to_sqlite(csv_path, db_path, verbose)
    run_sql_scripts(db_path, verbose)
    export_to_csv(db_path, export_dir, verbose)
    print_summary(db_path)
    print("SQL pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Run the SQL pipeline")
    parser.add_argument("--csv", default=str(RAW_CSV))
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--export-dir", default="data/processed")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    run_pipeline(
        csv_path=Path(args.csv),
        db_path=Path(args.db),
        export_dir=Path(args.export_dir),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
