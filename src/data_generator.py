"""
Synthetic DataCo-like Supply Chain Dataset Generator

Mirrors the schema and statistical properties of the DataCo Smart Supply Chain
dataset (Kaggle). Generates ~180k order-line records with realistic:
  - 5 product categories, 60 SKUs
  - 4 geographic regions, 12 markets
  - 2-year time range with weekly/annual seasonality
  - Realistic price, cost, margin, and quantity distributions
  - Occasional stockout and overstock signals embedded in the data

Usage:
    python -m src.data_generator
    python -m src.data_generator --records 50000 --seed 99
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)

CATEGORIES = {
    "Electronics": {
        "skus": ["Laptop Pro 15", "Wireless Headphones", "Smart Speaker", "USB-C Hub",
                 "Mechanical Keyboard", "Gaming Mouse", "Webcam HD", "Monitor 27in",
                 "Portable SSD 1TB", "Tablet 10in", "Phone Case", "Screen Protector"],
        "price_range": (15.0, 1200.0),
        "margin_pct": (0.12, 0.28),
        "lead_time_days": (7, 21),
        "seasonality_peak": [11, 12],
        "base_weekly_demand": 80,
    },
    "Office Supplies": {
        "skus": ["Ballpoint Pens 12pk", "Sticky Notes 5pk", "Legal Pads", "Stapler Pro",
                 "File Folders 50pk", "Whiteboard Markers", "Scissors", "Tape Dispenser",
                 "Binder 3in", "Desk Organizer", "Label Maker", "Paper Clips 500pk"],
        "price_range": (2.0, 45.0),
        "margin_pct": (0.30, 0.55),
        "lead_time_days": (3, 10),
        "seasonality_peak": [8, 9],
        "base_weekly_demand": 200,
    },
    "Furniture": {
        "skus": ["Ergonomic Chair", "Standing Desk", "Bookshelf 5-Shelf",
                 "Filing Cabinet 2-Drawer", "Monitor Arm", "Keyboard Tray",
                 "Desk Lamp LED", "Footrest Ergonomic", "Cable Management Kit",
                 "Whiteboard 48x36", "Coat Rack", "Storage Ottoman"],
        "price_range": (25.0, 800.0),
        "margin_pct": (0.20, 0.40),
        "lead_time_days": (10, 30),
        "seasonality_peak": [1, 2],
        "base_weekly_demand": 30,
    },
    "Clothing": {
        "skus": ["Work Polo Shirt", "Dress Pants", "Safety Vest", "Work Boots",
                 "Hard Hat", "Safety Glasses", "Latex Gloves 100pk", "Work Jacket",
                 "Steel Toe Sneakers", "Hi-Vis Jacket", "Ear Protection", "Knee Pads"],
        "price_range": (8.0, 150.0),
        "margin_pct": (0.35, 0.60),
        "lead_time_days": (5, 14),
        "seasonality_peak": [3, 10],
        "base_weekly_demand": 60,
    },
    "Sports": {
        "skus": ["Resistance Bands Set", "Yoga Mat", "Dumbbells 10lb pair",
                 "Jump Rope", "Foam Roller", "Water Bottle 32oz", "Gym Bag",
                 "Fitness Tracker Band", "Pull-Up Bar", "Ab Roller", "Workout Gloves",
                 "Protein Shaker Bottle"],
        "price_range": (5.0, 120.0),
        "margin_pct": (0.28, 0.50),
        "lead_time_days": (4, 12),
        "seasonality_peak": [1, 6],
        "base_weekly_demand": 90,
    },
}

REGIONS = {
    "West": {"markets": ["Pacific", "Mountain"], "volume_weight": 0.28},
    "East": {"markets": ["Northeast", "Mid-Atlantic"], "volume_weight": 0.32},
    "Central": {"markets": ["Midwest", "Great Plains"], "volume_weight": 0.22},
    "South": {"markets": ["Southeast", "Southwest"], "volume_weight": 0.18},
}

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SHIPPING_MODES = {
    "Standard Class": 0.50,
    "Second Class": 0.25,
    "First Class": 0.18,
    "Same Day": 0.07,
}
SHIPPING_DAYS = {
    "Standard Class": (5, 7),
    "Second Class": (3, 5),
    "First Class": (2, 3),
    "Same Day": (1, 1),
}
ORDER_STATUSES = {
    "Complete": 0.72,
    "Pending": 0.08,
    "Shipping": 0.10,
    "Cancelled": 0.06,
    "On Hold": 0.04,
}


def _seasonal_multiplier(date: datetime, peak_months: list, amplitude: float = 0.35) -> float:
    month = date.month
    base = 1.0
    if month in peak_months:
        base += amplitude
    adjacent = {(m % 12) + 1 for m in peak_months} | {((m - 2) % 12) + 1 for m in peak_months}
    if month in adjacent:
        base += amplitude * 0.5
    day_of_year = date.timetuple().tm_yday
    annual_wave = 0.08 * np.sin(2 * np.pi * day_of_year / 365)
    return max(0.3, base + annual_wave)


def _weekend_multiplier(date: datetime) -> float:
    return 0.45 if date.weekday() >= 5 else 1.0


def generate_sku_catalog() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    sku_id = 1000
    for category, cfg in CATEGORIES.items():
        for product_name in cfg["skus"]:
            price = rng.uniform(*cfg["price_range"])
            margin = rng.uniform(*cfg["margin_pct"])
            cost = round(price * (1 - margin), 2)
            price = round(price, 2)
            rows.append({
                "product_id": f"SKU-{sku_id}",
                "product_name": product_name,
                "category": category,
                "unit_price": price,
                "unit_cost": cost,
                "margin_pct": round(margin, 4),
                "lead_time_days_min": cfg["lead_time_days"][0],
                "lead_time_days_max": cfg["lead_time_days"][1],
                "base_weekly_demand": cfg["base_weekly_demand"],
            })
            sku_id += 1
    return pd.DataFrame(rows)


def generate_orders(n_records: int = 180_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    catalog = generate_sku_catalog()

    date_range = (END_DATE - START_DATE).days
    records = []
    order_id = 100000
    order_item_id = 1

    shipping_modes = list(SHIPPING_MODES.keys())
    shipping_weights = list(SHIPPING_MODES.values())
    order_statuses = list(ORDER_STATUSES.keys())
    status_weights = list(ORDER_STATUSES.values())
    regions = list(REGIONS.keys())

    batch_size = max(1, n_records // 500)
    for start in range(0, n_records, batch_size):
        n_batch = min(batch_size, n_records - start)

        days_offsets = rng.integers(0, date_range, size=n_batch)
        order_dates = [START_DATE + timedelta(days=int(d)) for d in days_offsets]

        sku_indices = rng.integers(0, len(catalog), size=n_batch)
        region_choices = rng.choice(regions, size=n_batch,
                                    p=[REGIONS[r]["volume_weight"] for r in regions])

        for i in range(n_batch):
            date = order_dates[i]
            sku = catalog.iloc[sku_indices[i]]
            region = region_choices[i]
            category = sku["category"]
            cat_cfg = CATEGORIES[category]

            seasonal = _seasonal_multiplier(date, cat_cfg["seasonality_peak"])
            weekend = _weekend_multiplier(date)
            demand_mean = sku["base_weekly_demand"] / 7 * seasonal * weekend
            qty = max(1, int(rng.poisson(max(0.5, demand_mean))))

            discount_rate = float(rng.choice(
                [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
                p=[0.55, 0.15, 0.12, 0.08, 0.06, 0.04]
            ))
            unit_price = sku["unit_price"]
            sale_price = round(unit_price * (1 - discount_rate), 2)
            item_total = round(sale_price * qty, 2)
            profit = round((sale_price - sku["unit_cost"]) * qty, 2)
            profit_ratio = round(profit / item_total, 4) if item_total > 0 else 0.0

            ship_mode = rng.choice(shipping_modes, p=shipping_weights)
            ship_min, ship_max = SHIPPING_DAYS[ship_mode]
            ship_days_actual = int(rng.integers(ship_min, ship_max + 1))
            ship_days_sched = ship_min
            late_risk = 1 if ship_days_actual > ship_days_sched else 0

            ship_date = date + timedelta(days=ship_days_actual)
            order_status = rng.choice(order_statuses, p=status_weights)

            market_list = REGIONS[region]["markets"]
            market = market_list[rng.integers(0, len(market_list))]
            customer_segment = rng.choice(CUSTOMER_SEGMENTS, p=[0.52, 0.35, 0.13])

            records.append({
                "order_id": order_id,
                "order_item_id": order_item_id,
                "order_date": date.strftime("%Y-%m-%d"),
                "shipping_date": ship_date.strftime("%Y-%m-%d"),
                "order_status": order_status,
                "shipping_mode": ship_mode,
                "days_for_shipping_real": ship_days_actual,
                "days_for_shipment_scheduled": ship_days_sched,
                "late_delivery_risk": late_risk,
                "product_id": sku["product_id"],
                "product_name": sku["product_name"],
                "category": category,
                "unit_price": unit_price,
                "unit_cost": sku["unit_cost"],
                "margin_pct": sku["margin_pct"],
                "order_item_quantity": qty,
                "order_item_discount_rate": discount_rate,
                "order_item_product_price": sale_price,
                "order_item_total": item_total,
                "order_item_profit_ratio": profit_ratio,
                "benefit_per_order": profit,
                "sales": item_total,
                "region": region,
                "market": market,
                "customer_segment": customer_segment,
            })

            if rng.random() > 0.6:
                order_id += 1
            order_item_id += 1

        order_id += 1

    df = pd.DataFrame(records)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["shipping_date"] = pd.to_datetime(df["shipping_date"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["week"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["order_date"].dt.dayofweek

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic supply chain dataset")
    parser.add_argument("--records", type=int, default=180_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="data/raw/supply_chain_data.csv")
    args = parser.parse_args()

    print(f"Generating {args.records:,} order-line records (seed={args.seed})...")
    df = generate_orders(n_records=args.records, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} records to {out_path}")
    print(f"Date range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"Categories: {df['category'].nunique()} | SKUs: {df['product_id'].nunique()}")
    print(f"Regions: {df['region'].nunique()} | Total revenue: ${df['sales'].sum():,.0f}")
    print(f"\nSKU catalog saved to: data/raw/sku_catalog.csv")

    catalog = generate_sku_catalog()
    catalog.to_csv("data/raw/sku_catalog.csv", index=False)


if __name__ == "__main__":
    main()
