#!/usr/bin/env python3
"""
Shared data loading utilities for factor mining scripts.
Supports: synthetic, CSV file, akshare (real A-share data).
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_factor_model import generate_synthetic_data


# Default stocks — diversified across sectors
DEFAULT_STOCKS = [
    "600519", "000858", "600036", "601318", "300750",
    "002594", "600276", "000333", "000651", "002415",
    "601899", "600900", "600690", "601888", "603501",
    "600809", "601166", "600030", "300760", "600438",
]


def fetch_akshare_data(
    symbols: Optional[List[str]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Fetch real A-share daily OHLCV data from akshare."""
    try:
        import akshare as ak
    except ImportError:
        print("⚠️  akshare not installed. Run: pip install akshare")
        print("   Falling back to synthetic data.")
        return generate_synthetic_data(3000)

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📡 Fetching real data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    all_data = []
    for i, sym in enumerate(stock_list):
        code = sym.split(".")[0]
        print(f"  [{i+1}/{len(stock_list)}] {code}...", end=" ", flush=True)
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty or len(df) < 100:
                print(f"skip ({0 if df is None else len(df)} bars)")
                continue
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df["symbol"] = code
            all_data.append(df)
            print(f"OK ({len(df)} bars)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not all_data:
        print("⚠️  No data fetched, falling back to synthetic")
        return generate_synthetic_data(3000)

    combined = pd.concat(all_data, axis=0).sort_index()
    print(f"✅ Total: {len(combined)} bars from {len(all_data)} stocks")
    return combined


def fetch_akshare_multi(
    symbols: Optional[List[str]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> Dict[str, pd.DataFrame]:
    """Fetch per-stock DataFrames for cross-stock mining."""
    try:
        import akshare as ak
    except ImportError:
        print("⚠️  akshare not installed, using synthetic multi-stock data")
        from factor_mining import generate_multi_stock_data
        return generate_multi_stock_data(10, 2000)

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📡 Fetching per-stock data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    stocks = {}
    for i, sym in enumerate(stock_list):
        code = sym.split(".")[0]
        print(f"  [{i+1}/{len(stock_list)}] {code}...", end=" ", flush=True)
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty or len(df) < 100:
                print(f"skip ({0 if df is None else len(df)} bars)")
                continue
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            stocks[code] = df
            print(f"OK ({len(df)} bars)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not stocks:
        print("⚠️  No data fetched, using synthetic")
        from factor_mining import generate_multi_stock_data
        return generate_multi_stock_data(10, 2000)

    print(f"✅ Loaded {len(stocks)} stocks")
    return stocks


def load_data(args) -> pd.DataFrame:
    """Unified data loading from argparse args.
    Supports: --data (CSV), --akshare, --synthetic (default)."""
    if getattr(args, 'data', None):
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in CSV")
        return df

    if getattr(args, 'akshare', False):
        symbols = None
        if getattr(args, 'symbols', None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        start = getattr(args, 'start_date', '2023-01-01')
        end = getattr(args, 'end_date', '2024-12-31')
        return fetch_akshare_data(symbols, start, end)

    n_bars = getattr(args, 'n_bars', 3000)
    print(f"Generating synthetic data ({n_bars} bars)...")
    return generate_synthetic_data(n_bars)


def add_data_args(parser):
    """Add common data source arguments to argparse parser."""
    parser.add_argument("--data", type=str, default=None,
                        help="Path to OHLCV CSV file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (default if no other source)")
    parser.add_argument("--akshare", action="store_true",
                        help="Fetch real A-share data via akshare")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated stock codes (e.g. 600519,000858)")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                        help="Start date for akshare (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                        help="End date for akshare (YYYY-MM-DD)")
    parser.add_argument("--n-bars", type=int, default=3000,
                        help="Number of bars (synthetic mode)")
