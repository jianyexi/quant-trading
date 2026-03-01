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
    """Load market data from local cache only (no network fetch).
    
    Data must be pre-cached via Pipeline Step 0 (Data Sync) or
    `python scripts/market_cache.py warm_symbols <symbols>`.
    Falls back to reading any cached data if exact range is missing.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from market_cache import get_cache

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📂 Loading cached data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    cache = get_cache()
    combined = cache.get_or_fetch_multi(stock_list, start_date, end_date, min_bars=30, cache_only=True)

    if combined is None or combined.empty:
        # Fallback: read any cached data for these symbols, ignoring date range
        print("⚠️  No exact-range cached data. Trying any cached data as fallback...")
        fallback_dfs = []
        for sym in stock_list:
            code = sym.split(".")[0]
            try:
                df = cache._read_cache(code, "2000-01-01", "2099-12-31")
                if df is not None and len(df) >= 30:
                    df["symbol"] = code
                    fallback_dfs.append(df)
                    print(f"  ✓ {code}: {len(df)} bars from cache")
            except Exception:
                pass
        if fallback_dfs:
            combined = pd.concat(fallback_dfs, axis=0).sort_index()
            print(f"📦 Fallback: using {len(combined)} cached bars from {len(fallback_dfs)} stocks")
        else:
            raise RuntimeError(
                "❌ No cached market data found. Please run data sync first:\n"
                "   Pipeline → Step 0: 数据准备 → 同步数据\n"
                "   or: python scripts/market_cache.py warm_symbols "
                + ",".join(stock_list) + f" {start_date} {end_date}"
            )

    return combined


def fetch_akshare_multi(
    symbols: Optional[List[str]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> Dict[str, pd.DataFrame]:
    """Load per-stock DataFrames from local cache only (no network fetch)."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from market_cache import get_cache

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📂 Loading per-stock cached data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    cache = get_cache()
    result = {}
    for i, sym in enumerate(stock_list):
        code = sym.split(".")[0]
        print(f"  [{i+1}/{len(stock_list)}] {code}...", end=" ", flush=True)
        try:
            df = cache.get_or_fetch(code, start_date, end_date, cache_only=True)
            if df is not None and len(df) >= 30:
                result[code] = df
                print(f"OK ({len(df)} bars)")
            else:
                print(f"skip ({0 if df is None else len(df)} bars)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not result:
        raise RuntimeError(
            "❌ No cached per-stock data found. Please run data sync first:\n"
            "   Pipeline → Step 0: 数据准备 → 同步数据\n"
            "   or: python scripts/market_cache.py warm_symbols "
            + ",".join(stock_list) + f" {start_date} {end_date}"
        )

    return result


def load_data(args) -> pd.DataFrame:
    """Unified data loading from argparse args.
    Supports: --data (CSV), --akshare (default: real data)."""
    if getattr(args, 'data', None):
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in CSV")
        return df

    # Default: always fetch real data via akshare/tushare
    symbols = None
    if getattr(args, 'symbols', None):
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = getattr(args, 'start_date', '2023-01-01')
    end = getattr(args, 'end_date', '2024-12-31')
    return fetch_akshare_data(symbols, start, end)


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
