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
    """Fetch real A-share daily OHLCV data, using local cache to avoid re-fetching."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from market_cache import get_cache

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📡 Fetching real data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    cache = get_cache()
    combined = cache.get_or_fetch_multi(stock_list, start_date, end_date)

    if combined is None or combined.empty:
        raise RuntimeError("❌ No market data fetched. Check network/API access and stock codes.")

    return combined


def fetch_akshare_multi(
    symbols: Optional[List[str]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> Dict[str, pd.DataFrame]:
    """Fetch per-stock DataFrames for cross-stock mining, using local cache."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from market_cache import get_cache

    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📡 Fetching per-stock data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    cache = get_cache()
    result = cache.get_or_fetch_multi_dict(stock_list, start_date, end_date)

    if not result:
        raise RuntimeError("❌ No per-stock data fetched. Check network/API access and stock codes.")

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
