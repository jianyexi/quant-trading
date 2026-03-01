#!/usr/bin/env python3
"""
Unified data loading interface for all quant-trading tasks.

All tasks (factor mining, ML training, backtesting) MUST use this module
to load market data. Data is always read from the local cache — no network
fetching happens here. Pre-cache data via Pipeline Step 0 (Data Sync) or:
    python scripts/market_cache.py warm_symbols <symbols> <start> <end>
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


# Default stocks — diversified across A-share sectors
DEFAULT_STOCKS = [
    # 白酒/食品
    "600519", "000858", "000568", "600809", "600887", "002304", "603288",
    "000895", "603369", "002568",
    # 金融
    "600036", "601318", "601166", "600030", "601398", "601288",
    "600000", "601601", "601688", "000776",
    # 新能源/汽车
    "300750", "002594", "600438", "601012", "002460",
    "600089", "002074", "300014", "601633",
    # 医药
    "600276", "000333", "300760", "603259", "300122",
    "000538", "600196", "002007", "300347",
    # 科技/电子
    "002415", "603501", "300782", "688981", "002049",
    "002371", "300433", "601138", "002241",
    # 消费/家电
    "000651", "600690", "002032", "601888",
    "000725", "002572", "603486",
    # 周期/材料
    "601899", "600031", "600309", "601225", "600585",
    "600019", "601600", "002466", "600348",
    # 公用事业/基建
    "600900", "601669", "600048", "601800",
    "600886", "601985", "003816",
    # 传媒/互联网
    "300059", "002230", "603444", "002555", "300413",
    # 军工
    "600760", "002179", "600893", "600150", "000768",
]


def _get_cache():
    """Get the MarketCache singleton."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from market_cache import get_cache
    return get_cache()


def load_cached_data(
    symbols: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    min_bars: int = 30,
) -> pd.DataFrame:
    """Load combined OHLCV data from local cache. No network access.

    Returns DataFrame with columns [open, high, low, close, volume, symbol].
    Raises RuntimeError if no cached data is available.
    """
    cache = _get_cache()
    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📂 Loading cached data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    combined = cache.get_or_fetch_multi(
        stock_list, start_date, end_date, min_bars=min_bars, cache_only=True,
    )

    if combined is None or combined.empty:
        # Fallback: read any cached data for these symbols, ignoring date range
        print("⚠️  No exact-range cached data. Trying any cached data as fallback...")
        fallback_dfs = []
        for sym in stock_list:
            code = sym.split(".")[0]
            try:
                df = cache._read_cache(code, "2000-01-01", "2099-12-31")
                if df is not None and len(df) >= min_bars:
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


def load_cached_multi(
    symbols: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    min_bars: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Load per-stock DataFrames from local cache. No network access.

    Returns dict of {symbol: DataFrame}.
    """
    cache = _get_cache()
    stock_list = symbols if symbols else DEFAULT_STOCKS
    print(f"📂 Loading per-stock cached data: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    result = {}
    for i, sym in enumerate(stock_list):
        code = sym.split(".")[0]
        print(f"  [{i+1}/{len(stock_list)}] {code}...", end=" ", flush=True)
        try:
            df = cache.get_or_fetch(code, start_date, end_date, cache_only=True)
            if df is not None and len(df) >= min_bars:
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


# ── Backward-compatible aliases ──────────────────────────────────────

def fetch_akshare_data(symbols=None, start_date="2020-01-01", end_date="2024-12-31"):
    """Alias for load_cached_data (backward compatibility)."""
    return load_cached_data(symbols, start_date, end_date)

def fetch_akshare_multi(symbols=None, start_date="2020-01-01", end_date="2024-12-31"):
    """Alias for load_cached_multi (backward compatibility)."""
    return load_cached_multi(symbols, start_date, end_date)


# ── CLI argument helpers ─────────────────────────────────────────────

def add_data_args(parser):
    """Add data arguments to argparse parser.

    Only --symbols, --start-date, --end-date are needed.
    Data is always loaded from cache (pre-synced via Pipeline Step 0).
    --data allows loading from a CSV file instead.
    """
    parser.add_argument("--data", type=str, default=None,
                        help="Path to OHLCV CSV file (overrides cache)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated stock codes (e.g. 600519,000858)")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                        help="End date (YYYY-MM-DD)")
    # Legacy flags — accepted but ignored
    parser.add_argument("--akshare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n-bars", type=int, default=3000, help=argparse.SUPPRESS)
    parser.add_argument("--data-source", type=str, default=None, help=argparse.SUPPRESS)


import argparse  # noqa: E402 (needed for SUPPRESS)


def load_data(args) -> pd.DataFrame:
    """Unified data loading from argparse args.

    Priority: --data (CSV file) > cache (default).
    """
    if getattr(args, 'data', None):
        print(f"📂 Loading data from CSV: {args.data}")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in CSV")
        return df

    symbols = None
    if getattr(args, 'symbols', None):
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = getattr(args, 'start_date', '2020-01-01')
    end = getattr(args, 'end_date', '2024-12-31')
    return load_cached_data(symbols, start, end)
