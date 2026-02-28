#!/usr/bin/env python3
"""
Yahoo Finance data provider for US and HK stocks.

Uses yfinance library (free, no auth required).
Provides same interface as tushare_provider.py for seamless integration.

Symbol conventions:
    US:  AAPL, GOOGL, MSFT, BRK-B  (yfinance uses dash for class shares)
    HK:  0700.HK, 9988.HK          (yfinance native format)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd


def is_available() -> bool:
    """Check if yfinance is installed."""
    try:
        import yfinance  # noqa: F401
        return True
    except ImportError:
        return False


def _to_yf_symbol(symbol: str, market: str = "US") -> str:
    """Convert our symbol format to yfinance format.

    US:  AAPL → AAPL, AAPL.US → AAPL, BRK.B → BRK-B
    HK:  0700 → 0700.HK, 0700.HK → 0700.HK
    """
    sym = symbol.upper().replace(".US", "")
    if market == "HK":
        sym = sym.replace(".HK", "")
        # Ensure 4-digit padding for HK stocks
        if sym.isdigit():
            sym = sym.zfill(4)
        return f"{sym}.HK"
    # US: replace dots with dashes for class shares (BRK.B → BRK-B)
    if "." in sym and not sym.endswith(".HK"):
        sym = sym.replace(".", "-")
    return sym


def fetch_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    market: str = "US",
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data via yfinance.

    Args:
        symbol: Stock symbol (e.g., "AAPL" or "0700.HK")
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        market: "US" or "HK"

    Returns:
        DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    import yfinance as yf

    yf_sym = _to_yf_symbol(symbol, market)
    # yfinance end date is exclusive, add 1 day
    end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d") + timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")

    # Suppress yfinance progress output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(start=start_date[:10], end=end_str, auto_adjust=True)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    if df is None or df.empty:
        return None

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    # Keep only OHLCV
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            return None
    df = df[cols].copy()
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    # Round prices
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].round(2)
    return df


def fetch_quote(symbol: str, market: str = "US") -> Dict:
    """Fetch latest quote via yfinance.

    Returns dict with price, change, volume, etc.
    """
    import yfinance as yf

    yf_sym = _to_yf_symbol(symbol, market)
    ticker = yf.Ticker(yf_sym)
    info = ticker.fast_info

    try:
        last_price = float(info.last_price)
        prev_close = float(info.previous_close)
        change = round(last_price - prev_close, 2)
        change_pct = round(change / prev_close * 100, 2) if prev_close else 0.0
    except Exception:
        last_price = 0.0
        prev_close = 0.0
        change = 0.0
        change_pct = 0.0

    try:
        volume = int(info.last_volume)
    except Exception:
        volume = 0

    try:
        name = ticker.info.get("shortName", yf_sym)
    except Exception:
        name = yf_sym

    return {
        "symbol": symbol,
        "name": name,
        "price": last_price,
        "open": round(float(getattr(info, "open", last_price)), 2),
        "high": round(float(getattr(info, "day_high", last_price)), 2),
        "low": round(float(getattr(info, "day_low", last_price)), 2),
        "close": last_price,
        "pre_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "market": market,
    }


def fetch_stock_info(symbol: str, market: str = "US") -> Dict:
    """Fetch basic stock info via yfinance."""
    import yfinance as yf

    yf_sym = _to_yf_symbol(symbol, market)
    ticker = yf.Ticker(yf_sym)

    try:
        info = ticker.info
    except Exception:
        info = {}

    exchange_map = {
        "NMS": "NASDAQ",
        "NYQ": "NYSE",
        "NGM": "NASDAQ",
        "PCX": "AMEX",
        "HKG": "HKEX",
    }
    raw_exchange = info.get("exchange", "")
    exchange = exchange_map.get(raw_exchange, raw_exchange)

    return {
        "symbol": symbol,
        "name": info.get("shortName", info.get("longName", yf_sym)),
        "industry": info.get("industry", ""),
        "sector": info.get("sector", ""),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "exchange": exchange,
        "market": market,
        "currency": info.get("currency", "USD" if market == "US" else "HKD"),
    }
