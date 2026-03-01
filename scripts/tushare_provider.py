#!/usr/bin/env python3
"""
Tushare data provider — primary market data source for the quant trading system.

Requires:
    pip install tushare
    export TUSHARE_TOKEN=<your_token>

Tushare API docs: https://tushare.pro/document/2

This module provides a unified interface that the cache layer uses to fetch
daily OHLCV, minute OHLCV, quotes, stock info, and index constituents.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# ── Token & Init ─────────────────────────────────────────────────────

_ts_api = None


def _get_api():
    """Lazily initialize and return tushare pro_api."""
    global _ts_api
    if _ts_api is not None:
        return _ts_api
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("tushare not installed. Run: pip install tushare")

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        # Try reading from config file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "tushare_token.txt"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                token = f.read().strip()

    if not token:
        raise ValueError(
            "Tushare token not found. Set TUSHARE_TOKEN env var or "
            "create config/tushare_token.txt"
        )

    ts.set_token(token)
    _ts_api = ts.pro_api()
    return _ts_api


_is_available_cache: Optional[bool] = None


def is_available() -> bool:
    """Check if tushare is available, token is configured, and API responds.

    Result is cached after first call to avoid repeated network checks.
    """
    global _is_available_cache
    if _is_available_cache is not None:
        return _is_available_cache
    try:
        api = _get_api()
        # Verify token actually works with a lightweight call
        df = api.trade_cal(exchange="SSE", start_date="20250101", end_date="20250102")
        _is_available_cache = df is not None
    except Exception:
        _is_available_cache = False
    return _is_available_cache


# ── Symbol Helpers ───────────────────────────────────────────────────

def to_ts_code(symbol: str) -> str:
    """Convert bare symbol (e.g. '600519') to tushare format ('600519.SH').

    Handles:
        600519     -> 600519.SH
        000001     -> 000001.SZ
        300750     -> 300750.SZ
        688981     -> 688981.SH
        600519.SH  -> 600519.SH  (pass-through)
        000001.SZ  -> 000001.SZ  (pass-through)
    """
    if "." in symbol:
        # Already has suffix — normalize to tushare format
        code, suffix = symbol.split(".", 1)
        return f"{code}.{suffix.upper()}"

    code = symbol.strip()
    if code.startswith("6") or code.startswith("688"):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def from_ts_code(ts_code: str) -> str:
    """Convert tushare code ('600519.SH') to bare symbol ('600519')."""
    return ts_code.split(".")[0] if "." in ts_code else ts_code


# ── Daily OHLCV ─────────────────────────────────────────────────────

def _fetch_daily_with_manual_adj(api, ts_code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """Fallback: fetch unadjusted daily + adj_factor and compute adjusted prices.

    This avoids pro_bar() which requires adj_factor API permissions (200+ credits).
    Uses daily() + adj_factor() separately, or just daily() if adj_factor fails.
    """
    import sys

    # 1) Fetch raw daily data
    try:
        df = api.daily(ts_code=ts_code, start_date=start, end_date=end)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 2) Try to fetch adjustment factors
    try:
        adj_df = api.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
        if adj_df is not None and not adj_df.empty:
            adj_df = adj_df[["trade_date", "adj_factor"]].drop_duplicates("trade_date")
            df = df.merge(adj_df, on="trade_date", how="left")
            df["adj_factor"] = df["adj_factor"].fillna(method="ffill").fillna(1.0)

            if adjust == "qfq":
                # Forward-adjusted: multiply by (adj_factor / latest_adj_factor)
                latest = df["adj_factor"].iloc[0]  # df is sorted desc from tushare
                ratio = df["adj_factor"] / latest
            else:
                # Backward-adjusted: multiply by adj_factor
                ratio = df["adj_factor"]

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * ratio
        else:
            print("  ⚠️ adj_factor unavailable, using unadjusted prices", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠️ adj_factor API failed ({e}), using unadjusted prices", file=sys.stderr)

    return df


def fetch_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch daily OHLCV data from tushare.

    Args:
        symbol: Stock code (e.g. '600519' or '600519.SH')
        start_date: Start date 'YYYY-MM-DD' or 'YYYYMMDD'
        end_date: End date 'YYYY-MM-DD' or 'YYYYMMDD'
        adjust: Price adjustment: 'qfq' (forward), 'hfq' (backward), '' (none)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        Index: DatetimeIndex named 'date'
    """
    api = _get_api()
    ts_code = to_ts_code(symbol)

    # Normalize dates to YYYYMMDD
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    df = None

    if adjust in ("qfq", "hfq"):
        # Try pro_bar first, but suppress stdout/stderr leaks from tushare
        # (adj_factor API needs 200+ credits; errors pollute stdout as text)
        import io
        import sys
        import warnings
        try:
            import tushare as ts
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        start_date=start,
                        end_date=end,
                        adj=adjust,
                        asset="E",
                        freq="D",
                    )
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        except Exception:
            df = None

        # Fallback: fetch unadjusted and manually apply adj_factor
        if df is None or df.empty:
            df = _fetch_daily_with_manual_adj(api, ts_code, start, end, adjust)
    else:
        # Unadjusted: use daily() API
        df = api.daily(ts_code=ts_code, start_date=start, end_date=end)

    if df is None or df.empty:
        return pd.DataFrame()

    # Standardize columns
    df = df.rename(columns={
        "trade_date": "date",
        "vol": "volume",
    })

    # Keep only needed columns
    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Parse dates and sort ascending
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    df = df.astype(float)

    # Tushare volume is in 手 (100 shares) — convert to shares
    if "volume" in df.columns:
        df["volume"] = df["volume"] * 100

    return df


# ── Minute OHLCV ────────────────────────────────────────────────────

def fetch_minute(
    symbol: str,
    start_date: str,
    end_date: str,
    freq: str = "5",
) -> pd.DataFrame:
    """Fetch minute-level OHLCV from tushare.

    Args:
        symbol: Stock code
        start_date, end_date: Date range 'YYYY-MM-DD' or 'YYYYMMDD'
        freq: '1', '5', '15', '30', '60' (minutes)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    import tushare as ts
    ts_code = to_ts_code(symbol)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    freq_map = {"1": "1min", "5": "5min", "15": "15min", "30": "30min", "60": "60min"}
    ts_freq = freq_map.get(freq, f"{freq}min")

    df = ts.pro_bar(
        ts_code=ts_code,
        start_date=start,
        end_date=end,
        freq=ts_freq,
        asset="E",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "trade_time": "date",
        "trade_date": "date",
        "vol": "volume",
    })

    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    df = df.astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"] * 100

    return df


# ── Real-time / Latest Quote ────────────────────────────────────────

def fetch_quote(symbol: str) -> Dict:
    """Fetch latest quote for a stock.

    Returns dict with keys: symbol, name, price, open, high, low,
    volume, turnover, change, change_percent, date
    """
    api = _get_api()
    ts_code = to_ts_code(symbol)

    # Fetch last 5 trading days
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")

    df = api.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError(f"No quote data for {symbol}")

    df.sort_values("trade_date", inplace=True)
    r = df.iloc[-1]

    # Get stock name from basic info (cached)
    name = _get_stock_name(ts_code)

    return {
        "symbol": ts_code.replace(".SH", ".SH").replace(".SZ", ".SZ"),
        "name": name,
        "price": float(r["close"]),
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "volume": float(r["vol"]) * 100,
        "turnover": float(r.get("amount", 0)) * 1000,  # tushare amount in 千元
        "change": float(r.get("change", 0)),
        "change_percent": float(r.get("pct_chg", 0)),
        "date": str(r["trade_date"]),
    }


# ── Stock Info ───────────────────────────────────────────────────────

_name_cache: Dict[str, str] = {}


def _get_stock_name(ts_code: str) -> str:
    """Get stock name, with in-memory caching."""
    if ts_code in _name_cache:
        return _name_cache[ts_code]

    try:
        api = _get_api()
        df = api.namechange(ts_code=ts_code)
        if df is not None and not df.empty:
            name = str(df.iloc[0]["name"])
            _name_cache[ts_code] = name
            return name
    except Exception:
        pass

    # Fallback: return code
    _name_cache[ts_code] = from_ts_code(ts_code)
    return _name_cache[ts_code]


def fetch_stock_info(symbol: str) -> Dict:
    """Get stock basic information."""
    api = _get_api()
    ts_code = to_ts_code(symbol)

    df = api.stock_basic(ts_code=ts_code, fields="ts_code,name,area,industry,list_date,market")
    if df is None or df.empty:
        return {"symbol": ts_code, "error": "not found"}

    r = df.iloc[0]
    return {
        "symbol": ts_code,
        "name": str(r.get("name", "")),
        "industry": str(r.get("industry", "")),
        "list_date": str(r.get("list_date", "")),
        "market": str(r.get("market", "")),
        "area": str(r.get("area", "")),
    }


# ── Index Constituents ───────────────────────────────────────────────

def fetch_index_members(index: str = "000300.SH") -> List[str]:
    """Fetch index constituents (CSI300, CSI500, etc.).

    Args:
        index: Index code. Common values:
            '000300.SH' — CSI 300
            '000905.SH' — CSI 500
            '000016.SH' — SSE 50

    Returns:
        List of stock codes (bare, e.g. ['600519', '000001', ...])
    """
    api = _get_api()
    df = api.index_weight(index_code=index)
    if df is None or df.empty:
        return []

    codes = df["con_code"].unique().tolist()
    return [from_ts_code(c) for c in codes]


# ── Stock List ───────────────────────────────────────────────────────

def fetch_stock_list(market: str = "") -> List[Dict]:
    """Fetch all listed stocks.

    Args:
        market: Filter by market ('主板', '创业板', '科创板', or '' for all)

    Returns:
        List of dicts with keys: symbol, name, industry, market
    """
    api = _get_api()
    df = api.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name,industry,market,list_date",
    )
    if df is None or df.empty:
        return []

    stocks = []
    for _, r in df.iterrows():
        mkt = str(r.get("market", ""))
        if market and mkt != market:
            continue
        code = from_ts_code(str(r["ts_code"]))
        suffix = ".SH" if str(r["ts_code"]).endswith(".SH") else ".SZ"
        stocks.append({
            "symbol": f"{code}{suffix}",
            "name": str(r.get("name", "")),
            "industry": str(r.get("industry", "")),
            "market": mkt,
        })
    return stocks


# ── Trading Calendar ─────────────────────────────────────────────────

_cal_cache: Optional[set] = None


def get_trading_dates(start: str, end: str) -> List[str]:
    """Get trading calendar dates.

    Returns list of 'YYYY-MM-DD' strings.
    """
    global _cal_cache
    api = _get_api()

    start_fmt = start.replace("-", "")
    end_fmt = end.replace("-", "")

    df = api.trade_cal(
        exchange="SSE",
        start_date=start_fmt,
        end_date=end_fmt,
        is_open=1,
    )
    if df is None or df.empty:
        return []

    dates = df["cal_date"].tolist()
    return [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sorted(dates)]


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not is_available():
        print("❌ Tushare not available. Set TUSHARE_TOKEN env var.")
        sys.exit(1)

    print("✅ Tushare connection OK")

    # Test daily
    print("\n--- Daily OHLCV (600519, last 5 days) ---")
    df = fetch_daily("600519", "2024-12-01", "2024-12-31")
    print(df.tail())

    # Test quote
    print("\n--- Quote (000001.SZ) ---")
    q = fetch_quote("000001")
    print(q)

    # Test stock info
    print("\n--- Stock Info (600519) ---")
    info = fetch_stock_info("600519")
    print(info)

    # Test index members
    print("\n--- CSI300 members (first 10) ---")
    members = fetch_index_members("000300.SH")
    print(members[:10])

    print("\n✅ All tests passed!")
