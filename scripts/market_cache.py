#!/usr/bin/env python3
"""
Persistent local cache for market OHLCV data (SQLite-backed).

Usage:
    from market_cache import MarketCache
    cache = MarketCache()              # defaults to data/market_cache.db
    df = cache.get_or_fetch("600519", "2020-01-01", "2024-12-31")

Key features:
  - Only fetches from akshare what's missing (incremental gap-filling)
  - Trading calendar aware (weekends/holidays are not gaps)
  - Thread-safe via SQLite WAL mode
  - Used by both scripts/market_data.py and ml_models/auto_retrain.py
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kline_daily (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,  -- YYYY-MM-DD
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS cache_meta (
    symbol       TEXT PRIMARY KEY,
    min_date     TEXT NOT NULL,
    max_date     TEXT NOT NULL,
    bar_count    INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kline_symbol_date ON kline_daily(symbol, date);
"""


class MarketCache:
    """SQLite-backed market data cache with incremental fetching."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            os.makedirs(_DB_DIR, exist_ok=True)
            db_path = os.path.join(_DB_DIR, "market_cache.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Public API ───────────────────────────────────────────────────

    def get_or_fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "daily",
        max_retries: int = 3,
        base_delay: float = 0.8,
        market: str = "CN",
    ) -> pd.DataFrame:
        """Get klines from cache, fetching missing ranges.

        Args:
            symbol: Stock code, e.g. "600519", "AAPL", "0700.HK"
            start_date: "YYYY-MM-DD" or "YYYYMMDD"
            end_date: "YYYY-MM-DD" or "YYYYMMDD"
            period: "daily" only for cache (minute data bypasses cache)
            max_retries: Retry count for API calls
            base_delay: Base delay between retries (exponential backoff)
            market: "CN" (default), "US", or "HK"

        Returns:
            DataFrame with columns [open, high, low, close, volume], date index
        """
        if period != "daily":
            if market in ("US", "HK"):
                return self._fetch_yfinance_direct(symbol, start_date, end_date, market)
            return self._fetch_raw(symbol, start_date, end_date, period)

        # Use market-prefixed cache key to avoid symbol collisions
        cache_key = f"{market}:{symbol}" if market != "CN" else symbol

        start = self._normalize_date(start_date)
        end = self._normalize_date(end_date)

        # Check what we already have
        meta = self._get_meta(cache_key)
        gaps = self._find_gaps(cache_key, start, end, meta)

        if gaps:
            self._fill_gaps(cache_key, gaps, max_retries, base_delay, market=market, raw_symbol=symbol)

        # Return from cache
        return self._read_cache(cache_key, start, end)

    def get_or_fetch_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        base_delay: float = 0.8,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Fetch multiple stocks, returning a combined DataFrame with 'symbol' column."""
        all_data = []
        n_ok, n_skip, n_fail = 0, 0, 0
        n_cached, n_fetched = 0, 0

        for i, sym in enumerate(symbols):
            code = sym.split(".")[0]
            if verbose:
                print(f"  [{i+1}/{len(symbols)}] {code}...", end=" ", flush=True)

            try:
                # Check if fully cached before fetching
                was_cached = self._is_fully_cached(code, start_date, end_date)
                df = self.get_or_fetch(code, start_date, end_date, max_retries=max_retries, base_delay=base_delay)

                if df is None or df.empty or len(df) < 200:
                    if verbose:
                        print(f"skip ({0 if df is None else len(df)} bars)")
                    n_skip += 1
                    continue

                df["symbol"] = code
                all_data.append(df)
                if was_cached:
                    n_cached += 1
                    if verbose:
                        print(f"cached ({len(df)} bars)")
                else:
                    n_fetched += 1
                    if verbose:
                        print(f"OK ({len(df)} bars)")
                n_ok += 1
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                n_fail += 1
                continue

        if verbose:
            print(f"\n📊 Summary: {n_ok} OK ({n_cached} cached, {n_fetched} fetched), "
                  f"{n_skip} skipped, {n_fail} failed")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, axis=0).sort_index()
        if verbose:
            print(f"✅ Total: {len(combined)} bars from {len(all_data)}/{len(symbols)} stocks")
        return combined

    def get_or_fetch_multi_dict(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        base_delay: float = 0.8,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple stocks as a dict of {symbol: DataFrame}."""
        result = {}
        for i, sym in enumerate(symbols):
            code = sym.split(".")[0]
            if verbose:
                print(f"  [{i+1}/{len(symbols)}] {code}...", end=" ", flush=True)
            try:
                was_cached = self._is_fully_cached(code, start_date, end_date)
                df = self.get_or_fetch(code, start_date, end_date, max_retries=max_retries, base_delay=base_delay)
                if df is None or df.empty or len(df) < 100:
                    if verbose:
                        print(f"skip ({0 if df is None else len(df)} bars)")
                    continue
                result[code] = df
                if verbose:
                    src = "cached" if was_cached else "fetched"
                    print(f"{src} ({len(df)} bars)")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                continue
        if verbose:
            print(f"✅ Loaded {len(result)} stocks")
        return result

    def cache_status(self) -> List[dict]:
        """Return per-symbol cache coverage info."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT symbol, min_date, max_date, bar_count, last_updated FROM cache_meta ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [
            {"symbol": r[0], "min_date": r[1], "max_date": r[2],
             "bar_count": r[3], "last_updated": r[4]}
            for r in rows
        ]

    def invalidate(self, symbol: Optional[str] = None):
        """Remove cached data for a symbol (or all if None)."""
        conn = self._conn()
        if symbol:
            conn.execute("DELETE FROM kline_daily WHERE symbol = ?", (symbol,))
            conn.execute("DELETE FROM cache_meta WHERE symbol = ?", (symbol,))
        else:
            conn.execute("DELETE FROM kline_daily")
            conn.execute("DELETE FROM cache_meta")
        conn.commit()
        conn.close()

    # ── Internal Methods ─────────────────────────────────────────────

    def _normalize_date(self, d: str) -> str:
        """Normalize date to YYYY-MM-DD."""
        d = d.strip()
        if len(d) == 8 and d.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return d

    def _get_meta(self, symbol: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT min_date, max_date, bar_count, last_updated FROM cache_meta WHERE symbol = ?",
            (symbol,)
        ).fetchone()
        conn.close()
        if row:
            return {"min_date": row[0], "max_date": row[1], "bar_count": row[2], "last_updated": row[3]}
        return None

    def _is_fully_cached(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if the requested range is fully covered by cache."""
        start = self._normalize_date(start_date)
        end = self._normalize_date(end_date)
        meta = self._get_meta(symbol)
        if meta is None:
            return False
        return meta["min_date"] <= start and meta["max_date"] >= end

    def _find_gaps(
        self, symbol: str, start: str, end: str, meta: Optional[dict]
    ) -> List[Tuple[str, str]]:
        """Find date ranges that need to be fetched.

        Returns list of (gap_start, gap_end) tuples.
        """
        if meta is None:
            # No cache at all — fetch entire range
            return [(start, end)]

        gaps = []
        cached_min = meta["min_date"]
        cached_max = meta["max_date"]

        # Need data before what we have?
        if start < cached_min:
            # Fetch from start to one day before cached_min
            day_before = (datetime.strptime(cached_min, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if start <= day_before:
                gaps.append((start, day_before))

        # Need data after what we have?
        if end > cached_max:
            # Fetch from one day after cached_max to end
            day_after = (datetime.strptime(cached_max, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if day_after <= end:
                gaps.append((day_after, end))

        return gaps

    def _fill_gaps(
        self, symbol: str, gaps: List[Tuple[str, str]],
        max_retries: int, base_delay: float,
        market: str = "CN", raw_symbol: str = "",
    ):
        """Fetch missing date ranges and store in cache.

        For CN: tushare (primary) → akshare (fallback).
        For US/HK: yfinance.
        """
        import sys

        for gap_start, gap_end in gaps:
            fetched = False

            # US/HK: use yfinance
            if market in ("US", "HK"):
                last_err = None
                for attempt in range(1, max_retries + 1):
                    try:
                        from yfinance_provider import fetch_daily as yf_fetch, is_available as yf_ok
                        if yf_ok():
                            df = yf_fetch(raw_symbol or symbol, gap_start, gap_end, market=market)
                            if df is not None and not df.empty:
                                self._store_klines_df(symbol, df)
                                fetched = True
                            break
                        else:
                            last_err = "yfinance not available"
                            break
                    except ImportError:
                        last_err = "yfinance not installed"
                        break
                    except Exception as e:
                        last_err = str(e)
                        if attempt < max_retries:
                            time.sleep(base_delay * (2 ** (attempt - 1)))
                if not fetched and last_err:
                    print(f"[cache] yfinance failed for {symbol} ({gap_start}~{gap_end}): {last_err}", file=sys.stderr)
                if fetched or market in ("US", "HK"):
                    if len(gaps) > 1:
                        time.sleep(base_delay)
                    continue

            # CN: tushare first
            ts_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    from tushare_provider import fetch_daily as ts_fetch_daily, is_available as ts_ok
                    if ts_ok():
                        df = ts_fetch_daily(symbol, gap_start, gap_end)
                        if df is not None and not df.empty:
                            self._store_klines_df(symbol, df)
                            fetched = True
                        break
                    else:
                        ts_err = "no tushare token"
                        break  # No token, skip to akshare
                except ImportError:
                    ts_err = "tushare not installed"
                    break  # tushare not installed
                except Exception as e:
                    ts_err = str(e)
                    if attempt < max_retries:
                        time.sleep(base_delay * (2 ** (attempt - 1)))

            # 2) Fallback to akshare
            ak_err = None
            if not fetched:
                for attempt in range(1, max_retries + 1):
                    try:
                        import akshare as ak
                        df = ak.stock_zh_a_hist(
                            symbol=symbol, period="daily",
                            start_date=gap_start.replace("-", ""),
                            end_date=gap_end.replace("-", ""),
                            adjust="qfq",
                        )
                        if df is not None and not df.empty:
                            self._store_klines(symbol, df)
                            fetched = True
                        break
                    except ImportError:
                        ak_err = "akshare not installed"
                        break
                    except Exception as e:
                        ak_err = str(e)
                        if attempt < max_retries:
                            time.sleep(base_delay * (2 ** (attempt - 1)))

            if not fetched:
                reasons = []
                if ts_err: reasons.append(f"tushare: {ts_err}")
                if ak_err: reasons.append(f"akshare: {ak_err}")
                print(f"[cache] All providers failed for {symbol} ({gap_start}~{gap_end}): {'; '.join(reasons) or 'empty data'}", file=sys.stderr)

            # Throttle between gap fetches
            if len(gaps) > 1:
                time.sleep(base_delay)

    def _store_klines_df(self, symbol: str, df: pd.DataFrame):
        """Store a standardized DataFrame (date index, OHLCV columns) into cache."""
        conn = self._conn()
        rows = []
        for date_val, row in df.iterrows():
            date_str = str(date_val.date()) if hasattr(date_val, 'date') else str(date_val)[:10]
            rows.append((
                symbol, date_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))
        if not rows:
            conn.close()
            return
        conn.executemany(
            """INSERT OR REPLACE INTO kline_daily (symbol, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

        # Update metadata
        all_dates = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM kline_daily WHERE symbol = ?",
            (symbol,)
        ).fetchone()
        if all_dates and all_dates[0]:
            conn.execute(
                """INSERT OR REPLACE INTO cache_meta (symbol, min_date, max_date, bar_count, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, all_dates[0], all_dates[1], all_dates[2],
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

        conn.commit()
        conn.close()

    def _store_klines(self, symbol: str, df: pd.DataFrame):
        """Store fetched akshare DataFrame into cache."""
        conn = self._conn()

        # Normalize column names (akshare uses Chinese)
        col_map = {
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        }
        df = df.rename(columns=col_map)

        rows = []
        for _, row in df.iterrows():
            date_str = str(row["date"])
            if len(date_str) > 10:
                date_str = date_str[:10]
            rows.append((
                symbol, date_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))

        conn.executemany(
            """INSERT OR REPLACE INTO kline_daily (symbol, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

        # Update metadata
        all_dates = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM kline_daily WHERE symbol = ?",
            (symbol,)
        ).fetchone()

        if all_dates and all_dates[0]:
            conn.execute(
                """INSERT OR REPLACE INTO cache_meta (symbol, min_date, max_date, bar_count, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, all_dates[0], all_dates[1], all_dates[2],
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

        conn.commit()
        conn.close()

    def _read_cache(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Read klines from cache for the given range."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
               FROM kline_daily
               WHERE symbol = ? AND date >= ? AND date <= ?
               ORDER BY date ASC""",
            (symbol, start, end),
        ).fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.astype(float)
        return df

    def _fetch_raw(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> pd.DataFrame:
        """Direct fetch without caching. Tries tushare first, then akshare."""
        start = self._normalize_date(start_date)
        end = self._normalize_date(end_date)

        # 1) Try tushare
        try:
            from tushare_provider import is_available as ts_ok
            if ts_ok():
                if period in ("1", "5", "15", "30", "60"):
                    from tushare_provider import fetch_minute
                    df = fetch_minute(symbol, start, end, freq=period)
                else:
                    from tushare_provider import fetch_daily
                    df = fetch_daily(symbol, start, end)
                if df is not None and not df.empty:
                    return df
        except (ImportError, Exception):
            pass

        # 2) Fallback to akshare
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        start_raw = start.replace("-", "")
        end_raw = end.replace("-", "")

        if period in ("1", "5", "15", "30", "60"):
            start_dt = f"{start_raw[:4]}-{start_raw[4:6]}-{start_raw[6:8]} 09:30:00"
            end_dt = f"{end_raw[:4]}-{end_raw[4:6]}-{end_raw[6:8]} 15:00:00"
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period=period,
                start_date=start_dt, end_date=end_dt, adjust="qfq",
            )
        else:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start_raw, end_date=end_raw, adjust="qfq",
            )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date", "时间": "date",
            "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    # Keep old name as alias for backward compat
    _fetch_from_akshare_raw = _fetch_raw

    def _fetch_yfinance_direct(
        self, symbol: str, start_date: str, end_date: str, market: str
    ) -> pd.DataFrame:
        """Direct fetch from yfinance without caching (for minute data)."""
        try:
            from yfinance_provider import fetch_daily as yf_fetch, is_available as yf_ok
            if yf_ok():
                df = yf_fetch(symbol, start_date, end_date, market=market)
                if df is not None and not df.empty:
                    return df
        except (ImportError, Exception):
            pass
        return pd.DataFrame()


# ── Convenience singleton ────────────────────────────────────────────

_default_cache: Optional[MarketCache] = None


def get_cache(db_path: Optional[str] = None) -> MarketCache:
    """Get or create the default MarketCache instance."""
    global _default_cache
    if _default_cache is None or (db_path and _default_cache.db_path != db_path):
        _default_cache = MarketCache(db_path)
    return _default_cache


# ── CLI for diagnostics ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cache = MarketCache()

    if len(sys.argv) < 2:
        print("Usage: market_cache.py <status|fetch|invalidate> [args...]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        import json
        status = cache.cache_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        total = sum(s["bar_count"] for s in status)
        print(f"\nTotal: {len(status)} symbols, {total} bars")

    elif cmd == "fetch":
        if len(sys.argv) < 5:
            print("Usage: market_cache.py fetch <symbol> <start> <end>")
            sys.exit(1)
        sym, start, end = sys.argv[2], sys.argv[3], sys.argv[4]
        df = cache.get_or_fetch(sym, start, end)
        print(f"Fetched {len(df)} bars for {sym}")
        if not df.empty:
            print(df.head())
            print("...")
            print(df.tail())

    elif cmd == "invalidate":
        sym = sys.argv[2] if len(sys.argv) > 2 else None
        cache.invalidate(sym)
        print(f"Invalidated cache for: {sym or 'ALL'}")

    else:
        print(f"Unknown command: {cmd}")
