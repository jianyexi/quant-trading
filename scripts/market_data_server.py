#!/usr/bin/env python3
"""Persistent market data server — avoids fork-per-quote overhead.

Runs as a long-lived Flask+WebSocket server:
  - GET  /quote/<symbol>        → latest quote (cached, <5ms)
  - GET  /klines/<symbol>       → historical klines
  - WS   /ws/quotes             → real-time push (subscribe symbols)
  - GET  /health                → health check

Usage:
    python scripts/market_data_server.py --port 18092

Latency improvement: 500-2000ms (fork python) → <5ms (HTTP) / <1ms (WS push)
"""

import argparse
import json
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from market_data import normalize_symbol, exchange_suffix, _STOCK_NAMES

try:
    from flask import Flask, jsonify, request
    from flask_sock import Sock
except ImportError:
    print("Installing dependencies: flask flask-sock")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-sock", "-q"])
    from flask import Flask, jsonify, request
    from flask_sock import Sock

import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
sock = Sock(app)

# ── In-memory cache ─────────────────────────────────────────────────

class QuoteCache:
    """Thread-safe quote cache with TTL."""
    def __init__(self, ttl_seconds: float = 3.0):
        self.ttl = ttl_seconds
        self._cache = {}  # symbol -> (timestamp, data)
        self._lock = threading.Lock()

    def get(self, symbol: str):
        with self._lock:
            if symbol in self._cache:
                ts, data = self._cache[symbol]
                if time.time() - ts < self.ttl:
                    return data
        return None

    def put(self, symbol: str, data: dict):
        with self._lock:
            self._cache[symbol] = (time.time(), data)

    def all_symbols(self):
        with self._lock:
            return list(self._cache.keys())


class KlineCache:
    """Cache historical klines to avoid re-fetching on warmup."""
    def __init__(self):
        self._cache = {}  # (symbol, period) -> (timestamp, DataFrame)
        self._lock = threading.Lock()

    def get(self, symbol: str, period: str = "daily", max_age: float = 300.0):
        key = (symbol, period)
        with self._lock:
            if key in self._cache:
                ts, df = self._cache[key]
                if time.time() - ts < max_age:
                    return df
        return None

    def put(self, symbol: str, period: str, df: pd.DataFrame):
        with self._lock:
            self._cache[(symbol, period)] = (time.time(), df)


quote_cache = QuoteCache(ttl_seconds=3.0)
kline_cache = KlineCache()

# ── WebSocket subscribers ────────────────────────────────────────────

ws_subscribers = []  # list of (ws, symbols_set)
ws_lock = threading.Lock()

# ── Akshare fetchers ────────────────────────────────────────────────

def fetch_quote_akshare(symbol: str) -> dict:
    """Fetch latest quote from akshare — uses hist for speed."""
    code = normalize_symbol(symbol)
    full = symbol if "." in symbol else exchange_suffix(code)

    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        if df is not None and not df.empty:
            r = df.iloc[-1]
            return {
                "symbol": full,
                "name": _STOCK_NAMES.get(code, code),
                "price": float(r["收盘"]),
                "open": float(r["开盘"]),
                "high": float(r["最高"]),
                "low": float(r["最低"]),
                "volume": float(r["成交量"]),
                "turnover": float(r.get("成交额", 0)),
                "change": float(r.get("涨跌额", 0)),
                "change_percent": float(r.get("涨跌幅", 0)),
                "timestamp": str(r["日期"]),
            }
    except Exception as e:
        return {"symbol": full, "error": str(e)}

    return {"symbol": full, "error": "No data"}


def fetch_klines_akshare(symbol: str, start: str, end: str, period: str = "daily") -> list:
    """Fetch historical klines."""
    code = normalize_symbol(symbol)
    full = symbol if "." in symbol else exchange_suffix(code)

    try:
        if period == "daily":
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
        else:
            df = ak.stock_zh_a_hist_min_em(
                symbol=code, period=period,
                start_date=f"{start} 09:30:00",
                end_date=f"{end} 15:00:00",
                adjust="qfq",
            )

        if df is None or df.empty:
            return []

        col_map = {"日期": "date", "时间": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns=col_map)
        if "date" not in df.columns:
            for c in df.columns:
                if "时间" in c or "日期" in c:
                    df = df.rename(columns={c: "date"})
                    break

        result = []
        for _, r in df.iterrows():
            result.append({
                "symbol": full,
                "date": str(r.get("date", "")),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]


# ── Background quote poller (for WS push) ───────────────────────────

def quote_poller_thread(interval: float = 3.0):
    """Background thread that polls quotes and pushes to WS subscribers."""
    print(f"📡 Quote poller started (interval={interval}s)")
    while True:
        time.sleep(interval)

        # Collect all subscribed symbols
        with ws_lock:
            all_syms = set()
            for _, syms in ws_subscribers:
                all_syms |= syms
        if not all_syms:
            continue

        # Fetch quotes
        for sym in all_syms:
            cached = quote_cache.get(sym)
            if cached:
                quote = cached
            else:
                try:
                    quote = fetch_quote_akshare(sym)
                    if "error" not in quote:
                        quote_cache.put(sym, quote)
                except Exception as e:
                    quote = {"symbol": sym, "error": str(e)}

            # Push to subscribers
            msg = json.dumps(quote, ensure_ascii=False)
            with ws_lock:
                dead = []
                for i, (ws, syms) in enumerate(ws_subscribers):
                    if sym in syms:
                        try:
                            ws.send(msg)
                        except Exception:
                            dead.append(i)
                for i in reversed(dead):
                    ws_subscribers.pop(i)


# ── HTTP Endpoints ──────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "market_data_server",
        "cached_symbols": len(quote_cache.all_symbols()),
        "ws_subscribers": len(ws_subscribers),
        "uptime_seconds": int(time.time() - _start_time),
    })


@app.route("/quote/<symbol>")
def get_quote(symbol):
    t0 = time.time()

    # Check cache first
    cached = quote_cache.get(symbol)
    if cached:
        cached["_cache"] = True
        cached["_latency_ms"] = round((time.time() - t0) * 1000, 2)
        return jsonify(cached)

    # Fetch from akshare
    quote = fetch_quote_akshare(symbol)
    if "error" not in quote:
        quote_cache.put(symbol, quote)

    quote["_cache"] = False
    quote["_latency_ms"] = round((time.time() - t0) * 1000, 2)
    return jsonify(quote)


@app.route("/klines/<symbol>")
def get_klines(symbol):
    start = request.args.get("start", (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
    end = request.args.get("end", datetime.now().strftime("%Y-%m-%d"))
    period = request.args.get("period", "daily")

    # Check cache
    cached = kline_cache.get(symbol, period, max_age=60.0)
    if cached is not None:
        return jsonify({"klines": cached, "_cache": True})

    klines = fetch_klines_akshare(symbol, start, end, period)
    if klines and "error" not in klines[0]:
        kline_cache.put(symbol, period, klines)

    return jsonify({"klines": klines, "_cache": False})


@app.route("/batch_quotes", methods=["POST"])
def batch_quotes():
    """Fetch multiple quotes in one call."""
    body = request.get_json(force=True, silent=True) or {}
    symbols = body.get("symbols", [])
    results = {}
    for sym in symbols:
        cached = quote_cache.get(sym)
        if cached:
            results[sym] = cached
        else:
            q = fetch_quote_akshare(sym)
            if "error" not in q:
                quote_cache.put(sym, q)
            results[sym] = q
    return jsonify(results)


# ── WebSocket Endpoint ──────────────────────────────────────────────

@sock.route("/ws/quotes")
def ws_quotes(ws):
    """WebSocket: client sends {"subscribe": ["600519.SH", "300750.SZ"]}
    Server pushes quote updates for subscribed symbols."""
    sub_symbols = set()
    with ws_lock:
        ws_subscribers.append((ws, sub_symbols))
    print(f"🔌 WS client connected (total: {len(ws_subscribers)})")

    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                # Send heartbeat
                ws.send(json.dumps({"type": "heartbeat", "ts": time.time()}))
                continue
            try:
                data = json.loads(msg)
                if "subscribe" in data:
                    new_syms = set(data["subscribe"])
                    sub_symbols.clear()
                    sub_symbols.update(new_syms)
                    ws.send(json.dumps({"type": "subscribed", "symbols": list(sub_symbols)}))
                    print(f"📡 WS subscribed: {sub_symbols}")
                elif "unsubscribe" in data:
                    for s in data["unsubscribe"]:
                        sub_symbols.discard(s)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        with ws_lock:
            ws_subscribers[:] = [(w, s) for w, s in ws_subscribers if w is not ws]
        print(f"🔌 WS client disconnected (remaining: {len(ws_subscribers)})")


# ── Main ────────────────────────────────────────────────────────────

_start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Market Data Server")
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--poll-interval", type=float, default=3.0,
                        help="Quote polling interval for WS push (seconds)")
    parser.add_argument("--cache-ttl", type=float, default=3.0,
                        help="Quote cache TTL (seconds)")
    args = parser.parse_args()

    quote_cache.ttl = args.cache_ttl

    # Start background poller
    poller = threading.Thread(target=quote_poller_thread, args=(args.poll_interval,), daemon=True)
    poller.start()

    print(f"🚀 Market Data Server starting on port {args.port}")
    print(f"   Cache TTL: {args.cache_ttl}s | WS poll: {args.poll_interval}s")
    print(f"   Endpoints:")
    print(f"     GET  /health")
    print(f"     GET  /quote/<symbol>")
    print(f"     GET  /klines/<symbol>?start=&end=&period=")
    print(f"     POST /batch_quotes  {{symbols: [...]}}")
    print(f"     WS   /ws/quotes")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
