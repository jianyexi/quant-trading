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


# ── TCP Message Queue Server ─────────────────────────────────────────
# Binary protocol: [4 bytes: msg_len (big-endian u32)] [msg_len bytes: JSON]
# Much faster than HTTP — zero parsing overhead, persistent connection.

import socket
import struct
import select

tcp_clients = []  # list of (conn, addr, subscribed_symbols)
tcp_lock = threading.Lock()


def tcp_send(conn, msg: dict):
    """Send a length-prefixed JSON message."""
    data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(data))
    try:
        conn.sendall(header + data)
    except Exception:
        pass


def tcp_recv(conn, timeout=0.1) -> dict | None:
    """Receive a length-prefixed JSON message (non-blocking)."""
    conn.setblocking(False)
    try:
        ready = select.select([conn], [], [], timeout)
        if not ready[0]:
            return None
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        msg_len = struct.unpack(">I", header)[0]
        if msg_len > 10_000_000:
            return None
        body = b""
        while len(body) < msg_len:
            chunk = conn.recv(min(msg_len - len(body), 65536))
            if not chunk:
                return None
            body += chunk
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None
    finally:
        conn.setblocking(True)


def handle_tcp_client(conn, addr):
    """Handle a single TCP client connection."""
    sub_symbols = set()
    with tcp_lock:
        tcp_clients.append((conn, addr, sub_symbols))
    print(f"🔗 TCP client connected: {addr} (total: {len(tcp_clients)})")

    # Send health response
    tcp_send(conn, {"type": "connected", "ts": time.time()})

    try:
        while True:
            msg = tcp_recv(conn, timeout=1.0)
            if msg is None:
                # Check if connection is still alive
                try:
                    conn.sendall(b"")
                except Exception:
                    break
                continue

            cmd = msg.get("cmd", "")

            if cmd == "subscribe":
                syms = msg.get("symbols", [])
                sub_symbols.clear()
                sub_symbols.update(syms)
                tcp_send(conn, {"type": "subscribed", "symbols": list(sub_symbols)})
                print(f"📡 TCP subscribed: {sub_symbols}")

            elif cmd == "warmup":
                sym = msg.get("symbol", "")
                days = msg.get("days", 80)
                start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                period = msg.get("period", "daily")
                klines = fetch_klines_akshare(sym, start, end, period)
                for kl in klines:
                    if "error" not in kl:
                        kl["type"] = "kline"
                        tcp_send(conn, kl)
                tcp_send(conn, {"type": "warmup_done", "symbol": sym, "count": len(klines)})

            elif cmd == "quote":
                sym = msg.get("symbol", "")
                cached = quote_cache.get(sym)
                if cached:
                    cached["type"] = "quote"
                    cached["_cache"] = True
                    tcp_send(conn, cached)
                else:
                    q = fetch_quote_akshare(sym)
                    if "error" not in q:
                        quote_cache.put(sym, q)
                    q["type"] = "quote"
                    q["_cache"] = False
                    tcp_send(conn, q)

            elif cmd == "ping":
                tcp_send(conn, {"type": "pong", "ts": time.time()})

    except Exception as e:
        print(f"🔗 TCP client error: {e}")
    finally:
        with tcp_lock:
            tcp_clients[:] = [(c, a, s) for c, a, s in tcp_clients if c is not conn]
        conn.close()
        print(f"🔗 TCP client disconnected: {addr} (remaining: {len(tcp_clients)})")


def tcp_push_thread(interval: float = 3.0):
    """Background thread that pushes quotes to TCP subscribers."""
    while True:
        time.sleep(interval)
        with tcp_lock:
            all_syms = set()
            for _, _, syms in tcp_clients:
                all_syms |= syms
        if not all_syms:
            continue

        for sym in all_syms:
            cached = quote_cache.get(sym)
            if not cached:
                try:
                    cached = fetch_quote_akshare(sym)
                    if "error" not in cached:
                        quote_cache.put(sym, cached)
                except Exception:
                    continue

            if cached and "error" not in cached:
                msg = cached.copy()
                msg["type"] = "quote"
                msg["_push"] = True
                with tcp_lock:
                    dead = []
                    for i, (conn, addr, syms) in enumerate(tcp_clients):
                        if sym in syms:
                            try:
                                tcp_send(conn, msg)
                            except Exception:
                                dead.append(i)
                    for i in reversed(dead):
                        tcp_clients.pop(i)


def tcp_server_thread(port: int):
    """TCP message queue server thread."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(8)
    print(f"🔗 TCP Message Queue listening on port {port}")

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t = threading.Thread(target=handle_tcp_client, args=(conn, addr), daemon=True)
        t.start()


# ── Main ────────────────────────────────────────────────────────────

_start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Market Data Server")
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--tcp-port", type=int, default=18093,
                        help="TCP message queue port")
    parser.add_argument("--poll-interval", type=float, default=3.0,
                        help="Quote polling interval for push (seconds)")
    parser.add_argument("--cache-ttl", type=float, default=3.0,
                        help="Quote cache TTL (seconds)")
    args = parser.parse_args()

    quote_cache.ttl = args.cache_ttl

    # Start background threads
    threading.Thread(target=quote_poller_thread, args=(args.poll_interval,), daemon=True).start()
    threading.Thread(target=tcp_server_thread, args=(args.tcp_port,), daemon=True).start()
    threading.Thread(target=tcp_push_thread, args=(args.poll_interval,), daemon=True).start()

    print(f"🚀 Market Data Server starting")
    print(f"   HTTP: port {args.port} | TCP MQ: port {args.tcp_port}")
    print(f"   Cache TTL: {args.cache_ttl}s | Push interval: {args.poll_interval}s")
    print(f"   HTTP endpoints: /health, /quote/<sym>, /klines/<sym>, /batch_quotes")
    print(f"   TCP protocol: length-prefixed JSON (subscribe/warmup/quote/ping)")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
