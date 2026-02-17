#!/usr/bin/env python3
"""
L2 Market Data Recorder & Server — Records tick/depth data from QMT and serves
to the Rust engine via TCP Message Queue (port 18095).

Features:
  - Records L2 tick-by-tick trades (逐笔成交) from QMT xtquant
  - Records L2 depth snapshots (十档盘口) at configurable intervals
  - Stores data in Parquet files for ML training
  - Serves real-time L2 data to Rust engine via TCP MQ
  - Replay mode for backtesting with recorded L2 data
  - Aggregates L2 features (OBI, big-order ratio, etc.) per interval

Data Sources:
  - QMT xtquant: Primary L2 source (requires miniQMT)
  - Simulated: Random tick/depth generator for testing

Storage layout:
  data/l2/tick/600519.SH_20260217.parquet
  data/l2/depth/600519.SH_20260217.parquet
  data/l2/features/l2_features_20260217.parquet

Usage:
  # Record live L2 data (requires QMT running)
  python scripts/l2_recorder.py --mode record --symbols 600519.SH,000858.SZ

  # Serve recorded data to Rust engine (replay mode)
  python scripts/l2_recorder.py --mode replay --date 20260217

  # Generate simulated L2 data for testing
  python scripts/l2_recorder.py --mode simulate --symbols 600519.SH

  # Record + serve simultaneously
  python scripts/l2_recorder.py --mode live --symbols 600519.SH
"""

import argparse
import json
import logging
import os
import random
import signal
import socket
import struct
import sys
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow", "-q"])
    import pandas as pd

# ── Configuration ────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "l2"
TCP_PORT = 18095  # L2 TCP MQ port (matches Rust DataMode::L2)
DEPTH_INTERVAL = 3.0  # Seconds between depth snapshots
FLUSH_INTERVAL = 60.0  # Seconds between Parquet flushes
BIG_ORDER_THRESHOLD = 100_000  # Volume threshold for "big order" (手)

logger = logging.getLogger("l2_recorder")


# ── Data Buffers ─────────────────────────────────────────────────────

class L2Buffer:
    """Thread-safe buffer for L2 data accumulation and periodic flush."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._tick_buf: list[dict] = []
        self._depth_buf: list[dict] = []
        self._feature_buf: list[dict] = []
        self._lock = threading.Lock()

        # Aggregation state per symbol
        self._agg: dict[str, dict] = {}  # symbol -> running stats

        # Ensure directories
        (data_dir / "tick").mkdir(parents=True, exist_ok=True)
        (data_dir / "depth").mkdir(parents=True, exist_ok=True)
        (data_dir / "features").mkdir(parents=True, exist_ok=True)

    def add_tick(self, tick: dict):
        """Add a tick record to the buffer."""
        with self._lock:
            self._tick_buf.append(tick)
        # Update aggregation stats
        sym = tick["symbol"]
        self._update_agg(sym, tick)

    def add_depth(self, depth: dict):
        """Add a depth snapshot to the buffer."""
        with self._lock:
            self._depth_buf.append(depth)

    def add_feature(self, feat: dict):
        """Add an aggregated feature row."""
        with self._lock:
            self._feature_buf.append(feat)

    def _update_agg(self, symbol: str, tick: dict):
        """Update running aggregation for a symbol (lock-free per symbol)."""
        if symbol not in self._agg:
            self._agg[symbol] = {
                "total_vol": 0.0, "buy_vol": 0.0, "sell_vol": 0.0,
                "big_buy_vol": 0.0, "big_sell_vol": 0.0,
                "tick_count": 0, "vwap_num": 0.0, "vwap_den": 0.0,
            }
        a = self._agg[symbol]
        vol = tick.get("volume", 0)
        price = tick.get("price", 0)
        direction = tick.get("direction", " ")

        a["total_vol"] += vol
        a["tick_count"] += 1
        a["vwap_num"] += price * vol
        a["vwap_den"] += vol

        if direction == "B":
            a["buy_vol"] += vol
            if vol >= BIG_ORDER_THRESHOLD:
                a["big_buy_vol"] += vol
        elif direction == "S":
            a["sell_vol"] += vol
            if vol >= BIG_ORDER_THRESHOLD:
                a["big_sell_vol"] += vol

    def compute_features(self, symbol: str, depth: dict) -> dict:
        """Compute L2 features from current aggregation + depth snapshot."""
        a = self._agg.get(symbol, {})
        total_vol = a.get("total_vol", 0) or 1.0

        # Depth features
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        bid_vol_5 = sum(l.get("volume", 0) for l in bids[:5])
        ask_vol_5 = sum(l.get("volume", 0) for l in asks[:5])
        total_depth = bid_vol_5 + ask_vol_5 or 1.0

        bid1_p = bids[0]["price"] if bids else 0
        ask1_p = asks[0]["price"] if asks else 0
        bid1_v = bids[0].get("volume", 0) if bids else 0
        ask1_v = asks[0].get("volume", 0) if asks else 0

        # Weighted mid price
        if bid1_v + ask1_v > 0:
            wmid = (bid1_p * ask1_v + ask1_p * bid1_v) / (bid1_v + ask1_v)
        else:
            wmid = (bid1_p + ask1_p) / 2 if (bid1_p + ask1_p) > 0 else 0

        vwap = a.get("vwap_num", 0) / a.get("vwap_den", 1) if a.get("vwap_den", 0) > 0 else 0

        feat = {
            "symbol": symbol,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Order Book Imbalance
            "obi": (bid_vol_5 - ask_vol_5) / total_depth,
            # Spread
            "spread": ask1_p - bid1_p,
            "spread_bps": (ask1_p - bid1_p) / ((ask1_p + bid1_p) / 2) * 10000 if (ask1_p + bid1_p) > 0 else 0,
            # Depth ratio
            "depth_ratio_1": bid1_v / ask1_v if ask1_v > 0 else 0,
            "depth_ratio_5": bid_vol_5 / ask_vol_5 if ask_vol_5 > 0 else 0,
            # Weighted mid
            "weighted_mid": wmid,
            # Trade flow
            "buy_ratio": a.get("buy_vol", 0) / total_vol,
            "big_order_ratio": (a.get("big_buy_vol", 0) + a.get("big_sell_vol", 0)) / total_vol,
            "big_buy_ratio": a.get("big_buy_vol", 0) / total_vol,
            "big_sell_ratio": a.get("big_sell_vol", 0) / total_vol,
            # Trade speed
            "tick_count": a.get("tick_count", 0),
            # VWAP
            "vwap": vwap,
            "vwap_deviation": (depth.get("last_price", 0) - vwap) / vwap if vwap > 0 else 0,
            # Volume
            "total_volume": a.get("total_vol", 0),
            "last_price": depth.get("last_price", 0),
        }
        return feat

    def reset_agg(self, symbol: str):
        """Reset aggregation counters for a new interval."""
        self._agg[symbol] = {
            "total_vol": 0.0, "buy_vol": 0.0, "sell_vol": 0.0,
            "big_buy_vol": 0.0, "big_sell_vol": 0.0,
            "tick_count": 0, "vwap_num": 0.0, "vwap_den": 0.0,
        }

    def flush(self, today: str = None):
        """Flush buffers to Parquet files."""
        if today is None:
            today = date.today().strftime("%Y%m%d")

        with self._lock:
            ticks = self._tick_buf.copy()
            depths = self._depth_buf.copy()
            feats = self._feature_buf.copy()
            self._tick_buf.clear()
            self._depth_buf.clear()
            self._feature_buf.clear()

        if ticks:
            self._append_parquet(ticks, "tick", today)
        if depths:
            self._append_parquet(depths, "depth", today)
        if feats:
            self._append_parquet(feats, "features", today, key="l2_features")

    def _append_parquet(self, rows: list[dict], subdir: str, today: str, key: str = None):
        """Append rows to a Parquet file, creating if needed."""
        # Group by symbol
        by_sym: dict[str, list] = {}
        for r in rows:
            sym = r.get("symbol", "unknown")
            by_sym.setdefault(sym, []).append(r)

        for sym, sym_rows in by_sym.items():
            if key:
                path = self.data_dir / subdir / f"{key}_{today}.parquet"
            else:
                path = self.data_dir / subdir / f"{sym}_{today}.parquet"

            df_new = pd.DataFrame(sym_rows)
            if path.exists():
                try:
                    df_old = pd.read_parquet(path)
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                except Exception:
                    pass

            df_new.to_parquet(path, index=False)
            logger.info(f"Flushed {len(sym_rows)} rows → {path}")


# ── TCP MQ Server (serves L2 to Rust engine) ────────────────────────

tcp_clients: list[tuple] = []
tcp_lock = threading.Lock()


def tcp_send(conn, msg: dict):
    """Send length-prefixed JSON message."""
    data = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
    header = struct.pack(">I", len(data))
    try:
        conn.sendall(header + data)
    except Exception:
        pass


def tcp_recv(conn, timeout=0.5) -> Optional[dict]:
    """Receive length-prefixed JSON message."""
    import select as sel
    conn.setblocking(False)
    try:
        ready = sel.select([conn], [], [], timeout)
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


def broadcast_l2(msg: dict):
    """Broadcast an L2 message to all connected TCP clients."""
    with tcp_lock:
        dead = []
        for i, (conn, addr) in enumerate(tcp_clients):
            try:
                tcp_send(conn, msg)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            tcp_clients.pop(i)


def handle_tcp_client(conn, addr):
    """Handle a TCP client — just keep-alive and subscribe confirmation."""
    with tcp_lock:
        tcp_clients.append((conn, addr))
    logger.info(f"L2 TCP client connected: {addr} (total: {len(tcp_clients)})")
    tcp_send(conn, {"type": "connected", "mode": "l2", "ts": time.time()})

    try:
        while True:
            msg = tcp_recv(conn, timeout=2.0)
            if msg is None:
                try:
                    conn.sendall(b"")
                except Exception:
                    break
                continue
            cmd = msg.get("cmd", "")
            if cmd == "ping":
                tcp_send(conn, {"type": "pong", "ts": time.time()})
            elif cmd == "subscribe":
                symbols = msg.get("symbols", [])
                tcp_send(conn, {"type": "subscribed", "symbols": symbols})
                logger.info(f"L2 client subscribed: {symbols}")
    except Exception as e:
        logger.debug(f"TCP client error: {e}")
    finally:
        with tcp_lock:
            tcp_clients[:] = [(c, a) for c, a in tcp_clients if c is not conn]
        conn.close()
        logger.info(f"L2 TCP client disconnected: {addr}")


def tcp_server_thread(port: int):
    """L2 TCP server — accepts connections from Rust engine."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    logger.info(f"L2 TCP MQ listening on port {port}")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t = threading.Thread(target=handle_tcp_client, args=(conn, addr), daemon=True)
        t.start()


# ── QMT L2 Data Source ──────────────────────────────────────────────

class QmtL2Source:
    """Connects to QMT xtquant for L2 tick and depth data."""

    def __init__(self, symbols: list[str], buf: L2Buffer):
        self.symbols = symbols
        self.buf = buf
        self._seq = 0
        self._running = False

    def start(self):
        """Start subscribing to QMT L2 data."""
        try:
            from xtquant import xtdata
        except ImportError:
            logger.error("xtquant not installed. Use --mode simulate for testing.")
            return False

        self._running = True
        logger.info(f"Subscribing to QMT L2 for: {self.symbols}")

        # Subscribe to tick-by-tick trades
        for sym in self.symbols:
            try:
                xtdata.subscribe_quote(
                    sym, period="tick", count=-1,
                    callback=self._on_tick_callback
                )
                logger.info(f"  Subscribed tick: {sym}")
            except Exception as e:
                logger.error(f"  Failed tick subscribe {sym}: {e}")

        # Start depth polling thread
        threading.Thread(target=self._depth_poll_loop, daemon=True).start()

        # Keep xtdata running
        try:
            from xtquant import xtdata
            xtdata.run()
        except Exception as e:
            logger.warning(f"xtdata.run() ended: {e}")

        return True

    def _on_tick_callback(self, data):
        """Callback from xtquant for tick data."""
        try:
            for sym, ticks in data.items():
                for t in ticks:
                    self._seq += 1
                    tick = {
                        "type": "tick",
                        "symbol": sym,
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "price": float(t.get("lastPrice", t.get("price", 0))),
                        "volume": float(t.get("volume", t.get("vol", 0))),
                        "direction": "B" if t.get("buyNo", 0) > t.get("sellNo", 0) else "S",
                        "seq": self._seq,
                        "bid1": float(t.get("bidPrice", [0])[0] if isinstance(t.get("bidPrice"), list) else t.get("bid1", 0)),
                        "ask1": float(t.get("askPrice", [0])[0] if isinstance(t.get("askPrice"), list) else t.get("ask1", 0)),
                    }
                    self.buf.add_tick(tick)
                    broadcast_l2(tick)
        except Exception as e:
            logger.error(f"Tick callback error: {e}")

    def _depth_poll_loop(self):
        """Poll QMT for depth snapshots at fixed interval."""
        from xtquant import xtdata

        while self._running:
            for sym in self.symbols:
                try:
                    q = xtdata.get_full_tick([sym])
                    if sym in q:
                        d = q[sym]
                        depth = self._parse_qmt_depth(sym, d)
                        if depth:
                            self.buf.add_depth(depth)
                            broadcast_l2(depth)

                            # Compute and store features
                            feat = self.buf.compute_features(sym, depth)
                            self.buf.add_feature(feat)
                except Exception as e:
                    logger.error(f"Depth poll error {sym}: {e}")

            time.sleep(DEPTH_INTERVAL)

    def _parse_qmt_depth(self, symbol: str, d: dict) -> Optional[dict]:
        """Parse QMT full tick into DepthData format."""
        try:
            bid_prices = d.get("bidPrice", d.get("bid", []))
            bid_vols = d.get("bidVol", d.get("bidVolume", []))
            ask_prices = d.get("askPrice", d.get("ask", []))
            ask_vols = d.get("askVol", d.get("askVolume", []))

            if not bid_prices or not ask_prices:
                return None

            n_levels = min(10, len(bid_prices), len(ask_prices))
            bids = [{"price": float(bid_prices[i]), "volume": float(bid_vols[i]), "order_count": 0}
                    for i in range(n_levels) if bid_prices[i] > 0]
            asks = [{"price": float(ask_prices[i]), "volume": float(ask_vols[i]), "order_count": 0}
                    for i in range(n_levels) if ask_prices[i] > 0]

            return {
                "type": "depth",
                "symbol": symbol,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "bids": bids,
                "asks": asks,
                "last_price": float(d.get("lastPrice", 0)),
                "total_volume": float(d.get("volume", 0)),
                "total_turnover": float(d.get("amount", d.get("turnover", 0))),
            }
        except Exception as e:
            logger.error(f"Parse depth error: {e}")
            return None

    def stop(self):
        self._running = False


# ── Simulated L2 Source (for testing) ────────────────────────────────

class SimulatedL2Source:
    """Generates realistic simulated L2 data for testing."""

    def __init__(self, symbols: list[str], buf: L2Buffer, tick_interval: float = 0.1):
        self.symbols = symbols
        self.buf = buf
        self.tick_interval = tick_interval
        self._running = False
        self._seq = 0
        # Initial prices
        self._prices = {s: 100.0 + random.random() * 1900 for s in symbols}
        self._volumes = {s: 0.0 for s in symbols}

    def start(self):
        self._running = True
        threading.Thread(target=self._tick_loop, daemon=True).start()
        threading.Thread(target=self._depth_loop, daemon=True).start()
        logger.info(f"Simulated L2 source started for: {self.symbols}")

    def _tick_loop(self):
        """Generate simulated ticks."""
        while self._running:
            for sym in self.symbols:
                self._seq += 1
                price = self._prices[sym]

                # Random walk
                change = random.gauss(0, price * 0.0003)
                price = max(0.01, price + change)
                self._prices[sym] = price

                vol = random.randint(1, 50) * 100  # 1-50手
                direction = "B" if random.random() > 0.48 else "S"
                self._volumes[sym] += vol

                spread = price * 0.0005
                bid1 = price - spread / 2
                ask1 = price + spread / 2

                tick = {
                    "type": "tick",
                    "symbol": sym,
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "price": round(price, 2),
                    "volume": vol,
                    "direction": direction,
                    "seq": self._seq,
                    "bid1": round(bid1, 2),
                    "ask1": round(ask1, 2),
                }
                self.buf.add_tick(tick)
                broadcast_l2(tick)

            time.sleep(self.tick_interval)

    def _depth_loop(self):
        """Generate simulated depth snapshots."""
        while self._running:
            for sym in self.symbols:
                price = self._prices[sym]
                tick_size = 0.01

                bids = []
                asks = []
                for i in range(10):
                    bids.append({
                        "price": round(price - (i + 1) * tick_size, 2),
                        "volume": random.randint(10, 500) * 100,
                        "order_count": random.randint(1, 50),
                    })
                    asks.append({
                        "price": round(price + (i + 1) * tick_size, 2),
                        "volume": random.randint(10, 500) * 100,
                        "order_count": random.randint(1, 50),
                    })

                depth = {
                    "type": "depth",
                    "symbol": sym,
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "bids": bids,
                    "asks": asks,
                    "last_price": round(price, 2),
                    "total_volume": self._volumes[sym],
                    "total_turnover": self._volumes[sym] * price,
                }
                self.buf.add_depth(depth)
                broadcast_l2(depth)

                # Compute features
                feat = self.buf.compute_features(sym, depth)
                self.buf.add_feature(feat)

            time.sleep(DEPTH_INTERVAL)

    def stop(self):
        self._running = False


# ── Replay Mode ──────────────────────────────────────────────────────

class ReplayL2Source:
    """Replay recorded L2 data from Parquet files to TCP clients."""

    def __init__(self, data_dir: Path, replay_date: str, speed: float = 1.0):
        self.data_dir = data_dir
        self.replay_date = replay_date
        self.speed = speed  # 1.0 = real-time, 10.0 = 10x speed
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._replay_loop, daemon=True).start()

    def _replay_loop(self):
        """Replay tick and depth data interleaved by timestamp."""
        tick_dir = self.data_dir / "tick"
        depth_dir = self.data_dir / "depth"

        # Load all tick files for the date
        tick_files = sorted(tick_dir.glob(f"*_{self.replay_date}.parquet"))
        depth_files = sorted(depth_dir.glob(f"*_{self.replay_date}.parquet"))

        all_events = []

        for f in tick_files:
            try:
                df = pd.read_parquet(f)
                for _, row in df.iterrows():
                    event = row.to_dict()
                    event["type"] = "tick"
                    all_events.append(event)
            except Exception as e:
                logger.error(f"Error loading {f}: {e}")

        for f in depth_files:
            try:
                df = pd.read_parquet(f)
                for _, row in df.iterrows():
                    event = row.to_dict()
                    event["type"] = "depth"
                    # Reconstruct bids/asks from flat columns if needed
                    if "bids" not in event:
                        event["bids"] = json.loads(event.get("bids_json", "[]"))
                        event["asks"] = json.loads(event.get("asks_json", "[]"))
                    all_events.append(event)
            except Exception as e:
                logger.error(f"Error loading {f}: {e}")

        if not all_events:
            logger.warning(f"No L2 data found for date {self.replay_date}")
            return

        # Sort by datetime
        all_events.sort(key=lambda e: str(e.get("datetime", "")))
        logger.info(f"Replaying {len(all_events)} events for {self.replay_date} at {self.speed}x speed")

        # Wait for at least one TCP client
        logger.info("Waiting for TCP client to connect...")
        while self._running and not tcp_clients:
            time.sleep(0.5)

        prev_dt = None
        for event in all_events:
            if not self._running:
                break

            # Pace replay
            dt_str = str(event.get("datetime", ""))
            try:
                dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
                if prev_dt:
                    gap = (dt - prev_dt).total_seconds()
                    if gap > 0:
                        time.sleep(gap / self.speed)
                prev_dt = dt
            except (ValueError, TypeError):
                pass

            # Ensure bids/asks are lists (might be serialized as JSON strings)
            if event.get("type") == "depth":
                for key in ("bids", "asks"):
                    if isinstance(event.get(key), str):
                        try:
                            event[key] = json.loads(event[key])
                        except Exception:
                            event[key] = []

            broadcast_l2(event)

        logger.info("Replay complete.")

    def stop(self):
        self._running = False


# ── Feature Export (for ML training) ─────────────────────────────────

def export_training_features(data_dir: Path, date_range: list[str] = None) -> str:
    """Merge all L2 feature files into a single training CSV.

    Returns path to the exported file.
    """
    feat_dir = data_dir / "features"
    if date_range:
        files = []
        for d in date_range:
            files.extend(feat_dir.glob(f"*_{d}.parquet"))
    else:
        files = sorted(feat_dir.glob("*.parquet"))

    if not files:
        logger.warning("No feature files found")
        return ""

    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)

    out_path = data_dir / "l2_training_features.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Exported {len(merged)} feature rows → {out_path}")
    return str(out_path)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="L2 Market Data Recorder & Server")
    parser.add_argument("--mode", choices=["record", "simulate", "replay", "live", "export"],
                        default="simulate",
                        help="Operating mode")
    parser.add_argument("--symbols", type=str, default="600519.SH",
                        help="Comma-separated stock symbols")
    parser.add_argument("--port", type=int, default=TCP_PORT,
                        help="TCP MQ port for Rust engine")
    parser.add_argument("--date", type=str, default=None,
                        help="Date for replay mode (YYYYMMDD)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier")
    parser.add_argument("--tick-interval", type=float, default=0.1,
                        help="Simulated tick interval (seconds)")
    parser.add_argument("--depth-interval", type=float, default=3.0,
                        help="Depth snapshot interval (seconds)")
    parser.add_argument("--flush-interval", type=float, default=60.0,
                        help="Parquet flush interval (seconds)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="L2 data storage directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    global DEPTH_INTERVAL
    DEPTH_INTERVAL = args.depth_interval

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    buf = L2Buffer(data_dir)

    # Export mode — just merge features and exit
    if args.mode == "export":
        path = export_training_features(data_dir)
        if path:
            print(f"✅ Exported to: {path}")
        else:
            print("❌ No feature files found")
        return

    # Start TCP server
    threading.Thread(target=tcp_server_thread, args=(args.port,), daemon=True).start()

    # Start flush thread
    def flush_loop():
        while True:
            time.sleep(args.flush_interval)
            buf.flush()
            for sym in symbols:
                buf.reset_agg(sym)
            logger.info(f"Periodic flush complete")

    threading.Thread(target=flush_loop, daemon=True).start()

    # Start data source
    source = None
    if args.mode == "simulate":
        source = SimulatedL2Source(symbols, buf, tick_interval=args.tick_interval)
        source.start()
        print(f"🎮 Simulated L2 source running")
        print(f"   Symbols: {symbols}")
        print(f"   Tick interval: {args.tick_interval}s | Depth interval: {args.depth_interval}s")

    elif args.mode in ("record", "live"):
        source = QmtL2Source(symbols, buf)
        print(f"📡 QMT L2 recorder starting")
        print(f"   Symbols: {symbols}")
        if not source.start():
            print("❌ Failed to connect to QMT. Falling back to simulate mode.")
            source = SimulatedL2Source(symbols, buf, tick_interval=args.tick_interval)
            source.start()

    elif args.mode == "replay":
        replay_date = args.date or date.today().strftime("%Y%m%d")
        source = ReplayL2Source(data_dir, replay_date, speed=args.speed)
        source.start()
        print(f"⏪ Replay mode: date={replay_date}, speed={args.speed}x")

    print(f"🔗 L2 TCP MQ: port {args.port}")
    print(f"💾 Data dir: {data_dir}")
    print(f"   Flush interval: {args.flush_interval}s")
    print(f"   Press Ctrl+C to stop")

    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n🛑 Shutting down...")
        if source:
            source.stop()
        buf.flush()
        print("✅ Final flush complete. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
