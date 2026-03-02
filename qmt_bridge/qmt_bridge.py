#!/usr/bin/env python3
"""
QMT Bridge — HTTP sidecar that wraps xtquant SDK for Rust integration.

Exposes a local REST API on port 18090 to allow the Rust quant-trading
system to place orders, query positions, and manage accounts via QMT
(迅投量化, miniQMT mode).

Usage:
    1. Start QMT client in miniQMT mode
    2. pip install flask xtquant
    3. python qmt_bridge.py --qmt-path "C:/QMT/userdata_mini" --account "YOUR_ACCOUNT"
"""

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, request

# ── Globals ──────────────────────────────────────────────────────────

app = Flask(__name__)
logger = logging.getLogger("qmt_bridge")

# Will be set after successful connection
_trader = None
_account_id: Optional[str] = None
_connected = False
_session_id = "quant_bridge_001"
_qmt_path: Optional[str] = None

# Track order callbacks
_order_results: dict = {}
_order_lock = threading.Lock()


# ── XtQuant Callback ────────────────────────────────────────────────

class BridgeCallback:
    """Callback handler for xtquant async events."""

    def on_disconnected(self):
        global _connected
        _connected = False
        logger.warning("QMT disconnected")

    def on_stock_order(self, response):
        """Called when an order status changes."""
        with _order_lock:
            _order_results[str(response.order_id)] = {
                "order_id": response.order_id,
                "stock_code": response.stock_code,
                "order_status": response.order_status,
                "order_sysid": getattr(response, "order_sysid", ""),
                "error_msg": getattr(response, "error_msg", ""),
                "traded_volume": getattr(response, "traded_volume", 0),
                "traded_price": getattr(response, "traded_price", 0.0),
            }
        logger.info(f"Order callback: id={response.order_id} status={response.order_status}")

    def on_order_error(self, order_error):
        logger.error(f"Order error: {order_error}")

    def on_cancel_error(self, cancel_error):
        logger.error(f"Cancel error: {cancel_error}")

    def on_order_stock_async_response(self, response):
        logger.info(f"Async order response: seq={response.seq}")

    def on_account_status(self, status):
        logger.info(f"Account status: {status}")


# ── Connection ──────────────────────────────────────────────────────

def connect_qmt(qmt_path: str, session_id: str, account_id: str) -> bool:
    """Connect to QMT client and subscribe to the account."""
    global _trader, _account_id, _connected, _qmt_path

    try:
        from xtquant import xttrader, xtconstant  # noqa: F401
    except ImportError:
        logger.error(
            "xtquant not installed. Install via: pip install xtquant "
            "or copy from QMT client directory."
        )
        return False

    _qmt_path = qmt_path
    _account_id = account_id

    try:
        _trader = xttrader.XtQuantTrader(qmt_path, session_id)
        callback = BridgeCallback()
        _trader.register_callback(callback)
        _trader.start()

        ret = _trader.connect()
        if ret != 0:
            logger.error(f"QMT connect failed with code {ret}")
            return False

        # Subscribe to account
        sub_ret = _trader.subscribe(_account_id)
        if sub_ret != 0:
            logger.warning(f"Account subscribe returned {sub_ret}, may still work")

        _connected = True
        logger.info(f"Connected to QMT: path={qmt_path} account={account_id}")
        return True

    except Exception as e:
        logger.error(f"QMT connection error: {e}")
        return False


# ── API Endpoints ───────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "connected": _connected,
        "account": _account_id,
        "qmt_path": _qmt_path,
    })


@app.route("/connect", methods=["POST"])
def api_connect():
    """(Re)connect to QMT client."""
    data = request.get_json(silent=True) or {}
    qmt_path = data.get("qmt_path", _qmt_path)
    account = data.get("account", _account_id)
    session = data.get("session_id", _session_id)

    if not qmt_path or not account:
        return jsonify({"error": "qmt_path and account are required"}), 400

    ok = connect_qmt(qmt_path, session, account)
    return jsonify({"connected": ok})


@app.route("/order", methods=["POST"])
def api_order():
    """
    Place a stock order via QMT.

    Body JSON:
        stock_code: str      — e.g. "000001.SZ"
        price: float         — order price (0 for market order)
        amount: int          — number of shares (must be multiple of 100)
        side: str            — "buy" or "sell"
        price_type: str      — "limit" or "market" (default: "limit")
    """
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    stock_code = data.get("stock_code")
    price = float(data.get("price", 0))
    amount = int(data.get("amount", 0))
    side = data.get("side", "buy").lower()
    price_type_str = data.get("price_type", "limit").lower()

    if not stock_code or amount <= 0:
        return jsonify({"error": "stock_code and amount > 0 required"}), 400

    from xtquant import xtconstant

    # Map side
    order_type = (
        xtconstant.STOCK_BUY if side == "buy" else xtconstant.STOCK_SELL
    )

    # Map price type
    if price_type_str == "market":
        xt_price_type = xtconstant.LATEST_PRICE
    else:
        xt_price_type = xtconstant.FIX_PRICE

    try:
        order_id = _trader.order_stock(
            _account_id,
            stock_code,
            order_type,
            amount,
            xt_price_type,
            price,
        )
        logger.info(
            f"Order placed: {side.upper()} {stock_code} x{amount} @ {price} → id={order_id}"
        )
        return jsonify({
            "order_id": order_id,
            "stock_code": stock_code,
            "side": side,
            "amount": amount,
            "price": price,
            "price_type": price_type_str,
            "status": "submitted",
        })
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/cancel", methods=["POST"])
def api_cancel():
    """Cancel an existing order. Body: { "order_id": int }"""
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    data = request.get_json()
    order_id = data.get("order_id")
    if order_id is None:
        return jsonify({"error": "order_id required"}), 400

    try:
        ret = _trader.cancel_order_stock(_account_id, int(order_id))
        return jsonify({"result": ret, "order_id": order_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/positions", methods=["GET"])
def api_positions():
    """Query current positions."""
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    try:
        positions = _trader.query_stock_positions(_account_id)
        result = []
        for pos in positions:
            result.append({
                "stock_code": pos.stock_code,
                "volume": pos.volume,
                "can_use_volume": pos.can_use_volume,
                "frozen_volume": pos.frozen_volume,
                "open_price": pos.open_price,
                "market_value": pos.market_value,
                "cost_price": getattr(pos, "cost_price", pos.open_price),
            })
        return jsonify({"positions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/account", methods=["GET"])
def api_account():
    """Query account asset information."""
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    try:
        asset = _trader.query_stock_asset(_account_id)
        return jsonify({
            "account_id": _account_id,
            "total_asset": asset.total_asset,
            "cash": asset.cash,
            "market_value": asset.market_value,
            "frozen_cash": getattr(asset, "frozen_cash", 0.0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/orders", methods=["GET"])
def api_orders():
    """Query today's orders."""
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    try:
        orders = _trader.query_stock_orders(_account_id)
        result = []
        for o in orders:
            result.append({
                "order_id": o.order_id,
                "stock_code": o.stock_code,
                "order_type": o.order_type,
                "order_volume": o.order_volume,
                "price": o.price,
                "traded_volume": o.traded_volume,
                "traded_price": o.traded_price,
                "order_status": o.order_status,
                "order_time": getattr(o, "order_time", ""),
            })
        return jsonify({"orders": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/order_result/<order_id>", methods=["GET"])
def api_order_result(order_id: str):
    """Get cached callback result for a specific order."""
    with _order_lock:
        result = _order_results.get(order_id)
    if result:
        return jsonify(result)
    return jsonify({"error": "No callback received yet for this order"}), 404


@app.route("/order_status/<order_id>", methods=["GET"])
def api_order_status(order_id: str):
    """
    Get order status from QMT. First checks callback cache, then queries QMT directly.
    Returns: { order_id, order_status, traded_volume, traded_price }
    """
    # Check callback cache first
    with _order_lock:
        cached = _order_results.get(order_id)
    if cached:
        return jsonify(cached)

    # Fall back to querying QMT directly for today's orders
    if not _connected or _trader is None:
        return jsonify({"error": "Not connected to QMT"}), 503

    try:
        orders = _trader.query_stock_orders(_account_id)
        for o in orders:
            if str(o.order_id) == order_id:
                return jsonify({
                    "order_id": o.order_id,
                    "stock_code": o.stock_code,
                    "order_status": o.order_status,
                    "traded_volume": o.traded_volume,
                    "traded_price": o.traded_price,
                })
        return jsonify({"error": "Order not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Market Data Endpoints ───────────────────────────────────────────

def _get_xtdata():
    """Lazily import xtdata. Returns the module or None."""
    try:
        from xtquant import xtdata
        return xtdata
    except ImportError:
        return None


@app.route("/market/kline", methods=["GET"])
def api_market_kline():
    """
    Fetch historical kline (OHLCV) data via xtdata.

    Query params:
        stock_code: str   — e.g. "000001.SZ"
        period: str       — "1d", "1m", "5m", "15m", "30m", "60m" (default: "1d")
        start_time: str   — e.g. "20240101" or "20240101093000" (default: "")
        end_time: str     — e.g. "20241231" (default: "")
        count: int        — max bars to return (default: -1 = all)
    """
    xtdata = _get_xtdata()
    if xtdata is None:
        return jsonify({"error": "xtdata not available (xtquant not installed)"}), 503

    stock_code = request.args.get("stock_code", "")
    period = request.args.get("period", "1d")
    start_time = request.args.get("start_time", "")
    end_time = request.args.get("end_time", "")
    count = int(request.args.get("count", -1))

    if not stock_code:
        return jsonify({"error": "stock_code is required"}), 400

    # Map user-friendly period names to xtdata period strings
    period_map = {
        "1d": "1d", "daily": "1d", "day": "1d",
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m",
        "1w": "1w", "week": "1w",
    }
    xt_period = period_map.get(period, period)

    try:
        # Download data first (ensures local cache is up to date)
        xtdata.download_history_data(stock_code, xt_period, start_time, end_time)

        data = xtdata.get_market_data(
            field_list=["open", "high", "low", "close", "volume"],
            stock_list=[stock_code],
            period=xt_period,
            start_time=start_time,
            end_time=end_time,
            count=count,
        )

        if data is None or not data:
            return jsonify({"stock_code": stock_code, "klines": [], "count": 0})

        # data is dict: { field: DataFrame(index=time, columns=stock_list) }
        klines = []
        opens = data.get("open", {}).get(stock_code, {})
        highs = data.get("high", {}).get(stock_code, {})
        lows = data.get("low", {}).get(stock_code, {})
        closes = data.get("close", {}).get(stock_code, {})
        volumes = data.get("volume", {}).get(stock_code, {})

        if hasattr(opens, "items"):
            for ts in opens.keys():
                dt_str = str(ts)
                klines.append({
                    "datetime": dt_str,
                    "open": float(opens.get(ts, 0)),
                    "high": float(highs.get(ts, 0)),
                    "low": float(lows.get(ts, 0)),
                    "close": float(closes.get(ts, 0)),
                    "volume": float(volumes.get(ts, 0)),
                })

        return jsonify({
            "stock_code": stock_code,
            "period": xt_period,
            "klines": klines,
            "count": len(klines),
        })
    except Exception as e:
        logger.error(f"Market kline error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/market/quote", methods=["GET"])
def api_market_quote():
    """
    Fetch latest snapshot quote for one or more stocks.

    Query params:
        stock_codes: str  — comma-separated, e.g. "000001.SZ,600519.SH"
    """
    xtdata = _get_xtdata()
    if xtdata is None:
        return jsonify({"error": "xtdata not available (xtquant not installed)"}), 503

    codes_str = request.args.get("stock_codes", "")
    if not codes_str:
        return jsonify({"error": "stock_codes is required"}), 400

    stock_list = [c.strip() for c in codes_str.split(",") if c.strip()]

    try:
        data = xtdata.get_full_tick(stock_list)

        quotes = []
        for code in stock_list:
            tick = data.get(code)
            if tick is None:
                continue
            quotes.append({
                "stock_code": code,
                "last_price": float(getattr(tick, "lastPrice", 0)),
                "open": float(getattr(tick, "open", 0)),
                "high": float(getattr(tick, "high", 0)),
                "low": float(getattr(tick, "low", 0)),
                "pre_close": float(getattr(tick, "lastClose", 0)),
                "volume": float(getattr(tick, "volume", 0)),
                "amount": float(getattr(tick, "amount", 0)),
                "bid_prices": [float(p) for p in getattr(tick, "bidPrice", [])[:5]],
                "ask_prices": [float(p) for p in getattr(tick, "askPrice", [])[:5]],
                "bid_vols": [int(v) for v in getattr(tick, "bidVol", [])[:5]],
                "ask_vols": [int(v) for v in getattr(tick, "askVol", [])[:5]],
                "timestamp": int(getattr(tick, "time", 0)),
            })
        return jsonify({"quotes": quotes, "count": len(quotes)})
    except Exception as e:
        logger.error(f"Market quote error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/market/subscribe", methods=["POST"])
def api_market_subscribe():
    """
    Subscribe to real-time quote updates (writes to in-memory cache).

    Body JSON:
        stock_codes: list[str]  — e.g. ["000001.SZ", "600519.SH"]
        period: str             — "tick", "1m", "5m" (default: "tick")
    """
    xtdata = _get_xtdata()
    if xtdata is None:
        return jsonify({"error": "xtdata not available (xtquant not installed)"}), 503

    data = request.get_json(silent=True) or {}
    stock_codes = data.get("stock_codes", [])
    period = data.get("period", "tick")

    if not stock_codes:
        return jsonify({"error": "stock_codes list is required"}), 400

    period_map = {"tick": "tick", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m"}
    xt_period = period_map.get(period, "tick")

    subscribed = []
    errors = []
    for code in stock_codes:
        try:
            seq = xtdata.subscribe_quote(code, period=xt_period, count=-1)
            subscribed.append({"stock_code": code, "seq": seq})
        except Exception as e:
            errors.append({"stock_code": code, "error": str(e)})

    return jsonify({
        "subscribed": subscribed,
        "errors": errors,
        "count": len(subscribed),
    })


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QMT Bridge HTTP Sidecar")
    parser.add_argument(
        "--qmt-path",
        required=True,
        help="Path to QMT miniQMT userdata directory",
    )
    parser.add_argument(
        "--account", required=True, help="QMT trading account ID"
    )
    parser.add_argument(
        "--session-id", default="quant_bridge_001", help="Session identifier"
    )
    parser.add_argument(
        "--port", type=int, default=18090, help="HTTP port (default: 18090)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="Start server without connecting to QMT (for testing)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not args.no_connect:
        ok = connect_qmt(args.qmt_path, args.session_id, args.account)
        if not ok:
            logger.error("Failed to connect to QMT. Start with --no-connect to skip.")
            sys.exit(1)
    else:
        global _qmt_path, _account_id
        _qmt_path = args.qmt_path
        _account_id = args.account
        logger.info("Starting in offline mode (--no-connect)")

    logger.info(f"QMT Bridge starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
