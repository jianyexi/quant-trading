#!/usr/bin/env python3
"""Fetch historical A-share klines via akshare and output JSON to stdout.

Usage:
    python fetch_klines.py <symbol> <start_date> <end_date>

    symbol:     Stock code, e.g. "600519" or "600519.SH"
    start_date: "YYYY-MM-DD" or "YYYYMMDD"
    end_date:   "YYYY-MM-DD" or "YYYYMMDD"

Output: JSON array of { symbol, datetime, open, high, low, close, volume }
"""

import json
import sys


def normalize_symbol(sym: str) -> str:
    """Strip exchange suffix (.SH / .SZ) — akshare uses pure numeric codes."""
    return sym.split(".")[0]


def normalize_date(d: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD, return YYYYMMDD for akshare."""
    return d.replace("-", "")


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "usage: fetch_klines.py <symbol> <start> <end>"}))
        sys.exit(1)

    raw_symbol = sys.argv[1]
    start = normalize_date(sys.argv[2])
    end = normalize_date(sys.argv[3])
    symbol = normalize_symbol(raw_symbol)

    try:
        import akshare as ak

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",  # 前复权 (forward-adjusted)
        )

        if df is None or df.empty:
            print(json.dumps([]))
            return

        records = []
        for _, row in df.iterrows():
            date_str = str(row["日期"])
            dt = date_str + " 15:00:00" if len(date_str) == 10 else date_str

            records.append(
                {
                    "symbol": raw_symbol,
                    "datetime": dt,
                    "open": round(float(row["开盘"]), 2),
                    "high": round(float(row["最高"]), 2),
                    "low": round(float(row["最低"]), 2),
                    "close": round(float(row["收盘"]), 2),
                    "volume": float(row["成交量"]),
                }
            )

        print(json.dumps(records))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
