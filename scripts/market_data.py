#!/usr/bin/env python3
"""Unified market data bridge for the quant trading system.

Commands:
    python market_data.py klines <symbol> <start> <end>
    python market_data.py quote <symbol>
    python market_data.py stock_info <symbol>
    python market_data.py stock_list

All output is JSON to stdout.
"""

import json
import sys


def normalize_symbol(sym: str) -> str:
    """Strip exchange suffix (.SH / .SZ) for akshare."""
    return sym.split(".")[0]


def normalize_date(d: str) -> str:
    return d.replace("-", "")


def exchange_suffix(code: str) -> str:
    """Infer .SH/.SZ from numeric code."""
    if code.startswith(("6", "5", "9")):
        return code + ".SH"
    return code + ".SZ"


def cmd_klines(args):
    if len(args) < 3:
        return {"error": "usage: klines <symbol> <start> <end> [period]"}
    raw_symbol, start, end = args[0], normalize_date(args[1]), normalize_date(args[2])
    period = args[3] if len(args) > 3 else "daily"
    symbol = normalize_symbol(raw_symbol)
    full_symbol = raw_symbol if "." in raw_symbol else exchange_suffix(raw_symbol)

    import akshare as ak

    # Minute-level data uses a different API
    if period in ("1", "5", "15", "30", "60"):
        # Convert dates to datetime format for minute API
        start_dt = start[:4] + "-" + start[4:6] + "-" + start[6:8] + " 09:30:00"
        end_dt = end[:4] + "-" + end[4:6] + "-" + end[6:8] + " 15:00:00"
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol, period=period,
            start_date=start_dt, end_date=end_dt, adjust="qfq",
        )
        if df is None or df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            dt_str = str(row["时间"])
            records.append({
                "symbol": full_symbol,
                "datetime": dt_str,
                "open": round(float(row["开盘"]), 2),
                "high": round(float(row["最高"]), 2),
                "low": round(float(row["最低"]), 2),
                "close": round(float(row["收盘"]), 2),
                "volume": float(row["成交量"]),
            })
        return records

    # Daily data
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="qfq",
    )
    if df is None or df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        date_str = str(row["日期"])
        dt = date_str + " 15:00:00" if len(date_str) == 10 else date_str
        records.append({
            "symbol": full_symbol,
            "datetime": dt,
            "open": round(float(row["开盘"]), 2),
            "high": round(float(row["最高"]), 2),
            "low": round(float(row["最低"]), 2),
            "close": round(float(row["收盘"]), 2),
            "volume": float(row["成交量"]),
        })
    return records


def cmd_quote(args):
    """Get latest quote by fetching last 5 trading days of klines."""
    if not args:
        return {"error": "usage: quote <symbol>"}
    raw_symbol = args[0]
    symbol = normalize_symbol(raw_symbol)
    full_symbol = raw_symbol if "." in raw_symbol else exchange_suffix(symbol)

    import akshare as ak
    from datetime import datetime, timedelta
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="qfq",
    )
    if df is None or df.empty:
        return {"error": f"No quote data for {symbol}"}
    r = df.iloc[-1]

    # Get stock name via stock_individual_info_em
    name = symbol
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        name_row = info[info["item"] == "股票简称"]
        if not name_row.empty:
            name = str(name_row.iloc[0]["value"])
    except Exception:
        pass

    return {
        "symbol": full_symbol,
        "name": name,
        "price": float(r["收盘"]),
        "open": float(r["开盘"]),
        "high": float(r["最高"]),
        "low": float(r["最低"]),
        "volume": float(r["成交量"]),
        "turnover": float(r["成交额"]),
        "change": float(r["涨跌额"]),
        "change_percent": float(r["涨跌幅"]),
        "date": str(r["日期"]),
    }


def cmd_stock_info(args):
    """Get individual stock info."""
    if not args:
        return {"error": "usage: stock_info <symbol>"}
    symbol = normalize_symbol(args[0])
    full_symbol = args[0] if "." in args[0] else exchange_suffix(symbol)

    import akshare as ak
    info = ak.stock_individual_info_em(symbol=symbol)
    result = {"symbol": full_symbol}
    for _, row in info.iterrows():
        item, value = str(row["item"]), row["value"]
        if item == "股票简称":
            result["name"] = str(value)
        elif item == "行业":
            result["industry"] = str(value)
        elif item == "上市时间":
            result["list_date"] = str(value)
        elif item == "总市值":
            result["market_cap"] = float(value)
        elif item == "流通市值":
            result["float_market_cap"] = float(value)
    return result


def cmd_stock_list(args):
    """Get list of A-share stocks (top by market cap, fast)."""
    import akshare as ak
    # Use a small, focused list of major A-share stocks
    # stock_zh_a_spot_em is too slow, so we use a curated approach
    codes = [
        "600519", "000858", "601318", "000001", "600036",
        "300750", "600276", "000333", "601888", "002594",
        "601012", "600900", "000568", "600809", "002475",
        "600030", "601166", "000661", "002714", "600585",
        "603259", "601899", "600031", "000002", "600309",
        "002304", "601668", "300059", "002230", "600887",
    ]
    results = []
    for code in codes:
        try:
            info = ak.stock_individual_info_em(symbol=code)
            rec = {"symbol": exchange_suffix(code)}
            for _, row in info.iterrows():
                item = str(row["item"])
                if item == "股票简称":
                    rec["name"] = str(row["value"])
                elif item == "行业":
                    rec["industry"] = str(row["value"])
            if "name" in rec:
                market = "SSE" if code.startswith("6") else "SZSE"
                if code.startswith("300"):
                    market = "ChiNext"
                rec["market"] = market
                results.append(rec)
        except Exception:
            continue
    return {"stocks": results}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: market_data.py <command> [args...]"}))
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    try:
        if cmd == "klines":
            result = cmd_klines(args)
        elif cmd == "quote":
            result = cmd_quote(args)
        elif cmd == "stock_info":
            result = cmd_stock_info(args)
        elif cmd == "stock_list":
            result = cmd_stock_list(args)
        else:
            result = {"error": f"Unknown command: {cmd}"}
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
