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

# Pre-known stock names to avoid slow individual API calls
_STOCK_NAMES = {
    "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安",
    "000001": "平安银行", "600036": "招商银行", "300750": "宁德时代",
    "600276": "恒瑞医药", "000333": "美的集团", "601888": "中国中免",
    "002594": "比亚迪", "601012": "隆基绿能", "600900": "长江电力",
    "000568": "泸州老窖", "600809": "山西汾酒", "002475": "立讯精密",
    "600030": "中信证券", "601166": "兴业银行", "000661": "长春高新",
    "002714": "牧原股份", "600585": "海螺水泥", "603259": "药明康德",
    "601899": "紫金矿业", "600031": "三一重工", "000002": "万科A",
    "600309": "万华化学", "002304": "洋河股份", "601668": "中国建筑",
    "300059": "东方财富", "002230": "科大讯飞", "600887": "伊利股份",
}


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

    # Use pre-known name map for common stocks (avoid slow API call)
    name = _STOCK_NAMES.get(symbol, symbol)

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
    """Get list of A-share stocks (curated, with batch spot data for speed)."""
    import akshare as ak

    # Curated list with pre-known names/industries (avoid 30 individual API calls)
    CURATED = [
        ("600519", "贵州茅台", "白酒"), ("000858", "五粮液", "白酒"),
        ("601318", "中国平安", "保险"), ("000001", "平安银行", "银行"),
        ("600036", "招商银行", "银行"), ("300750", "宁德时代", "电池"),
        ("600276", "恒瑞医药", "医药"), ("000333", "美的集团", "家电"),
        ("601888", "中国中免", "零售"), ("002594", "比亚迪", "汽车"),
        ("601012", "隆基绿能", "光伏"), ("600900", "长江电力", "电力"),
        ("000568", "泸州老窖", "白酒"), ("600809", "山西汾酒", "白酒"),
        ("002475", "立讯精密", "电子"), ("600030", "中信证券", "证券"),
        ("601166", "兴业银行", "银行"), ("000661", "长春高新", "医药"),
        ("002714", "牧原股份", "农牧"), ("600585", "海螺水泥", "建材"),
        ("603259", "药明康德", "医药"), ("601899", "紫金矿业", "矿业"),
        ("600031", "三一重工", "机械"), ("000002", "万科A", "房产"),
        ("600309", "万华化学", "化工"), ("002304", "洋河股份", "白酒"),
        ("601668", "中国建筑", "建筑"), ("300059", "东方财富", "金融IT"),
        ("002230", "科大讯飞", "AI"), ("600887", "伊利股份", "乳业"),
    ]

    results = []
    for code, name, industry in CURATED:
        market = "SSE" if code.startswith("6") else "SZSE"
        if code.startswith("300"):
            market = "ChiNext"
        results.append({
            "symbol": exchange_suffix(code),
            "name": name,
            "industry": industry,
            "market": market,
        })
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
