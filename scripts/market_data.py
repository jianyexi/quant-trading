#!/usr/bin/env python3
"""Unified market data bridge for the quant trading system.

Commands:
    python market_data.py klines <symbol> <start> <end>
    python market_data.py quote <symbol>
    python market_data.py stock_info <symbol>
    python market_data.py stock_list
    python market_data.py stock_pool <csi300|csi500|custom>

All output is JSON to stdout.
"""

import json
import sys

import pandas as pd

# Pre-known stock names to avoid slow individual API calls
_STOCK_NAMES = {
    # 主板 (Main Board)
    "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安",
    "000001": "平安银行", "600036": "招商银行", "600276": "恒瑞医药",
    "000333": "美的集团", "601888": "中国中免", "002594": "比亚迪",
    "601012": "隆基绿能", "600900": "长江电力", "000568": "泸州老窖",
    "600809": "山西汾酒", "002475": "立讯精密", "600030": "中信证券",
    "601166": "兴业银行", "000661": "长春高新", "002714": "牧原股份",
    "600585": "海螺水泥", "603259": "药明康德", "601899": "紫金矿业",
    "600031": "三一重工", "000002": "万科A", "600309": "万华化学",
    "002304": "洋河股份", "601668": "中国建筑", "600887": "伊利股份",
    "000651": "格力电器", "601398": "工商银行", "601288": "农业银行",
    "600690": "海尔智家", "601669": "中国电建", "600048": "保利发展",
    "601800": "中国交建", "601225": "陕西煤业", "600438": "通威股份",
    "002460": "赣锋锂业", "002032": "苏泊尔", "002415": "海康威视",
    # 创业板 (ChiNext, 300xxx, ±20%)
    "300750": "宁德时代", "300760": "迈瑞医疗", "300059": "东方财富",
    "300122": "智飞生物", "300782": "卓胜微", "300015": "爱尔眼科",
    "300274": "阳光电源", "300498": "温氏股份",
    # 科创板 (STAR Market, 688xxx, ±20%)
    "688981": "中芯国际", "688111": "金山办公", "688036": "传音控股",
    "688561": "奇安信", "688005": "容百科技", "688012": "中微公司",
    "688185": "康希诺", "688599": "天合光能",
    # 传媒
    "002230": "科大讯飞", "603444": "吉比特",
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

    # Use local cache for daily data to avoid redundant API calls
    if period == "daily":
        from market_cache import get_cache
        cache = get_cache()
        start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        df = cache.get_or_fetch(symbol, start_fmt, end_fmt)
        if df is None or df.empty:
            return []
        records = []
        for date_val, row in df.iterrows():
            dt = str(date_val.date()) + " 15:00:00"
            records.append({
                "symbol": full_symbol,
                "datetime": dt,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": float(row["volume"]),
            })
        return records

    # Minute-level data — try tushare first, then akshare
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    df = None
    try:
        from tushare_provider import fetch_minute, is_available as ts_ok
        if ts_ok():
            df = fetch_minute(symbol, start_fmt, end_fmt, freq=period)
    except (ImportError, Exception):
        pass

    if df is None or df.empty:
        try:
            import akshare as ak
            start_dt = start_fmt + " 09:30:00"
            end_dt = end_fmt + " 15:00:00"
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period=period,
                start_date=start_dt, end_date=end_dt, adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "时间": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
        except (ImportError, Exception):
            pass

    if df is None or df.empty:
        return []

    records = []
    for date_val, row in df.iterrows():
        dt_str = str(date_val)
        records.append({
            "symbol": full_symbol,
            "datetime": dt_str,
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": float(row["volume"]),
        })
    return records


def cmd_quote(args):
    """Get latest quote. Tries tushare first, then akshare."""
    if not args:
        return {"error": "usage: quote <symbol>"}
    raw_symbol = args[0]
    symbol = normalize_symbol(raw_symbol)
    full_symbol = raw_symbol if "." in raw_symbol else exchange_suffix(symbol)

    # 1) Try tushare
    try:
        from tushare_provider import fetch_quote, is_available as ts_ok
        if ts_ok():
            q = fetch_quote(symbol)
            q["symbol"] = full_symbol
            name = _STOCK_NAMES.get(symbol)
            if name:
                q["name"] = name
            return q
    except (ImportError, Exception):
        pass

    # 2) Fallback to akshare
    try:
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
    except (ImportError, Exception) as e:
        return {"error": f"No quote data for {symbol}: {e}"}


def cmd_stock_info(args):
    """Get individual stock info. Tries tushare first, then akshare."""
    if not args:
        return {"error": "usage: stock_info <symbol>"}
    symbol = normalize_symbol(args[0])
    full_symbol = args[0] if "." in args[0] else exchange_suffix(symbol)

    # 1) Try tushare
    try:
        from tushare_provider import fetch_stock_info, is_available as ts_ok
        if ts_ok():
            info = fetch_stock_info(symbol)
            info["symbol"] = full_symbol
            return info
    except (ImportError, Exception):
        pass

    # 2) Fallback to akshare
    try:
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
    except (ImportError, Exception) as e:
        return {"symbol": full_symbol, "error": str(e)}


def cmd_stock_pool(args):
    """Get stock pool by index: csi300, csi500, or custom (curated list).

    Usage: stock_pool [csi300|csi500|custom]
    Returns cached result if available and fresh (< 1 day old).
    """
    import os, time
    pool = args[0] if args else "custom"

    # Check cache first
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"stock_pool_{pool}.json")
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 86400:  # 1 day
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

    if pool == "custom":
        result = _curated_pool()
    elif pool in ("csi300", "csi500"):
        result = _index_pool(pool)
    else:
        return {"error": f"Unknown pool: {pool}. Use csi300, csi500, or custom."}

    # Cache result
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result


def _index_pool(pool: str) -> dict:
    """Fetch index constituents. Tries tushare first, then akshare."""
    idx_map_ts = {"csi300": "000300.SH", "csi500": "000905.SH"}
    idx_map_ak = {"csi300": "000300", "csi500": "000905"}

    # 1) Try tushare
    try:
        from tushare_provider import fetch_index_members, is_available as ts_ok
        if ts_ok():
            ts_idx = idx_map_ts[pool]
            codes = fetch_index_members(ts_idx)
            if codes:
                stocks = []
                for code in codes:
                    name = _STOCK_NAMES.get(code, code)
                    stocks.append({
                        "symbol": exchange_suffix(code),
                        "name": name,
                        "industry": "",
                    })
                return {"pool": pool, "count": len(stocks), "stocks": stocks}
    except (ImportError, Exception):
        pass

    # 2) Fallback to akshare
    try:
        import akshare as ak
        idx_code = idx_map_ak[pool]
        df = ak.index_stock_cons_csindex_df(symbol=idx_code)

        if df is None or df.empty:
            return {"error": f"No data for {pool}", "stocks": []}

        stocks = []
        code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
        name_col = "成分券名称" if "成分券名称" in df.columns else df.columns[1]

        for _, row in df.iterrows():
            code = str(row[code_col]).zfill(6)
            name = str(row[name_col]) if name_col in df.columns else _STOCK_NAMES.get(code, code)
            industry = ""
            for col in df.columns:
                if "行业" in str(col):
                    industry = str(row[col]) if row[col] and str(row[col]) != "nan" else ""
                    break
            stocks.append({
                "symbol": exchange_suffix(code),
                "name": name,
                "industry": industry or _STOCK_NAMES.get(code, ""),
            })
        return {"pool": pool, "count": len(stocks), "stocks": stocks}
    except (ImportError, Exception) as e:
        return {"error": f"Failed to fetch {pool}: {e}", "stocks": []}


def _curated_pool() -> dict:
    """Return curated stock list with industry info."""
    CURATED = [
        ("600519", "贵州茅台", "白酒"), ("000858", "五粮液", "白酒"),
        ("601318", "中国平安", "保险"), ("000001", "平安银行", "银行"),
        ("600036", "招商银行", "银行"), ("600276", "恒瑞医药", "医药"),
        ("000333", "美的集团", "家电"), ("601888", "中国中免", "零售"),
        ("002594", "比亚迪", "汽车"), ("601012", "隆基绿能", "光伏"),
        ("600900", "长江电力", "电力"), ("000568", "泸州老窖", "白酒"),
        ("600809", "山西汾酒", "白酒"), ("002475", "立讯精密", "电子"),
        ("600030", "中信证券", "证券"), ("601166", "兴业银行", "银行"),
        ("000661", "长春高新", "医药"), ("002714", "牧原股份", "农牧"),
        ("600585", "海螺水泥", "建材"), ("603259", "药明康德", "医药"),
        ("601899", "紫金矿业", "矿业"), ("600031", "三一重工", "机械"),
        ("600309", "万华化学", "化工"), ("002304", "洋河股份", "白酒"),
        ("600887", "伊利股份", "乳业"), ("000651", "格力电器", "家电"),
        ("002415", "海康威视", "安防"),
        ("300750", "宁德时代", "电池"), ("300760", "迈瑞医疗", "医疗器械"),
        ("300059", "东方财富", "金融IT"), ("300122", "智飞生物", "疫苗"),
        ("300782", "卓胜微", "芯片"), ("300015", "爱尔眼科", "医疗"),
        ("300274", "阳光电源", "光伏"),
        ("688981", "中芯国际", "半导体"), ("688111", "金山办公", "软件"),
        ("688036", "传音控股", "手机"), ("688561", "奇安信", "网络安全"),
        ("688005", "容百科技", "锂电材料"), ("688012", "中微公司", "半导体设备"),
        ("002230", "科大讯飞", "AI"),
    ]
    stocks = []
    for code, name, industry in CURATED:
        stocks.append({
            "symbol": exchange_suffix(code),
            "name": name,
            "industry": industry,
        })
    return {"pool": "custom", "count": len(stocks), "stocks": stocks}


def cmd_stock_list(args):
    """Get list of A-share stocks (curated, with batch spot data for speed)."""
    import akshare as ak

    # Curated list with pre-known names/industries
    CURATED = [
        # 主板 (SSE/SZSE)
        ("600519", "贵州茅台", "白酒"), ("000858", "五粮液", "白酒"),
        ("601318", "中国平安", "保险"), ("000001", "平安银行", "银行"),
        ("600036", "招商银行", "银行"), ("600276", "恒瑞医药", "医药"),
        ("000333", "美的集团", "家电"), ("601888", "中国中免", "零售"),
        ("002594", "比亚迪", "汽车"), ("601012", "隆基绿能", "光伏"),
        ("600900", "长江电力", "电力"), ("000568", "泸州老窖", "白酒"),
        ("600809", "山西汾酒", "白酒"), ("002475", "立讯精密", "电子"),
        ("600030", "中信证券", "证券"), ("601166", "兴业银行", "银行"),
        ("000661", "长春高新", "医药"), ("002714", "牧原股份", "农牧"),
        ("600585", "海螺水泥", "建材"), ("603259", "药明康德", "医药"),
        ("601899", "紫金矿业", "矿业"), ("600031", "三一重工", "机械"),
        ("600309", "万华化学", "化工"), ("002304", "洋河股份", "白酒"),
        ("600887", "伊利股份", "乳业"), ("000651", "格力电器", "家电"),
        ("002415", "海康威视", "安防"),
        # 创业板 (ChiNext, 300xxx, ±20%)
        ("300750", "宁德时代", "电池"), ("300760", "迈瑞医疗", "医疗器械"),
        ("300059", "东方财富", "金融IT"), ("300122", "智飞生物", "疫苗"),
        ("300782", "卓胜微", "芯片"), ("300015", "爱尔眼科", "医疗"),
        ("300274", "阳光电源", "光伏"),
        # 科创板 (STAR Market, 688xxx, ±20%)
        ("688981", "中芯国际", "半导体"), ("688111", "金山办公", "软件"),
        ("688036", "传音控股", "手机"), ("688561", "奇安信", "网络安全"),
        ("688005", "容百科技", "锂电材料"), ("688012", "中微公司", "半导体设备"),
        # 其他
        ("002230", "科大讯飞", "AI"),
    ]

    results = []
    for code, name, industry in CURATED:
        if code.startswith("688"):
            market = "科创板"
        elif code.startswith("300"):
            market = "创业板"
        elif code.startswith("6"):
            market = "SSE"
        else:
            market = "SZSE"
        results.append({
            "symbol": exchange_suffix(code),
            "name": name,
            "industry": industry,
            "market": market,
        })
    return {"stocks": results}


def cmd_cache_status(args):
    """Return cache coverage info for all cached symbols."""
    from market_cache import get_cache
    cache = get_cache()
    status = cache.cache_status()
    total_bars = sum(s["bar_count"] for s in status)
    return {
        "symbols": status,
        "total_symbols": len(status),
        "total_bars": total_bars,
    }


def cmd_sync_cache(args):
    """Pre-fetch and cache data for given symbols.

    Usage: sync_cache <symbols_csv> <start> <end>
    """
    if len(args) < 3:
        return {"error": "usage: sync_cache <symbols_csv> <start> <end>"}
    symbols = [s.strip() for s in args[0].split(",") if s.strip()]
    start_date = args[1]
    end_date = args[2]
    from market_cache import get_cache
    cache = get_cache()
    df = cache.get_or_fetch_multi(symbols, start_date, end_date)
    status = cache.cache_status()
    return {
        "status": "ok",
        "synced_symbols": len(symbols),
        "total_bars": len(df) if df is not None else 0,
        "cache": status,
    }


def cmd_data_source_status(args):
    """Return status of available data sources."""
    result = {"primary": None, "available": [], "akshare": False, "tushare": False}
    try:
        from tushare_provider import is_available as ts_ok
        if ts_ok():
            result["tushare"] = True
            result["available"].append("tushare")
    except ImportError:
        pass
    try:
        import akshare
        result["akshare"] = True
        result["available"].append("akshare")
    except ImportError:
        pass
    result["primary"] = result["available"][0] if result["available"] else None
    return result


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
        elif cmd == "stock_pool":
            result = cmd_stock_pool(args)
        elif cmd == "cache_status":
            result = cmd_cache_status(args)
        elif cmd == "sync_cache":
            result = cmd_sync_cache(args)
        elif cmd == "data_source_status":
            result = cmd_data_source_status(args)
        else:
            result = {"error": f"Unknown command: {cmd}"}
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
