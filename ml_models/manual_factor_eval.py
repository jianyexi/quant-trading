#!/usr/bin/env python3
"""Manual factor evaluation: safe expression parsing + single-factor test.

Usage:
    python manual_factor_eval.py \
        --expression "pct_change(close, 20) / rolling_std(pct_change(close, 1), 20)" \
        --symbols 300750,601318 \
        --start-date 2022-01-01 --end-date 2024-12-31 \
        --horizon 5
"""
import argparse
import ast
import json
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# ── Safe Expression Engine ────────────────────────────────────────────

ALLOWED_NAMES = frozenset({
    "close", "open", "high", "low", "volume",
})

ALLOWED_FUNCTIONS = frozenset({
    "pct_change", "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
    "rolling_sum", "rolling_rank", "rolling_skew",
    "shift", "delay", "delta",
    "abs", "log", "sqrt", "sign", "square", "clip",
    "max", "min", "rank",
    "atr", "ema",
})

FORBIDDEN_STRINGS = {"import", "exec", "eval", "open(", "os.", "sys.", "__", "lambda"}


class ExpressionValidator(ast.NodeVisitor):
    """Walk AST and reject anything outside whitelist."""

    def generic_visit(self, node):
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
            ast.Name, ast.Constant, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Unknown name: '{node.id}'. Allowed: {sorted(ALLOWED_NAMES)}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Forbidden function: '{node.func.id}'")
        elif isinstance(node.func, ast.Attribute):
            raise ValueError("Attribute calls (e.g. x.method()) not allowed; use function syntax")
        else:
            raise ValueError("Only simple function calls allowed")
        for arg in node.args:
            self.visit(arg)


def validate_expression(expr: str) -> ast.Expression:
    """Parse and validate expression, return AST."""
    for forbidden in FORBIDDEN_STRINGS:
        if forbidden in expr:
            raise ValueError(f"Forbidden pattern: '{forbidden}'")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    ExpressionValidator().visit(tree)
    return tree


def _safe_eval_node(node, ns: dict) -> pd.Series:
    """Recursively evaluate AST node using pandas operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, ns)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand, ns)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left, ns)
        right = _safe_eval_node(node.right, ns)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if isinstance(right, (int, float)):
                return left / right if right != 0 else left * 0
            return left / right.replace(0, np.nan)
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right

    if isinstance(node, ast.Call):
        func_name = node.func.id
        args = [_safe_eval_node(a, ns) for a in node.args]
        return _apply_function(func_name, args)

    raise ValueError(f"Cannot evaluate: {ast.dump(node)}")


def _apply_function(name: str, args: list):
    """Map function name to pandas/numpy operation."""
    if name == "pct_change":
        series, period = args[0], int(args[1]) if len(args) > 1 else 1
        return series.pct_change(period)
    if name == "rolling_mean":
        return args[0].rolling(int(args[1]), min_periods=1).mean()
    if name == "rolling_std":
        return args[0].rolling(int(args[1]), min_periods=2).std()
    if name == "rolling_max":
        return args[0].rolling(int(args[1]), min_periods=1).max()
    if name == "rolling_min":
        return args[0].rolling(int(args[1]), min_periods=1).min()
    if name == "rolling_sum":
        return args[0].rolling(int(args[1]), min_periods=1).sum()
    if name == "rolling_rank":
        return args[0].rolling(int(args[1]), min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    if name == "rolling_skew":
        return args[0].rolling(int(args[1]), min_periods=3).skew()
    if name == "shift" or name == "delay":
        return args[0].shift(int(args[1]) if len(args) > 1 else 1)
    if name == "delta":
        period = int(args[1]) if len(args) > 1 else 1
        return args[0] - args[0].shift(period)
    if name == "abs":
        return args[0].abs() if isinstance(args[0], pd.Series) else abs(args[0])
    if name == "log":
        return np.log(args[0].clip(lower=1e-10)) if isinstance(args[0], pd.Series) else np.log(max(args[0], 1e-10))
    if name == "sqrt":
        return np.sqrt(args[0].clip(lower=0)) if isinstance(args[0], pd.Series) else np.sqrt(max(args[0], 0))
    if name == "sign":
        return np.sign(args[0])
    if name == "square":
        return args[0] ** 2
    if name == "clip":
        return args[0].clip(lower=-float(args[1]), upper=float(args[1])) if len(args) > 1 else args[0].clip(lower=-3, upper=3)
    if name == "rank":
        return args[0].rank(pct=True)
    if name == "max":
        return pd.concat([args[0], args[1]], axis=1).max(axis=1)
    if name == "min":
        return pd.concat([args[0], args[1]], axis=1).min(axis=1)
    if name == "ema":
        return args[0].ewm(span=int(args[1]), adjust=False).mean()
    if name == "atr":
        # Expects (high, low, close, period) or just (period) if h/l/c in namespace
        raise ValueError("atr() requires explicit args: use rolling_mean(max(high - low, abs(high - shift(close, 1)), abs(low - shift(close, 1))), N)")
    raise ValueError(f"Unknown function: {name}")


def compute_factor(expr_str: str, ohlcv: pd.DataFrame) -> pd.Series:
    """Compute factor values from expression string on OHLCV data."""
    tree = validate_expression(expr_str)
    ns = {
        "close": ohlcv["close"].astype(float),
        "open": ohlcv["open"].astype(float),
        "high": ohlcv["high"].astype(float),
        "low": ohlcv["low"].astype(float),
        "volume": ohlcv["volume"].astype(float),
    }
    result = _safe_eval_node(tree, ns)
    if isinstance(result, (int, float)):
        return pd.Series(result, index=ohlcv.index)
    return result


# ── Factor Evaluation ─────────────────────────────────────────────────

def evaluate_single_factor(
    factor: pd.Series, fwd_ret: pd.Series, window: int = 20
) -> dict:
    """Evaluate factor with Rank IC, returns metrics + IC time series."""
    aligned = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret}).dropna()
    if len(aligned) < 60:
        return {
            "metrics": {"ic_mean": 0, "ic_std": 1, "ir": 0, "ic_pos_rate": 0,
                        "turnover": 0, "decay": 0, "n_obs": 0},
            "ic_series": [],
            "warning": f"Insufficient data: {len(aligned)} rows (need ≥60)",
        }

    f = aligned["factor"]
    r = aligned["fwd_ret"]
    f_rank = f.rank(pct=True)
    r_rank = r.rank(pct=True)

    # Monthly IC with period labels
    ic_series = []
    ics = []
    for i in range(0, len(aligned) - window, window):
        chunk_f = f_rank.iloc[i:i + window]
        chunk_r = r_rank.iloc[i:i + window]
        if chunk_f.std() > 1e-10 and chunk_r.std() > 1e-10:
            ic = chunk_f.corr(chunk_r)
            if not np.isnan(ic):
                ics.append(ic)
                period_date = aligned.index[i]
                label = period_date.strftime("%Y-%m") if hasattr(period_date, "strftime") else str(period_date)[:7]
                ic_series.append({"period": label, "ic": round(ic, 4)})

    if len(ics) < 3:
        return {
            "metrics": {"ic_mean": 0, "ic_std": 1, "ir": 0, "ic_pos_rate": 0,
                        "turnover": 0, "decay": 0, "n_obs": len(aligned)},
            "ic_series": ic_series,
            "warning": f"Too few IC periods: {len(ics)} (need ≥3)",
        }

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics)) + 1e-10
    ir = ic_mean / ic_std
    ic_pos_rate = float(np.mean([1 if ic > 0 else 0 for ic in ics]))

    rank_diff = f_rank.diff().abs()
    turnover = float(rank_diff.mean())

    lag5_ic = float(f_rank.shift(5).corr(r_rank)) if len(aligned) > 60 else 0
    decay = abs(ic_mean) / (abs(lag5_ic) + 1e-10) if abs(lag5_ic) > 1e-10 else 1.0

    return {
        "metrics": {
            "ic_mean": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "ir": round(ir, 3),
            "ic_pos_rate": round(ic_pos_rate, 3),
            "turnover": round(turnover, 4),
            "decay": round(decay, 3),
            "n_obs": len(aligned),
        },
        "ic_series": ic_series,
    }


def compute_quintile_returns(
    factor: pd.Series, fwd_ret: pd.Series, n_groups: int = 5
) -> list:
    """Split by factor quintile, compute average forward return per group."""
    aligned = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret}).dropna()
    if len(aligned) < n_groups * 10:
        return []
    try:
        aligned["quintile"] = pd.qcut(aligned["factor"], n_groups, labels=False, duplicates="drop") + 1
    except ValueError:
        return []
    result = []
    for q, grp in aligned.groupby("quintile"):
        result.append({
            "quintile": int(q),
            "avg_return": round(float(grp["fwd_ret"].mean()), 5),
            "count": len(grp),
        })
    return result


def rate_factor(metrics: dict) -> int:
    """Rate factor quality 1-5 stars."""
    score = 0
    ic = abs(metrics.get("ic_mean", 0))
    ir = abs(metrics.get("ir", 0))
    pos = metrics.get("ic_pos_rate", 0)
    if ic >= 0.02: score += 1
    if ic >= 0.04: score += 1
    if ir >= 0.5:  score += 1
    if ir >= 1.0:  score += 1
    if pos >= 0.55: score += 1
    return min(score, 5)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manual factor evaluation")
    parser.add_argument("--expression", required=True, help="Factor expression")
    parser.add_argument("--name", default="manual_factor", help="Factor name")
    parser.add_argument("--symbols", default=None, help="Comma-separated stock codes")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon")
    args = parser.parse_args()

    # Validate expression first (fast fail)
    try:
        validate_expression(args.expression)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Load data
    try:
        from data_utils import load_cached_data
        symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
        df = load_cached_data(symbols, args.start_date, args.end_date)
    except Exception as e:
        print(json.dumps({"error": f"Data loading failed: {e}"}))
        sys.exit(1)

    has_symbol = "symbol" in df.columns
    all_ics = []
    all_ic_series = []
    all_quintiles = []
    per_stock_ic = {}
    total_obs = 0

    stocks = df.groupby("symbol") if has_symbol else [("single", df)]

    for sym, grp in stocks:
        if has_symbol:
            grp = grp.drop(columns=["symbol"])
        if len(grp) < 60:
            continue

        try:
            factor_vals = compute_factor(args.expression, grp)
        except Exception as e:
            print(f"⚠ {sym}: expression error: {e}", file=sys.stderr)
            continue

        fwd_ret = grp["close"].pct_change(args.horizon).shift(-args.horizon)
        result = evaluate_single_factor(factor_vals, fwd_ret)

        if result["metrics"]["n_obs"] > 0:
            per_stock_ic[str(sym)] = round(result["metrics"]["ic_mean"], 4)
            all_ics.append(result["metrics"]["ic_mean"])
            total_obs += result["metrics"]["n_obs"]
            if not all_ic_series:
                all_ic_series = result["ic_series"]

        quintiles = compute_quintile_returns(factor_vals, fwd_ret)
        if quintiles and not all_quintiles:
            all_quintiles = quintiles

    if not all_ics:
        print(json.dumps({"error": "No valid results from any stock", "n_stocks": 0}))
        sys.exit(1)

    # Aggregate across stocks
    avg_ic = float(np.mean(all_ics))
    std_ic = float(np.std(all_ics)) + 1e-10
    agg_metrics = {
        "ic_mean": round(avg_ic, 5),
        "ic_std": round(std_ic, 5),
        "ir": round(avg_ic / std_ic, 3),
        "ic_pos_rate": round(float(np.mean([1 if ic > 0 else 0 for ic in all_ics])), 3),
        "turnover": 0.0,
        "decay": 1.0,
        "n_obs": total_obs,
        "n_stocks": len(all_ics),
    }
    agg_metrics["rating"] = rate_factor(agg_metrics)

    output = {
        "name": args.name,
        "expression": args.expression,
        "metrics": agg_metrics,
        "ic_series": all_ic_series,
        "quintile_returns": all_quintiles,
        "per_stock_ic": per_stock_ic,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
