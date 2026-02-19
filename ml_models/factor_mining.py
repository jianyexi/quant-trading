#!/usr/bin/env python3
"""
Automated Factor Mining — discovers profitable factors via parameterized search.

Pipeline:
  1. Generate candidate factors (template × parameter grid)
  2. Evaluate each factor: IC, IR, turnover, decay
  3. Filter: IC > threshold, multi-test correction
  4. De-correlate: remove redundant factors (corr > 0.7)
  5. Export: best factors → feature list for ML training

Supports:
  - Parameterized search (window sweep on known templates)
  - Composite factors (ratio/diff of two base factors)
  - Statistical validation (Bonferroni, IC stability)
  - Auto-integration with existing ML pipeline

Usage:
    python ml_models/factor_mining.py --data market_data.csv
    python ml_models/factor_mining.py --synthetic --n-stocks 10
    python ml_models/factor_mining.py --data data.csv --export-top 30 --retrain
"""

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from train_factor_model import compute_features, generate_synthetic_data

# ── Factor Templates ─────────────────────────────────────────────────

WINDOWS = [3, 5, 10, 15, 20, 30, 60]
SHORT_WINDOWS = [3, 5, 10, 15, 20]
LONG_WINDOWS = [20, 30, 60, 120]


def compute_all_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all candidate factors from OHLCV data via template × parameter grid."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()

    factors = pd.DataFrame(index=df.index)

    # ── 1. Momentum (returns at various horizons) ────────────────────
    for n in WINDOWS:
        factors[f"momentum_{n}"] = c.pct_change(n)

    # ── 2. Volatility (rolling std of returns) ───────────────────────
    for n in WINDOWS:
        factors[f"volatility_{n}"] = ret.rolling(n).std()

    # ── 3. MA ratio (price relative to moving average) ───────────────
    for n in WINDOWS:
        ma = c.rolling(n).mean()
        factors[f"ma_ratio_{n}"] = c / ma - 1

    # ── 4. MA cross (short MA / long MA) ─────────────────────────────
    for s in SHORT_WINDOWS:
        for l_win in LONG_WINDOWS:
            if s >= l_win:
                continue
            ma_s = c.rolling(s).mean()
            ma_l = c.rolling(l_win).mean()
            factors[f"ma_cross_{s}_{l_win}"] = ma_s / ma_l - 1

    # ── 5. RSI at various periods ────────────────────────────────────
    delta = c.diff()
    for n in [5, 7, 10, 14, 21]:
        gain = delta.where(delta > 0, 0.0).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        factors[f"rsi_{n}"] = (100 - 100 / (1 + rs)) / 100.0  # Normalize to [0,1]

    # ── 6. MACD variants ─────────────────────────────────────────────
    for fast, slow, sig in [(8, 21, 5), (12, 26, 9), (5, 34, 5), (19, 39, 9)]:
        ema_f = c.ewm(span=fast).mean()
        ema_s = c.ewm(span=slow).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig).mean()
        factors[f"macd_hist_{fast}_{slow}_{sig}"] = (macd - signal) / c

    # ── 7. Volume features ───────────────────────────────────────────
    for n in WINDOWS:
        vol_ma = v.rolling(n).mean()
        factors[f"volume_ratio_{n}"] = v / vol_ma.replace(0, np.nan)

    for s in SHORT_WINDOWS:
        for l_win in LONG_WINDOWS:
            if s >= l_win:
                continue
            vs = v.rolling(s).mean()
            vl = v.rolling(l_win).mean()
            factors[f"vol_cross_{s}_{l_win}"] = vs / vl.replace(0, np.nan) - 1

    # ── 8. Price range position ──────────────────────────────────────
    for n in WINDOWS:
        hi = h.rolling(n).max()
        lo = l.rolling(n).min()
        rng = hi - lo
        factors[f"price_pos_{n}"] = (c - lo) / rng.replace(0, np.nan)

    # ── 9. Bollinger %B ──────────────────────────────────────────────
    for n in [10, 20, 30]:
        for k in [1.5, 2.0, 2.5]:
            ma = c.rolling(n).mean()
            std = c.rolling(n).std()
            bb_upper = ma + k * std
            bb_lower = ma - k * std
            bw = bb_upper - bb_lower
            factors[f"boll_pctb_{n}_{k}"] = (c - bb_lower) / bw.replace(0, np.nan)

    # ── 10. ATR (Average True Range) normalized ──────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    for n in WINDOWS:
        atr = tr.rolling(n).mean()
        factors[f"atr_norm_{n}"] = atr / c

    # ── 11. Intraday patterns ────────────────────────────────────────
    body = (c - o).abs()
    total = h - l
    factors["body_ratio"] = body / total.replace(0, np.nan)
    factors["upper_shadow"] = (h - c.where(c > o, o)) / total.replace(0, np.nan)
    factors["lower_shadow"] = (c.where(c < o, o) - l) / total.replace(0, np.nan)
    factors["intraday_range"] = (h - l) / o
    factors["gap"] = o / c.shift(1) - 1
    factors["close_to_open"] = c / o - 1

    # ── 12. Volume-weighted momentum ─────────────────────────────────
    for n in [5, 10, 20]:
        vwap_approx = (c * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)
        factors[f"vwap_dev_{n}"] = c / vwap_approx - 1

    # ── 13. Rate of change acceleration ──────────────────────────────
    for n in [5, 10, 20]:
        roc = c.pct_change(n)
        roc_prev = c.shift(n).pct_change(n)
        factors[f"roc_accel_{n}"] = roc - roc_prev

    # ── 14. Skewness and kurtosis of returns ─────────────────────────
    for n in [10, 20, 60]:
        factors[f"ret_skew_{n}"] = ret.rolling(n).skew()
        factors[f"ret_kurt_{n}"] = ret.rolling(n).kurt()

    # ── 15. High-low ratio ───────────────────────────────────────────
    for n in [5, 10, 20]:
        factors[f"hl_ratio_{n}"] = h.rolling(n).max() / l.rolling(n).min() - 1

    # ── 16. Close-to-high / close-to-low ─────────────────────────────
    for n in [5, 10, 20]:
        factors[f"close_to_high_{n}"] = c / h.rolling(n).max() - 1
        factors[f"close_to_low_{n}"] = c / l.rolling(n).min() - 1

    # ── 17. Amihud illiquidity ───────────────────────────────────────
    for n in [5, 10, 20]:
        illiq = (ret.abs() / (v * c).replace(0, np.nan)).rolling(n).mean()
        factors[f"amihud_{n}"] = illiq

    # Replace inf with NaN
    factors.replace([np.inf, -np.inf], np.nan, inplace=True)

    return factors


# ── Factor Evaluation ────────────────────────────────────────────────

def evaluate_factor(factor: pd.Series, forward_ret: pd.Series) -> Dict[str, float]:
    """
    Evaluate a single factor's predictive power.

    Returns: {ic_mean, ic_std, ir, ic_positive_rate, turnover, decay_ratio}
    """
    # Align and drop NaN
    aligned = pd.DataFrame({"factor": factor, "fwd_ret": forward_ret}).dropna()
    if len(aligned) < 120:
        return {"ic_mean": 0, "ic_std": 1, "ir": 0, "ic_pos_rate": 0, "turnover": 1, "decay": 0, "n_obs": 0}

    f = aligned["factor"]
    r = aligned["fwd_ret"]

    # Monthly IC (rolling 20-bar windows)
    window = 20
    ics = []
    for i in range(0, len(aligned) - window, window):
        chunk_f = f.iloc[i:i+window]
        chunk_r = r.iloc[i:i+window]
        if chunk_f.std() > 1e-10 and chunk_r.std() > 1e-10:
            ic = chunk_f.corr(chunk_r)
            if not np.isnan(ic):
                ics.append(ic)

    if len(ics) < 3:
        return {"ic_mean": 0, "ic_std": 1, "ir": 0, "ic_pos_rate": 0, "turnover": 1, "decay": 0, "n_obs": 0}

    ic_mean = np.mean(ics)
    ic_std = np.std(ics) + 1e-10
    ir = ic_mean / ic_std

    # IC positive rate (how often IC > 0)
    ic_pos_rate = np.mean([1 if ic > 0 else 0 for ic in ics])

    # Turnover: how much the factor ranking changes period-to-period
    rank = f.rank(pct=True)
    rank_diff = rank.diff().abs()
    turnover = rank_diff.mean()

    # Decay: compare IC of immediate vs lagged factor
    # High IC at lag=0, low IC at lag=5 → factor has fast decay (good for short-term)
    lag5_ic = f.shift(5).corr(r) if len(aligned) > 60 else 0
    decay = abs(ic_mean) / (abs(lag5_ic) + 1e-10) if abs(lag5_ic) > 1e-10 else 1.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "ic_pos_rate": ic_pos_rate,
        "turnover": turnover,
        "decay": decay,
        "n_obs": len(aligned),
    }


def mine_factors(
    df: pd.DataFrame,
    horizon: int = 5,
    ic_threshold: float = 0.02,
    ir_threshold: float = 0.3,
    max_corr: float = 0.7,
    top_n: int = 30,
    bonferroni: bool = True,
) -> pd.DataFrame:
    """
    Full factor mining pipeline:
      1. Generate candidates
      2. Evaluate IC/IR
      3. Filter by thresholds
      4. Bonferroni correction
      5. De-correlate
      6. Rank and return top N

    Returns DataFrame with columns: [factor_name, ic_mean, ir, ic_pos_rate, turnover, decay]
    """
    print(f"\n{'='*60}")
    print(f"  AUTOMATED FACTOR MINING")
    print(f"  Horizon: {horizon} bars | IC threshold: {ic_threshold}")
    print(f"  Data: {len(df)} rows")
    print(f"{'='*60}\n")

    t0 = time.time()

    # 1. Generate candidates
    print("📐 Generating candidate factors...")
    candidates = compute_all_candidates(df)
    n_candidates = len(candidates.columns)
    print(f"   Generated {n_candidates} candidate factors")

    # 2. Forward returns (label)
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

    # 3. Evaluate all factors
    print(f"📊 Evaluating {n_candidates} factors (horizon={horizon})...")
    results = []
    for i, col in enumerate(candidates.columns):
        if (i + 1) % 50 == 0:
            print(f"   ... {i+1}/{n_candidates}")
        ev = evaluate_factor(candidates[col], fwd_ret)
        ev["factor_name"] = col
        results.append(ev)

    results_df = pd.DataFrame(results)
    results_df.sort_values("ic_mean", key=abs, ascending=False, inplace=True)

    # 4. Filter by thresholds
    abs_ic = results_df["ic_mean"].abs()
    mask = (abs_ic >= ic_threshold) & (results_df["ir"].abs() >= ir_threshold) & (results_df["ic_pos_rate"] >= 0.5)

    # Bonferroni correction: require higher IC if testing many factors
    if bonferroni and n_candidates > 10:
        # Approximate: IC t-stat > 2 after correction
        correction_factor = np.sqrt(np.log(n_candidates))
        adjusted_threshold = ic_threshold * correction_factor
        mask = mask & (abs_ic >= adjusted_threshold)
        print(f"   Bonferroni adjusted IC threshold: {adjusted_threshold:.4f} (correction={correction_factor:.2f})")

    filtered = results_df[mask].copy()
    print(f"\n✅ {len(filtered)}/{n_candidates} factors passed filters")

    if filtered.empty:
        print("⚠️  No factors passed. Relaxing thresholds...")
        # Fallback: take top 20 by |IC|
        filtered = results_df.head(min(20, len(results_df))).copy()

    # 5. De-correlate: remove highly correlated factors
    print(f"🔄 De-correlating (max_corr={max_corr})...")
    kept_names = []
    kept_data = pd.DataFrame(index=df.index)

    for _, row in filtered.iterrows():
        name = row["factor_name"]
        if name not in candidates.columns:
            continue
        series = candidates[name]

        # Check correlation with already-kept factors
        too_correlated = False
        for kept in kept_names:
            corr = series.corr(kept_data[kept])
            if abs(corr) > max_corr:
                too_correlated = True
                break

        if not too_correlated:
            kept_names.append(name)
            kept_data[name] = series

        if len(kept_names) >= top_n:
            break

    final = filtered[filtered["factor_name"].isin(kept_names)].copy()
    final = final.sort_values("ir", key=abs, ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  RESULTS: {len(final)} factors selected ({elapsed:.1f}s)")
    print(f"{'='*60}\n")

    # Pretty print results
    print(f"{'Rank':>4} {'Factor':>30} {'IC':>8} {'IR':>8} {'IC+%':>6} {'Turn':>6} {'Decay':>6}")
    print("-" * 75)
    for i, row in final.iterrows():
        print(f"{i+1:4d} {row['factor_name']:>30} {row['ic_mean']:8.4f} {row['ir']:8.3f} "
              f"{row['ic_pos_rate']:6.1%} {row['turnover']:6.3f} {row['decay']:6.2f}")

    return final


# ── Export & Integration ─────────────────────────────────────────────

def export_factors(
    results: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str = "ml_models",
    retrain: bool = False,
):
    """Export mined factors for ML integration."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save factor report
    report_path = out / f"factor_mining_report_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "n_candidates_tested": None,  # filled externally
        "n_selected": len(results),
        "factors": results.to_dict(orient="records"),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Report: {report_path}")

    # 2. Save factor names for Rust integration
    feature_path = out / "mined_factor_features.txt"
    with open(feature_path, "w") as f:
        for name in results["factor_name"]:
            f.write(name + "\n")
    print(f"📄 Feature list: {feature_path}")

    # 3. Generate factor data for retraining
    candidates = compute_all_candidates(df)
    selected_cols = [n for n in results["factor_name"] if n in candidates.columns]
    factor_data = candidates[selected_cols].copy()
    factor_data["close"] = df["close"]

    data_path = out / f"mined_factors_data_{timestamp}.csv"
    factor_data.to_csv(data_path)
    print(f"📄 Factor data: {data_path}")

    # 4. Generate Rust code snippet
    rust_path = out / "mined_factors_rust_snippet.rs"
    generate_rust_snippet(results, rust_path)
    print(f"📄 Rust snippet: {rust_path}")

    # 5. Retrain ML model with new factors
    if retrain:
        print("\n🔄 Retraining ML model with mined factors...")
        retrain_with_factors(df, factor_data, selected_cols, out, timestamp)

    return str(data_path)


def generate_rust_snippet(results: pd.DataFrame, output_path: Path):
    """Generate Rust code for computing mined factors in fast_factors.rs."""
    lines = [
        "// Auto-generated by factor_mining.py",
        "// Add these factors to IncrementalFactorEngine::update()",
        "",
        "pub const MINED_FACTOR_NAMES: &[&str] = &[",
    ]
    for name in results["factor_name"]:
        lines.append(f'    "{name}",')
    lines.append("];")
    lines.append("")
    lines.append(f"pub const NUM_MINED_FACTORS: usize = {len(results)};")
    lines.append("")

    # Generate compute hints
    lines.append("/*")
    lines.append("Factor computation guide:")
    lines.append("")
    for _, row in results.iterrows():
        name = row["factor_name"]
        ic = row["ic_mean"]
        ir = row["ir"]
        lines.append(f"  {name}: IC={ic:.4f}, IR={ir:.3f}")

        # Parse template and generate computation hint
        if name.startswith("momentum_"):
            n = name.split("_")[1]
            lines.append(f"    → close / delay(close, {n}) - 1")
        elif name.startswith("volatility_"):
            n = name.split("_")[1]
            lines.append(f"    → rolling_std(returns, {n})")
        elif name.startswith("ma_ratio_"):
            n = name.split("_")[2]
            lines.append(f"    → close / rolling_mean(close, {n}) - 1")
        elif name.startswith("ma_cross_"):
            parts = name.split("_")
            lines.append(f"    → rolling_mean(close, {parts[2]}) / rolling_mean(close, {parts[3]}) - 1")
        elif name.startswith("rsi_"):
            n = name.split("_")[1]
            lines.append(f"    → RSI({n}) / 100")
        elif name.startswith("boll_pctb_"):
            parts = name.split("_")
            lines.append(f"    → bollinger_pctb(close, n={parts[2]}, k={parts[3]})")
        elif name.startswith("atr_norm_"):
            n = name.split("_")[2]
            lines.append(f"    → ATR({n}) / close")
        elif name.startswith("vwap_dev_"):
            n = name.split("_")[2]
            lines.append(f"    → close / VWAP({n}) - 1")
        elif name.startswith("amihud_"):
            n = name.split("_")[1]
            lines.append(f"    → rolling_mean(|ret| / (volume * close), {n})")
        lines.append("")

    lines.append("*/")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def retrain_with_factors(
    df: pd.DataFrame,
    factor_data: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    timestamp: str,
):
    """Retrain LightGBM model using mined factors."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        print("⚠️  lightgbm or sklearn not installed, skipping retrain")
        return

    # Create labels
    horizon = 5
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = (fwd_ret > 0.01).astype(int)

    # Combine
    combined = factor_data[feature_cols].copy()
    combined["label"] = labels.values
    combined.dropna(inplace=True)

    X = combined[feature_cols].values.astype(np.float32)
    y = combined["label"].values.astype(int)

    if len(X) < 200:
        print("⚠️  Not enough data for training")
        return

    # Walk-forward split
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))

    print(f"\n   🎯 Mined-factor model: AUC={auc:.4f}, Accuracy={acc:.4f}")

    # Save model
    model_path = output_dir / f"mined_factor_model_{timestamp}.model"
    model.save_model(str(model_path))
    print(f"   💾 Model saved: {model_path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\n   Top 10 mined factors by model importance:")
    for fname, imp in feat_imp[:10]:
        print(f"     {fname:30s} {imp:10.1f}")

    # Save report
    report = {
        "timestamp": timestamp,
        "auc": float(auc),
        "accuracy": float(acc),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_importance": {n: float(i) for n, i in feat_imp},
    }
    report_path = output_dir / f"mined_model_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


# ── Multi-Stock Mining ───────────────────────────────────────────────

def generate_multi_stock_data(n_stocks: int = 10, n_bars: int = 2000) -> Dict[str, pd.DataFrame]:
    """Generate synthetic multi-stock data for cross-stock factor validation."""
    stocks = {}
    for i in range(n_stocks):
        np.random.seed(42 + i)
        base_price = 50 + i * 30
        df = generate_synthetic_data(n_bars)
        # Scale to different price levels
        scale = base_price / df["close"].iloc[0]
        for col in ["open", "high", "low", "close"]:
            df[col] *= scale
        df["volume"] *= (0.5 + np.random.random())
        symbol = f"stock_{i:03d}"
        stocks[symbol] = df
    return stocks


def mine_cross_stock(
    stocks: Dict[str, pd.DataFrame],
    horizon: int = 5,
    ic_threshold: float = 0.02,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Mine factors across multiple stocks for robustness.
    A factor must be effective in >50% of stocks to pass.
    """
    print(f"\n{'='*60}")
    print(f"  CROSS-STOCK FACTOR MINING ({len(stocks)} stocks)")
    print(f"{'='*60}\n")

    # Evaluate each factor on each stock
    all_stock_results = {}  # factor_name → [ic_per_stock]

    for sym, df in stocks.items():
        print(f"📊 Processing {sym}...")
        candidates = compute_all_candidates(df)
        fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

        for col in candidates.columns:
            ev = evaluate_factor(candidates[col], fwd_ret)
            if col not in all_stock_results:
                all_stock_results[col] = []
            all_stock_results[col].append(ev)

    # Aggregate: factor is good if IC is consistent across stocks
    agg_results = []
    for factor_name, evals in all_stock_results.items():
        ics = [e["ic_mean"] for e in evals if e["n_obs"] > 0]
        if len(ics) < 2:
            continue

        avg_ic = np.mean(ics)
        ic_std = np.std(ics) + 1e-10
        cross_ir = avg_ic / ic_std  # Cross-stock IR
        win_rate = np.mean([1 if abs(ic) > ic_threshold else 0 for ic in ics])

        agg_results.append({
            "factor_name": factor_name,
            "ic_mean": avg_ic,
            "ic_std": ic_std,
            "ir": cross_ir,
            "ic_pos_rate": np.mean([e["ic_pos_rate"] for e in evals]),
            "turnover": np.mean([e["turnover"] for e in evals]),
            "decay": np.mean([e["decay"] for e in evals]),
            "stock_win_rate": win_rate,
            "n_stocks": len(ics),
        })

    agg_df = pd.DataFrame(agg_results)
    agg_df = agg_df[agg_df["stock_win_rate"] >= 0.5]  # Must work in ≥50% of stocks
    agg_df.sort_values("ir", key=abs, ascending=False, inplace=True)
    agg_df = agg_df.head(top_n).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  CROSS-STOCK RESULTS: {len(agg_df)} robust factors")
    print(f"{'='*60}\n")

    print(f"{'Rank':>4} {'Factor':>30} {'IC':>8} {'IR':>8} {'StockWin':>9}")
    print("-" * 65)
    for i, row in agg_df.iterrows():
        print(f"{i+1:4d} {row['factor_name']:>30} {row['ic_mean']:8.4f} {row['ir']:8.3f} {row['stock_win_rate']:9.1%}")

    return agg_df


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automated Factor Mining")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to OHLCV CSV (optional)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--n-stocks", type=int, default=10,
                        help="Number of stocks for cross-stock mining")
    parser.add_argument("--n-bars", type=int, default=3000,
                        help="Number of bars per stock (synthetic)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon (bars)")
    parser.add_argument("--ic-threshold", type=float, default=0.02,
                        help="Minimum |IC| threshold")
    parser.add_argument("--ir-threshold", type=float, default=0.3,
                        help="Minimum |IR| threshold")
    parser.add_argument("--max-corr", type=float, default=0.7,
                        help="Max factor correlation (de-dup)")
    parser.add_argument("--export-top", type=int, default=30,
                        help="Number of top factors to export")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain ML model with mined factors")
    parser.add_argument("--cross-stock", action="store_true",
                        help="Run cross-stock validation")
    parser.add_argument("--output-dir", type=str, default="ml_models",
                        help="Output directory")
    args = parser.parse_args()

    # Load data
    if args.data:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                print(f"ERROR: Missing column '{col}' in CSV")
                return
    else:
        print(f"Generating synthetic data ({args.n_bars} bars)...")
        df = generate_synthetic_data(args.n_bars)

    # Cross-stock or single-stock mining
    if args.cross_stock:
        if args.data:
            print("Cross-stock mining requires multiple stocks. Using synthetic data.")
        stocks = generate_multi_stock_data(args.n_stocks, args.n_bars)
        results = mine_cross_stock(
            stocks,
            horizon=args.horizon,
            ic_threshold=args.ic_threshold,
            top_n=args.export_top,
        )
    else:
        results = mine_factors(
            df,
            horizon=args.horizon,
            ic_threshold=args.ic_threshold,
            ir_threshold=args.ir_threshold,
            max_corr=args.max_corr,
            top_n=args.export_top,
        )

    if results.empty:
        print("\n❌ No factors found. Try relaxing thresholds.")
        return

    # Export
    export_factors(results, df, args.output_dir, retrain=args.retrain)
    print("\n✅ Factor mining complete!")


if __name__ == "__main__":
    main()
