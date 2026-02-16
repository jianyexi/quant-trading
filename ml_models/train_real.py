#!/usr/bin/env python3
"""Train LightGBM model on real A-share data via akshare.

Usage:
    python ml_models/train_real.py

Fetches 3 years of daily data for top stocks, computes 24 features,
trains LightGBM to predict 5-day forward returns, saves model.
"""

import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "volatility_5d", "volatility_20d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio", "ma5_ma20_cross",
    "rsi_14",
    "macd_histogram", "macd_normalized",
    "volume_ratio_5_20", "volume_change",
    "price_position",
    "gap",
    "intraday_range",
    "upper_shadow_ratio", "lower_shadow_ratio",
    "bollinger_pctb",
    "body_ratio",
    "close_to_open",
]

STOCKS = [
    "600519", "000858", "000001", "600036", "300750",
    "002594", "601318", "600276", "000333", "601888",
    "600030", "601166", "600900", "000568", "600809",
    "601899", "600031", "600309", "300059", "600887",
]


def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from akshare."""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24 factor features from OHLCV."""
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    feat = pd.DataFrame(index=df.index)

    feat["ret_1d"] = c.pct_change(1)
    feat["ret_5d"] = c.pct_change(5)
    feat["ret_10d"] = c.pct_change(10)
    feat["ret_20d"] = c.pct_change(20)
    feat["volatility_5d"] = c.pct_change().rolling(5).std()
    feat["volatility_20d"] = c.pct_change().rolling(20).std()

    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    feat["ma5_ratio"] = c / ma5 - 1
    feat["ma10_ratio"] = c / ma10 - 1
    feat["ma20_ratio"] = c / ma20 - 1
    feat["ma60_ratio"] = c / ma60 - 1
    feat["ma5_ma20_cross"] = ma5 / ma20 - 1

    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    feat["macd_histogram"] = macd_line - signal_line
    feat["macd_normalized"] = feat["macd_histogram"] / c

    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    feat["volume_ratio_5_20"] = vol_ma5 / vol_ma20.replace(0, np.nan)
    feat["volume_change"] = v.pct_change(1)

    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    range_20 = high_20 - low_20
    feat["price_position"] = (c - low_20) / range_20.replace(0, np.nan)

    feat["gap"] = o / c.shift(1) - 1
    feat["intraday_range"] = (h - l) / o

    body = (c - o).abs()
    total_range = (h - l).replace(0, np.nan)
    feat["upper_shadow_ratio"] = (h - c.where(c > o, o)) / total_range
    feat["lower_shadow_ratio"] = (c.where(c < o, o) - l) / total_range

    bb_mid = ma20
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bollinger_pctb"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    feat["body_ratio"] = (c - o) / total_range
    feat["close_to_open"] = c / o - 1

    return feat


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score

    start, end = "2021-01-01", "2024-12-31"
    horizon = 5
    threshold = 0.01  # 1% forward return = positive label

    # Phase 1: Fetch data
    all_data = []
    for i, sym in enumerate(STOCKS):
        print(f"  [{i+1}/{len(STOCKS)}] Fetching {sym}...", end=" ", flush=True)
        try:
            df = fetch_stock_data(sym, start, end)
            if len(df) < 200:
                print(f"skip ({len(df)} bars)")
                continue
            features = compute_features(df)
            fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
            labels = (fwd_ret > threshold).astype(int)
            combined = features.copy()
            combined["label"] = labels
            combined["symbol"] = sym
            combined.dropna(inplace=True)
            all_data.append(combined)
            print(f"OK ({len(combined)} samples)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not all_data:
        print("No data fetched!")
        sys.exit(1)

    full = pd.concat(all_data, axis=0).sort_index()
    feature_cols = FEATURE_NAMES
    # Ensure only the feature columns we want + handle any extra cols like body_ratio, close_to_open
    available_cols = [c for c in feature_cols if c in full.columns]
    if len(available_cols) < 20:
        print(f"Warning: only {len(available_cols)} features available")

    X = full[available_cols].values.astype(np.float32)
    y = full["label"].values.astype(int)

    print(f"\n{'='*60}")
    print(f"Total: {len(X)} samples from {len(all_data)} stocks")
    print(f"Features: {len(available_cols)}")
    print(f"Label dist: 0={np.sum(y==0)}, 1={np.sum(y==1)} ({np.mean(y)*100:.1f}% positive)")

    # Phase 2: Train (time-series split — last 20% as validation)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }

    print(f"\nTraining LightGBM (200 rounds)...")
    model = lgb.train(
        params, train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    # Evaluate
    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    print(f"\nValidation AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(available_cols, importance), key=lambda x: -x[1])
    print("\nTop features by importance:")
    for fname, imp in feat_imp[:10]:
        print(f"  {fname:25s} {imp:10.1f}")

    # Save model
    model_path = "ml_models/factor_model.lgb.txt"
    model.save_model(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"   AUC={auc:.4f}, Acc={acc:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ML Factor Model Training — Real A-Share Data")
    print("=" * 60)
    main()
