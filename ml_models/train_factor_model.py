#!/usr/bin/env python3
"""
ML Factor Model Training Script

Trains a LightGBM model on technical factor features derived from OHLCV data,
then exports the model to ONNX format for Rust-side GPU/CPU inference.

Usage:
    pip install lightgbm onnxmltools skl2onnx numpy pandas
    python train_factor_model.py [--output model.onnx] [--csv data.csv]

If no CSV is provided, generates synthetic training data for demonstration.
"""

import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 20+ factor features from OHLCV DataFrame.
    Columns expected: open, high, low, close, volume
    Returns DataFrame with feature columns (NaN rows at start due to lookback).
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    feat = pd.DataFrame(index=df.index)

    # === Returns ===
    feat["ret_1d"] = c.pct_change(1)
    feat["ret_5d"] = c.pct_change(5)
    feat["ret_10d"] = c.pct_change(10)
    feat["ret_20d"] = c.pct_change(20)

    # === Volatility ===
    feat["volatility_5d"] = c.pct_change().rolling(5).std()
    feat["volatility_20d"] = c.pct_change().rolling(20).std()

    # === MA ratios ===
    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    feat["ma5_ratio"] = c / ma5 - 1
    feat["ma10_ratio"] = c / ma10 - 1
    feat["ma20_ratio"] = c / ma20 - 1
    feat["ma60_ratio"] = c / ma60 - 1
    feat["ma5_ma20_cross"] = ma5 / ma20 - 1

    # === RSI(14) ===
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    # === MACD ===
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    feat["macd_histogram"] = macd_line - signal_line
    feat["macd_normalized"] = feat["macd_histogram"] / c

    # === Volume features ===
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    feat["volume_ratio_5_20"] = vol_ma5 / vol_ma20.replace(0, np.nan)
    feat["volume_change"] = v.pct_change(1)

    # === Price range position ===
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    range_20 = high_20 - low_20
    feat["price_position"] = (c - low_20) / range_20.replace(0, np.nan)

    # === Gap ===
    feat["gap"] = o / c.shift(1) - 1

    # === Intraday range ===
    feat["intraday_range"] = (h - l) / o

    # === Upper/lower shadow ===
    body = (c - o).abs()
    total_range = h - l
    feat["upper_shadow_ratio"] = (h - c.where(c > o, o)) / total_range.replace(0, np.nan)
    feat["lower_shadow_ratio"] = (c.where(c < o, o) - l) / total_range.replace(0, np.nan)

    # === Bollinger %B ===
    bb_mid = ma20
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bollinger_pctb"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    return feat


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for training demonstration."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")

    close = 100.0
    data = []
    for d in dates:
        ret = np.random.normal(0.0003, 0.02)
        close *= 1 + ret
        intra_vol = abs(ret) * 0.5 + 0.005
        high = close * (1 + np.random.uniform(0, intra_vol))
        low = close * (1 - np.random.uniform(0, intra_vol))
        open_ = close * (1 + np.random.normal(0, 0.003))
        volume = np.random.uniform(5e6, 50e6) * (1 + abs(ret) * 10)
        data.append([d, open_, high, low, close, volume])

    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    df.set_index("date", inplace=True)
    return df


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.Series:
    """
    Create binary labels: 1 if forward return > threshold, 0 otherwise.
    This is the target for the ML model.
    """
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = (fwd_ret > threshold).astype(int)
    return labels


def train_and_export(output_path: str, csv_path: str = None):
    """Train LightGBM model and export to ONNX."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        return

    # Load or generate data
    if csv_path:
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print("Generating synthetic training data (5000 bars)...")
        df = generate_synthetic_data(5000)

    # Compute features and labels
    features = compute_features(df)
    labels = create_labels(df, horizon=5, threshold=0.01)

    # Merge and drop NaN
    combined = features.copy()
    combined["label"] = labels
    combined.dropna(inplace=True)

    feature_cols = [c for c in combined.columns if c != "label"]
    X = combined[feature_cols].values.astype(np.float32)
    y = combined["label"].values.astype(int)

    print(f"Training set: {len(X)} samples, {len(feature_cols)} features")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Features: {feature_cols}")

    # Train/val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
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
    }

    print("\nTraining LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    print(f"\nValidation AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\nTop 10 features by importance:")
    for fname, imp in feat_imp[:10]:
        print(f"  {fname:25s} {imp:10.1f}")

    # Export to ONNX
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=11)
        onnxmltools.utils.save_model(onnx_model, output_path)
        print(f"\n✅ Model exported to: {output_path}")
        print(f"   Input shape: [batch_size, {len(feature_cols)}]")
        print(f"   Output: probability of positive return")
    except ImportError:
        print("\nERROR: onnxmltools not installed. Run: pip install onnxmltools skl2onnx")
        print("Saving LightGBM model as .txt instead...")
        model.save_model(output_path.replace(".onnx", ".lgb.txt"))

    # Save feature list for Rust
    feature_list_path = output_path.replace(".onnx", "_features.txt")
    with open(feature_list_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    print(f"   Feature list saved to: {feature_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Factor Model")
    parser.add_argument("--output", "-o", default="ml_models/factor_model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--csv", default=None,
                       help="Path to OHLCV CSV data (optional, generates synthetic if omitted)")
    args = parser.parse_args()

    train_and_export(args.output, args.csv)
