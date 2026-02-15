#!/usr/bin/env python3
"""
Auto-retrain pipeline for ML factor model.

Collects training data from:
  1. Trade journal (data/trade_journal.db) — real trade outcomes as labels
  2. Historical OHLCV data (CSV or API) — features
  3. Walk-forward cross-validation — prevents overfitting

Outputs:
  - Retrained model (ONNX / LightGBM)
  - Feature importance report (JSON)
  - Walk-forward validation report
  - Notifies ml_serve to hot-reload

Usage:
    python auto_retrain.py --data market_data.csv
    python auto_retrain.py --journal data/trade_journal.db --data market_data.csv
    python auto_retrain.py --schedule  # run as daily cron
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import feature computation from existing training script
sys.path.insert(0, str(Path(__file__).parent))
from train_factor_model import compute_features, generate_synthetic_data


# ── Training Data Collection ────────────────────────────────────────

def collect_journal_labels(journal_db: str) -> pd.DataFrame:
    """
    Read trade journal and extract trade outcomes as training labels.
    
    Returns DataFrame with columns: [symbol, timestamp, side, price, pnl, label]
    where label = 1 if the trade was profitable, 0 otherwise.
    """
    if not os.path.exists(journal_db):
        print(f"⚠️  Journal DB not found: {journal_db}")
        return pd.DataFrame()

    conn = sqlite3.connect(journal_db)
    try:
        df = pd.read_sql_query("""
            SELECT timestamp, symbol, entry_type, side, price, pnl, quantity
            FROM journal
            WHERE entry_type IN ('order_filled', 'signal')
            AND symbol IS NOT NULL AND symbol != ''
            ORDER BY timestamp
        """, conn)
    except Exception as e:
        print(f"⚠️  Error reading journal: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # For filled orders, mark profitable sells as label=1
    fills = df[df["entry_type"] == "order_filled"].copy()
    if not fills.empty:
        fills["label"] = (fills["pnl"].fillna(0) > 0).astype(int)
        return fills[["timestamp", "symbol", "side", "price", "pnl", "label"]]

    return pd.DataFrame()


def merge_journal_with_ohlcv(
    ohlcv_df: pd.DataFrame,
    journal_df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.01,
) -> tuple:
    """
    Merge OHLCV features with journal-based labels where available,
    fall back to forward-return labels elsewhere.
    
    Returns (features, labels) DataFrames.
    """
    features = compute_features(ohlcv_df)

    # Default: forward return labels
    fwd_ret = ohlcv_df["close"].pct_change(horizon).shift(-horizon)
    labels = (fwd_ret > threshold).astype(int)

    # Override with journal-based labels where available
    if not journal_df.empty:
        # Align journal to OHLCV dates
        journal_df = journal_df.set_index("timestamp").sort_index()
        for idx in journal_df.index:
            closest = features.index.get_indexer([idx], method="nearest")
            if len(closest) > 0 and closest[0] >= 0:
                pos = closest[0]
                labels.iloc[pos] = journal_df.loc[idx, "label"]

    combined = features.copy()
    combined["label"] = labels
    combined.dropna(inplace=True)

    X = combined.drop(columns=["label"])
    y = combined["label"]
    return X, y


# ── Walk-Forward Cross Validation ───────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    params: dict = None,
) -> dict:
    """
    Time-series walk-forward cross-validation.
    
    Splits data into n_splits sequential windows, each with train_ratio
    for training and the rest for validation. No future leakage.
    
    Returns dict with per-fold metrics and overall summary.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        print("ERROR: lightgbm/sklearn required. pip install lightgbm scikit-learn")
        return {"error": "dependencies missing"}

    if params is None:
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

    n = len(X)
    fold_size = n // n_splits
    results = []
    feature_cols = list(X.columns)

    print(f"\n📊 Walk-forward CV: {n_splits} folds, {n} total samples")
    print(f"   Fold size: ~{fold_size}, train ratio: {train_ratio:.0%}")

    for fold in range(n_splits):
        start = fold * fold_size
        end = min(start + fold_size, n)
        if end - start < 100:
            continue

        split = start + int((end - start) * train_ratio)
        X_tr = X.iloc[start:split].values.astype(np.float32)
        y_tr = y.iloc[start:split].values.astype(int)
        X_va = X.iloc[split:end].values.astype(np.float32)
        y_va = y.iloc[split:end].values.astype(int)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            print(f"   Fold {fold+1}: skipped (single class)")
            continue

        # Handle class imbalance
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)
        fold_params = {**params, "scale_pos_weight": scale_pos_weight}

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

        model = lgb.train(
            fold_params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        pred = model.predict(X_va)
        auc = roc_auc_score(y_va, pred)
        acc = accuracy_score(y_va, (pred > 0.5).astype(int))

        results.append({
            "fold": fold + 1,
            "train_size": len(y_tr),
            "val_size": len(y_va),
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "pos_ratio_train": round(pos_count / len(y_tr), 3),
            "best_iteration": model.best_iteration,
        })
        print(f"   Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}, best_iter={model.best_iteration}")

    if not results:
        return {"error": "no valid folds", "folds": []}

    avg_auc = np.mean([r["auc"] for r in results])
    avg_acc = np.mean([r["accuracy"] for r in results])
    std_auc = np.std([r["auc"] for r in results])

    summary = {
        "n_folds": len(results),
        "avg_auc": round(avg_auc, 4),
        "std_auc": round(std_auc, 4),
        "avg_accuracy": round(avg_acc, 4),
        "folds": results,
    }
    print(f"\n   📈 Overall: AUC={avg_auc:.4f} ± {std_auc:.4f}, Acc={avg_acc:.4f}")
    return summary


# ── Full Training Pipeline ──────────────────────────────────────────

def retrain(
    data_path: Optional[str] = None,
    journal_path: str = "data/trade_journal.db",
    output_dir: str = "ml_models",
    n_cv_folds: int = 5,
    horizon: int = 5,
    threshold: float = 0.01,
    notify_serve: bool = True,
    serve_url: str = "http://127.0.0.1:18091",
) -> dict:
    """
    Full retrain pipeline:
      1. Load OHLCV data (CSV or synthetic)
      2. Collect journal labels (if available)
      3. Walk-forward cross-validation
      4. Train final model on all data
      5. Export ONNX + feature importance
      6. Notify ml_serve to hot-reload
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        return {"error": "lightgbm/sklearn required"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": timestamp, "status": "started"}

    # 1. Load data
    if data_path and os.path.exists(data_path):
        print(f"📂 Loading market data from: {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print("📂 No market data provided, using synthetic (5000 bars)")
        df = generate_synthetic_data(5000)
    report["data_rows"] = len(df)

    # 2. Collect journal labels
    journal_df = collect_journal_labels(journal_path)
    report["journal_trades"] = len(journal_df)
    if not journal_df.empty:
        print(f"📋 Found {len(journal_df)} trade outcomes in journal")

    # 3. Build features + labels
    X, y = merge_journal_with_ohlcv(df, journal_df, horizon, threshold)
    feature_cols = list(X.columns)
    report["n_features"] = len(feature_cols)
    report["n_samples"] = len(X)
    report["label_distribution"] = {"positive": int(y.sum()), "negative": int(len(y) - y.sum())}
    print(f"\n🔧 Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"   Label dist: {report['label_distribution']}")

    # 4. Walk-forward CV
    cv_report = walk_forward_cv(X, y, n_splits=n_cv_folds)
    report["walk_forward_cv"] = cv_report

    if "error" in cv_report:
        report["status"] = "cv_failed"
        return report

    # 5. Train final model on all data with early stopping
    print("\n🏋️ Training final model on full dataset...")
    split_idx = int(len(X) * 0.85)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(int)

    pos_count = y_arr[:split_idx].sum()
    neg_count = split_idx - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

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
        "scale_pos_weight": scale_pos_weight,
    }

    train_data = lgb.Dataset(X_arr[:split_idx], label=y_arr[:split_idx], feature_name=feature_cols)
    val_data = lgb.Dataset(X_arr[split_idx:], label=y_arr[split_idx:], reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    val_pred = model.predict(X_arr[split_idx:])
    final_auc = roc_auc_score(y_arr[split_idx:], val_pred)
    final_acc = accuracy_score(y_arr[split_idx:], (val_pred > 0.5).astype(int))
    report["final_model"] = {
        "auc": round(final_auc, 4),
        "accuracy": round(final_acc, 4),
        "best_iteration": model.best_iteration,
        "scale_pos_weight": round(scale_pos_weight, 3),
    }
    print(f"\n✅ Final model: AUC={final_auc:.4f}, Acc={final_acc:.4f}, iters={model.best_iteration}")

    # 6. Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance.tolist()), key=lambda x: -x[1])
    report["feature_importance"] = [{"feature": f, "importance": round(imp, 2)} for f, imp in feat_imp]
    print("\n📊 Top 10 features:")
    for f, imp in feat_imp[:10]:
        print(f"   {f:25s} {imp:10.1f}")

    # 7. Export model
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"factor_model_{timestamp}"

    # Save LightGBM native
    lgb_path = os.path.join(output_dir, f"{model_name}.lgb.txt")
    model.save_model(lgb_path)
    report["lgb_model_path"] = lgb_path

    # Try ONNX export
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=11)
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        report["onnx_model_path"] = onnx_path
        print(f"\n💾 ONNX model: {onnx_path}")
    except ImportError:
        print("⚠️  onnxmltools not installed, ONNX export skipped")

    # Also save as the default model path for ml_serve
    default_lgb = os.path.join(output_dir, "factor_model.lgb.txt")
    model.save_model(default_lgb)
    print(f"💾 LightGBM model: {default_lgb}")

    # Save feature importance JSON
    report_path = os.path.join(output_dir, f"retrain_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Report: {report_path}")

    # Save feature list
    feat_list_path = os.path.join(output_dir, "factor_model_features.txt")
    with open(feat_list_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    # 8. Notify ml_serve to reload
    if notify_serve:
        try:
            import requests
            resp = requests.post(f"{serve_url}/reload", json={
                "model_path": default_lgb,
            }, timeout=5)
            if resp.ok:
                print("🔄 ml_serve notified to reload model")
            else:
                print(f"⚠️  ml_serve reload failed: {resp.status_code}")
        except Exception as e:
            print(f"⚠️  Could not notify ml_serve: {e}")

    report["status"] = "completed"
    print(f"\n🎉 Retrain complete! CV AUC: {cv_report['avg_auc']:.4f}, Final AUC: {final_auc:.4f}")
    return report


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-retrain ML Factor Model")
    parser.add_argument("--data", default=None,
                        help="Path to OHLCV CSV data")
    parser.add_argument("--journal", default="data/trade_journal.db",
                        help="Path to trade journal SQLite DB")
    parser.add_argument("--output-dir", default="ml_models",
                        help="Output directory for models")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of walk-forward CV folds")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon (bars)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Forward return threshold for positive label")
    parser.add_argument("--no-notify", action="store_true",
                        help="Don't notify ml_serve to reload")
    parser.add_argument("--serve-url", default="http://127.0.0.1:18091",
                        help="ml_serve URL for hot-reload notification")

    args = parser.parse_args()

    retrain(
        data_path=args.data,
        journal_path=args.journal,
        output_dir=args.output_dir,
        n_cv_folds=args.folds,
        horizon=args.horizon,
        threshold=args.threshold,
        notify_serve=not args.no_notify,
        serve_url=args.serve_url,
    )
