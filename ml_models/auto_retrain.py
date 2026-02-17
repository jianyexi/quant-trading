#!/usr/bin/env python3
"""
Auto-retrain pipeline for ML factor model.

Supports multiple algorithms:
  - LightGBM (gradient boosting)
  - XGBoost (gradient boosting)
  - CatBoost (gradient boosting)
  - LSTM (deep learning, PyTorch)
  - Transformer (deep learning, PyTorch)

Collects training data from:
  1. Trade journal (data/trade_journal.db) — real trade outcomes as labels
  2. Historical OHLCV data (CSV or API) — features
  3. Walk-forward cross-validation — prevents overfitting

All models compete on the same walk-forward CV; the best AUC wins.

Usage:
    python auto_retrain.py --data market_data.csv
    python auto_retrain.py --data market_data.csv --algorithms lgb,xgb,catboost,lstm,transformer
    python auto_retrain.py --journal data/trade_journal.db --data market_data.csv
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

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
    """Merge OHLCV features with journal-based labels where available."""
    features = compute_features(ohlcv_df)

    fwd_ret = ohlcv_df["close"].pct_change(horizon).shift(-horizon)
    labels = (fwd_ret > threshold).astype(int)

    if not journal_df.empty:
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


# ══════════════════════════════════════════════════════════════════════
# Model Trainers — unified interface for all algorithms
# ══════════════════════════════════════════════════════════════════════

class ModelTrainer(ABC):
    """Base class for all model trainers."""
    name: str = "base"

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs) -> Any:
        """Train model, return fitted model object."""
        ...

    @abstractmethod
    def predict(self, model, X) -> np.ndarray:
        """Return probabilities [0, 1] for each row."""
        ...

    @abstractmethod
    def save(self, model, path: str, feature_cols: List[str]):
        """Save model to disk."""
        ...

    def feature_importance(self, model, feature_cols) -> List[Dict]:
        """Return sorted feature importance list. Optional."""
        return []

    @staticmethod
    def available() -> bool:
        """Check if dependencies are installed."""
        return True


# ── LightGBM ────────────────────────────────────────────────────────

class LightGBMTrainer(ModelTrainer):
    name = "lightgbm"

    @staticmethod
    def available() -> bool:
        try:
            import lightgbm  # noqa
            return True
        except ImportError:
            return False

    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs):
        import lightgbm as lgb

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = neg / max(pos, 1)

        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 20, "max_depth": 5,
            "min_child_samples": 50,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 1.0,
            "scale_pos_weight": spw, "verbose": -1,
        }
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params, train_data, num_boost_round=800, valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def predict(self, model, X):
        return model.predict(X)

    def save(self, model, path, feature_cols):
        model.save_model(path)

    def feature_importance(self, model, feature_cols):
        imp = model.feature_importance(importance_type="gain")
        return sorted(
            [{"feature": f, "importance": round(float(v), 2)} for f, v in zip(feature_cols, imp)],
            key=lambda x: -x["importance"],
        )


# ── XGBoost ─────────────────────────────────────────────────────────

class XGBoostTrainer(ModelTrainer):
    name = "xgboost"

    @staticmethod
    def available() -> bool:
        try:
            import xgboost  # noqa
            return True
        except ImportError:
            return False

    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs):
        import xgboost as xgb

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = neg / max(pos, 1)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": spw, "verbosity": 0,
        }
        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        return model

    def predict(self, model, X):
        import xgboost as xgb
        return model.predict(xgb.DMatrix(X))

    def save(self, model, path, feature_cols):
        model.save_model(path)

    def feature_importance(self, model, feature_cols):
        scores = model.get_score(importance_type="gain")
        return sorted(
            [{"feature": f, "importance": round(scores.get(f, 0), 2)} for f in feature_cols],
            key=lambda x: -x["importance"],
        )


# ── CatBoost ────────────────────────────────────────────────────────

class CatBoostTrainer(ModelTrainer):
    name = "catboost"

    @staticmethod
    def available() -> bool:
        try:
            import catboost  # noqa
            return True
        except ImportError:
            return False

    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs):
        from catboost import CatBoostClassifier

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = neg / max(pos, 1)

        model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            loss_function="Logloss", eval_metric="AUC",
            scale_pos_weight=spw,
            early_stopping_rounds=30, verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return model

    def predict(self, model, X):
        return model.predict_proba(X)[:, 1]

    def save(self, model, path, feature_cols):
        model.save_model(path)

    def feature_importance(self, model, feature_cols):
        imp = model.get_feature_importance()
        return sorted(
            [{"feature": f, "importance": round(float(v), 2)} for f, v in zip(feature_cols, imp)],
            key=lambda x: -x["importance"],
        )


# ── LSTM (PyTorch) ──────────────────────────────────────────────────

class LSTMTrainer(ModelTrainer):
    name = "lstm"

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa
            return True
        except ImportError:
            return False

    def _build_sequences(self, X, y, seq_len=20):
        """Convert tabular data to sliding-window sequences for LSTM."""
        seqs, labels = [], []
        for i in range(seq_len, len(X)):
            seqs.append(X[i - seq_len:i])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs):
        import torch
        import torch.nn as nn

        seq_len = min(20, len(X_train) // 5)
        X_tr_seq, y_tr_seq = self._build_sequences(X_train, y_train, seq_len)
        X_va_seq, y_va_seq = self._build_sequences(X_val, y_val, seq_len)

        if len(X_tr_seq) < 50 or len(X_va_seq) < 10:
            raise ValueError("Not enough data for LSTM sequences")

        n_features = X_train.shape[1]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden=64, layers=2, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
                self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = LSTMModel(n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        pos = y_tr_seq.sum()
        neg = len(y_tr_seq) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_t = torch.tensor(X_tr_seq, device=device)
        y_t = torch.tensor(y_tr_seq, device=device)
        X_v = torch.tensor(X_va_seq, device=device)
        y_v = torch.tensor(y_va_seq, device=device)

        best_auc, patience, wait = 0.0, 15, 0
        best_state = None

        for epoch in range(100):
            model.train()
            # Mini-batch training
            bs = min(256, len(X_t))
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            for i in range(0, len(X_t), bs):
                idx = perm[i:i + bs]
                logits = model(X_t[idx])
                loss = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_va_seq, val_probs)
            except ValueError:
                auc = 0.5

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Store metadata for prediction
        model._seq_len = seq_len
        model._device = device
        return model

    def predict(self, model, X):
        import torch
        seq_len = getattr(model, "_seq_len", 20)
        device = getattr(model, "_device", "cpu")

        seqs, _ = self._build_sequences(X, np.zeros(len(X)), seq_len)
        if len(seqs) == 0:
            return np.full(len(X), 0.5)

        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(seqs, device=device)
            logits = model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Pad the beginning (no sequences available)
        result = np.full(len(X), 0.5)
        result[seq_len:seq_len + len(probs)] = probs
        return result

    def save(self, model, path, feature_cols):
        import torch
        # Save as TorchScript for ml_serve compatibility
        model.eval()
        model_cpu = model.cpu()
        seq_len = getattr(model, "_seq_len", 20)
        dummy = torch.randn(1, seq_len, len(feature_cols))
        scripted = torch.jit.trace(model_cpu, dummy)
        scripted.save(path)


# ── Transformer (PyTorch) ───────────────────────────────────────────

class TransformerTrainer(ModelTrainer):
    name = "transformer"

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa
            return True
        except ImportError:
            return False

    def _build_sequences(self, X, y, seq_len=20):
        seqs, labels = [], []
        for i in range(seq_len, len(X)):
            seqs.append(X[i - seq_len:i])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, X_train, y_train, X_val, y_val, feature_cols, **kwargs):
        import torch
        import torch.nn as nn

        seq_len = min(20, len(X_train) // 5)
        X_tr_seq, y_tr_seq = self._build_sequences(X_train, y_train, seq_len)
        X_va_seq, y_va_seq = self._build_sequences(X_val, y_val, seq_len)

        if len(X_tr_seq) < 50 or len(X_va_seq) < 10:
            raise ValueError("Not enough data for Transformer sequences")

        n_features = X_train.shape[1]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class FactorTransformer(nn.Module):
            def __init__(self, input_dim, d_model=64, nhead=4, layers=2, dropout=0.2):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=128,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

            def forward(self, x):
                x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
                x = self.encoder(x)
                return self.fc(x[:, -1, :]).squeeze(-1)

        model = FactorTransformer(n_features).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

        pos = y_tr_seq.sum()
        neg = len(y_tr_seq) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_t = torch.tensor(X_tr_seq, device=device)
        y_t = torch.tensor(y_tr_seq, device=device)
        X_v = torch.tensor(X_va_seq, device=device)
        y_v = torch.tensor(y_va_seq, device=device)

        best_auc, patience, wait = 0.0, 15, 0
        best_state = None

        for epoch in range(80):
            model.train()
            bs = min(256, len(X_t))
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), bs):
                idx = perm[i:i + bs]
                logits = model(X_t[idx])
                loss = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy()
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_va_seq, val_probs)
            except ValueError:
                auc = 0.5

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model._seq_len = seq_len
        model._device = device
        return model

    def predict(self, model, X):
        import torch
        seq_len = getattr(model, "_seq_len", 20)
        device = getattr(model, "_device", "cpu")

        seqs, _ = self._build_sequences(X, np.zeros(len(X)), seq_len)
        if len(seqs) == 0:
            return np.full(len(X), 0.5)

        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(seqs, device=device)
            probs = torch.sigmoid(model(X_t)).cpu().numpy()

        result = np.full(len(X), 0.5)
        result[seq_len:seq_len + len(probs)] = probs
        return result

    def save(self, model, path, feature_cols):
        import torch
        model.eval()
        model_cpu = model.cpu()
        seq_len = getattr(model, "_seq_len", 20)
        dummy = torch.randn(1, seq_len, len(feature_cols))
        scripted = torch.jit.trace(model_cpu, dummy)
        scripted.save(path)


# ══════════════════════════════════════════════════════════════════════
# Registry & Competition
# ══════════════════════════════════════════════════════════════════════

TRAINER_REGISTRY = {
    "lgb": LightGBMTrainer,
    "xgb": XGBoostTrainer,
    "catboost": CatBoostTrainer,
    "lstm": LSTMTrainer,
    "transformer": TransformerTrainer,
}

MODEL_EXTENSIONS = {
    "lgb": ".lgb.txt",
    "xgb": ".xgb.json",
    "catboost": ".catboost.bin",
    "lstm": ".lstm.pt",
    "transformer": ".transformer.pt",
}


def walk_forward_cv_single(
    trainer: ModelTrainer,
    X: pd.DataFrame, y: pd.Series,
    n_splits: int = 5, train_ratio: float = 0.6,
) -> dict:
    """Walk-forward CV for a single trainer. Returns summary dict."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    n = len(X)
    fold_size = n // n_splits
    results = []
    feature_cols = list(X.columns)

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
            continue

        try:
            model = trainer.train(X_tr, y_tr, X_va, y_va, feature_cols)
            pred = trainer.predict(model, X_va)
            # Align predictions to validation length for sequence models
            if len(pred) > len(y_va):
                pred = pred[:len(y_va)]
            elif len(pred) < len(y_va):
                # Pad front for sequence models
                y_va = y_va[len(y_va) - len(pred):]
            auc = roc_auc_score(y_va, pred)
            acc = accuracy_score(y_va, (pred > 0.5).astype(int))
            results.append({"fold": fold + 1, "auc": round(auc, 4), "accuracy": round(acc, 4)})
        except Exception as e:
            results.append({"fold": fold + 1, "error": str(e)})

    valid = [r for r in results if "auc" in r]
    if not valid:
        return {"algorithm": trainer.name, "error": "no valid folds", "folds": results}

    avg_auc = np.mean([r["auc"] for r in valid])
    std_auc = np.std([r["auc"] for r in valid])
    return {
        "algorithm": trainer.name,
        "avg_auc": round(avg_auc, 4),
        "std_auc": round(std_auc, 4),
        "n_folds": len(valid),
        "folds": results,
    }


# Default stock list for akshare training — 50 stocks across sectors
DEFAULT_TRAIN_STOCKS = [
    # 白酒/食品 (Consumer Staples)
    "600519", "000858", "000568", "600809", "600887", "002304", "603288",
    # 金融 (Financials)
    "600036", "601318", "601166", "600030", "601398", "601288",
    # 新能源/汽车 (New Energy / Auto)
    "300750", "002594", "600438", "601012", "002460",
    # 医药 (Healthcare)
    "600276", "000333", "300760", "603259", "300122",
    # 科技/电子 (Tech / Electronics)
    "002415", "603501", "300782", "688981", "002049",
    # 消费/家电 (Consumer Discretionary)
    "000651", "600690", "002032", "601888",
    # 周期/材料 (Materials / Industrials)
    "601899", "600031", "600309", "601225", "600585",
    # 公用事业/基建 (Utilities / Infra)
    "600900", "601669", "600048", "601800",
    # 传媒/互联网 (Media / Internet)
    "300059", "002230", "603444",
    # 军工 (Defense)
    "600760", "002179", "600893",
]


def _fetch_akshare_data(
    symbols: Optional[List[str]] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Fetch real A-share daily data from akshare for multiple stocks."""
    import akshare as ak

    stock_list = symbols if symbols else DEFAULT_TRAIN_STOCKS
    print(f"📡 Fetching real data from akshare: {len(stock_list)} stocks, {start_date} ~ {end_date}")

    all_data = []
    for i, sym in enumerate(stock_list):
        # Strip exchange suffix (600519.SH -> 600519)
        code = sym.split(".")[0]
        print(f"  [{i+1}/{len(stock_list)}] {code}...", end=" ", flush=True)
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty or len(df) < 200:
                print(f"skip ({0 if df is None else len(df)} bars)")
                continue
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            all_data.append(df)
            print(f"OK ({len(df)} bars)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not all_data:
        print("⚠️  No data fetched from akshare, falling back to synthetic")
        return generate_synthetic_data(5000)

    combined = pd.concat(all_data, axis=0).sort_index()
    print(f"✅ Total: {len(combined)} bars from {len(all_data)} stocks")
    return combined


# ── Full Training Pipeline ──────────────────────────────────────────

def retrain(
    data_path: Optional[str] = None,
    data_source: str = "synthetic",
    symbols: Optional[List[str]] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    journal_path: str = "data/trade_journal.db",
    output_dir: str = "ml_models",
    n_cv_folds: int = 5,
    horizon: int = 5,
    threshold: float = 0.01,
    algorithms: Optional[List[str]] = None,
    notify_serve: bool = True,
    serve_url: str = "http://127.0.0.1:18091",
) -> dict:
    """
    Full multi-algorithm retrain pipeline:
      1. Load OHLCV data
      2. Collect journal labels
      3. Walk-forward CV for each algorithm
      4. Pick best algorithm
      5. Train final model on all data
      6. Export + notify
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": timestamp, "status": "started"}

    # 1. Load data
    if data_path and os.path.exists(data_path):
        print(f"📂 Loading market data from: {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    elif data_source == "akshare":
        df = _fetch_akshare_data(symbols, start_date, end_date)
    else:
        print("📂 No market data provided, using synthetic (5000 bars)")
        df = generate_synthetic_data(5000)
    report["data_rows"] = len(df)
    report["data_source"] = data_source if not data_path else "csv"

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

    # 4. Determine algorithms to try
    if algorithms is None:
        algorithms = ["lgb"]  # default to LightGBM only

    trainers = []
    for algo in algorithms:
        cls = TRAINER_REGISTRY.get(algo)
        if cls is None:
            print(f"⚠️  Unknown algorithm: {algo}")
            continue
        if not cls.available():
            print(f"⚠️  {algo} not available (dependencies not installed)")
            continue
        trainers.append(cls())

    if not trainers:
        report["status"] = "no_trainers"
        report["error"] = "No valid algorithms available"
        return report

    print(f"\n🏁 Algorithms to compete: {[t.name for t in trainers]}")

    # 5. Walk-forward CV competition
    cv_results = []
    for trainer in trainers:
        print(f"\n{'='*60}")
        print(f"📊 Walk-forward CV: {trainer.name}")
        cv = walk_forward_cv_single(trainer, X, y, n_splits=n_cv_folds)
        cv_results.append(cv)
        if "avg_auc" in cv:
            print(f"   ➜ AUC: {cv['avg_auc']:.4f} ± {cv['std_auc']:.4f}")
        else:
            print(f"   ➜ FAILED: {cv.get('error', 'unknown')}")

    report["cv_competition"] = cv_results

    # Pick best algorithm by avg AUC
    valid_cvs = [c for c in cv_results if "avg_auc" in c]
    if not valid_cvs:
        report["status"] = "all_cv_failed"
        return report

    best_cv = max(valid_cvs, key=lambda c: c["avg_auc"])
    best_algo = best_cv["algorithm"]
    report["best_algorithm"] = best_algo
    print(f"\n🏆 Winner: {best_algo} (AUC={best_cv['avg_auc']:.4f})")

    # 6. Train final model with best algorithm
    best_trainer = next(t for t in trainers if t.name == best_algo)
    split_idx = int(len(X) * 0.85)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(int)

    print(f"\n🏋️ Training final {best_algo} model on full dataset...")
    final_model = best_trainer.train(
        X_arr[:split_idx], y_arr[:split_idx],
        X_arr[split_idx:], y_arr[split_idx:],
        feature_cols,
    )

    val_pred = best_trainer.predict(final_model, X_arr[split_idx:])
    # Align for sequence models
    val_labels = y_arr[split_idx:]
    if len(val_pred) < len(val_labels):
        val_labels = val_labels[len(val_labels) - len(val_pred):]
    elif len(val_pred) > len(val_labels):
        val_pred = val_pred[:len(val_labels)]

    final_auc = roc_auc_score(val_labels, val_pred)
    final_acc = accuracy_score(val_labels, (val_pred > 0.5).astype(int))
    report["final_model"] = {
        "algorithm": best_algo,
        "auc": round(final_auc, 4),
        "accuracy": round(final_acc, 4),
    }
    print(f"✅ Final {best_algo}: AUC={final_auc:.4f}, Acc={final_acc:.4f}")

    # 7. Feature importance
    feat_imp = best_trainer.feature_importance(final_model, feature_cols)
    report["feature_importance"] = feat_imp
    if feat_imp:
        print("\n📊 Top 10 features:")
        for fi in feat_imp[:10]:
            print(f"   {fi['feature']:25s} {fi['importance']:10.1f}")

    # 8. Export
    os.makedirs(output_dir, exist_ok=True)
    ext = MODEL_EXTENSIONS.get(best_algo, ".model")
    model_name = f"factor_model_{timestamp}{ext}"
    model_path = os.path.join(output_dir, model_name)
    best_trainer.save(final_model, model_path, feature_cols)
    report["model_path"] = model_path
    print(f"\n💾 Model saved: {model_path}")

    # Also save to default location for easy loading
    default_path = os.path.join(output_dir, f"factor_model{ext}")
    best_trainer.save(final_model, default_path, feature_cols)

    # For LightGBM, also try ONNX export
    if best_algo == "lgb":
        try:
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            onnx_path = os.path.join(output_dir, f"factor_model_{timestamp}.onnx")
            initial_type = [("features", FloatTensorType([None, len(feature_cols)]))]
            onnx_model = convert_lightgbm(final_model, initial_types=initial_type, target_opset=11)
            onnxmltools.utils.save_model(onnx_model, onnx_path)
            # Also save to default location
            default_onnx = os.path.join(output_dir, "factor_model.onnx")
            onnxmltools.utils.save_model(onnx_model, default_onnx)
            report["onnx_model_path"] = onnx_path
            print(f"💾 ONNX model: {onnx_path}")
            print(f"💾 ONNX default: {default_onnx}")
        except ImportError:
            pass

    # Save report
    report_path = os.path.join(output_dir, f"retrain_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Report: {report_path}")

    # Save feature list
    feat_path = os.path.join(output_dir, "factor_model_features.txt")
    with open(feat_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    # 9. Notify ml_serve to reload
    if notify_serve:
        try:
            import requests
            resp = requests.post(f"{serve_url}/reload", json={"model_path": default_path}, timeout=5)
            if resp.ok:
                print("🔄 ml_serve notified to reload model")
        except Exception as e:
            print(f"⚠️  Could not notify ml_serve: {e}")

    # If we trained multiple models, set up ensemble
    all_models_paths = []
    for trainer in trainers:
        algo = trainer.name
        ext_a = MODEL_EXTENSIONS.get(algo, ".model")
        p = os.path.join(output_dir, f"factor_model_{timestamp}{ext_a}")
        if algo != best_algo:
            try:
                m = trainer.train(
                    X_arr[:split_idx], y_arr[:split_idx],
                    X_arr[split_idx:], y_arr[split_idx:],
                    feature_cols,
                )
                trainer.save(m, p, feature_cols)
                all_models_paths.append({"path": p, "algorithm": algo})
            except Exception:
                pass
        else:
            all_models_paths.append({"path": model_path, "algorithm": algo})
    report["all_models"] = all_models_paths

    report["status"] = "completed"
    print(f"\n🎉 Retrain complete! Best: {best_algo} AUC={final_auc:.4f}")
    return report


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-retrain ML Factor Model (multi-algorithm)")
    parser.add_argument("--data", default=None, help="Path to OHLCV CSV data")
    parser.add_argument("--data-source", default="synthetic", choices=["synthetic", "akshare"],
                        help="Data source: synthetic (random GBM) or akshare (real A-share data)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated stock codes for akshare (e.g. 600519,000858). Default: top 20 A-shares")
    parser.add_argument("--start-date", default="2022-01-01", help="Start date for akshare data (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for akshare data (YYYY-MM-DD)")
    parser.add_argument("--journal", default="data/trade_journal.db", help="Path to trade journal DB")
    parser.add_argument("--output-dir", default="ml_models", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward CV folds")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon (bars)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Positive label threshold")
    parser.add_argument("--algorithms", default="lgb",
                        help="Comma-separated algorithms: lgb,xgb,catboost,lstm,transformer")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--serve-url", default="http://127.0.0.1:18091")

    args = parser.parse_args()

    algos = [a.strip() for a in args.algorithms.split(",")]
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    retrain(
        data_path=args.data,
        data_source=args.data_source,
        symbols=syms,
        start_date=args.start_date,
        end_date=args.end_date,
        journal_path=args.journal,
        output_dir=args.output_dir,
        n_cv_folds=args.folds,
        horizon=args.horizon,
        threshold=args.threshold,
        algorithms=algos,
        notify_serve=not args.no_notify,
        serve_url=args.serve_url,
    )
