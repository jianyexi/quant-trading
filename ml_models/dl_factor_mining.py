#!/usr/bin/env python3
"""
Deep Learning Factor Mining — discovers latent alpha factors using neural networks.

Three complementary approaches:
  1. **FactorAutoencoder** — unsupervised: compress OHLCV features into latent factors,
     then evaluate which latent dimensions predict forward returns (Rank IC).
  2. **AttentionFactorMiner** — supervised: temporal attention over feature windows
     to learn which features × time-lags matter; extract attention weights as factor
     importance and hidden states as latent factors.
  3. **ResidualFactorNet** — supervised: learn non-linear factor combinations that
     predict returns; extract intermediate layer activations as synthetic factors.

All discovered factors are evaluated with Rank IC (from factor_mining.evaluate_factor)
and can be registered into the GP factor registry for lifecycle management.

Usage:
    python dl_factor_mining.py --synthetic                    # synthetic data demo
    python dl_factor_mining.py --data market_data.csv         # from CSV
    python dl_factor_mining.py --akshare --cross-stock        # multi-stock A-share
    python dl_factor_mining.py --synthetic --method all       # run all 3 methods
"""

import argparse
import json
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
from factor_mining import evaluate_factor, compute_all_candidates
from data_utils import load_data, add_data_args

# ── Check PyTorch availability ───────────────────────────────────────

def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch>=2.0")
        return False


# ── 1. Factor Autoencoder ────────────────────────────────────────────

class FactorAutoencoder:
    """
    Unsupervised factor discovery via autoencoder.

    Idea: compress high-dimensional feature space into a low-dimensional
    latent space. Each latent dimension is a "learned factor". We then
    evaluate which latent factors have predictive power (Rank IC) for
    forward returns.

    Architecture:
        Input (n_features) → Encoder → Latent (n_latent) → Decoder → Output (n_features)

    The encoder learns to extract the most informative signal combinations.
    """

    def __init__(self, n_latent: int = 8, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.2,
                 seq_len: int = 20, lr: float = 1e-3, epochs: int = 100):
        self.n_latent = n_latent
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = None
        self.feature_cols = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        class TemporalAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden, n_layers, dropout):
                super().__init__()
                # Encoder: LSTM to capture temporal patterns → project to latent
                self.encoder_lstm = nn.LSTM(
                    input_dim, hidden, n_layers,
                    batch_first=True, dropout=dropout if n_layers > 1 else 0
                )
                self.encoder_proj = nn.Sequential(
                    nn.Linear(hidden, latent_dim),
                    nn.Tanh(),
                )
                # Decoder: latent → LSTM → reconstruct features
                self.decoder_proj = nn.Linear(latent_dim, hidden)
                self.decoder_lstm = nn.LSTM(
                    hidden, hidden, 1,
                    batch_first=True
                )
                self.decoder_out = nn.Linear(hidden, input_dim)

            def encode(self, x):
                # x: (batch, seq_len, n_features)
                out, _ = self.encoder_lstm(x)
                # Use last hidden state
                latent = self.encoder_proj(out[:, -1, :])
                return latent

            def decode(self, latent, seq_len):
                h = self.decoder_proj(latent).unsqueeze(1).repeat(1, seq_len, 1)
                out, _ = self.decoder_lstm(h)
                return self.decoder_out(out)

            def forward(self, x):
                latent = self.encode(x)
                recon = self.decode(latent, x.size(1))
                return recon, latent

        return TemporalAutoencoder(
            n_features, self.n_latent, self.hidden_dim,
            self.n_layers, self.dropout
        )

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        seqs = []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
        return np.array(seqs, dtype=np.float32)

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Train the autoencoder on feature data."""
        import torch
        import torch.nn as nn

        # Compute features
        features = compute_features(df)
        self.feature_cols = list(features.columns)
        X = features.values.astype(np.float32)

        # Normalize
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0) + 1e-8
        X = np.nan_to_num((X - self._mean) / self._std, 0)

        # Build sequences
        X_seq = self._build_sequences(X)
        if len(X_seq) < 100:
            raise ValueError(f"Not enough data: {len(X_seq)} sequences (need ≥100)")

        # Train/val split (temporal)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model(len(self.feature_cols)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, device=self.device)
        X_v = torch.tensor(X_val, device=self.device)

        best_val_loss = float('inf')
        patience, wait = 15, 0
        best_state = None

        if verbose:
            print(f"\n  Autoencoder: {len(self.feature_cols)} features → {self.n_latent} latent factors")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Device: {self.device}")

        for epoch in range(self.epochs):
            self.model.train()
            bs = min(256, len(X_t))
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            n_batch = 0
            for i in range(0, len(X_t), bs):
                idx = perm[i:i + bs]
                recon, _ = self.model(X_t[idx])
                loss = criterion(recon, X_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            self.model.eval()
            with torch.no_grad():
                recon_v, _ = self.model(X_v)
                val_loss = criterion(recon_v, X_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch}")
                    break

            if verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"  Epoch {epoch:3d} | Train: {epoch_loss/n_batch:.6f} | Val: {val_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)

        return {"val_loss": best_val_loss, "epochs": epoch + 1}

    def extract_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract latent factors from trained autoencoder."""
        import torch

        features = compute_features(df)
        X = features.values.astype(np.float32)
        X = np.nan_to_num((X - self._mean) / self._std, 0)
        X_seq = self._build_sequences(X)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, device=self.device)
            _, latent = self.model(X_t)
            latent_np = latent.cpu().numpy()

        # Align with original DataFrame index
        result = pd.DataFrame(
            np.full((len(df), self.n_latent), np.nan),
            index=df.index,
            columns=[f"ae_factor_{i}" for i in range(self.n_latent)],
        )
        result.iloc[self.seq_len:self.seq_len + len(latent_np)] = latent_np
        return result


# ── 2. Attention Factor Miner ────────────────────────────────────────

class AttentionFactorMiner:
    """
    Supervised factor discovery using temporal attention.

    Idea: Train a model to predict forward returns using multi-head attention
    over feature sequences. The attention weights reveal which features at
    which time lags are most important. Hidden states serve as learned factors.

    Bonus: attention weight analysis gives interpretable feature × lag importance.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_latent: int = 8, seq_len: int = 20, dropout: float = 0.2,
                 lr: float = 5e-4, epochs: int = 100):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.seq_len = seq_len
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = None
        self.feature_cols = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        class AttentionFactorNet(nn.Module):
            def __init__(self, input_dim, d_model, n_heads, n_layers,
                         n_latent, seq_len, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                # Factor extraction head: project to latent factors
                self.factor_head = nn.Sequential(
                    nn.Linear(d_model, n_latent),
                    nn.Tanh(),
                )
                # Prediction head: from latent factors to return prediction
                self.pred_head = nn.Sequential(
                    nn.Linear(n_latent, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                )
                # Feature importance: learnable weights per input feature
                self.feature_gate = nn.Parameter(torch.ones(input_dim))

            def forward(self, x, return_factors=False):
                # Apply feature gating
                x = x * torch.sigmoid(self.feature_gate)
                x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
                encoded = self.encoder(x)
                # Use last time step
                factors = self.factor_head(encoded[:, -1, :])
                pred = self.pred_head(factors).squeeze(-1)
                if return_factors:
                    return pred, factors, torch.sigmoid(self.feature_gate)
                return pred

        return AttentionFactorNet(
            n_features, self.d_model, self.n_heads, self.n_layers,
            self.n_latent, self.seq_len, self.dropout
        )

    def _build_sequences(self, X, y):
        seqs, labels = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def fit(self, df: pd.DataFrame, horizon: int = 5, verbose: bool = True) -> Dict:
        """Train attention model to predict forward returns."""
        import torch
        import torch.nn as nn

        features = compute_features(df)
        self.feature_cols = list(features.columns)
        X = features.values.astype(np.float32)

        # Normalize
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0) + 1e-8
        X = np.nan_to_num((X - self._mean) / self._std, 0)

        # Continuous return target (not binary) — better for factor discovery
        fwd_ret = df["close"].pct_change(horizon).shift(-horizon).values.astype(np.float32)
        fwd_ret = np.nan_to_num(fwd_ret, 0)

        X_seq, y_seq = self._build_sequences(X, fwd_ret)
        if len(X_seq) < 100:
            raise ValueError(f"Not enough data: {len(X_seq)} sequences")

        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model(len(self.feature_cols)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, device=self.device)
        y_t = torch.tensor(y_train, device=self.device)
        X_v = torch.tensor(X_val, device=self.device)
        y_v = torch.tensor(y_val, device=self.device)

        best_val_loss = float('inf')
        patience, wait = 15, 0
        best_state = None

        if verbose:
            print(f"\n  Attention Factor Miner: {len(self.feature_cols)} features")
            print(f"  d_model={self.d_model}, heads={self.n_heads}, latent={self.n_latent}")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Device: {self.device}")

        for epoch in range(self.epochs):
            self.model.train()
            bs = min(256, len(X_t))
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            n_batch = 0
            for i in range(0, len(X_t), bs):
                idx = perm[i:i + bs]
                pred = self.model(X_t[idx])
                loss = criterion(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_v)
                val_loss = criterion(val_pred, y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch}")
                    break

            if verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"  Epoch {epoch:3d} | Train: {epoch_loss/n_batch:.6f} | Val: {val_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)

        return {"val_loss": best_val_loss, "epochs": epoch + 1}

    def extract_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract learned factors from trained model."""
        import torch

        features = compute_features(df)
        X = features.values.astype(np.float32)
        X = np.nan_to_num((X - self._mean) / self._std, 0)
        X_seq, _ = self._build_sequences(X, np.zeros(len(X)))

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, device=self.device)
            _, factors, _ = self.model(X_t, return_factors=True)
            factors_np = factors.cpu().numpy()

        result = pd.DataFrame(
            np.full((len(df), self.n_latent), np.nan),
            index=df.index,
            columns=[f"attn_factor_{i}" for i in range(self.n_latent)],
        )
        result.iloc[self.seq_len:self.seq_len + len(factors_np)] = factors_np
        return result

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Get learned feature importance from gate weights."""
        import torch
        self.model.eval()
        with torch.no_grad():
            gates = torch.sigmoid(self.model.feature_gate).cpu().numpy()
        ranked = sorted(zip(self.feature_cols, gates), key=lambda x: -x[1])
        return ranked


# ── 3. Residual Factor Network ───────────────────────────────────────

class ResidualFactorNet:
    """
    Supervised non-linear factor combination via residual network.

    Idea: Learn non-linear combinations of existing factors that predict returns.
    The intermediate activations of hidden layers serve as new synthetic factors.
    Residual connections ensure the network can learn both simple and complex
    factor interactions.

    Unlike the Autoencoder (unsupervised) and Attention (temporal), this is a
    direct "factor combination" approach: it takes the ~205 parametric factors
    as input and learns the best non-linear blends.
    """

    def __init__(self, n_latent: int = 10, hidden_dim: int = 128,
                 n_blocks: int = 3, dropout: float = 0.3,
                 lr: float = 1e-3, epochs: int = 150):
        self.n_latent = n_latent
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = None
        self.factor_names = None

    def _build_model(self, n_features: int):
        import torch
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, dim, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(x + self.net(x))

        class FactorResNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_blocks, n_latent, dropout):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.blocks = nn.ModuleList([
                    ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
                ])
                # Factor bottleneck
                self.factor_layer = nn.Sequential(
                    nn.Linear(hidden_dim, n_latent),
                    nn.Tanh(),
                )
                # Prediction from factors
                self.pred_head = nn.Sequential(
                    nn.Linear(n_latent, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x, return_factors=False):
                h = self.input_proj(x)
                for block in self.blocks:
                    h = block(h)
                factors = self.factor_layer(h)
                pred = self.pred_head(factors).squeeze(-1)
                if return_factors:
                    return pred, factors
                return pred

        return FactorResNet(n_features, self.hidden_dim, self.n_blocks,
                            self.n_latent, self.dropout)

    def fit(self, df: pd.DataFrame, horizon: int = 5, verbose: bool = True) -> Dict:
        """Train ResNet on parametric factors to predict returns."""
        import torch
        import torch.nn as nn

        # Use the full parametric factor set as input
        candidates = compute_all_candidates(df)
        self.factor_names = list(candidates.columns)
        X = candidates.values.astype(np.float32)

        # Normalize
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0) + 1e-8
        X = np.nan_to_num((X - self._mean) / self._std, 0)

        # Forward return target
        fwd_ret = df["close"].pct_change(horizon).shift(-horizon).values.astype(np.float32)
        fwd_ret = np.nan_to_num(fwd_ret, 0)

        # Drop rows with no valid data
        valid = np.isfinite(X).all(axis=1) & np.isfinite(fwd_ret)
        X, fwd_ret = X[valid], fwd_ret[valid]

        if len(X) < 200:
            raise ValueError(f"Not enough valid data: {len(X)} rows")

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = fwd_ret[:split], fwd_ret[split:]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model(len(self.factor_names)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, device=self.device)
        y_t = torch.tensor(y_train, device=self.device)
        X_v = torch.tensor(X_val, device=self.device)
        y_v = torch.tensor(y_val, device=self.device)

        best_val_loss = float('inf')
        patience, wait = 20, 0
        best_state = None

        if verbose:
            print(f"\n  Residual Factor Net: {len(self.factor_names)} input factors → {self.n_latent} latent")
            print(f"  ResBlocks={self.n_blocks}, hidden={self.hidden_dim}")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Device: {self.device}")

        for epoch in range(self.epochs):
            self.model.train()
            bs = min(512, len(X_t))
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            n_batch = 0
            for i in range(0, len(X_t), bs):
                idx = perm[i:i + bs]
                pred = self.model(X_t[idx])
                loss = criterion(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_v)
                val_loss = criterion(val_pred, y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch}")
                    break

            if verbose and (epoch % 30 == 0 or epoch == self.epochs - 1):
                print(f"  Epoch {epoch:3d} | Train: {epoch_loss/n_batch:.8f} | Val: {val_loss:.8f}")

        if best_state:
            self.model.load_state_dict(best_state)

        return {"val_loss": best_val_loss, "epochs": epoch + 1}

    def extract_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract non-linear factor combinations from trained ResNet."""
        import torch

        candidates = compute_all_candidates(df)
        X = candidates.values.astype(np.float32)
        X = np.nan_to_num((X - self._mean) / self._std, 0)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, device=self.device)
            _, factors = self.model(X_t, return_factors=True)
            factors_np = factors.cpu().numpy()

        result = pd.DataFrame(
            factors_np, index=df.index,
            columns=[f"resnet_factor_{i}" for i in range(self.n_latent)],
        )
        return result


# ── Unified Mining Pipeline ──────────────────────────────────────────

def mine_dl_factors(
    df: pd.DataFrame,
    methods: List[str] = None,
    horizon: int = 5,
    n_latent: int = 8,
    ic_threshold: float = 0.02,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run DL factor mining pipeline.

    Args:
        df: OHLCV DataFrame
        methods: list of methods to run: ["autoencoder", "attention", "resnet"]
        horizon: forward return horizon for evaluation
        n_latent: number of latent factors per method
        ic_threshold: minimum |Rank IC| to keep a factor
        verbose: print progress

    Returns:
        DataFrame with columns: [factor_name, ic_mean, ir, method, ...]
    """
    if not _check_torch():
        return pd.DataFrame()

    if methods is None:
        methods = ["autoencoder", "attention", "resnet"]

    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

    all_results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DEEP LEARNING FACTOR MINING")
        print(f"  Methods: {', '.join(methods)}")
        print(f"  Data: {len(df)} bars | Horizon: {horizon} | Latent: {n_latent}")
        print(f"{'='*60}")

    # ── Autoencoder ──────────────────────────────────────────────
    if "autoencoder" in methods:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"  Method 1/3: Factor Autoencoder")
            print(f"{'─'*40}")
        try:
            t0 = time.time()
            ae = FactorAutoencoder(n_latent=n_latent, epochs=80)
            train_info = ae.fit(df, verbose=verbose)
            factors = ae.extract_factors(df)

            for col in factors.columns:
                ev = evaluate_factor(factors[col].dropna(), fwd_ret)
                ev["factor_name"] = col
                ev["method"] = "autoencoder"
                ev["train_loss"] = train_info["val_loss"]
                all_results.append(ev)

            elapsed = time.time() - t0
            if verbose:
                print(f"  Autoencoder done ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  Autoencoder failed: {e}")

    # ── Attention ────────────────────────────────────────────────
    if "attention" in methods:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"  Method 2/3: Attention Factor Miner")
            print(f"{'─'*40}")
        try:
            t0 = time.time()
            attn = AttentionFactorMiner(n_latent=n_latent, epochs=80)
            train_info = attn.fit(df, horizon=horizon, verbose=verbose)
            factors = attn.extract_factors(df)

            for col in factors.columns:
                ev = evaluate_factor(factors[col].dropna(), fwd_ret)
                ev["factor_name"] = col
                ev["method"] = "attention"
                ev["train_loss"] = train_info["val_loss"]
                all_results.append(ev)

            # Feature importance from attention gates
            if verbose:
                importance = attn.feature_importance()
                print(f"\n  Feature importance (attention gate):")
                for name, score in importance[:10]:
                    bar = "#" * int(score * 30)
                    print(f"    {name:>25}: {score:.3f} {bar}")

            elapsed = time.time() - t0
            if verbose:
                print(f"  Attention done ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  Attention failed: {e}")

    # ── Residual Factor Net ──────────────────────────────────────
    if "resnet" in methods:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"  Method 3/3: Residual Factor Network")
            print(f"{'─'*40}")
        try:
            t0 = time.time()
            rnet = ResidualFactorNet(n_latent=n_latent, epochs=100)
            train_info = rnet.fit(df, horizon=horizon, verbose=verbose)
            factors = rnet.extract_factors(df)

            for col in factors.columns:
                ev = evaluate_factor(factors[col].dropna(), fwd_ret)
                ev["factor_name"] = col
                ev["method"] = "resnet"
                ev["train_loss"] = train_info["val_loss"]
                all_results.append(ev)

            elapsed = time.time() - t0
            if verbose:
                print(f"  ResNet done ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  ResNet failed: {e}")

    # ── Aggregate results ────────────────────────────────────────
    if not all_results:
        if verbose:
            print("\n  No factors discovered!")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df["abs_ic"] = results_df["ic_mean"].abs()
    results_df.sort_values("abs_ic", ascending=False, inplace=True)
    results_df = results_df[results_df["abs_ic"] >= ic_threshold].reset_index(drop=True)

    if verbose:
        n_total = len(all_results)
        n_pass = len(results_df)
        print(f"\n{'='*60}")
        print(f"  DL FACTOR MINING RESULTS")
        print(f"  Total: {n_total} | Passed IC>{ic_threshold}: {n_pass}")
        print(f"{'='*60}\n")

        print(f"{'Rank':>4} {'Factor':>25} {'Method':>12} {'IC':>8} {'IR':>8} {'Turn':>6}")
        print("-" * 70)
        for i, row in results_df.head(20).iterrows():
            print(f"{i+1:4d} {row['factor_name']:>25} {row['method']:>12} "
                  f"{row['ic_mean']:8.4f} {row['ir']:8.3f} {row['turnover']:6.3f}")

    return results_df


# ── Export & Save ────────────────────────────────────────────────────

def save_dl_factors(results: pd.DataFrame, output_dir: str = "ml_models"):
    """Save DL-mined factor report."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "n_factors": len(results),
        "methods": list(results["method"].unique()) if len(results) > 0 else [],
        "factors": results.to_dict(orient="records") if len(results) > 0 else [],
    }
    report_path = out / f"dl_factor_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Factor Mining")
    add_data_args(parser)
    parser.add_argument("--method", type=str, default="all",
                        help="Methods: autoencoder, attention, resnet, all")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon")
    parser.add_argument("--n-latent", type=int, default=8,
                        help="Number of latent factors per method")
    parser.add_argument("--ic-threshold", type=float, default=0.02,
                        help="Minimum |Rank IC| to keep")
    parser.add_argument("--output-dir", type=str, default="ml_models",
                        help="Output directory")
    args = parser.parse_args()

    df = load_data(args)

    methods = args.method.split(",") if args.method != "all" else None
    results = mine_dl_factors(
        df, methods=methods, horizon=args.horizon,
        n_latent=args.n_latent, ic_threshold=args.ic_threshold,
    )

    if len(results) > 0:
        save_dl_factors(results, args.output_dir)


if __name__ == "__main__":
    main()
