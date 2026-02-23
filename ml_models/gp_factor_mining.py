#!/usr/bin/env python3
"""
Genetic Programming Factor Mining — Phase 2

Evolves expression trees to discover novel alpha factors.
Includes a factor registry for lifecycle management:
  candidate → validated → promoted → retired

Pipeline:
  1. GP evolution: random trees → crossover/mutation → IC fitness
  2. Evaluate discovered factors: IC, IR, turnover, decay
  3. Register in factor_registry.json with lifecycle tracking
  4. Self-manage: periodic re-evaluation, auto-promote/demote/retire
  5. Export: best factors → Rust snippet + retrain ML model

Usage:
    python ml_models/gp_factor_mining.py --synthetic --generations 50
    python ml_models/gp_factor_mining.py --data market_data.csv --pop-size 500
    python ml_models/gp_factor_mining.py --manage              # run lifecycle management
    python ml_models/gp_factor_mining.py --export-promoted      # export promoted factors
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from train_factor_model import generate_synthetic_data
from factor_mining import evaluate_factor, retrain_with_factors
from data_utils import load_data, add_data_args

# ── GP Primitives ────────────────────────────────────────────────────

# Binary operators
def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mul(a, b): return a * b
def _div(a, b): return np.where(np.abs(b) > 1e-10, a / b, 0.0)
def _max(a, b): return np.maximum(a, b)
def _min(a, b): return np.minimum(a, b)

# Unary operators
def _neg(a): return -a
def _abs(a): return np.abs(a)
def _log(a): return np.where(a > 1e-10, np.log(a), 0.0)
def _sqrt(a): return np.where(a > 1e-10, np.sqrt(a), 0.0)
def _sign(a): return np.sign(a)
def _inv(a): return np.where(np.abs(a) > 1e-10, 1.0 / a, 0.0)
def _square(a): return a * a
def _clip(a): return np.clip(a, -3.0, 3.0)  # bound to ±3 std

# Rolling operators (unary with window)
def _rolling_mean(a, w):
    s = pd.Series(a)
    return s.rolling(w, min_periods=1).mean().values

def _rolling_std(a, w):
    s = pd.Series(a)
    return s.rolling(w, min_periods=2).std().fillna(0).values

def _rolling_max(a, w):
    s = pd.Series(a)
    return s.rolling(w, min_periods=1).max().values

def _rolling_min(a, w):
    s = pd.Series(a)
    return s.rolling(w, min_periods=1).min().values

def _delay(a, d):
    s = pd.Series(a)
    return s.shift(d).fillna(method="bfill").values

def _delta(a, d):
    s = pd.Series(a)
    return s.diff(d).fillna(0).values

def _ts_rank(a, w):
    """Rolling percentile rank within window."""
    s = pd.Series(a)
    return s.rolling(w, min_periods=1).rank(pct=True).values

def _ts_skew(a, w):
    """Rolling skewness within window."""
    s = pd.Series(a)
    return s.rolling(w, min_periods=3).skew().fillna(0).values

def _ema(a, w):
    """Exponential moving average."""
    s = pd.Series(a)
    return s.ewm(span=w, min_periods=1).mean().values

def _decay_linear(a, w):
    """Linearly decaying weighted mean."""
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    s = pd.Series(a)
    return s.rolling(w, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) if len(x) > 0 else 0,
        raw=True
    ).values

def _rank(a):
    s = pd.Series(a)
    return s.rank(pct=True).values

def _ts_corr(a, b, w):
    sa, sb = pd.Series(a), pd.Series(b)
    return sa.rolling(w, min_periods=5).corr(sb).fillna(0).values


BINARY_OPS = {
    "add": (_add, "+"),
    "sub": (_sub, "-"),
    "mul": (_mul, "*"),
    "div": (_div, "/"),
    "max": (_max, "max"),
    "min": (_min, "min"),
}

UNARY_OPS = {
    "neg": (_neg, "neg"),
    "abs": (_abs, "abs"),
    "log": (_log, "log"),
    "sqrt": (_sqrt, "sqrt"),
    "sign": (_sign, "sign"),
    "inv": (_inv, "inv"),
    "square": (_square, "square"),
    "clip": (_clip, "clip"),
}

ROLLING_OPS = {
    "ts_mean": (_rolling_mean, "ts_mean"),
    "ts_std": (_rolling_std, "ts_std"),
    "ts_max": (_rolling_max, "ts_max"),
    "ts_min": (_rolling_min, "ts_min"),
    "delay": (_delay, "delay"),
    "delta": (_delta, "delta"),
    "ts_rank": (_ts_rank, "ts_rank"),
    "ts_skew": (_ts_skew, "ts_skew"),
    "ema": (_ema, "ema"),
    "decay_linear": (_decay_linear, "decay_linear"),
}

# Raw price terminals are excluded by default to prevent price-level bias.
# GP should discover factors from normalized data (returns, ratios, ranges).
TERMINALS = ["returns", "volume", "vwap", "tr",
             "hl_ratio", "co_ratio", "cl_ratio", "vol_chg",
             "mom_5", "mom_20", "vol_x_ret"]  # composite terminals
WINDOWS = [2, 3, 5, 10, 20, 30, 60, 120]


# ── Expression Tree ──────────────────────────────────────────────────

class Node:
    """Expression tree node."""
    __slots__ = ["op_type", "op_name", "children", "terminal", "window"]

    def __init__(self, op_type: str, op_name: str = "",
                 children: list = None, terminal: str = "",
                 window: int = 5):
        self.op_type = op_type      # "binary", "unary", "rolling", "terminal"
        self.op_name = op_name
        self.children = children or []
        self.terminal = terminal
        self.window = window

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "Node":
        return Node(
            self.op_type, self.op_name,
            [c.copy() for c in self.children],
            self.terminal, self.window
        )

    def to_expr(self) -> str:
        if self.op_type == "terminal":
            return self.terminal
        if self.op_type == "unary":
            return f"{self.op_name}({self.children[0].to_expr()})"
        if self.op_type == "rolling":
            return f"{self.op_name}({self.children[0].to_expr()}, {self.window})"
        if self.op_type == "binary":
            sym = BINARY_OPS.get(self.op_name, (None, self.op_name))[1]
            if sym in "+-*/":
                return f"({self.children[0].to_expr()} {sym} {self.children[1].to_expr()})"
            return f"{sym}({self.children[0].to_expr()}, {self.children[1].to_expr()})"
        return "?"

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.op_type == "terminal":
            return data[self.terminal].astype(np.float64)

        if self.op_type == "unary":
            a = self.children[0].evaluate(data)
            fn = UNARY_OPS[self.op_name][0]
            return fn(a)

        if self.op_type == "rolling":
            a = self.children[0].evaluate(data)
            fn = ROLLING_OPS[self.op_name][0]
            return fn(a, self.window)

        if self.op_type == "binary":
            a = self.children[0].evaluate(data)
            b = self.children[1].evaluate(data)
            fn = BINARY_OPS[self.op_name][0]
            return fn(a, b)

        return np.zeros(len(data["close"]))


def random_terminal() -> Node:
    return Node("terminal", terminal=random.choice(TERMINALS))


def random_tree(max_depth: int = 4) -> Node:
    """Generate a random expression tree."""
    if max_depth <= 1 or random.random() < 0.3:
        return random_terminal()

    r = random.random()
    if r < 0.35:
        # Binary op
        op = random.choice(list(BINARY_OPS.keys()))
        left = random_tree(max_depth - 1)
        right = random_tree(max_depth - 1)
        return Node("binary", op, [left, right])
    elif r < 0.6:
        # Unary op
        op = random.choice(list(UNARY_OPS.keys()))
        child = random_tree(max_depth - 1)
        return Node("unary", op, [child])
    elif r < 0.85:
        # Rolling op
        op = random.choice(list(ROLLING_OPS.keys()))
        child = random_tree(max_depth - 1)
        w = random.choice(WINDOWS)
        return Node("rolling", op, [child], window=w)
    else:
        return random_terminal()


# ── GP Operations ────────────────────────────────────────────────────

def _all_nodes(node: Node) -> List[Tuple[Node, int]]:
    """Collect all nodes with their depth."""
    result = [(node, 0)]
    for c in node.children:
        for n, d in _all_nodes(c):
            result.append((n, d + 1))
    return result


def _random_subtree(node: Node) -> Tuple[Node, Optional[Node], int]:
    """Return (parent, child_to_replace, child_index). If root selected, parent is None."""
    nodes = _all_nodes(node)
    if len(nodes) <= 1:
        return None, node, -1
    # Pick non-root
    idx = random.randint(1, len(nodes) - 1)
    target = nodes[idx][0]
    # Find parent
    def find_parent(n, t):
        for i, c in enumerate(n.children):
            if c is t:
                return n, i
            result = find_parent(c, t)
            if result:
                return result
        return None
    result = find_parent(node, target)
    if result:
        return result[0], target, result[1]
    return None, node, -1


def crossover(p1: Node, p2: Node, max_depth: int = 6) -> Node:
    """Subtree crossover: replace random subtree of p1 with random subtree of p2."""
    child = p1.copy()
    donor = p2.copy()

    # Get subtree from donor
    donor_nodes = _all_nodes(donor)
    donor_sub = random.choice(donor_nodes)[0].copy()

    # Pick insertion point in child
    parent, _, idx = _random_subtree(child)
    if parent is None:
        child = donor_sub
    else:
        parent.children[idx] = donor_sub

    # Depth control
    if child.depth() > max_depth:
        return p1.copy()
    return child


def mutate(node: Node, max_depth: int = 6) -> Node:
    """Mutation: replace random subtree with new random tree."""
    child = node.copy()
    r = random.random()

    if r < 0.4:
        # Subtree replacement
        parent, _, idx = _random_subtree(child)
        new_sub = random_tree(max_depth=3)
        if parent is None:
            child = new_sub
        else:
            parent.children[idx] = new_sub
    elif r < 0.7:
        # Point mutation: change operator
        nodes = _all_nodes(child)
        op_nodes = [(n, d) for n, d in nodes if n.op_type != "terminal"]
        if op_nodes:
            target = random.choice(op_nodes)[0]
            if target.op_type == "binary":
                target.op_name = random.choice(list(BINARY_OPS.keys()))
            elif target.op_type == "unary":
                target.op_name = random.choice(list(UNARY_OPS.keys()))
            elif target.op_type == "rolling":
                target.op_name = random.choice(list(ROLLING_OPS.keys()))
                target.window = random.choice(WINDOWS)
    else:
        # Terminal mutation: swap terminal
        nodes = _all_nodes(child)
        term_nodes = [n for n, _ in nodes if n.op_type == "terminal"]
        if term_nodes:
            target = random.choice(term_nodes)
            target.terminal = random.choice(TERMINALS)

    if child.depth() > max_depth:
        return node.copy()
    return child


# ── Fitness Evaluation ───────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Prepare terminal data arrays from OHLCV DataFrame.

    All terminals are normalized (ratios, returns, ranges) to prevent
    price-level bias.  Raw close/open/high/low are NOT exposed.
    """
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    ret = np.diff(c, prepend=c[0]) / np.maximum(np.abs(c), 1e-10)
    vwap = np.where(v > 0, (h + l + c) / 3.0, c)  # simplified VWAP
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))

    # Normalized terminals — immune to absolute price level
    hl_ratio = (h - l) / np.maximum(c, 1e-10)          # intraday range / close
    co_ratio = (c - o) / np.maximum(np.abs(o), 1e-10)  # close-to-open return
    cl_ratio = c / np.maximum(np.roll(c, 1), 1e-10)    # close / prev_close
    vol_chg = np.diff(v, prepend=v[0]) / np.maximum(np.abs(v), 1e-10)  # volume change rate

    # Normalize vwap and tr to ratios so they are scale-free
    vwap_ratio = c / np.maximum(vwap, 1e-10)           # close / vwap
    tr_norm = tr / np.maximum(c, 1e-10)                # true range / close

    result = {
        "returns": ret, "volume": v, "vwap": vwap_ratio, "tr": tr_norm,
        "hl_ratio": hl_ratio, "co_ratio": co_ratio,
        "cl_ratio": cl_ratio, "vol_chg": vol_chg,
    }

    # Composite terminals — pre-computed interactions for GP to combine
    mom_5 = pd.Series(c).pct_change(5).fillna(0).values
    mom_20 = pd.Series(c).pct_change(20).fillna(0).values
    vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values
    vol_ratio = v / np.maximum(vol_ma, 1e-10)
    result["mom_5"] = mom_5
    result["mom_20"] = mom_20
    result["vol_x_ret"] = vol_ratio * ret  # volume-weighted return

    # Legacy aliases for backward compatibility with existing registry expressions
    result["close"] = c
    result["open"] = o
    result["high"] = h
    result["low"] = l
    return result


def evaluate_tree_fitness(
    tree: Node,
    data: Dict[str, np.ndarray],
    fwd_ret: pd.Series,
    parsimony_coeff: float = 0.001,
    close_prices: Optional[np.ndarray] = None,
) -> float:
    """
    Fitness = |IC| - parsimony_coeff * tree_size - price_level_penalty
    Returns negative fitness for invalid trees.
    """
    try:
        values = tree.evaluate(data)
        # Sanitize
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(values) < 1e-10:
            return -1.0

        factor = pd.Series(values, index=fwd_ret.index)
        ev = evaluate_factor(factor, fwd_ret)
        ic = abs(ev["ic_mean"])
        if np.isnan(ic):
            return -1.0

        # Parsimony pressure: penalize large trees
        fitness = ic - parsimony_coeff * tree.size()

        # Price-level penalty: if factor rank-correlates >0.8 with raw close,
        # it's likely tracking price levels rather than signals.
        if close_prices is not None:
            from scipy.stats import spearmanr
            mask = np.isfinite(values) & np.isfinite(close_prices)
            if mask.sum() > 60:
                corr, _ = spearmanr(values[mask], close_prices[mask])
                if abs(corr) > 0.8:
                    fitness *= 0.1  # heavy penalty
        return fitness

    except Exception:
        return -1.0


# ── Evolution Engine ─────────────────────────────────────────────────

def tournament_select(population: List[Tuple[Node, float]], k: int = 3) -> Node:
    """Tournament selection: pick best from k random individuals."""
    tournament = random.sample(population, min(k, len(population)))
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]


def evolve(
    df: pd.DataFrame,
    pop_size: int = 500,
    generations: int = 100,
    max_depth: int = 8,
    tournament_size: int = 5,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    elite_ratio: float = 0.05,
    parsimony_coeff: float = 0.001,
    horizon: int = 5,
    ic_threshold: float = 0.03,
    verbose: bool = True,
) -> List[Tuple[Node, Dict]]:
    """
    Main GP evolution loop.

    Returns: list of (tree, evaluation_dict) for discovered factors.
    """
    data = prepare_data(df)
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    close_prices = df["close"].values.astype(np.float64)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GP FACTOR EVOLUTION")
        print(f"  Population: {pop_size} | Generations: {generations}")
        print(f"  Max depth: {max_depth} | Horizon: {horizon}")
        print(f"{'='*60}\n")

    # Initialize population
    population = []
    for _ in range(pop_size):
        tree = random_tree(max_depth=max_depth)
        fitness = evaluate_tree_fitness(tree, data, fwd_ret, parsimony_coeff, close_prices)
        population.append((tree, fitness))

    # Track unique discovered factors
    discovered = {}  # expr → (tree, eval_dict)
    best_fitness_history = []
    n_elite = max(1, int(pop_size * elite_ratio))

    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)
        best_fit = population[0][1]
        avg_fit = np.mean([f for _, f in population if f > -1])
        best_fitness_history.append(best_fit)

        if verbose and (gen % 5 == 0 or gen == generations - 1):
            best_expr = population[0][0].to_expr()
            if len(best_expr) > 60:
                best_expr = best_expr[:57] + "..."
            print(f"  Gen {gen:3d}/{generations} | Best: {best_fit:.4f} | "
                  f"Avg: {avg_fit:.4f} | Pop: {len(population)} | "
                  f"Found: {len(discovered)}")

        # Collect good individuals
        for tree, fit in population:
            if fit >= ic_threshold:
                expr = tree.to_expr()
                if expr not in discovered or fit > discovered[expr][1].get("fitness", -1):
                    # Full evaluation
                    try:
                        values = tree.evaluate(data)
                        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                        factor = pd.Series(values, index=fwd_ret.index)
                        ev = evaluate_factor(factor, fwd_ret)
                        ev["expression"] = expr
                        ev["tree_size"] = tree.size()
                        ev["tree_depth"] = tree.depth()
                        ev["fitness"] = fit
                        discovered[expr] = (tree.copy(), ev)
                    except Exception:
                        pass

        # Elitism: keep top N
        next_pop = [(t.copy(), f) for t, f in population[:n_elite]]

        # Breed new generation
        while len(next_pop) < pop_size:
            r = random.random()
            if r < crossover_rate:
                p1 = tournament_select(population, tournament_size)
                p2 = tournament_select(population, tournament_size)
                child = crossover(p1, p2, max_depth)
            elif r < crossover_rate + mutation_rate:
                parent = tournament_select(population, tournament_size)
                child = mutate(parent, max_depth)
            else:
                # Reproduction
                child = tournament_select(population, tournament_size).copy()

            fitness = evaluate_tree_fitness(child, data, fwd_ret, parsimony_coeff, close_prices)
            next_pop.append((child, fitness))

        population = next_pop

    # Final results
    results = []
    for expr, (tree, ev) in discovered.items():
        if ev.get("ic_mean", 0) != 0:
            results.append((tree, ev))

    results.sort(key=lambda x: abs(x[1]["ic_mean"]), reverse=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EVOLUTION COMPLETE: {len(results)} factors discovered")
        print(f"  Best fitness: {best_fitness_history[-1]:.4f}")
        print(f"{'='*60}\n")

    return results


# ── Factor Registry ──────────────────────────────────────────────────

REGISTRY_PATH = Path(__file__).parent / "factor_registry.json"

# Lifecycle states
STATE_CANDIDATE = "candidate"       # Newly discovered, needs validation
STATE_VALIDATED = "validated"        # Passed initial IC/IR checks
STATE_PROMOTED  = "promoted"         # In active use by ML model
STATE_RETIRED   = "retired"          # IC decayed, removed from model

# Auto-promotion thresholds
PROMOTE_MIN_VALIDATIONS = 3          # Must pass N consecutive validations
PROMOTE_MIN_IC = 0.03                # Minimum average IC
RETIRE_IC_THRESHOLD = 0.01           # Below this → retire
RETIRE_DECAY_CHECKS = 3             # Must fail N consecutive checks


def load_registry() -> Dict:
    """Load or initialize factor registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 2,
        "created": datetime.now().isoformat(),
        "factors": {},
        "stats": {
            "total_discovered": 0,
            "total_promoted": 0,
            "total_retired": 0,
        },
    }


def save_registry(registry: Dict):
    """Persist registry to disk."""
    registry["updated"] = datetime.now().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def register_factor(
    registry: Dict,
    factor_id: str,
    expression: str,
    evaluation: Dict,
    source: str = "gp",
) -> bool:
    """Register a newly discovered factor. Returns True if new."""
    if factor_id in registry["factors"]:
        # Update existing
        entry = registry["factors"][factor_id]
        entry["ic_history"].append({
            "timestamp": datetime.now().isoformat(),
            "ic": evaluation.get("ic_mean", 0),
            "ir": evaluation.get("ir", 0),
        })
        return False

    registry["factors"][factor_id] = {
        "expression": expression,
        "source": source,
        "state": STATE_CANDIDATE,
        "created": datetime.now().isoformat(),
        "tree_size": evaluation.get("tree_size", 0),
        "tree_depth": evaluation.get("tree_depth", 0),
        "ic_mean": evaluation.get("ic_mean", 0),
        "ir": evaluation.get("ir", 0),
        "ic_pos_rate": evaluation.get("ic_pos_rate", 0),
        "turnover": evaluation.get("turnover", 0),
        "decay": evaluation.get("decay", 0),
        "low_turnover_warning": evaluation.get("turnover", 1) < 0.02,
        "ic_history": [{
            "timestamp": datetime.now().isoformat(),
            "ic": evaluation.get("ic_mean", 0),
            "ir": evaluation.get("ir", 0),
        }],
        "validation_count": 0,
        "fail_count": 0,
        "last_validated": None,
        "promoted_at": None,
        "retired_at": None,
    }
    registry["stats"]["total_discovered"] += 1
    return True


def _factor_id_from_expr(expr: str) -> str:
    """Generate a stable ID from expression string."""
    import hashlib
    h = hashlib.md5(expr.encode()).hexdigest()[:8]
    # Create a short readable prefix
    for prefix in ["ts_rank", "ts_skew", "ema", "decay_linear",
                   "ts_mean", "ts_std", "div", "mul", "sub", "add",
                   "log", "delta", "delay", "square", "clip"]:
        if prefix in expr:
            return f"gp_{prefix}_{h}"
    return f"gp_{h}"


# ── Self-Management ──────────────────────────────────────────────────

def manage_lifecycle(
    registry: Dict,
    df: pd.DataFrame,
    horizon: int = 5,
    verbose: bool = True,
    oos_ratio: float = 0.3,
) -> Dict[str, int]:
    """
    Run lifecycle management on all registered factors:
      1. Split data temporally: first (1-oos_ratio) for training, last oos_ratio for validation
      2. Re-evaluate each factor's IC on OOS data only
      3. Auto-promote: candidate/validated → promoted if OOS IC stable
      4. Auto-demote: promoted → retired if OOS IC decayed
      5. Prune: remove highly correlated promoted factors

    Returns: counts of promotions, demotions, retirements
    """
    # Temporal split: decisions based on OOS portion only
    n = len(df)
    split_idx = int(n * (1 - oos_ratio))
    df_oos = df.iloc[split_idx:].copy()

    data_oos = prepare_data(df_oos)
    fwd_ret_oos = df_oos["close"].pct_change(horizon).shift(-horizon)

    # Also keep full data for pruning correlation check
    data_full = prepare_data(df)
    now = datetime.now().isoformat()

    counts = {"promoted": 0, "retired": 0, "validated": 0, "pruned": 0}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  FACTOR LIFECYCLE MANAGEMENT (OOS split={oos_ratio:.0%})")
        print(f"  Registered: {len(registry['factors'])} factors")
        print(f"  Total bars: {n}, OOS bars: {n - split_idx}")
        print(f"{'='*60}\n")

    for fid, entry in list(registry["factors"].items()):
        if entry["state"] == STATE_RETIRED:
            continue

        expr = entry["expression"]

        # Parse and evaluate on OOS data
        try:
            tree = parse_expression(expr)
            if tree is None:
                continue
            values = tree.evaluate(data_oos)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if np.std(values) < 1e-10:
                entry["fail_count"] += 1
                continue

            factor = pd.Series(values, index=fwd_ret_oos.index)
            ev = evaluate_factor(factor, fwd_ret_oos)
            current_ic = abs(ev["ic_mean"])

            # Also compute in-sample IC for diagnostics
            data_is = prepare_data(df.iloc[:split_idx])
            fwd_is = df.iloc[:split_idx]["close"].pct_change(horizon).shift(-horizon)
            vals_is = tree.evaluate(data_is)
            vals_is = np.nan_to_num(vals_is, nan=0.0, posinf=0.0, neginf=0.0)
            ev_is = evaluate_factor(pd.Series(vals_is, index=fwd_is.index), fwd_is)

            entry["ic_history"].append({
                "timestamp": now,
                "ic_oos": ev["ic_mean"],
                "ir_oos": ev["ir"],
                "ic_is": ev_is["ic_mean"],
                "ir_is": ev_is["ir"],
                # Legacy fields for backward compat
                "ic": ev["ic_mean"],
                "ir": ev["ir"],
            })
            entry["last_validated"] = now

        except Exception as e:
            entry["fail_count"] += 1
            if verbose:
                print(f"  ⚠️  {fid}: evaluation failed ({e})")
            continue

        # State transitions — based on OOS IC only
        if entry["state"] in (STATE_CANDIDATE, STATE_VALIDATED):
            if current_ic >= PROMOTE_MIN_IC:
                entry["validation_count"] += 1
                entry["fail_count"] = 0
                if entry["state"] == STATE_CANDIDATE:
                    entry["state"] = STATE_VALIDATED
                    counts["validated"] += 1
                    if verbose:
                        print(f"  ✓ {fid}: candidate → validated (OOS IC={current_ic:.4f})")

                if entry["validation_count"] >= PROMOTE_MIN_VALIDATIONS:
                    entry["state"] = STATE_PROMOTED
                    entry["promoted_at"] = now
                    registry["stats"]["total_promoted"] += 1
                    counts["promoted"] += 1
                    if verbose:
                        print(f"  ⬆ {fid}: validated → PROMOTED (OOS IC={current_ic:.4f}, "
                              f"validations={entry['validation_count']})")
            else:
                entry["fail_count"] += 1
                if verbose:
                    print(f"  ✗ {fid}: OOS IC={current_ic:.4f} below threshold")

        elif entry["state"] == STATE_PROMOTED:
            if current_ic < RETIRE_IC_THRESHOLD:
                entry["fail_count"] += 1
                if entry["fail_count"] >= RETIRE_DECAY_CHECKS:
                    entry["state"] = STATE_RETIRED
                    entry["retired_at"] = now
                    registry["stats"]["total_retired"] += 1
                    counts["retired"] += 1
                    if verbose:
                        print(f"  ⬇ {fid}: promoted → RETIRED (OOS IC={current_ic:.4f}, "
                              f"fails={entry['fail_count']})")
                elif verbose:
                    print(f"  ⚠ {fid}: IC declining ({current_ic:.4f}), "
                          f"fail {entry['fail_count']}/{RETIRE_DECAY_CHECKS}")
            else:
                entry["fail_count"] = 0
                entry["validation_count"] += 1

    # Prune correlated promoted factors (using full data for stable estimates)
    promoted = {fid: e for fid, e in registry["factors"].items()
                if e["state"] == STATE_PROMOTED}

    if len(promoted) > 1:
        promoted_sorted = sorted(promoted.items(),
                                  key=lambda x: abs(x[1]["ic_mean"]), reverse=True)
        kept_ids = []
        kept_values = {}

        for fid, entry in promoted_sorted:
            try:
                tree = parse_expression(entry["expression"])
                if tree is None:
                    continue
                vals = tree.evaluate(data_full)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                s = pd.Series(vals)

                too_corr = False
                for kid, kvals in kept_values.items():
                    corr = s.corr(pd.Series(kvals))
                    if abs(corr) > 0.7:
                        too_corr = True
                        break

                if not too_corr:
                    kept_ids.append(fid)
                    kept_values[fid] = vals
                else:
                    entry["state"] = STATE_RETIRED
                    entry["retired_at"] = now
                    counts["pruned"] += 1
                    if verbose:
                        print(f"  🔄 {fid}: pruned (correlated with {kid})")
            except Exception:
                pass

    if verbose:
        print(f"\n  Summary: +{counts['promoted']} promoted, "
              f"-{counts['retired']} retired, ~{counts['pruned']} pruned")

    return counts


# ── Expression Parser (for re-evaluation from registry) ──────────────

def parse_expression(expr: str) -> Optional[Node]:
    """
    Parse expression string back into a Node tree.
    Supports: (a + b), func(a), ts_func(a, w)
    """
    expr = expr.strip()
    if not expr:
        return None

    # Terminal
    # Also accept legacy terminal names (close, open, high, low) for backward compat
    LEGACY_TERMINALS = TERMINALS + ["close", "open", "high", "low"]
    if expr in LEGACY_TERMINALS:
        return Node("terminal", terminal=expr)

    # Parenthesized binary: (expr op expr)
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        # Find the operator at the correct nesting level
        depth = 0
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and i > 0 and i < len(inner) - 1:
                for op_sym in [" + ", " - ", " * ", " / "]:
                    if inner[i:i+len(op_sym)] == op_sym:
                        left_str = inner[:i]
                        right_str = inner[i+len(op_sym):]
                        left = parse_expression(left_str)
                        right = parse_expression(right_str)
                        if left and right:
                            op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
                            return Node("binary", op_map[op_sym.strip()], [left, right])
        # If we can't parse as binary, try inner as expression
        return parse_expression(inner)

    # Function call: func(args)
    paren_idx = expr.find("(")
    if paren_idx > 0 and expr.endswith(")"):
        func_name = expr[:paren_idx]
        args_str = expr[paren_idx+1:-1]

        # Split args respecting nesting
        args = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())

        if func_name in UNARY_OPS and len(args) == 1:
            child = parse_expression(args[0])
            if child:
                return Node("unary", func_name, [child])

        if func_name in ROLLING_OPS and len(args) == 2:
            child = parse_expression(args[0])
            try:
                window = int(args[1])
            except ValueError:
                window = 5
            if child:
                return Node("rolling", func_name, [child], window=window)

        if func_name in ("max", "min") and len(args) == 2:
            left = parse_expression(args[0])
            right = parse_expression(args[1])
            if left and right:
                return Node("binary", func_name, [left, right])

    return None


# ── Export ────────────────────────────────────────────────────────────

def export_promoted_factors(
    registry: Dict,
    df: pd.DataFrame,
    output_dir: str = "ml_models",
    retrain: bool = False,
):
    """Export all promoted factors for ML integration."""
    promoted = {fid: e for fid, e in registry["factors"].items()
                if e["state"] == STATE_PROMOTED}

    if not promoted:
        print("⚠️  No promoted factors to export")
        return

    out = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = prepare_data(df)

    print(f"\n📦 Exporting {len(promoted)} promoted GP factors...\n")

    # Compute factor values
    factor_df = pd.DataFrame(index=df.index)
    factor_names = []
    factor_exprs = {}

    for fid, entry in sorted(promoted.items(), key=lambda x: abs(x[1]["ic_mean"]), reverse=True):
        try:
            tree = parse_expression(entry["expression"])
            if tree is None:
                continue
            vals = tree.evaluate(data)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            factor_df[fid] = vals
            factor_names.append(fid)
            factor_exprs[fid] = entry["expression"]
            print(f"  ✓ {fid}: IC={entry['ic_mean']:.4f} | {entry['expression'][:60]}")
        except Exception as e:
            print(f"  ✗ {fid}: {e}")

    if not factor_names:
        print("⚠️  No factors could be computed")
        return

    # Save feature list
    feat_path = out / "gp_factor_features.txt"
    with open(feat_path, "w", encoding="utf-8") as f:
        for name in factor_names:
            f.write(f"{name}\t{factor_exprs[name]}\n")
    print(f"\n📄 Feature list: {feat_path}")

    # Save factor data
    factor_df["close"] = df["close"]
    data_path = out / f"gp_factors_data_{timestamp}.csv"
    factor_df.to_csv(data_path)
    print(f"📄 Factor data: {data_path}")

    # Generate Rust snippet
    rust_path = out / "gp_factors_rust_snippet.rs"
    _generate_gp_rust_snippet(promoted, factor_names, factor_exprs, rust_path)
    print(f"📄 Rust snippet: {rust_path}")

    # Retrain if requested
    if retrain and len(factor_names) >= 2:
        print("\n🔄 Retraining ML model with GP factors...")
        retrain_with_factors(df, factor_df[factor_names], factor_names, out, timestamp)

    print("\n✅ GP factor export complete!")


def _generate_gp_rust_snippet(
    promoted: Dict, factor_names: List[str],
    factor_exprs: Dict, output_path: Path,
):
    """Generate Rust code snippet for GP-discovered factors."""
    lines = [
        "// Auto-generated by gp_factor_mining.py (Phase 2: Genetic Programming)",
        "// These factors were discovered via evolutionary search and validated",
        "// through multiple lifecycle checks.",
        "",
        "pub const GP_FACTOR_NAMES: &[&str] = &[",
    ]
    for name in factor_names:
        lines.append(f'    "{name}",')
    lines.append("];")
    lines.append(f"\npub const NUM_GP_FACTORS: usize = {len(factor_names)};")
    lines.append("")
    lines.append("/*")
    lines.append("GP-discovered factor expressions:")
    lines.append("")
    for name in factor_names:
        expr = factor_exprs.get(name, "?")
        ic = promoted[name]["ic_mean"]
        lines.append(f"  {name}: IC={ic:.4f}")
        lines.append(f"    expr = {expr}")
        lines.append("")
    lines.append("Implementation guide:")
    lines.append("  Each expression uses: close, open, high, low, volume, returns, vwap, tr")
    lines.append("  Operators: +, -, *, / (protected), log, sqrt, abs, sign, inv, neg")
    lines.append("  Rolling: ts_mean(x,w), ts_std(x,w), ts_max(x,w), ts_min(x,w), delta(x,d), delay(x,d)")
    lines.append("*/")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── De-duplicate discovered factors ──────────────────────────────────

def deduplicate_results(
    results: List[Tuple[Node, Dict]],
    df: pd.DataFrame,
    max_corr: float = 0.7,
    top_n: int = 30,
) -> List[Tuple[Node, Dict]]:
    """Remove duplicate/correlated factors from GP results."""
    if not results:
        return []

    data = prepare_data(df)
    kept = []
    kept_values = []

    for tree, ev in results[:min(len(results), top_n * 3)]:
        try:
            vals = tree.evaluate(data)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            s = pd.Series(vals)

            if np.std(vals) < 1e-10:
                continue

            too_corr = False
            for kv in kept_values:
                corr = s.corr(pd.Series(kv))
                if abs(corr) > max_corr:
                    too_corr = True
                    break

            if not too_corr:
                kept.append((tree, ev))
                kept_values.append(vals)

            if len(kept) >= top_n:
                break
        except Exception:
            continue

    return kept


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GP Factor Mining (Phase 2)")
    add_data_args(parser)
    parser.add_argument("--pop-size", type=int, default=300,
                        help="GP population size")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of GP generations")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max expression tree depth")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon")
    parser.add_argument("--ic-threshold", type=float, default=0.03,
                        help="Min |IC| for factor discovery")
    parser.add_argument("--parsimony", type=float, default=0.001,
                        help="Parsimony pressure coefficient")
    parser.add_argument("--max-corr", type=float, default=0.7,
                        help="Max correlation between factors")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Max factors to keep")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain ML model with discovered factors")
    parser.add_argument("--manage", action="store_true",
                        help="Run lifecycle management only")
    parser.add_argument("--export-promoted", action="store_true",
                        help="Export promoted factors only")
    parser.add_argument("--output-dir", type=str, default="ml_models",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    df = load_data(args)

    registry = load_registry()

    # Mode: lifecycle management only
    if args.manage:
        counts = manage_lifecycle(registry, df, args.horizon)
        save_registry(registry)
        print(f"\n📊 Registry: {len(registry['factors'])} factors total")
        _print_registry_summary(registry)
        return

    # Mode: export promoted only
    if args.export_promoted:
        export_promoted_factors(registry, df, args.output_dir, args.retrain)
        return

    # Mode: GP evolution
    t0 = time.time()
    results = evolve(
        df,
        pop_size=args.pop_size,
        generations=args.generations,
        max_depth=args.max_depth,
        horizon=args.horizon,
        ic_threshold=args.ic_threshold,
        parsimony_coeff=args.parsimony,
        verbose=True,
    )

    if not results:
        print("\n❌ No factors discovered. Try more generations or lower thresholds.")
        return

    # De-duplicate
    print(f"\n🔄 De-duplicating {len(results)} raw factors (max_corr={args.max_corr})...")
    results = deduplicate_results(results, df, args.max_corr, args.top_n)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  GP RESULTS: {len(results)} unique factors ({elapsed:.1f}s)")
    print(f"{'='*70}\n")

    print(f"{'Rank':>4} {'IC':>8} {'IR':>8} {'Size':>5} {'Depth':>5}  Expression")
    print("-" * 80)
    for i, (tree, ev) in enumerate(results[:30]):
        expr = ev.get("expression", tree.to_expr())
        if len(expr) > 50:
            expr = expr[:47] + "..."
        print(f"{i+1:4d} {ev['ic_mean']:8.4f} {ev['ir']:8.3f} "
              f"{ev.get('tree_size', 0):5d} {ev.get('tree_depth', 0):5d}  {expr}")

    # Register all discovered factors
    new_count = 0
    for tree, ev in results:
        expr = ev.get("expression", tree.to_expr())
        fid = _factor_id_from_expr(expr)
        if register_factor(registry, fid, expr, ev, source="gp"):
            new_count += 1

    save_registry(registry)
    print(f"\n📝 Registered {new_count} new factors (total: {len(registry['factors'])})")

    # Run lifecycle management to promote good ones
    print("\n🔄 Running lifecycle management...")
    counts = manage_lifecycle(registry, df, args.horizon)
    save_registry(registry)

    _print_registry_summary(registry)

    # Export promoted factors
    promoted = {fid: e for fid, e in registry["factors"].items()
                if e["state"] == STATE_PROMOTED}
    if promoted:
        export_promoted_factors(registry, df, args.output_dir, args.retrain)
    else:
        print("\nℹ️  No promoted factors yet. Run --manage again with new data to accumulate validations.")

    print("\n✅ GP factor mining complete!")


def _print_registry_summary(registry: Dict):
    """Print registry state summary."""
    states = {}
    for entry in registry["factors"].values():
        s = entry["state"]
        states[s] = states.get(s, 0) + 1

    print(f"\n📊 Registry Summary:")
    print(f"   Candidates:  {states.get(STATE_CANDIDATE, 0)}")
    print(f"   Validated:   {states.get(STATE_VALIDATED, 0)}")
    print(f"   Promoted:    {states.get(STATE_PROMOTED, 0)}")
    print(f"   Retired:     {states.get(STATE_RETIRED, 0)}")
    print(f"   Total:       {len(registry['factors'])}")
    print(f"   Lifetime discovered: {registry['stats']['total_discovered']}")
    print(f"   Lifetime promoted:   {registry['stats']['total_promoted']}")
    print(f"   Lifetime retired:    {registry['stats']['total_retired']}")


if __name__ == "__main__":
    main()
