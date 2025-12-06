import numpy as np
import pandas as pd
import os
import json

# ============================================================
# Utility and Strategy Functions for DEUS Framework
# ============================================================

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()

def calculate_max_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / rolling_max - 1
    return drawdowns.min()

def count_trades(signals):
    transitions = np.abs(np.diff(signals))
    return np.sum(transitions > 0)

# ============================================================
# Simple Momentum Strategy (for Learning Agent Testing)
# ============================================================

def simple_momentum_strategy(X, y, params):
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.5)

    close = X['Close'] if 'Close' in X.columns else y.copy()
    ma = close.rolling(window=lookback).mean()
    momentum = (close / ma - 1).fillna(0)

    signals = np.where(momentum > threshold / 100, 1, np.where(momentum < -threshold / 100, -1, 0))

    returns = y.copy()
    strategy_returns = signals[:-1] * returns[1:].values

    equity = (1 + pd.Series(strategy_returns, index=close.index[1:])).cumprod()
    return equity, pd.Series(strategy_returns, index=close.index[1:])

# ============================================================
# Evaluation Function
# ============================================================

def evaluate_strategy(equity_curve):
    if equity_curve is None or len(equity_curve) == 0:
        return {'sharpe': 0, 'return': 0, 'drawdown': 0}

    returns = equity_curve.pct_change().fillna(0)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    return {
        'sharpe': sharpe,
        'return': total_return,
        'drawdown': abs(max_dd)
    }

# ============================================================
# Utility for Learning Agent Persistence
# ============================================================

def save_run_results(params, metrics, experiments_dir="experiments"):
    os.makedirs(experiments_dir, exist_ok=True)
    fname = os.path.join(experiments_dir, f"run_{len(os.listdir(experiments_dir)) + 1}.json")
    with open(fname, 'w') as f:
        json.dump({'params': params, 'metrics': metrics}, f, indent=2)

def load_previous_runs(experiments_dir="experiments"):
    if not os.path.exists(experiments_dir):
        return []
    runs = []
    for file in os.listdir(experiments_dir):
        if file.endswith('.json'):
            path = os.path.join(experiments_dir, file)
            try:
                with open(path, 'r') as f:
                    runs.append(json.load(f))
            except Exception:
                continue
    return runs