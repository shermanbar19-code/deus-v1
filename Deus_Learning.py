import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from strategy_lib import (
    simple_momentum_strategy,
    evaluate_strategy,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    count_trades,
    load_previous_runs,
    save_run_results
)

class Learning:
    def __init__(self, strategy=None, logger=None):
        """
        Learning Agent — Responsible for optimizing strategy parameters.
        Now includes a 'strategy' argument to avoid duplicate logger errors.
        """
        self.strategy = strategy
        self.log = logger if logger is not None else print
        self.previous_runs = []

    # ============================================================
    # Load / Save previous optimization runs
    # ============================================================

    def use_previous_runs(self, experiments_dir="experiments"):
        """Load results from previous optimization runs."""
        self.previous_runs = load_previous_runs(experiments_dir)
        if len(self.previous_runs) > 0:
            self.log(f"Learning: Loaded {len(self.previous_runs)} previous runs from '{experiments_dir}'.")
        else:
            self.log("Learning: No previous runs found, starting fresh.")

    def save_run(self, params, metrics, experiments_dir="experiments"):
        """Save the result of an optimization run."""
        save_run_results(params, metrics, experiments_dir)
        self.log(f"Learning: Saved run with params {params} and metrics {metrics}.")

    # ============================================================
    # Core Optimization Logic
    # ============================================================

    def optimize_strategy(self, data):
        """Grid search optimization for the simple momentum strategy."""
        try:
            y = data['Return'].fillna(0)
            X = data.copy()
        except Exception as e:
            self.log(f"Learning Error: Invalid data for optimization: {e}")
            return None, None, None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
        self.log(f"Learning: Data split into {len(X_train)} train / {len(X_val)} validation points.")

        param_grid = {
            'lookback': [10, 20, 30],
            'threshold': [0.3, 0.5, 1.0]
        }

        best_params = None
        best_score = -np.inf
        best_metrics = None

        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            try:
                equity, _ = simple_momentum_strategy(X_train, y_train, params)
                metrics = evaluate_strategy(equity)
                score = metrics['sharpe'] - metrics['drawdown']

                self.log(f"Learning: Tested {params} | Sharpe={metrics['sharpe']:.2f}, DD={metrics['drawdown']:.2f}")
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
            except Exception as e:
                self.log(f"Learning Warning: Failed for {params} -> {e}")

        if best_params is None:
            self.log("Learning: No valid parameter combination found.")
            return None, None, None

        # Validation phase
        equity_val, _ = simple_momentum_strategy(X_val, y_val, best_params)
        val_metrics = evaluate_strategy(equity_val)

        # Save run result
        self.save_run(best_params, val_metrics)

        return best_params, best_metrics, val_metrics
