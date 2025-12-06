import numpy as np
import pandas as pd
from strategy_lib import simple_momentum_strategy, evaluate_strategy

class StrategyConfig:
    """
    StrategyConfig — אחראי על הגדרת האסטרטגיה והרצת בדיקות (Backtest)
    מחובר ישירות אל Deus_PM ואל Deus_Architect.
    """

    def __init__(self, features=None, long_k=0.6, short_k=0.6, logger=None):
        self.features = features
        self.long_k = long_k
        self.short_k = short_k
        self.log = logger if logger is not None else print

    def _build_signals(self, data, params):
        """יוצר אותות קנייה/מכירה על בסיס מומנטום."""
        lookback = int(params.get('lookback', 20))
        threshold_pct = float(params.get('threshold', 0.5)) / 100.0

        close = data['Close'].astype(float)
        ma = close.rolling(window=lookback, min_periods=lookback).mean()
        momentum = (close / ma - 1.0)

        signals = np.where(momentum > threshold_pct, 1,
                  np.where(momentum < -threshold_pct, -1, 0))
        return pd.Series(signals, index=data.index, dtype=int)

    def backtest(self, data, params, initial_capital=10000):
        """מריץ בק-טסט מלא ומחזיר (signals, equity_curve, returns)."""
        self.log("[Deus_StrategyConfig] Backtest התחיל...")

        signals = self._build_signals(data, params)
        shifted = signals.shift(1).fillna(0)
        daily_returns = (shifted * data['Return'].astype(float)).fillna(0.0)
        equity_curve = (1 + daily_returns).cumprod() * initial_capital

        self.log(f"[Deus_StrategyConfig] הסתיים ✅ | lookback={params.get('lookback',20)}, threshold={params.get('threshold',0.5)}")
        return signals, equity_curve, daily_returns
