import numpy as np
import pandas as pd

def run_backtest(df: pd.DataFrame, signals: pd.Series, rr: float,
                 slippage_bps: float, fees_bps: float) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["signal"] = signals
    trades = []
    daily_returns = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        s = df["signal"].iloc[i-1]
        if s == 0:
            continue
        entry = df["Close"].iloc[i]
        atr = df["atr"].iloc[i]
        if np.isnan(atr) or atr <= 0:
            continue

        sl = entry - s*atr
        tp = entry + s*atr*rr
        entry *= (1 + s * (slippage_bps + fees_bps) / 10000.0)

        exit_price = df["Close"].iloc[i]
        hit_tp = (s == 1 and df["High"].iloc[i] >= tp) or (s == -1 and df["Low"].iloc[i] <= tp)
        hit_sl = (s == 1 and df["Low"].iloc[i] <= sl) or (s == -1 and df["High"].iloc[i] >= sl)
        if hit_tp:
            exit_price = tp
        elif hit_sl:
            exit_price = sl

        exit_price *= (1 - s * (slippage_bps + fees_bps) / 10000.0)
        ret = (exit_price - entry) / entry * s
        daily_returns.iloc[i] = ret
        trades.append(ret)

    trades = np.array(trades) if trades else np.array([0.0])
    equity_curve = (1 + daily_returns).cumprod()

    metrics = {
        "trades": int((signals != 0).sum()),
        "avg_trade": float(trades.mean()) if trades.size else 0.0,
        "win_rate": float((trades > 0).mean()) if trades.size else 0.0,
        "cagr": float(equity_curve.iloc[-1]**(252/len(df)) - 1) if len(df) > 50 else 0.0,
        "drawdown": float((equity_curve.cummax() - equity_curve).max()),
        "sharpe": float((daily_returns.mean() / (daily_returns.std()+1e-9)) * np.sqrt(252))
    }
    return pd.DataFrame({"daily_returns": daily_returns}), metrics
