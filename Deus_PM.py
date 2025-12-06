import pandas as pd
import numpy as np

class PM:
    def __init__(self, initial_capital=10000, logger=None):
        self.initial_capital = initial_capital
        self.log = logger if logger else print

    def run_strategy(self, strategy, params, data):
        signals, equity, returns = strategy.backtest(data, params, initial_capital=self.initial_capital)

        trade_log = []
        position = None
        entry_price = None
        entry_date = None

        for i in range(1, len(signals)):
            signal = signals.iloc[i]

            if signal == 1 and position != 'LONG':
                position = 'LONG'
                entry_price = data['Close'].iloc[i]
                entry_date = data.index[i]
                if isinstance(entry_date, (pd.Timestamp, np.datetime64)):
                    entry_date_str = pd.to_datetime(entry_date).strftime('%Y-%m-%d')
                else:
                    entry_date_str = str(entry_date)
                self.log(f"PM: Entered LONG on {entry_date_str} at {entry_price:.2f}")

            elif signal == -1 and position != 'SHORT':
                position = 'SHORT'
                entry_price = data['Close'].iloc[i]
                entry_date = data.index[i]
                if isinstance(entry_date, (pd.Timestamp, np.datetime64)):
                    entry_date_str = pd.to_datetime(entry_date).strftime('%Y-%m-%d')
                else:
                    entry_date_str = str(entry_date)
                self.log(f"PM: Entered SHORT on {entry_date_str} at {entry_price:.2f}")

            elif signal == 0 and position is not None:
                exit_price = data['Close'].iloc[i]
                exit_date = data.index[i]
                if isinstance(exit_date, (pd.Timestamp, np.datetime64)):
                    exit_date_str = pd.to_datetime(exit_date).strftime('%Y-%m-%d')
                else:
                    exit_date_str = str(exit_date)
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                trade_log.append({'type': position, 'entry': entry_price, 'exit': exit_price, 'pnl': pnl})
                self.log(f"PM: Exited {position} on {exit_date_str} at {exit_price:.2f} | PnL: {pnl:.2f}")
                position = None

        # עדכון equity אם צריך
        return trade_log, equity