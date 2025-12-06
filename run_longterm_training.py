import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# --- נתיב בסיס ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(BASE_DIR, 'agents')
if AGENTS_DIR not in sys.path:
    sys.path.append(AGENTS_DIR)

# --- ייבוא ---
from agents.Deus_Architect import Architect
from agents.Deus_PM import PM
from agents.Deus_StrategyConfig import StrategyConfig

print("🚀 מריץ למידה ובדיקת רווחיות על נתוני 5 השנים האחרונות...")

# --- טעינת נתונים ---
data_path = os.path.join(BASE_DIR, 'data', 'daily_5y', 'SPY_QQQ_5y_features.csv')
if not os.path.exists(data_path):
    print(f"⚠️ קובץ הנתונים לא נמצא: {data_path}")
    sys.exit(1)

data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
print(f"[DEBUG] Loaded {len(data)} rows from {data_path}")

# --- אתחול ארכיטקט ---
arch = Architect()
print("[DEBUG] Architect initialized. Config path:", arch.config_dir)

# --- בניית אסטרטגיה ---
features = [c for c in data.columns if c not in ['Date', 'Symbol', 'Close']]
print(f"[DEBUG] Building StrategyConfig with {len(features)} features.")
strategy = StrategyConfig(features=features)

# --- פרמטרים לבדיקה ---
param_grid = [
    {'lookback': 10, 'threshold': 0.3},
    {'lookback': 10, 'threshold': 0.5},
    {'lookback': 10, 'threshold': 1.0},
    {'lookback': 20, 'threshold': 0.3},
    {'lookback': 20, 'threshold': 0.5},
    {'lookback': 20, 'threshold': 1.0},
    {'lookback': 30, 'threshold': 0.3},
    {'lookback': 30, 'threshold': 0.5},
    {'lookback': 30, 'threshold': 1.0},
]

train_size = int(len(data) * 0.7)
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:]
print(f"Learning: Data split into {len(train_data)} train / {len(val_data)} validation points.")

best_params = None
best_sharpe = -np.inf

# --- בדיקת פרמטרים ---
for params in param_grid:
    lookback, threshold = params['lookback'], params['threshold']
    data['Signal'] = np.where(data['Close'].pct_change(lookback) > threshold / 100, 1,
                              np.where(data['Close'].pct_change(lookback) < -threshold / 100, -1, 0))
    returns = data['Signal'].shift(1) * data['Close'].pct_change()
    sharpe = returns.mean() / (returns.std() + 1e-9)
    dd = (1 - (1 + returns.fillna(0)).cumprod() / (1 + returns.fillna(0)).cumprod().cummax()).max()

    print(f"Learning: Tested {params} | Sharpe={sharpe:.2f}, DD={dd:.2f}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = params

arch._save_config_yaml({'best_params': best_params}, os.path.join(arch.config_dir, f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"))
print(f"✅ פרמטרים מיטביים שנמצאו: {best_params}")

# --- הפעלת ניהול תיק ---
pm = PM()
trade_log, equity = pm.run_strategy(strategy, best_params, data)

# --- ניתוח תוצאות ---
if isinstance(trade_log, list):
    trade_log = pd.DataFrame(trade_log)

if equity is not None and len(equity) > 1 and not trade_log.empty:
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    max_dd = (1 - equity / equity.cummax()).max() * 100
    avg_trade = trade_log['PnL'].mean() if 'PnL' in trade_log.columns else 0
    win_rate = (trade_log['PnL'] > 0).mean() * 100 if 'PnL' in trade_log.columns else 0

    print("\n📊 דו""ח ביצועים מסכם:")
    print(f"סה""כ טריידים שבוצעו: {len(trade_log)}")
    print(f"אחוז הצלחות בטריידים: {win_rate:.2f}%")
    print(f"רווח ממוצע לטרייד: {avg_trade:.2f}")
    print(f"משיכה מקסימלית (Drawdown): {max_dd:.2f}%")
    print(f"רווח סופי: {total_return:.2f}%")

    # --- שמירת דו""ח ביצועים ---
    report_dir = os.path.join(BASE_DIR, 'experiments')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv')

    summary_df = pd.DataFrame({
        'Total Trades': [len(trade_log)],
        'Win Rate %': [win_rate],
        'Average Trade PnL': [avg_trade],
        'Max Drawdown %': [max_dd],
        'Total Return %': [total_return]
    })
    summary_df.to_csv(report_path, index=False)
    print(f"\n📁 דו""ח ביצועים נשמר אל: {report_path}")

    # --- שמירת טריידים מפורטים ---
    trades_path = os.path.join(report_dir, f'trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv')
    if not trade_log.empty:
        trade_log.to_csv(trades_path, index=False)
        print(f"📄 קובץ טריידים נשמר אל: {trades_path}")

else:
    print("⚠️ לא נמצאו נתוני ביצועים תקינים או שאין טריידים סגורים.")