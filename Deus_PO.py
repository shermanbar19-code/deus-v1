# agents/po_agent.py

def execute_order(symbol: str, strategy: dict, data, verbose=False):
    """
    מדמה ביצוע פקודות מסחר לפי אסטרטגיה נתונה.
    כרגע מדובר רק בהדמיה - לא שולח פקודות אמיתיות.
    """
    entry_price = data.iloc[-1]['close']
    stop_loss = entry_price - strategy.get('atr', 1) * 1.5
    take_profit = entry_price + strategy.get('atr', 1) * strategy.get('risk_reward', 2)

    result = {
        "symbol": symbol,
        "entry": entry_price,
        "stop_loss": round(stop_loss, 3),
        "take_profit": round(take_profit, 3),
        "status": "executed"
    }

    if verbose:
        print(f"[POAgent] Executed simulated trade: {result}")

    return result
