import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

def backtest(symbol):
    df = pd.read_csv(f"data/{symbol}_data.csv", parse_dates=["Date"])
    df.dropna(inplace=True)

    df["Signal"] = np.where(df["Close"] > df["Open"], 1, -1)
    df["Return"] = df["Close"].pct_change() * df["Signal"].shift(1)
    df["Equity"] = (1 + df["Return"]).cumprod()

    total_return = (df["Equity"].iloc[-1] - 1) * 100
    win_rate = (df["Return"] > 0).sum() / len(df["Return"]) * 100

    print(f"\n📈 תוצאות הבק-טסט ({symbol})")
    print(f"סה\"כ רווח: {total_return:.2f}%")
    print(f"אחוז הצלחה: {win_rate:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Equity"], color="lime", linewidth=2)
    plt.title(f"Deus ICT Backtest – {symbol}", color="gold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"symbol": symbol, "return": total_return, "win_rate": win_rate}

if __name__ == "__main__":
    backtest("^GSPC")
