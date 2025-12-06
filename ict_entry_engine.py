import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

def load_data(symbol):
    path = f"data/{symbol}_data.csv"
    df = pd.read_csv(path)

    # מזהה אם הקובץ כולל עמודת Date או Index
    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df.dropna(inplace=True)
    return df



def detect_signals(df):
    df["Entry_Long"] = False
    df["Entry_Short"] = False
    df["Exit"] = False

    for i in range(2, len(df)):
        # לונג: שפל חדש ונר ירוק
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Close"].iloc[i] > df["Open"].iloc[i]:
            df.loc[df.index[i], "Entry_Long"] = True

        # שורט: שיא חדש ונר אדום
        if df["High"].iloc[i] > df["High"].iloc[i-1] and df["Close"].iloc[i] < df["Open"].iloc[i]:
            df.loc[df.index[i], "Entry_Short"] = True

        # יציאה לפי שינוי מגמה פשוט
        if df["Entry_Long"].iloc[i-1] and df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            df.loc[df.index[i], "Exit"] = True
        elif df["Entry_Short"].iloc[i-1] and df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            df.loc[df.index[i], "Exit"] = True

    return df

def simulate_trades(df, rr_ratio=2.0, risk_percent=1.0):
    trades = []
    for i in range(len(df)):
        if df["Entry_Long"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            stop_loss = df["Low"].iloc[i] * (1 - risk_percent / 100)
            take_profit = entry_price + (entry_price - stop_loss) * rr_ratio
            trades.append(("LONG", df["Date"].iloc[i], entry_price, take_profit, stop_loss))
        elif df["Entry_Short"].iloc[i]:
            entry_price = df["Close"].iloc[i]
            stop_loss = df["High"].iloc[i] * (1 + risk_percent / 100)
            take_profit = entry_price - (stop_loss - entry_price) * rr_ratio
            trades.append(("SHORT", df["Date"].iloc[i], entry_price, take_profit, stop_loss))
    return trades

def plot_signals(df, symbol):
    plt.figure(figsize=(13, 6))
    plt.plot(df["Date"], df["Close"], color="white", label="Price", linewidth=1.2)

    plt.scatter(df.loc[df["Entry_Long"], "Date"], df.loc[df["Entry_Long"], "Close"], color="lime", marker="^", s=80, label="Buy Entry")
    plt.scatter(df.loc[df["Entry_Short"], "Date"], df.loc[df["Entry_Short"], "Close"], color="red", marker="v", s=80, label="Sell Entry")
    plt.scatter(df.loc[df["Exit"], "Date"], df.loc[df["Exit"], "Close"], color="orange", marker="x", s=100, label="Exit")

    plt.title(f"🧠 DEUS - ICT Entry Engine ({symbol})", fontsize=15, color="gold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def run(symbol):
    print(f"\n📈 טוען נתונים עבור {symbol}...")
    df = load_data(symbol)
    df = detect_signals(df)

    rr_ratio = float(input("📊 הכנס יחס RR (למשל 2.0): ") or 2.0)
    risk_percent = float(input("⚙️  הכנס אחוז סיכון (למשל 1.0): ") or 1.0)

    trades = simulate_trades(df, rr_ratio=rr_ratio, risk_percent=risk_percent)
    plot_signals(df, symbol)

    print(f"\n📋 סיכום עסקאות ({symbol}):")
    for t in trades[-5:]:
        print(f"{t[0]} | {t[1].strftime('%Y-%m-%d')} | כניסה: {t[2]:.2f} | TP: {t[3]:.2f} | SL: {t[4]:.2f}")

if __name__ == "__main__":
    run("^GSPC")
