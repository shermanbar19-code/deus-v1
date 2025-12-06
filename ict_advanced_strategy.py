import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# ---------- טעינת נתונים בצורה בטוחה ----------
def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # טיפול בעמודת תאריך
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.date_range(start="2020-01-01", periods=len(df))
    
    # המרת עמודות מחיר למספרים
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"❌ Missing column: {col}")
    
    df = df.dropna(subset=["High", "Low", "Close"])
    return df

# ---------- ICT: מבנה שוק ----------
def detect_market_structure(df):
    df["HH"], df["LL"] = False, False
    for i in range(2, len(df)):
        if df["High"].iloc[i] > df["High"].iloc[i-1] > df["High"].iloc[i-2]:
            df.loc[df.index[i], "HH"] = True
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] < df["Low"].iloc[i-2]:
            df.loc[df.index[i], "LL"] = True
    return df

# ---------- ICT: Fair Value Gap ----------
def detect_fvg(df):
    df["FVG"] = False
    for i in range(2, len(df)):
        if df["Low"].iloc[i] > df["High"].iloc[i-2]:
            df.loc[df.index[i], "FVG"] = True
    return df

# ---------- ICT: Liquidity Grab ----------
def detect_liquidity(df):
    df["Liquidity_Grab"] = False
    for i in range(1, len(df)):
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Close"].iloc[i] > df["Open"].iloc[i]:
            df.loc[df.index[i], "Liquidity_Grab"] = True
    return df

# ---------- ניתוח ושרטוט ----------
def analyze(symbol: str):
    path = f"data/{symbol}_data.csv"
    df = load_price_csv(path)

    df = detect_market_structure(df)
    df = detect_fvg(df)
    df = detect_liquidity(df)

    print(f"\n📊 {symbol} — דוגמה אחרונה:")
    print(df[["Date", "Close", "HH", "LL", "FVG", "Liquidity_Grab"]].tail(5))

    plt.figure(figsize=(13, 6))
    plt.plot(df["Date"], df["Close"], color="white", linewidth=1.2, label="Price")

    plt.scatter(df.loc[df["HH"], "Date"], df.loc[df["HH"], "Close"], color="lime", s=40, label="Higher Highs")
    plt.scatter(df.loc[df["LL"], "Date"], df.loc[df["LL"], "Close"], color="orange", s=40, label="Lower Lows")
    plt.scatter(df.loc[df["FVG"], "Date"], df.loc[df["FVG"], "Close"], color="red", s=60, marker="x", label="FVG")
    plt.scatter(df.loc[df["Liquidity_Grab"], "Date"], df.loc[df["Liquidity_Grab"], "Close"], color="cyan", s=60, marker="^", label="Liquidity Grab")

    plt.title(f"ICT Advanced Strategy — {symbol}", fontsize=14, color="gold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze("^GSPC")
    analyze("^NDX")
