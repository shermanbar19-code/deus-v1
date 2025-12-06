import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# ============================
# Config
# ============================
DATA_DIR = os.path.join(os.getcwd(), "data", "minute")
SYMBOLS = ["SPY_1min.csv", "QQQ_1min.csv"]  # אתה יכול להוסיף/לשנות
TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "11:00"
DAYS = {0,1,2,3,4}  # Mon-Fri

# תנאים מהדרישה שלך
MAX_TRADE_MINUTES = 90
RR_TARGETS = [2.0, 3.0]  # 1:2 או 1:3
MIN_WINRATE_TARGET = 0.50  # נבדוק ונציג בדו"ח

OUT_DIR = os.path.join("experiments", "intraday")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# Utils
# ============================

def _to_ny_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            raise ValueError("Input dataframe must have DatetimeIndex or 'Date' column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").tz_convert(TZ)
    else:
        df.index = df.index.tz_convert(TZ)
    df = df.sort_index()
    return df


def load_minute_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    rename = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=rename)
    # Allow both 'Date' and 'Datetime'
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df = _to_ny_time_index(df)
    # enforce numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def in_session(ts: pd.Timestamp) -> bool:
    if ts.weekday() not in DAYS:
        return False
    hhmm = ts.strftime("%H:%M")
    return (SESSION_START <= hhmm <= SESSION_END)


def resample_bias(df: pd.DataFrame, rule: str = "60min") -> pd.Series:
    """Bias לפי ממוצע נע לעומת מחיר (HTF: שעה/4 שעות)."""
    close = df["Close"]
    ohlc = close.resample(rule).last().dropna()
    ma = ohlc.rolling(20).mean()
    bias = (ohlc / ma - 1.0).fillna(0.0)
    return bias


def compute_bias_signal(df: pd.DataFrame) -> pd.Series:
    # 1H ו-4H
    bias_h1 = resample_bias(df, "60min")
    bias_h4 = resample_bias(df, "240min")
    # bring back to 1-min index via ffill
    b1 = bias_h1.reindex(df.index, method="ffill").fillna(0)
    b4 = bias_h4.reindex(df.index, method="ffill").fillna(0)
    # bias חיובי אם שניהם >= 0, שלילי אם שניהם <= 0
    sig = pd.Series(0, index=df.index)
    sig[(b1 >= 0) & (b4 >= 0)] = 1
    sig[(b1 <= 0) & (b4 <= 0)] = -1
    return sig


def liquidity_sweep_flags(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """סימון sweep של לואו/היי אחרונים: ירידה מתחת ללואו N נרות וסגירה מעל (ללונג), ולהפך לשורט."""
    low_n = df["Low"].rolling(lookback).min()
    high_n = df["High"].rolling(lookback).max()

    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    long_sweep = (low < low_n.shift(1)) & (close > low_n.shift(1))
    short_sweep = (high > high_n.shift(1)) & (close < high_n.shift(1))

    return pd.DataFrame({
        "sweep_long": long_sweep.fillna(False),
        "sweep_short": short_sweep.fillna(False)
    }, index=df.index)


def run_intraday_trades(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    bias = compute_bias_signal(df)
    sweeps = liquidity_sweep_flags(df)

    trades = []
    i = 0
    while i < len(df):
        ts = df.index[i]
        if not in_session(ts):
            i += 1
            continue

        dir_sig = bias.iloc[i]
        if dir_sig == 1 and sweeps.loc[ts, "sweep_long"]:
            # כניסת לונג אחרי sweep לואו
            entry = df.iloc[i]
            entry_price = float(entry["Close"])
            stop_price = float(entry["Low"])  # קצת מתחת ללואו הנר
            if stop_price >= entry_price:
                i += 1
                continue
            risk = entry_price - stop_price
            rr = RR_TARGETS[0]
            target_price = entry_price + rr * risk
            exit_ts, exit_price, outcome, rr_hit = simulate_exit(df, i, entry_price, stop_price, target_price)
            trades.append({
                "symbol": symbol,
                "side": "LONG",
                "entry_time": ts.isoformat(),
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "exit_time": exit_ts.isoformat() if pd.notna(exit_ts) else None,
                "exit": exit_price,
                "outcome": outcome,
                "rr": rr_hit,
                "minutes": minutes_between(ts, exit_ts) if exit_ts is not None else None,
                "confluences": [
                    "HTF bias H1&H4 up",
                    "Liquidity sweep of prior lows",
                    "Session 09:30-11:00"
                ]
            })
            # דילוג קדימה אחרי יציאה
            i = df.index.get_loc(exit_ts) + 1 if exit_ts in df.index else i + 1
            continue

        if dir_sig == -1 and sweeps.loc[ts, "sweep_short"]:
            # כניסת שורט אחרי sweep היי
            entry = df.iloc[i]
            entry_price = float(entry["Close"])
            stop_price = float(entry["High"])  # קצת מעל היי הנר
            if stop_price <= entry_price:
                i += 1
                continue
            risk = stop_price - entry_price
            rr = RR_TARGETS[0]
            target_price = entry_price - rr * risk
            exit_ts, exit_price, outcome, rr_hit = simulate_exit(df, i, entry_price, stop_price, target_price, short=True)
            trades.append({
                "symbol": symbol,
                "side": "SHORT",
                "entry_time": ts.isoformat(),
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "exit_time": exit_ts.isoformat() if pd.notna(exit_ts) else None,
                "exit": exit_price,
                "outcome": outcome,
                "rr": rr_hit,
                "minutes": minutes_between(ts, exit_ts) if exit_ts is not None else None,
                "confluences": [
                    "HTF bias H1&H4 down",
                    "Liquidity sweep of prior highs",
                    "Session 09:30-11:00"
                ]
            })
            i = df.index.get_loc(exit_ts) + 1 if exit_ts in df.index else i + 1
            continue

        i += 1

    return pd.DataFrame(trades)


def simulate_exit(df: pd.DataFrame, start_idx: int, entry: float, stop: float, target: float, short: bool = False):
    start_ts = df.index[start_idx]
    max_end_ts = start_ts + timedelta(minutes=MAX_TRADE_MINUTES)

    rr_hit = None
    for j in range(start_idx + 1, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        if ts > max_end_ts:
            # יציאה כפויה על הזמן
            return ts, float(row["Close"]), "TIME_EXIT", rr_hit

        h = float(row["High"])
        l = float(row["Low"])

        if not short:
            # לונג: פגיעה בסטופ/טייק-פרופיט לפי סדר הופעה
            # אם היי הגיע ליעד לפני שהלואו הגיע לסטופ -> TP
            if l <= stop:
                rr_hit = (stop - entry) / (entry - stop)  # בערך -1
                return ts, stop, "SL", rr_hit
            if h >= target:
                rr_hit = (target - entry) / (entry - stop)  # אמור להיות RR_TARGETS[0]
                return ts, target, "TP", rr_hit
        else:
            # שורט
            if h >= stop:
                rr_hit = (entry - stop) / (stop - entry)  # ~ -1
                return ts, stop, "SL", rr_hit
            if l <= target:
                rr_hit = (entry - target) / (stop - entry)  # RR חיובי
                return ts, target, "TP", rr_hit

    # אם נגמר הדאטה
    last_ts = df.index[-1]
    return last_ts, float(df.iloc[-1]["Close"]), "EOD_EXIT", rr_hit


def minutes_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    if a is None or b is None:
        return np.nan
    return (b - a).total_seconds() / 60.0


def metrics_from_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "avg_minutes": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
        }

    wins = trades[trades["outcome"] == "TP"]
    losses = trades[trades["outcome"] == "SL"]

    win_rate = len(wins) / len(trades)

    # נחשב פיי&אל פשוט פר טרייד ביחידות סיכון (R)
    def rr_for_row(r):
        if r["outcome"] == "TP":
            return r["rr"] if pd.notna(r["rr"]) else RR_TARGETS[0]
        elif r["outcome"] == "SL":
            return -1.0
        else:
            # TIME_EXIT/EOD_EXIT — נחלק לפי יחס מרחק מחירי
            entry = r["entry"]; exitp = r["exit"]; stop = r["stop"]
            if r["side"] == "LONG":
                R = (entry - stop)
                return (exitp - entry) / R if R != 0 else 0.0
            else:
                R = (stop - entry)
                return (entry - exitp) / R if R != 0 else 0.0

    rr_list = trades.apply(rr_for_row, axis=1).values

    total_R = float(np.nansum(rr_list))
    avg_rr = float(np.nanmean(rr_list))

    gross_win = float(np.nansum([x for x in rr_list if x > 0]))
    gross_loss = -float(np.nansum([x for x in rr_list if x < 0]))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf

    avg_minutes = float(trades["minutes"].dropna().mean()) if "minutes" in trades else np.nan

    # המרה לרווח באחוזים בהנחה של 1R=1% חשיפה להמחשה
    total_return = total_R / len(trades) if len(trades) > 0 else 0.0

    return {
        "trades": int(len(trades)),
        "win_rate": float(win_rate),
        "avg_rr": float(avg_rr),
        "avg_minutes": avg_minutes,
        "profit_factor": float(profit_factor),
        "total_return": float(total_return),
    }


def plot_trades(df: pd.DataFrame, trades: pd.DataFrame, title: str, out_path: str):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Close"], label="Close")

    for _, r in trades.iterrows():
        et = pd.to_datetime(r["entry_time"])  # כבר timezone ניו-יורק
        if et not in df.index:
            continue
        plt.scatter(et, r["entry"], marker="^" if r["side"]=="LONG" else "v")
        if r["exit_time"]:
            xt = pd.to_datetime(r["exit_time"])
            if xt in df.index:
                plt.scatter(xt, r["exit"], marker="x")
        # קווים אופקיים סטופ/טארגט
        plt.hlines([r["stop"], r["target"]], xmin=et, xmax=et+timedelta(minutes=MAX_TRADE_MINUTES), linestyles="dotted")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================
# Main
# ============================
if __name__ == "__main__":
    all_trades = []

    for fname in SYMBOLS:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠️  Missing file: {path}")
            continue
        df = load_minute_csv(path)
        # סינון חלון הזמן היומי
        mask = df.index.map(in_session)
        dfi = df.loc[mask].copy()
        if dfi.empty:
            print(f"⚠️  No in-session data for {fname}")
            continue

        trades = run_intraday_trades(dfi, symbol=fname.split("_")[0])
        metrics = metrics_from_trades(trades)

        print(f"\n=== {fname} ===")
        print(json.dumps(metrics, indent=2))
        if not trades.empty:
            out_img = os.path.join(OUT_DIR, f"{fname.replace('.csv','')}_intraday_trades.png")
            plot_trades(dfi, trades, title=f"{fname} intraday trades", out_path=out_img)
            print(f"Saved plot -> {out_img}")

        trades["symbol_file"] = fname
        all_trades.append(trades)

    if all_trades:
        out_df = pd.concat(all_trades, ignore_index=True)
        out_csv = os.path.join(OUT_DIR, "intraday_trades_log.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved trades log -> {out_csv}")

        # בדיקה מול היעד שלך (win-rate >= 50% ו-RR 1:2/1:3)
        m = metrics_from_trades(out_df)
        ok_win = m["win_rate"] >= MIN_WINRATE_TARGET
        ok_rr = m["avg_rr"] >= 0.5 * (RR_TARGETS[0] - 1)  # בדיקה רכה
        print("\n=== SUMMARY (ALL) ===")
        print(json.dumps(m, indent=2))
        if ok_win:
            print("✅ Win-rate >= 50% (as requested)")
        else:
            print("❌ Win-rate < 50% — consider tuning filters/parameters")
        print("Note: avg_rr is measured in R; Targets used:", RR_TARGETS)
    else:
        print("No trades found across files — verify CSVs/time window.")
