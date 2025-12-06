import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# טוען נתונים היסטוריים
def fetch_market_data(symbol, days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(f"data/{symbol}_data.csv")
    print(f"✅ {symbol} data saved successfully!")

# הורדת נתונים של S&P 500 ו-NASDAQ 100
if __name__ == "__main__":
    fetch_market_data("^GSPC")   # S&P 500
    fetch_market_data("^NDX")    # NASDAQ 100
