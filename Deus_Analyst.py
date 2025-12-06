import pandas as pd
import numpy as np

class Analyst:
    def __init__(self, sp500_path='data/sp500.csv', nasdaq_path='data/nasdaq100.csv', logger=None):
        self.sp500_path = sp500_path
        self.nasdaq_path = nasdaq_path
        self.log = logger if logger is not None else print
        self.data = None
        self.data_nasdaq = None
        self.features = None

    def load_data(self):
        try:
            df_sp = pd.read_csv(
                self.sp500_path,
                skiprows=2,
                names=["Date", "Close", "High", "Low", "Open", "Volume"],
                header=None,
                parse_dates=["Date"],
                date_format="%Y-%m-%d"
            )
            df_sp.set_index("Date", inplace=True)
            df_sp.sort_index(inplace=True)

            df_ndx = pd.read_csv(
                self.nasdaq_path,
                skiprows=2,
                names=["Date", "Close", "High", "Low", "Open", "Volume"],
                header=None,
                parse_dates=["Date"],
                date_format="%Y-%m-%d"
            )
            df_ndx.set_index("Date", inplace=True)
            df_ndx.sort_index(inplace=True)

            common_idx = df_sp.index.intersection(df_ndx.index)
            self.data = df_sp.loc[common_idx].copy()
            self.data_nasdaq = df_ndx.loc[common_idx].copy()
            self.log(f"Analyst: Loaded {len(self.data)} records of S&P500 and {len(self.data_nasdaq)} of NASDAQ100.")
            return self.data
        except Exception as e:
            self.log(f"Analyst: Error loading data: {e}")
            raise

    def compute_features(self):
        if self.data is None:
            raise ValueError("Data not loaded.")
        df = self.data.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Return'] = df['Close'].pct_change(fill_method=None)
        df['RollingVolatility'] = df['Return'].rolling(window=20).std()
        df['VolumeMA20'] = df['Volume'].rolling(window=20).mean()
        df['NormalizedVolume'] = df['Volume'] / df['VolumeMA20']

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
        df['TrueRange'] = true_range.max(axis=1)
        df['ATR'] = df['TrueRange'].rolling(window=14).mean()

        pivot_raw = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['Pivot'] = pivot_raw.combine_first((df['High'] + df['Low'] + df['Close']) / 3)
        df['Resistance1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['Support1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['Resistance2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['Support2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))

        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['PriceDev_MA50'] = (df['Close'] - df['MA50']) / df['MA50']
        df['PriceDev_Pivot'] = (df['Close'] - df['Pivot']) / df['Pivot']

        if self.data_nasdaq is not None:
            df['NASDAQ_Close'] = self.data_nasdaq['Close']
            df['NASDAQ_Return'] = df['NASDAQ_Close'].pct_change(fill_method=None)
            df['Corr_SP500_NASDAQ_20'] = df['Return'].rolling(window=20).corr(df['NASDAQ_Return'])
            df['RelStrength_SP500_NASDAQ'] = df['Close'] / df['NASDAQ_Close']

        df.dropna(inplace=True)
        self.features = df
        self.log(f"Analyst: Computed features columns: {list(df.columns)}")
        return self.features

    def detect_anomalies(self):
        if self.features is None:
            raise ValueError("Features not computed.")
        df = self.features
        issues = []

        if df.isnull().any().any():
            issues.append("Missing values present in data.")

        if 'Return' in df.columns:
            mu = df['Return'].mean()
            sigma = df['Return'].std()
            outliers = df[np.abs(df['Return'] - mu) > 6 * sigma]
            if not outliers.empty:
                try:
                    dates = outliers.index.strftime('%Y-%m-%d').tolist()
                except Exception:
                    dates = [str(idx) for idx in outliers.index]
                issues.append(f"Extreme return outliers on: {dates}")

        if 'Volume' in df.columns:
            bad_vol = df[df['Volume'] <= 0]
            if not bad_vol.empty:
                try:
                    dates = bad_vol.index.strftime('%Y-%m-%d').tolist()
                except Exception:
                    dates = [str(idx) for idx in bad_vol.index]
                issues.append(f"Non-positive volume on dates: {dates}")

        if issues:
            for issue in issues:
                self.log(f"Analyst Warning: {issue}")
        else:
            self.log("Analyst: No anomalies detected.")

        return issues
