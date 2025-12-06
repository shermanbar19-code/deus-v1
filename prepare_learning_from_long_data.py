import pandas as pd
import os

# נתיב התיקייה שנשמרו בה הנתונים
DATA_DIR = os.path.join(os.getcwd(), 'data', 'daily_5y')
FILES = ['SPY_5y.csv', 'QQQ_5y.csv']

print("📊 טוען נתונים ארוכי טווח...")

merged = []
for f in FILES:
    path = os.path.join(DATA_DIR, f)
    if not os.path.exists(path):
        print(f'⚠️ קובץ חסר: {f}')
        continue
    df = pd.read_csv(path, parse_dates=['Date'])

    # המרת עמודות מספריות לוודא שאין טקסט
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Symbol'] = f.split('_')[0]
    merged.append(df)

if not merged:
    print("❌ לא נמצאו נתונים.")
else:
    combined = pd.concat(merged)
    combined.sort_values(['Date', 'Symbol'], inplace=True)

    # חישובי תכונות פשוטים ללמידה
    combined['Return'] = combined.groupby('Symbol')['Close'].pct_change()
    combined['MA20'] = combined.groupby('Symbol')['Close'].transform(lambda x: x.rolling(20).mean())
    combined['Volatility20'] = combined.groupby('Symbol')['Return'].transform(lambda x: x.rolling(20).std())

    save_path = os.path.join(DATA_DIR, 'SPY_QQQ_5y_features.csv')
    combined.to_csv(save_path, index=False)
    print(f'✅ קובץ תכונות ללמידה נשמר בהצלחה: {save_path}')

print("\n🎯 כעת תוכל להטעין את הקובץ הזה לתוך המנוע הלומד של הבוט כדי לבדוק ביצועים אמיתיים על 5 השנים האחרונות!")