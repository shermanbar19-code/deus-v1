import yfinance as yf
import pandas as pd
import os

# יצירת תיקייה לשמירת הנתונים
DATA_DIR = os.path.join(os.getcwd(), 'data', 'daily_5y')
os.makedirs(DATA_DIR, exist_ok=True)

# סמלים להורדה (S&P500, NASDAQ100)
symbols = ['SPY', 'QQQ']

for symbol in symbols:
    print(f'📥 מוריד נתונים היסטוריים עבור {symbol} (5 שנים אחרונות)...')
    try:
        data = yf.download(symbol, period='5y', interval='1d')  # נתונים יומיים לחמש שנים
        if data.empty:
            print(f'⚠️ לא התקבלו נתונים עבור {symbol}.')
            continue
        data.reset_index(inplace=True)
        save_path = os.path.join(DATA_DIR, f'{symbol}_5y.csv')
        data.to_csv(save_path, index=False)
        print(f'✅ נשמר בהצלחה: {save_path}')
    except Exception as e:
        print(f'❌ שגיאה בהורדת נתונים עבור {symbol}: {e}')

print('\n🎯 הנתונים נשמרו בהצלחה! כעת תוכל להזין אותם למנוע הלמידה של הבוט.')
