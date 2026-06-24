import pandas as pd
import os

# Load data
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file_path = os.path.join(file_dir, "Nifty 50 Historical Data.csv")
df = pd.read_csv(input_file_path)

# Sort by date (oldest first)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date').reset_index(drop=True)
df['Price'] = (df['Price'].astype(str).str.replace(',', '', regex=False).astype(float))
close = df['Price']

# EMA Calculation
df['EMA20'] = close.ewm(span=20, adjust=False).mean()
df['EMA50'] = close.ewm(span=50, adjust=False).mean()

# RSI 14 Calculation
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Signal generation
df['Signal'] = 0
position = 0
for i in range(1, len(df)):
    buy_if = (
        (df.loc[i, 'EMA20'] > df.loc[i, 'EMA50']) and
        (df.loc[i-1, 'EMA20'] <= df.loc[i-1, 'EMA50']) and
        (df.loc[i, 'RSI'] > 50)
    )
    sell_if = (
        (df.loc[i, 'EMA20'] < df.loc[i, 'EMA50']) or
        (df.loc[i, 'RSI'] < 45)
    )
    # Buy
    if position == 0 and buy_if:
        df.loc[i, 'Signal'] = 1
        position = 1

    # Sell
    elif position == 1 and sell_if:
        df.loc[i, 'Signal'] = -1
        position = 0

# Save signals in csv file
signal_df = df[['Date', 'Signal']]
signal_df.to_csv("nifty_signals.csv", index=False)