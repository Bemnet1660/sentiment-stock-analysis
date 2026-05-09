# Load stock price dataset
df_stock = pd.read_csv('../data/stock_prices.csv', parse_dates=['Date'])

# Set Date as index
df_stock.set_index('Date', inplace=True)
df_stock.sort_index(inplace=True)

# Check for missing values
print("Missing values per column:")
print(df_stock.isnull().sum())

# Handle missing values (forward fill or drop)
df_stock.fillna(method='ffill', inplace=True)
df_stock.dropna(inplace=True)

# Preview the data
df_stock.head()
import talib
import pynance as pn

# Extract numpy arrays for TA-Lib
close_prices = df_stock['Close'].values
high_prices = df_stock['High'].values
low_prices = df_stock['Low'].values
volume = df_stock['Volume'].values

# Simple Moving Average (SMA) and Exponential Moving Average (EMA)
df_stock['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
df_stock['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
df_stock['EMA_20'] = talib.EMA(close_prices, timeperiod=20)

# Relative Strength Index (RSI) - typically 14-day period
df_stock['RSI_14'] = talib.RSI(close_prices, timeperiod=14)

# Moving Average Convergence Divergence (MACD)
df_stock['MACD'], df_stock['MACD_signal'], df_stock['MACD_hist'] = talib.MACD(
    close_prices, fastperiod=12, slowperiod=26, signalperiod=9
)

# Bollinger Bands
df_stock['BB_upper'], df_stock['BB_middle'], df_stock['BB_lower'] = talib.BBANDS(
    close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
)
# Calculate daily returns using PyNance or manually
df_stock['daily_return'] = pn.metrics.return_(df_stock['Close'])

# Alternatively, manual calculation
# df_stock['daily_return'] = df_stock['Close'].pct_change()

# Calculate rolling volatility (20-day standard deviation of returns)
df_stock['volatility'] = df_stock['daily_return'].rolling(window=20).std() * np.sqrt(252)

# Average True Range (ATR) for volatility measurement
df_stock['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
