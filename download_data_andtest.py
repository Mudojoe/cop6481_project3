import yfinance as yf

# Define the ticker symbol
tickerSymbol = 'AAPL'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2020-09-01', end='2023-08-31')

# Save the data to a CSV file
tickerDf.to_csv('AAPL_3Y_Historical_Data.csv')
print(f"Historical data for {tickerSymbol} downloaded")
test_tickerDf = tickerData.history(period='1d', start='2023-09-01', end='2024-02-29')

# Save the test data to a CSV file
test_tickerDf.to_csv('AAPL_6M_Test_Data.csv')
print(f"Test data for {tickerSymbol} downloaded")
