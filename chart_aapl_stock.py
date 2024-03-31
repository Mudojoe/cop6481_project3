
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_buy_and_hold(csv_file_path='AAPL_6M_Test_Data.csv'):
    # Load the historical data from CSV
    data = pd.read_csv(csv_file_path)

    # Assuming the 'Date' column exists and your data might need to be sorted by date
    data['Date'] = pd.to_datetime(data['Date'],utc=True)
    data.sort_values('Date', inplace=True)

    # Calculate values
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    percent_difference = ((end_price - start_price) / start_price) * 100

    # Print values
    print("Start Price:", start_price)
    print("End Price:", end_price)
    print("Percent Difference:", percent_difference, "%")

    # Chart creation
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='AAPL Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('AAPL Stock Closing Price Chart')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally, return the percent difference for further analysis or display
    return percent_difference
