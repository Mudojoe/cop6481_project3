import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_save_data(input_file, output_file):
    # Load the cleaned data
    df = pd.read_csv(input_file)

    # Calculate additional features
    df['Daily_Change'] = df['Close'].diff()
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volume_Change'] = df['Volume'].diff()
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()

    # Drop the first 10 rows because of rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Standardize select columns
    features_to_standardize = ['Close', 'Daily_Change', 'Daily_Return', 'MA_5', 'MA_10', 'Volume_Change', 'Volatility']
    scaler = StandardScaler()
    df[features_to_standardize] = scaler.fit_transform(df[features_to_standardize])

    # Shift the 'Close' column to create the target
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)  # Drop the last row with a NaN target

    # Select features and target
    features = ['Close', 'Daily_Change', 'Daily_Return', 'MA_5', 'MA_10', 'Volume_Change', 'Volatility', 'Target']
    df_final = df[features]

    # Save the preprocessed data to a new CSV file
    df_final.to_csv(output_file, index=False)

    print(f"Preprocessed data saved to {output_file}.")
    print(df_final.head())
    for column in df_final.columns:
        print(f"{column}: {df_final[column].head().values}")

# File paths for training data
training_input_file = 'AAPL_3Y_Historical_Data_Cleaned.csv'
training_output_file = 'AAPL_3Y_Historical_Data_Reformatted.csv'

# File paths for test data
test_input_file = 'AAPL_6M_Test_Data_Cleaned.csv'
test_output_file = 'AAPL_6M_Test_Data_Reformatted.csv'

# Call the function for both training and test data
preprocess_and_save_data(training_input_file, training_output_file)
preprocess_and_save_data(test_input_file, test_output_file)
