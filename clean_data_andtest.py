import pandas as pd
from scipy import stats

def clean_dataset(file_path, output_file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    original_count = len(df)

    # Convert the 'Date' column to datetime format, specifying UTC to handle time zones uniformly
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Handling missing values with forward fill
    missing_values_before = df.isnull().sum().sum()  # Total number of missing values before filling
    df.ffill(inplace=True)
    missing_values_filled = missing_values_before - df.isnull().sum().sum()  # Total filled

    # Removing duplicates
    duplicates_count_before = len(df)
    df.drop_duplicates(subset='Date', inplace=True)
    duplicates_removed = duplicates_count_before - len(df)

    # Identifying outliers using Z-score for 'Close' prices
    close_z_scores = stats.zscore(df['Close'])
    outliers = df[(close_z_scores > 3) | (close_z_scores < -3)]
    num_outliers = len(outliers)

    # Report
    print(f"File: {file_path}")
    print(f"Original number of records: {original_count}")
    print(f"Missing values filled: {missing_values_filled}")
    print(f"Duplicate records removed: {duplicates_removed}")
    print(f"Identified outliers: {num_outliers}")

    # remove these outliers
    df = df[(close_z_scores <= 3) & (close_z_scores >= -3)]

    # Save the cleaned data to a new file, overwriting any existing file
    df.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}.")
    print(df.head())

# Paths for the datasets
training_data_path = 'AAPL_3Y_Historical_Data.csv'
test_data_path = 'AAPL_6M_Test_Data.csv'

# Paths for the output
cleaned_training_data_path = 'AAPL_3Y_Historical_Data_Cleaned.csv'
cleaned_test_data_path = 'AAPL_6M_Test_Data_Cleaned.csv'

# Clean the datasets
clean_dataset(training_data_path, cleaned_training_data_path)
clean_dataset(test_data_path, cleaned_test_data_path)
