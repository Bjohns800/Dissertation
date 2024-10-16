import os
import pandas as pd
import numpy as np

# Set display precision
pd.set_option('display.precision', 8)


#%%

def calculate_daily_volatility(df):
    """
    Calculates daily realized volatility from a DataFrame of stock price data.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date', 'Time', and 'Close' price columns.

    Returns:
        pd.Series: Daily realized volatility values indexed by 'Date'.
    """
    start_time = '09:30'
    end_time = '16:00'
    
    # Ensure 'Date' is in the correct format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the date range and create a copy to avoid SettingWithCopyWarning
    df = df.loc[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2024-01-01')].copy()

    # Create the 'DateTime' column safely using .loc
    df.loc[:, 'DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

    # Filter data within trading hours and create a copy to avoid SettingWithCopyWarning
    df = df.loc[(df['DateTime'].dt.time >= pd.to_datetime(start_time).time()) &
                         (df['DateTime'].dt.time <= pd.to_datetime(end_time).time())]#.copy()

    # Calculate log returns
    df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

    # Group by 'Date' and calculate daily realized volatility
    df['SquaredLogReturn'] = df['LogReturn']**2

    daily_realized_variance = df.groupby('Date')['SquaredLogReturn'].sum()
    
    daily_realized_volatility = np.sqrt(daily_realized_variance)
    
    return daily_realized_volatility


# Function to process all files in the folder
def process_folder(folder_path):
    """
    Processes all CSV files in the given folder to calculate daily realized volatility.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        output_file (str): Path to save the output CSV file.

    Returns:
        pd.DataFrame: DataFrame with daily realized volatility for all files.
    """
    result_df = None
    i=0
    for file_name in os.listdir(folder_path):
        print(file_name)
        i+=1
        if file_name.endswith('.csv'):  # Ensure that only CSV files are processed
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Calculate daily realized volatility
            daily_volatility = calculate_daily_volatility(df)

            # Convert to DataFrame
            daily_volatility_df = daily_volatility.to_frame(name=file_name.split('.')[0])
                
            # Outer join with the result DataFrame
            if result_df is None:
                result_df = daily_volatility_df
            else:
                result_df = result_df.join(daily_volatility_df, how='outer')

    return result_df

folder_path = 'CSVFolder'  # Replace with your folder path
daily_volatility_table = process_folder(folder_path)

# Save the result to a CSV file or print it
daily_volatility_table.to_csv('daily_volatility2.csv')
print(daily_volatility_table)

#%%

df = pd.read_csv('daily_volatility2.csv')


# Step 1: Count the number of NaN values in each column
nan_counts = df.isna().sum()
nan_counts_sorted = nan_counts.sort_values(ascending=False)

# Print the number of NaN values in each colum
print("NaN counts in each column:\n", nan_counts_sorted)

# Step 2: Drop columns with more than half of values being NaN
df_cleaned = df.dropna(axis=1, thresh=(3522/2)) # thresh = min values to be kept

# Save filtered dataframe as CSV
df_cleaned.to_csv('filtered_df.csv', index=False)




