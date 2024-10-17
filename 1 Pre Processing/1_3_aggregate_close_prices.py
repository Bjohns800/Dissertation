import os
import pandas as pd

# Set display precision
pd.set_option('display.precision', 8)


#%%

def daily_close_prices(df):
    start_time = '09:30'
    end_time = '16:00'
    
    # Ensure 'Time' and 'Date' columns are in the correct format
    df['Time'] = df['Time'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Create 'DateTime' column by combining 'Date' and 'Time'
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

    # Drop any rows with invalid Date or missing Close values
    df.dropna(subset=['DateTime', 'Close'], inplace=True)
    
    # Filter rows within trading hours (start_time to end_time)
    df = df.loc[(df['DateTime'].dt.time >= pd.to_datetime(start_time).time()) &
                (df['DateTime'].dt.time <= pd.to_datetime(end_time).time())]
    
    # Group by 'Date' and get the last 'Close' price for each day
    daily_close = df.groupby('Date')['Close'].last().reset_index()

    # Convert to list of lists (date, close price)
    close_prices = daily_close[['Date', 'Close']].values.tolist()

    return close_prices



# Function to process all files in the folder
def process_folder(folder_path):
    result_df = None
    i=0
    for file_name in os.listdir(folder_path):
        print(file_name)
        i+=1
        if file_name.endswith('.csv'):  # Ensure that only CSV files are processed
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Calculate daily realized volatility
            daily_close = daily_close_prices(df)

            # Convert to DataFrame
            daily_close_df = pd.DataFrame(daily_close, columns=['Date', file_name.split('.')[0]])
            daily_close_df.set_index('Date', inplace=True)
                
            # Outer join with the result DataFrame
            if result_df is None:
                result_df = daily_close_df
            else:
                result_df = result_df.join(daily_close_df, how='outer')
    
    result_df.sort_index(inplace=True)
    return result_df

folder_path = 'CSVFolder'  # Replace with your folder path
daily_price_table = process_folder(folder_path)


#%%





start_date='2010-01-01'
end_date='2023-12-31'

df_filtered = daily_price_table.loc[(daily_price_table.index >= start_date) & (daily_price_table.index <= end_date)]

df_filtered_cleaned = df_filtered.dropna(axis=1, how='any')
print(df_filtered_cleaned)



#%%


# Save the result to a CSV file or print it
df_filtered_cleaned.to_csv('daily_close.csv')
print(df_filtered_cleaned)




