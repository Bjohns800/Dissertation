import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats as stats

# Load your DataFrame with 100 time series
df = pd.read_csv('filtered_df.csv')
df = df.drop(columns=['Date'])


# These outliers are identified later and ammended here
df.at[2785, 'GME'] = 0.404
df.at[2786, 'GME'] = 0.405
df.at[2789, 'GME'] = 0.405

#%%

#Histogram
flattened_data = df.values.flatten()


plt.figure(figsize=(14, 6))

# First histogram with full range
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.hist(flattened_data, bins=50, color='blue', edgecolor='black', range=(0, 0.3))
plt.title('Histogram of the Flattened Data (Full Range)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Second histogram with cut x-axis
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.hist(flattened_data, bins=50, color='blue', edgecolor='black', range=(0,0.05))
plt.title('Histogram of the Flattened Data (Cut Range)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()


#%%

# Count outliers above 10%
count_above_0_3 = (df > 0.1).sum()
sorted_count_above_0_3 = count_above_0_3.sort_values(ascending=False)

# Display the result
print(sorted_count_above_0_3)


#%%

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(flattened_data, bins=50, color='blue', edgecolor='black', range=(0, 0.05)) #max(flattened_data)/1))
#plt.title('Figure 1: Distribution of All Individual Data Points Across the Entire Dataset')
plt.xlabel('Realised Volitility')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



#%%

# Remove both 0.0 and NaN values
flattened_data = [x for x in flattened_data if x != 0.0 and not np.isnan(x)]


# Find the 100 Smallest values
bottom_100_values = np.sort(flattened_data)[:100]

# Alternatively, using Pandas (slightly more convenient)
top_100_values_pd = pd.Series(flattened_data).nsmallest(100).values

# Print the 100 largest values
print("The 100 Smallest values are:")
print(bottom_100_values)


#%%

# Find the 100 largest values
top_100_values = np.sort(flattened_data)[-100:]

# Alternatively, using Pandas (slightly more convenient)
top_100_values_pd = pd.Series(flattened_data).nlargest(100).values

# Print the 100 largest values
print("The 100 largest values are:")
print(top_100_values)


#%%

# Descriptive statictics for individual colmns

def column_statistics(df: pd.DataFrame, column_name: str) -> dict:
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    column_data = df[column_name]*100
    
    # Calculate the required statistics and round to 3 decimal places
    stats = {
        'Min': round(column_data.min(), 3),
        'Max': round(column_data.max(), 3),
        '1st Quantile': round(column_data.quantile(0.25), 3),
        'Median': round(column_data.median(), 3),
        '3rd Quantile': round(column_data.quantile(0.75), 3),
        'Mean': round(column_data.mean(), 3),
        'STD': round(column_data.std(), 3),
        'Kurtosis': round(column_data.kurt(), 3),
        'Skewness': round(column_data.skew(), 3)
    }
    
    return stats


stats = column_statistics(df, 'AAPL')
print(stats)
stats = column_statistics(df, 'FDX')
print(stats)
stats = column_statistics(df, 'URI')
print(stats)
stats = column_statistics(df, 'GME')
print(stats)
stats = column_statistics(df, 'MSFT')
print(stats)
stats = column_statistics(df, 'NKE')
print(stats)


#%%

# Descriptive statistics for full dataset

flattened_data_na = flattened_data[~np.isnan(flattened_data)]

dfstats = {
        'Min': round(np.min(flattened_data_na)*100, 3),
        'Max': round(np.max(flattened_data_na)*100, 3),
        '1st Quantile': round(np.percentile(flattened_data_na, 25)*100, 3),
        'Median': round(np.median(flattened_data_na)*100, 3),
        '3rd Quantile': round(np.percentile(flattened_data_na, 75)*100, 3),
        'Mean': round(np.mean(flattened_data_na)*100, 3),
        'STD': round(np.std(flattened_data_na)*100, 3),
        'Kurtosis': round(np.float64(pd.Series(flattened_data_na).kurt()), 3),
        'Skewness': round(np.float64(pd.Series(flattened_data_na).skew()), 3)
    }
    
print(dfstats)


