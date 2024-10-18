import pickle
import numpy as np
import pandas as pd

with open('Targets_Backup.pkl', 'rb') as file:
    Actuals = pickle.load(file)

with open('GARCH_Backup.pkl', 'rb') as file:
    GARCH = pickle.load(file)

with open('HAR_Backup.pkl', 'rb') as file:
    HAR = pickle.load(file)
with open('HAR_with_Jumps.pkl', 'rb') as file:
    HAR_jump = pickle.load(file)
    
with open('CHAR.pkl', 'rb') as file:
    CHAR = pickle.load(file)
with open('ARQ_predictions.pkl', 'rb') as file:
    ARQ = pickle.load(file)    
    
with open('HARQ_predictions_V2.pkl', 'rb') as file:
    HARQ = pickle.load(file)
with open('HARQF_predictions.pkl', 'rb') as file:
    HARQF = pickle.load(file)    
    
with open('LSTM_Backup.pkl', 'rb') as file:
    LSTM = pickle.load(file)

with open('Encoder_Backup.pkl', 'rb') as file:
    Encoder = pickle.load(file)
with open('Transformer_Backup.pkl', 'rb') as file:
    Transformer = pickle.load(file)
    
with open('Autoformer_encoder_Results_NO_ATTENTION_Backup.pkl', 'rb') as file:
    Autoformer_encoder = pickle.load(file)    
with open('Autoformer_Full_Backup_model.pkl', 'rb') as file:
    Autoformer_full = pickle.load(file)    



for key in GARCH.keys():
    #removes last 6 rows 
    GARCH[key] = GARCH[key].iloc[:-6]
    
    HAR[key] = HAR[key].iloc[:-6]
    HAR_jump[key] = HAR_jump[key].iloc[:-6]
    CHAR[key] = CHAR[key].iloc[:-6]
    ARQ[key] = ARQ[key].iloc[:-6]
    HARQ[key] = HARQ[key].iloc[:-6]
    HARQF[key] = HARQF[key].iloc[:-6]
    
    LSTM[key] = LSTM[key].iloc[:-6]
    Encoder[key] = Encoder[key].iloc[:-6]
    Transformer[key] = Transformer[key].iloc[:-6]
    Autoformer_encoder[key] = Autoformer_encoder[key].iloc[:-6]
    Autoformer_full[key] = Autoformer_full[key].iloc[:-6]
    
    
    
    

all_match = True
# Iterate through each key in the dictionaries
for key in GARCH.keys():
    #print(key)
    # Get the shapes of the dataframes for the current key
    shape1 = GARCH[key].shape
    shape2 = HAR[key].shape
    shape3 = HAR_jump[key].shape
    shape4 = CHAR[key].shape
    shape5 = ARQ[key].shape
    shape6 = HARQ[key].shape
    shape7 = HARQF[key].shape
    
    shape8 = LSTM[key].shape
    shape9 = Actuals[key].shape
    shape10 = Encoder[key].shape
    shape11 = Transformer[key].shape
    shape12 = Autoformer_encoder[key].shape
    shape13 = Autoformer_full[key].shape
    
    # Check if all shapes are the same
    if any(shape1 != shape for shape in [shape2, shape3, shape4, shape5, shape6, shape7, shape8, shape9, shape10, shape11, shape12, shape13]):
        print(f"Dataframes for key '{key}' do not have the same size.")
        all_match = False

if all_match:
    print("All dataframes have the same size for all keys.")
else:
    print("There are discrepancies in dataframe sizes.")


#GARCH scale FIX
new_column_names = [f'Day_{i}_ahead' for i in range(1, 8)] 
for key in GARCH.keys():
    GARCH[key].columns = new_column_names
    GARCH[key] = GARCH[key] / 100

def popfunction(dictionary):
    dictionary.pop('SLE', None) 
    dictionary.pop('SBNY', None) 
    dictionary.pop('CPWR', None) 
    return dictionary

GARCH = popfunction(GARCH)
HAR = popfunction(HAR)
HAR_jump = popfunction(HAR_jump)
CHAR = popfunction(CHAR)
ARQ = popfunction(ARQ)
HARQ = popfunction(HARQ)
HARQF = popfunction(HARQF)

LSTM = popfunction(LSTM)
Actuals = popfunction(Actuals)
Encoder = popfunction(Encoder)
Transformer = popfunction(Transformer)
Autoformer_encoder = popfunction(Autoformer_encoder)
Autoformer_full = popfunction(Autoformer_full)


#%%

def MSEdayN (Actuals,model,day):
    day -=1
    TT = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the first column of the DataFrame for the real and model forecasts
        actual_day_1 = Actuals[stock].iloc[:, day]
        model_day_1 = model[stock].iloc[:, day]
    
        mse_first_day = np.mean((actual_day_1.values - model_day_1.values) ** 2)
    
        TT.append(mse_first_day)

    return(np.mean(TT))

def MDA_day1(Actuals, model):
    df = pd.read_csv('filtered_df.csv')
    df = df.drop(columns=['Date'])
    
    correct_directions = []
    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():

        # Extract the relevant columns of the DataFrame for the real and model forecasts
        actual_stock = Actuals[stock].iloc[:, 0]
        model_stock = model[stock].iloc[:, 0]
        
        stock_data = df[stock].dropna()
        split_index = int(len(stock_data) * 0.8)
        stock_data = stock_data.iloc[split_index-1:-7]

        actual_change = np.array(actual_stock) - np.array(stock_data)
        model_change = np.array(model_stock) - np.array(stock_data)

        # Calculate the number of times the direction of the change matches
        correct_direction = np.sum(np.sign(actual_change) == np.sign(model_change))
        # Append the percentage of correct directions for this stock
        correct_directions.append(correct_direction / len(actual_change))

    return np.mean(correct_directions)


def QLIKEdayN(Actuals, model, day):
    epsilon = 1e-10
    day -= 1  # Adjust for zero-based indexing
    QLIKE_values = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the values for the specified day from both Actuals and model
        RVt = Actuals[stock].iloc[:, day]+ epsilon
        RVdt = model[stock].iloc[:, day] + epsilon  # Add epsilon to prevent division by zero

        # Calculate QLIKE for the day
        qlike_day = (RVt / RVdt) - np.log(RVt / RVdt) - 1
        
        # Check if any of the values result in an invalid operation
        if np.any(np.isinf(qlike_day)) or np.any(np.isnan(qlike_day)):
            print(f"Warning: Found inf or NaN in QLIKE calculation for stock {stock}")

        # Store the mean QLIKE for the current stock
        QLIKE_values.append(np.mean(qlike_day))

    # Return the mean QLIKE across all stocks for the specified day
    return np.mean(QLIKE_values)


#%%

#day 1 dataframe 

metrics_dict_day1 = {
    'Model': ['GARCH', 'HAR', 'HAR Jumps', 'CHAR', 'ARQ', 'HARQ', 'HARQF', 'LSTM', 'Encoder', 'Transformer', 'Autoformer Encoder', 'Autoformer Full'],
    'MSE': [
        MSEdayN(Actuals, GARCH, 1), MSEdayN(Actuals, HAR, 1), MSEdayN(Actuals, HAR_jump, 1),
        MSEdayN(Actuals, CHAR, 1), MSEdayN(Actuals, ARQ, 1), MSEdayN(Actuals, HARQ, 1), MSEdayN(Actuals, HARQF, 1),
        MSEdayN(Actuals, LSTM, 1), MSEdayN(Actuals, Encoder, 1),
        MSEdayN(Actuals, Transformer, 1), MSEdayN(Actuals, Autoformer_encoder, 1),
        MSEdayN(Actuals, Autoformer_full, 1)
    ],
    'QLIKE': [
        QLIKEdayN(Actuals, GARCH, 1), QLIKEdayN(Actuals, HAR, 1), QLIKEdayN(Actuals, HAR_jump, 1),
        QLIKEdayN(Actuals, CHAR, 1), QLIKEdayN(Actuals, ARQ, 1), QLIKEdayN(Actuals, HARQ, 1), QLIKEdayN(Actuals, HARQF, 1),
        QLIKEdayN(Actuals, LSTM, 1), QLIKEdayN(Actuals, Encoder, 1),
        QLIKEdayN(Actuals, Transformer, 1), QLIKEdayN(Actuals, Autoformer_encoder, 1),
        QLIKEdayN(Actuals, Autoformer_full, 1)
    ],
    'MDA': [
        MDA_day1(Actuals, GARCH), MDA_day1(Actuals, HAR), MDA_day1(Actuals, HAR_jump),
        MDA_day1(Actuals, CHAR), MDA_day1(Actuals, ARQ), MDA_day1(Actuals, HARQ), MDA_day1(Actuals, HARQF),
        MDA_day1(Actuals, LSTM), MDA_day1(Actuals, Encoder),
        MDA_day1(Actuals, Transformer), MDA_day1(Actuals, Autoformer_encoder),
        MDA_day1(Actuals, Autoformer_full)
    ]
}



# Convert the dictionary to a DataFrame
metrics_dict_day1 = pd.DataFrame(metrics_dict_day1)#.transpose()

# Display the resulting DataFrame
print(metrics_dict_day1)

#%%


def MDA_Weekly(Actuals, model):
    df = pd.read_csv('filtered_df.csv')
    df = df.drop(columns=['Date'])
    
    correct_directions = []
    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the relevant columns of the DataFrame for the real and model forecasts
        actual_stock = Actuals[stock]
        model_stock = model[stock]
        
        weekly_sum = []
        
        stock_data = df[stock].dropna()
        # Define the split index to separate train and test data (80%-20%)
        split_index = int(len(stock_data) * 0.8)
        
        
        for i in range (split_index,len(stock_data)-6 ):
            weekly_sum.append(sum(stock_data[i-6:i+1]))
        weekly_sum = np.array(weekly_sum)
        
        actual_sum = actual_stock.sum(axis=1)
        model_sum = model_stock.sum(axis=1)
            
        actual_change = actual_sum - weekly_sum
        model_change = model_sum - weekly_sum

        # Calculate the number of times the direction of the change matches
        correct_direction = np.sum(np.sign(actual_change) == np.sign(model_change))
        # Append the percentage of correct directions for this stock
        correct_directions.append(correct_direction / len(actual_change))

    # Calculate and return the average MDA across all stocks and days
    print("Done")
    return np.mean(correct_directions)


def MSEweekly(Actuals, model):
    MSE = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the DataFrames for the actual and model forecasts
        actual_stock = Actuals[stock]
        model_stock = model[stock]
        
        # Step 1: Calculate the mean of each row (i.e., across the 7 days) for actuals and model forecasts
        actual_mean = actual_stock.mean(axis=1)
        model_mean = model_stock.mean(axis=1)
        
        # Step 2: Calculate the squared differences between the means
        squared_diff = (actual_mean - model_mean) ** 2
        
        # Step 3: Calculate the Mean Squared Error (MSE) as the mean of these squared differences
        mse = squared_diff.mean()
        
        MSE.append(mse)

    # Calculate the average MSE across all stocks
    return np.mean(MSE)


def QLIKEweekly(Actuals, model):
    epsilon = 1e-10
    QLIKE = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the DataFrames for the actual and model forecasts
        actual_stock = Actuals[stock] + epsilon
        model_stock = model[stock] + epsilon
        
        # Calculate the mean of each row (i.e., across the 7 days) for actuals and model forecasts
        actual_mean = actual_stock.mean(axis=1)
        model_mean = model_stock.mean(axis=1)
        
        # Calculate the QLIKE loss for each element
        qlike_stock = (actual_mean / model_mean) - np.log(actual_mean / model_mean) - 1

        # Check for any infinite or NaN values and print a warning if found
        if np.any(np.isinf(qlike_stock)) or np.any(np.isnan(qlike_stock)):
            print(f"Warning: Found inf or NaN in QLIKE calculation for stock {stock}")
            
        # Calculate the mean QLIKE for this stock
        qlike_mean = qlike_stock.mean()

        QLIKE.append(qlike_mean)

    # Return the average QLIKE across all stocks
    return np.mean(QLIKE)



#%%

metrics_dict_weekly = {
    'Model': ['GARCH', 'HAR', 'HAR Jumps', 'CHAR', 'ARQ', 'HARQ', 'HARQF', 'LSTM', 'Encoder', 'Transformer', 'Autoformer Encoder', 'Autoformer Full'],
    'MSE': [
        MSEweekly(Actuals, GARCH), MSEweekly(Actuals, HAR), MSEweekly(Actuals, HAR_jump),
        MSEweekly(Actuals, CHAR), MSEweekly(Actuals, ARQ), MSEweekly(Actuals, HARQ), MSEweekly(Actuals, HARQF),
        MSEweekly(Actuals, LSTM), MSEweekly(Actuals, Encoder),
        MSEweekly(Actuals, Transformer), MSEweekly(Actuals, Autoformer_encoder),
        MSEweekly(Actuals, Autoformer_full)
    ],
    'QLIKE': [
        QLIKEweekly(Actuals, GARCH), QLIKEweekly(Actuals, HAR), QLIKEweekly(Actuals, HAR_jump),
        QLIKEweekly(Actuals, CHAR), QLIKEweekly(Actuals, ARQ), QLIKEweekly(Actuals, HARQ), QLIKEweekly(Actuals, HARQF),
        QLIKEweekly(Actuals, LSTM), QLIKEweekly(Actuals, Encoder),
        QLIKEweekly(Actuals, Transformer), QLIKEweekly(Actuals, Autoformer_encoder),
        QLIKEweekly(Actuals, Autoformer_full)
    ],
    'MDA': [
        MDA_Weekly(Actuals, GARCH), MDA_Weekly(Actuals, HAR), MDA_Weekly(Actuals, HAR_jump),
        MDA_Weekly(Actuals, CHAR), MDA_Weekly(Actuals, ARQ), MDA_Weekly(Actuals, HARQ), MDA_Weekly(Actuals, HARQF),
        MDA_Weekly(Actuals, LSTM), MDA_Weekly(Actuals, Encoder),
        MDA_Weekly(Actuals, Transformer), MDA_Weekly(Actuals, Autoformer_encoder),
        MDA_Weekly(Actuals, Autoformer_full)
    ]
}



# Convert the dictionary to a DataFrame
metrics_dict_weekly = pd.DataFrame(metrics_dict_weekly)#.transpose()

# Display the resulting DataFrame
print(metrics_dict_weekly)


#%%

def MSE_7th_day(Actuals, model):
    MSE = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the DataFrames for the actual and model forecasts
        actual_stock = Actuals[stock]
        model_stock = model[stock]
        
        # Extract the 7th day's actual and model forecasts
        actual_7th_day = actual_stock.iloc[:, 6]  # 7th day is the 7th column (index 6)
        model_7th_day = model_stock.iloc[:, 6]
        
        # Calculate the squared differences between the actual and forecasted values for the 7th day
        squared_diff = (actual_7th_day - model_7th_day) ** 2
        
        # Calculate the Mean Squared Error (MSE) for the 7th day
        mse = squared_diff.mean()
        
        MSE.append(mse)

    # Calculate the average MSE for the 7th day across all stocks
    return np.mean(MSE)

def QLIKE_7th_day(Actuals, model):
    epsilon = 1e-10
    QLIKE = []

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the DataFrames for the actual and model forecasts
        actual_stock = Actuals[stock] + epsilon
        model_stock = model[stock] + epsilon
        
        # Extract the 7th day's actual and model forecasts
        actual_7th_day = actual_stock.iloc[:, 6]  # 7th day is the 7th column (index 6)
        model_7th_day = model_stock.iloc[:, 6]
        
        # Calculate the QLIKE loss for the 7th day
        qlike_stock = (actual_7th_day / model_7th_day) - np.log(actual_7th_day / model_7th_day) - 1

        # Check for any infinite or NaN values and print a warning if found
        if np.any(np.isinf(qlike_stock)) or np.any(np.isnan(qlike_stock)):
            print(f"Warning: Found inf or NaN in QLIKE calculation for stock {stock}")
            
        # Calculate the mean QLIKE for this stock for the 7th day
        qlike_mean = qlike_stock.mean()

        QLIKE.append(qlike_mean)

    # Return the average QLIKE for the 7th day across all stocks
    return np.mean(QLIKE)


#%%

metrics_dict_7th_day = {
    'Model': ['GARCH', 'HAR', 'HAR Jumps', 'CHAR', 'ARQ', 'HARQ', 'HARQF', 'LSTM', 'Encoder', 'Transformer', 'Autoformer Encoder', 'Autoformer Full'],
    'MSE_7th_day': [
        MSE_7th_day(Actuals, GARCH), MSE_7th_day(Actuals, HAR), MSE_7th_day(Actuals, HAR_jump),
        MSE_7th_day(Actuals, CHAR), MSE_7th_day(Actuals, ARQ), MSE_7th_day(Actuals, HARQ), MSE_7th_day(Actuals, HARQF),
        MSE_7th_day(Actuals, LSTM), MSE_7th_day(Actuals, Encoder),
        MSE_7th_day(Actuals, Transformer), MSE_7th_day(Actuals, Autoformer_encoder),
        MSE_7th_day(Actuals, Autoformer_full)
    ],
    'QLIKE_7th_day': [
        QLIKE_7th_day(Actuals, GARCH), QLIKE_7th_day(Actuals, HAR), QLIKE_7th_day(Actuals, HAR_jump),
        QLIKE_7th_day(Actuals, CHAR), QLIKE_7th_day(Actuals, ARQ), QLIKE_7th_day(Actuals, HARQ), QLIKE_7th_day(Actuals, HARQF),
        QLIKE_7th_day(Actuals, LSTM), QLIKE_7th_day(Actuals, Encoder),
        QLIKE_7th_day(Actuals, Transformer), QLIKE_7th_day(Actuals, Autoformer_encoder),
        QLIKE_7th_day(Actuals, Autoformer_full)
    ]
}


# Example usage:
# Convert the dictionary to a DataFrame
metrics_dict_7th_day = pd.DataFrame(metrics_dict_7th_day)#.transpose()

# Display the resulting DataFrame
print(metrics_dict_7th_day)



#%%

def MSE_first_column(Actuals, model):
    # Initialize an empty dictionary to store MSE for each stock
    mse_dict = {}
    qlike_dict = {}
    epsilon = 1e-10

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        actual_stock = Actuals[stock]
        model_stock = model[stock]

        # Ensure the DataFrame length is 699
        if len(actual_stock) == 699 and len(model_stock) == 699:
            # Calculate MSE for the first column
            actual_first_column = actual_stock.iloc[:, 0]  # First column of actual data
            model_first_column = model_stock.iloc[:, 0]    # First column of model data

            # Calculate the squared differences for each row
            squared_diff = (actual_first_column - model_first_column) ** 2
            # Store the squared differences in the dictionary
            mse_dict[stock] = squared_diff.values

            # Calculate QLIKE for the day
            qlike_day = ((actual_first_column+ epsilon) / (model_first_column + epsilon))- np.log((actual_first_column + epsilon) / (model_first_column+ epsilon)) - 1
            qlike_dict[stock] = qlike_day.values


    # Convert the dictionary to a DataFrame
    mse_df = pd.DataFrame(mse_dict).transpose()
    qlike_df = pd.DataFrame(qlike_dict).transpose()

    return mse_df, qlike_df

# Example usage:
test_mse_first_column, test_qlike_first_column = MSE_first_column(Actuals, Encoder)
print(test_mse_first_column)
print(test_qlike_first_column)



#%%

import matplotlib.pyplot as plt

def plot_mse_statistics_grouped(mse_df, group_size=40):
    # Calculate statistics for each day across all stocks
    mean = mse_df.mean(axis=0)
    median = mse_df.median(axis=0)


    # Group by specified group_size (e.g., 10)
    grouped_indices = np.arange(0, mse_df.shape[1], group_size)
    mean_grouped = [mean[i:i+group_size].mean() for i in grouped_indices]
    median_grouped = [median[i:i+group_size].mean() for i in grouped_indices]
    x_grouped = np.arange(1, 700, group_size)

    # Plotting the statistics
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_grouped, mean_grouped, label='Mean', linestyle='-')
    plt.plot(x_grouped, median_grouped, label='Median', linestyle='-')
    #print(mean_grouped)
    # Adding logarithmic scale to the y-axis
    #plt.yscale('log')

    # Adding labels and title
    plt.xlabel('Index (1-699, grouped by 10)')
    plt.ylabel('MSE (Log Scale)')
    plt.title('MSE Statistics Across All Stocks (Grouped and Logarithmic Scale)')
    plt.legend(title='Statistics')
    plt.grid(True)
    plt.show()



#%%

    
HAR_MSE_first_column, HAR_QLIKE_first_column = MSE_first_column(Actuals, HAR)
LSTM_MSE_first_column, LSTM_QLIKE_first_column = MSE_first_column(Actuals, LSTM)
Encoder_MSE_first_column, Encoder_QLIKE_first_column = MSE_first_column(Actuals, Encoder)
    
plot_mse_statistics_grouped( HAR_MSE_first_column)
#plot_mse_statistics_grouped( LSTM_MSE_first_column)
#plot_mse_statistics_grouped( Encoder_MSE_first_column)




#%%
import statsmodels.api as sm

def p_value(values):
    X = np.arange(len(values))
    X_with_const = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(values, X_with_const).fit()
    #print(model.summary())

    # Extract the p-value for the independent variable (x1)
    p_value_x1 = model.pvalues['x1'].round(4)
    
    # Return both the coefficient and the p-value as a list
    return p_value_x1


def Coef(values):
    X = np.arange(len(values))
    X_with_const = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(values, X_with_const).fit()
    #print(model.summary())
    # Extract and return the p-value for the independent variable (x1)
    coeff_x1 = (model.params['x1'])

    return coeff_x1




#%%


Pvalues_dict = {
    'Model': ['HAR', 'LSTM', 'Encoder'],
    'MSE median': [
        p_value(HAR_MSE_first_column.median()), 
        p_value(LSTM_MSE_first_column.median()), 
        p_value(Encoder_MSE_first_column.median())
    ],
    'MSE mean': [
        p_value(HAR_MSE_first_column.mean()), 
        p_value(LSTM_MSE_first_column.mean()), 
        p_value(Encoder_MSE_first_column.mean())
    ],
    'QLIKE median': [
        p_value(HAR_QLIKE_first_column.median()), 
        p_value(LSTM_QLIKE_first_column.median()), 
        p_value(Encoder_QLIKE_first_column.median())
    ],
    'QLIKE mean': [
        p_value(HAR_QLIKE_first_column.mean()), 
        p_value(LSTM_QLIKE_first_column.mean()), 
        p_value(Encoder_QLIKE_first_column.mean())
    ]
}

# Convert the dictionary to a DataFrame
Pvalues_df = pd.DataFrame(Pvalues_dict)

# Display the resulting DataFrame
print(Pvalues_df)



print(Coef(LSTM_MSE_first_column.mean()))
print(Coef(Encoder_MSE_first_column.mean()))
print(Coef(Encoder_QLIKE_first_column.mean()))



#%%


def MSE_and_QLIKE_per_day(Actuals, model):
    # Initialize dictionaries to store MSE and QLIKE values for each stock and each day
    mse_dict = {stock: [] for stock in Actuals.keys()}
    qlike_dict = {stock: [] for stock in Actuals.keys()}
    epsilon = 1e-10
    
    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        # Extract the DataFrames for the actual and model forecasts
        actual_stock = Actuals[stock]
        model_stock = model[stock]
        
        # Loop through the 7 forecast days
        for day in range(7):
            # Extract the nth day's actual and model forecasts
            actual_day = actual_stock.iloc[:, day]
            model_day = model_stock.iloc[:, day]
            
            # Calculate the squared differences between the actual and forecasted values for the nth day (MSE)
            squared_diff = (actual_day - model_day) ** 2
            mse = squared_diff.mean()
            
            # QLIKE calculation: -log(model_day) + actual_day / model_day
            # Ensure no divide-by-zero or log(0) errors
            #model_day_safe = model_day.replace(0, 1e-10)  # Replace 0 in model_day to avoid divide by zero
            #qlike = (-np.log(model_day_safe) + actual_day / model_day_safe).mean()
            
            qlike_day = ((actual_day+ epsilon) / (model_day + epsilon))- np.log((actual_day + epsilon) / (model_day+ epsilon)) - 1
            qlike = qlike_day.mean()
            
            # Append the MSE and QLIKE to the corresponding stock in the dictionaries
            mse_dict[stock].append(mse)
            qlike_dict[stock].append(qlike)
    
    # Convert the dictionaries to DataFrames for easy viewing and analysis
    mse_df = pd.DataFrame.from_dict(mse_dict, orient='index', columns=[f'Day_{i+1}' for i in range(7)])
    qlike_df = pd.DataFrame.from_dict(qlike_dict, orient='index', columns=[f'Day_{i+1}' for i in range(7)])
    
    return mse_df, qlike_df



def plot_mse_qlike_boxplots(Actuals, model1, model2):
    model_1_MSE,  model_1_QLIKE = MSE_and_QLIKE_per_day(Actuals, model1)
    model_2_MSE,  model_2_QLIKE = MSE_and_QLIKE_per_day(Actuals, model2)
    # Create a figure with a 2x2 grid for the subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # First subplot: MSE DataFrame 1
    model_1_MSE.boxplot(column=[f'Day_{i+1}' for i in range(7)], showfliers=False, ax=axs[0, 0])
    #axs[0, 0].set_title('MSE LSTM')
    axs[0, 0].set_title('LSTM')
    #axs[0, 0].set_xlabel('Forecast Day')
    axs[0, 0].set_ylabel('MSE')
    axs[0, 0].set_ylim(0, 0.0005)

    # Second subplot: MSE DataFrame 2
    model_2_MSE.boxplot(column=[f'Day_{i+1}' for i in range(7)], showfliers=False, ax=axs[0, 1])
    #axs[0, 1].set_title('MSE Encoder')
    axs[0, 1].set_title('Encoder')
    #axs[0, 1].set_xlabel('Forecast Day')
    #axs[0, 1].set_ylabel('MSE ')
    axs[0, 0].set_ylim(0, 0.0005)

    # Third subplot: QLIKE DataFrame 1
    model_1_QLIKE.boxplot(column=[f'Day_{i+1}' for i in range(7)], showfliers=False, ax=axs[1, 0])
    #axs[1, 0].set_title('QLIKE LSTM')
    axs[1, 0].set_xlabel('Forecast Day')
    axs[1, 0].set_ylabel('QLIKE ')
    axs[1, 0].set_ylim(0, 0.35) 

    # Fourth subplot: QLIKE DataFrame 2
    model_2_QLIKE.boxplot(column=[f'Day_{i+1}' for i in range(7)], showfliers=False, ax=axs[1, 1])
    #axs[1, 1].set_title('QLIKE Encoder')
    axs[1, 1].set_xlabel('Forecast Day')
    #axs[1, 1].set_ylabel('QLIKE')
    axs[1, 0].set_ylim(0, 0.35)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    # Show the plot
    plt.show()


plot_mse_qlike_boxplots(Actuals, LSTM, Encoder)

#%%


total_rows = sum(df.shape[0] for df in Encoder.values())

print(f"Total number of rows in all DataFrames: {total_rows}")















