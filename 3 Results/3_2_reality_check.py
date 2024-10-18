import pickle
#from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

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


#GARCH FIX
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
            actual_first_column = actual_stock.iloc[:, 6]  # First column of actual data
            model_first_column = model_stock.iloc[:, 6]    # First column of model data

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


def stationary_bootstrap_test_mse(model_mse_df, benchmark_mse_df, block_size=5, n_bootstrap=999):
    # Ensure the two DataFrames have the same shape
    assert model_mse_df.shape == benchmark_mse_df.shape, "Model and Benchmark DataFrames must have the same shape."
    
    # Calculate loss differences (benchmark MSE - model MSE)
    loss_diff = benchmark_mse_df - model_mse_df
    
    # Flatten the loss differences into a 1D array for bootstrapping
    loss_diff_flat = loss_diff.values.flatten()
    
    # Initialize stationary bootstrap
    bs = StationaryBootstrap(block_size, loss_diff_flat)
    
    # Bootstrap distribution of loss differences
    bootstrap_distributions = []
    
    for data in bs.bootstrap(n_bootstrap):
        resampled_loss_diff = data[0]  # Resampled loss differences
        bootstrap_distributions.append(np.mean(resampled_loss_diff))  # Mean of resampled loss differences
    
    # Convert bootstrap results to numpy array for further analysis
    bootstrap_distributions = np.array(bootstrap_distributions)
    print(bootstrap_distributions)
    # Calculate p-value (two-tailed test)
    p_value = np.mean(bootstrap_distributions <= 0)  # Proportion of times where the difference is >= 0
    
    # Calculate 95% confidence interval
    conf_interval = np.percentile(bootstrap_distributions, [2.5, 97.5])
    
    return p_value, conf_interval



#%%


GARCH_mse_first_column, GARCH_qlike_first_column = MSE_first_column(Actuals, GARCH)
HAR_mse_first_column, HAR_qlike_first_column = MSE_first_column(Actuals, HAR)
HAR_jump_mse_first_column, HAR_jump_qlike_first_column = MSE_first_column(Actuals, HAR_jump)
CHAR_mse_first_column, CHAR_qlike_first_column = MSE_first_column(Actuals, CHAR)
ARQ_mse_first_column, ARQ_qlike_first_column = MSE_first_column(Actuals, ARQ)
HARQ_mse_first_column, HARQ_qlike_first_column = MSE_first_column(Actuals, HARQ)

LSTM_mse_first_column, LSTM_qlike_first_column = MSE_first_column(Actuals, LSTM)
Encoder_mse_first_column, Encoder_qlike_first_column = MSE_first_column(Actuals, Encoder)
Transformer_mse_first_column, Transformer_qlike_first_column = MSE_first_column(Actuals, Transformer)
Autoformer_encoder_mse_first_column, Autoformer_encoder_qlike_first_column = MSE_first_column(Actuals, Autoformer_encoder)
Autoformer_full_mse_first_column, Autoformer_full_qlike_first_column = MSE_first_column(Actuals, Autoformer_full)

#%%

p_value, conf_interval = stationary_bootstrap_test_mse(HAR_mse_first_column, HARQ_mse_first_column)
        
print(p_value)

#%%

result = HAR_mse_first_column - HARQ_mse_first_column

print(result)


positive_mask = result > 0

# Step 2: Sum the True values to count positives
positive_count = positive_mask.sum().sum()

print(f"Number of positive values: {positive_count}")
print(positive_count/350000)
#%%



garch_har_models_mse = {
    'GARCH': GARCH_mse_first_column,
    'HAR': HAR_mse_first_column,
    'HAR Jump': HAR_jump_mse_first_column,
    'CHAR': CHAR_mse_first_column,
    'ARQ': ARQ_mse_first_column,
    'HARQ': HARQ_mse_first_column,
    'HARQF': HARQ_mse_first_column
}

ml_models_mse = {
    'LSTM': LSTM_mse_first_column,
    'Encoder': Encoder_mse_first_column,
    'Transformer': Transformer_mse_first_column,
    'Autoformer Encoder': Autoformer_encoder_mse_first_column,
    'Autoformer Full': Autoformer_full_mse_first_column
}

# Create an empty list to store results
results = []

# Loop through each ML model and compare it to each GARCH/HAR model
for ml_model_name, ml_mse in ml_models_mse.items():
    print(ml_model_name)
    print("$$")
    for benchmark_name, benchmark_mse in garch_har_models_mse.items():
        print(benchmark_name)
        # Perform stationary bootstrap test
        p_value, conf_interval = stationary_bootstrap_test_mse(ml_mse, benchmark_mse)
        
        # Store the results in a dictionary
        result = {
            'ML Model': ml_model_name,
            'Benchmark Model': benchmark_name,
            'P-value': p_value,
            '95% Confidence Interval': conf_interval
        }
        results.append(result)

# Convert the results to a DataFrame for easier visualization
results_df = pd.DataFrame(results)

p_values_pivot = results_df.pivot(index='ML Model', columns='Benchmark Model', values='P-value')

# Display the new DataFrame
print(p_values_pivot)

#%%

garch_har_models_Qlike = {
    'GARCH': GARCH_qlike_first_column,
    'HAR': HAR_qlike_first_column,
    'HAR Jump': HAR_jump_qlike_first_column,
    'CHAR': CHAR_qlike_first_column,
    'ARQ': ARQ_qlike_first_column,
    'HARQ': HARQ_qlike_first_column,
    'HARQF': HARQ_qlike_first_column
}

ml_models_Qlike = {
    'LSTM': LSTM_qlike_first_column,
    'Encoder': Encoder_qlike_first_column,
    'Transformer': Transformer_qlike_first_column,
    'Autoformer Encoder': Autoformer_encoder_qlike_first_column,
    'Autoformer Full': Autoformer_full_qlike_first_column
}


# Create an empty list to store results
results_Qlike = []

# Loop through each ML model and compare it to each GARCH/HAR model
for ml_model_name, ml_Qlike in ml_models_Qlike.items():
    print(ml_model_name)
    print("$$")
    for benchmark_name, benchmark_Qlike in garch_har_models_Qlike.items():
        print(benchmark_name)
        # Perform stationary bootstrap test
        p_value, conf_interval = stationary_bootstrap_test_mse(ml_Qlike, benchmark_Qlike)
        
        # Store the results in a dictionary
        result = {
            'ML Model': ml_model_name,
            'Benchmark Model': benchmark_name,
            'P-value': p_value,
            '95% Confidence Interval': conf_interval
        }
        results_Qlike.append(result)

# Convert the results to a DataFrame for easier visualization
results_df_Qlike = pd.DataFrame(results_Qlike)

# Display the resulting DataFrame
print(results_df_Qlike)

#%%

p_values_pivot_Qlike = results_df_Qlike.pivot(index='ML Model', columns='Benchmark Model', values='P-value')

# Display the new DataFrame
print(p_values_pivot_Qlike)


#%%

p_value, conf_interval = stationary_bootstrap_test_mse(LSTM_mse_first_column, Encoder_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(LSTM_mse_first_column, Transformer_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(LSTM_mse_first_column, Autoformer_encoder_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(LSTM_mse_first_column, Autoformer_full_mse_first_column)
print(p_value)

print("$$")

p_value, conf_interval = stationary_bootstrap_test_mse(Encoder_mse_first_column, Transformer_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(Encoder_mse_first_column, Autoformer_encoder_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(Encoder_mse_first_column, Autoformer_full_mse_first_column)
print(p_value)


print("$$")

p_value, conf_interval = stationary_bootstrap_test_mse(Transformer_mse_first_column, Autoformer_encoder_mse_first_column)
print(p_value)
p_value, conf_interval = stationary_bootstrap_test_mse(Transformer_mse_first_column, Autoformer_full_mse_first_column)
print(p_value)

print("$$")

p_value, conf_interval = stationary_bootstrap_test_mse(Autoformer_encoder_mse_first_column, Autoformer_full_mse_first_column)
print(p_value)

#%%

def MSE_weekly(Actuals, model):
    MSE = []
    epsilon = 1e-10
    QLIKE = []
    processed_stocks = []  # To store names of stocks that passed the length check

    # Loop through each stock in the dictionaries
    for stock in Actuals.keys():
        actual_stock = Actuals[stock]
        model_stock = model[stock]

        # Ensure the DataFrame length is 699
        if len(actual_stock) == 699 and len(model_stock) == 699:
            actual_stock += epsilon
            model_stock += epsilon

            # Step 1: Calculate the mean of each row (i.e., across the 7 days) for actuals and model forecasts
            actual_mean = actual_stock.mean(axis=1)
            model_mean = model_stock.mean(axis=1)

            # Step 2: Calculate the squared differences between the means
            squared_diff = (actual_mean - model_mean) ** 2
            MSE.append(squared_diff)

            # Calculate the QLIKE loss for each element
            qlike_stock = (actual_mean / model_mean) - np.log(actual_mean / model_mean) - 1

            if np.any(np.isinf(qlike_stock)) or np.any(np.isnan(qlike_stock)):
                print(f"Warning: Found inf or NaN in QLIKE calculation for stock {stock}")

            QLIKE.append(qlike_stock)
            processed_stocks.append(stock)  # Store the name of the processed stock

    # Step 4: Convert the lists of Series into DataFrames
    if MSE:  # Check if the list is not empty
        MSE_df = pd.concat(MSE, axis=1)
        QLIKE_df = pd.concat(QLIKE, axis=1)

        # Use only the processed stock names
        MSE_df.columns = processed_stocks
        QLIKE_df.columns = processed_stocks

        return MSE_df, QLIKE_df
    else:
        print("No stocks with length 699 found.")
        return None, None






#%%


GARCH_mse_weekly, GARCH_qlike_weekly = MSE_weekly(Actuals, GARCH)
HAR_mse_weekly, HAR_qlike_weekly = MSE_weekly(Actuals, HAR)
HAR_jump_mse_weekly, HAR_jump_qlike_weekly = MSE_weekly(Actuals, HAR_jump)
CHAR_mse_weekly, CHAR_qlike_weekly = MSE_weekly(Actuals, CHAR)
ARQ_mse_weekly, ARQ_qlike_weekly = MSE_weekly(Actuals, ARQ)
HARQ_mse_weekly, HARQ_qlike_weekly = MSE_weekly(Actuals, HARQ)

LSTM_mse_weekly, LSTM_qlike_weekly = MSE_weekly(Actuals, LSTM)
Encoder_mse_weekly, Encoder_qlike_weekly = MSE_weekly(Actuals, Encoder)
Transformer_mse_weekly, Transformer_qlike_weekly = MSE_weekly(Actuals, Transformer)
Autoformer_encoder_mse_weekly, Autoformer_encoder_qlike_weekly = MSE_weekly(Actuals, Autoformer_encoder)
Autoformer_full_mse_weekly, Autoformer_full_qlike_weekly = MSE_weekly(Actuals, Autoformer_full)





#%%


garch_har_models_mse = {
    'GARCH': GARCH_mse_weekly,
    'HAR': HAR_mse_weekly,
    'HAR Jump': HAR_jump_mse_weekly,
    'CHAR': CHAR_mse_weekly,
    'ARQ': ARQ_mse_weekly,
    'HARQ': HARQ_mse_weekly,
    'HARQF': HARQ_mse_weekly
}

ml_models_mse_weekly = {
    'LSTM': LSTM_mse_weekly,
    'Encoder': Encoder_mse_weekly,
    'Transformer': Transformer_mse_weekly,
    'Autoformer Encoder': Autoformer_encoder_mse_weekly,
    'Autoformer Full': Autoformer_full_mse_weekly
}

# Create an empty list to store results
results_weekly = []

# Loop through each ML model and compare it to each GARCH/HAR model
for ml_model_name, ml_mse in ml_models_mse_weekly.items():
    print(ml_model_name)
    print("$$")
    for benchmark_name, benchmark_mse in garch_har_models_mse.items():
        print(benchmark_name)
        # Perform stationary bootstrap test
        p_value, conf_interval = stationary_bootstrap_test_mse(ml_mse, benchmark_mse)
        
        # Store the results in a dictionary
        result = {
            'ML Model': ml_model_name,
            'Benchmark Model': benchmark_name,
            'P-value': p_value,
            '95% Confidence Interval': conf_interval
        }
        # Append the result to the correct list (results_weekly)
        results_weekly.append(result)

# Convert the results to a DataFrame for easier visualization
results_df_weekly = pd.DataFrame(results_weekly)

# Pivot the results DataFrame to create a comparison table of p-values
p_values_pivot_weekly = results_df_weekly.pivot(index='ML Model', columns='Benchmark Model', values='P-value')

# Display the new DataFrame
print(p_values_pivot_weekly)













