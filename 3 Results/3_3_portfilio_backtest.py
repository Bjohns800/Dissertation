import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load data
prices = pd.read_csv('daily_close.csv', index_col='Date', parse_dates=True)

with open('LSTM_Backup.pkl', 'rb') as file:
    LSTM_forecasts_dict = pickle.load(file)

with open('Encoder_Backup.pkl', 'rb') as file:
    Encoder_forecasts_dict = pickle.load(file)

for key in LSTM_forecasts_dict.keys():
    LSTM_forecasts_dict[key] = LSTM_forecasts_dict[key].iloc[:-6]
    Encoder_forecasts_dict[key] = Encoder_forecasts_dict[key].iloc[:-6]

def popfunction(dictionary):
    dictionary.pop('SLE', None) 
    dictionary.pop('SBNY', None) 
    dictionary.pop('CPWR', None) 
    return dictionary

Encoder_forecasts_dict = popfunction(Encoder_forecasts_dict)
LSTM_forecasts_dict = popfunction(LSTM_forecasts_dict)

column_names = prices.columns
LSTM_forecasts_dict = {key: LSTM_forecasts_dict[key] for key in column_names if key in LSTM_forecasts_dict}
Encoder_forecasts_dict = {key: Encoder_forecasts_dict[key] for key in column_names if key in Encoder_forecasts_dict}

test_period_start_idx = int(len(prices) * 0.8)
test_period_start_date = prices.index[test_period_start_idx]

# Step 1: Calculate daily log returns from price data
log_returns = np.log(prices / prices.shift(1)).dropna()

# Filter the returns to only include the test period
test_returns = log_returns.loc[test_period_start_date:].iloc[:-6]

# Step 2: Function to extract the relevant forecast for each stock for a given period
def get_forecast_for_period(stock, rebalance_days, i, vol_forecasts_dict):
    forecast_df = vol_forecasts_dict[stock]
    try:
        forecast = forecast_df.iloc[i, 0:rebalance_days].tolist()
    except KeyError:
        print(f"Missing forecast for {stock} for {rebalance_days} days")
        forecast = [np.nan] * rebalance_days  # Return NaNs if missing data
    return forecast

# Function to calculate the volatility-weighted portfolio
def volatility_weighted_portfolio(vol_forecasts_dict, rebalance_days, i):
    num_safest=250
    stocks = vol_forecasts_dict.keys()
    avg_vols = {}
    
    # Step 1: Calculate the average forecasted volatility for each stock
    for stock in stocks:
        forecast_vols = get_forecast_for_period(stock, rebalance_days, i, vol_forecasts_dict)
        
        # Average volatility over the rebalance period
        avg_vol = np.mean([vol for vol in forecast_vols if not np.isnan(vol)])
        
        if avg_vol > 0:  # Avoid division by zero
            avg_vols[stock] = avg_vol
    
    # Step 2: Filter to only include the 100 stocks with the lowest volatility
    safest_stocks = sorted(avg_vols, key=avg_vols.get)[:num_safest]
    
    # Step 3: Calculate inverse volatilities for the 100 safest stocks
    inverse_vols = {stock: 1 / avg_vols[stock] for stock in safest_stocks}
    
    # Step 4: Normalize the inverse volatilities to sum to 1 (creating the weights)
    total_inverse_vol = sum(inverse_vols.values())
    
    weights = {stock: inverse_vol / total_inverse_vol for stock, inverse_vol in inverse_vols.items() if total_inverse_vol > 0}
    
    return weights

# Function to simulate the volatility-weighted portfolio
def simulate_volatility_weighted_portfolio(returns, vol_forecasts_dict, rebalance_days, test_period_days, transaction_cost_per_trade=0.001):
    portfolio_values = [1]  # Start with an initial portfolio value of 1
    previous_weights = None  # To store weights from the previous period
    transaction_costs = []  # To store daily transaction costs
    post_transaction_portfolio_values = [1]  # Start with initial post-transaction portfolio value of 1
    
    for i in range(0, test_period_days, rebalance_days):
        # Calculate volatility-weighted portfolio weights for the current period
        current_weights = volatility_weighted_portfolio(vol_forecasts_dict, rebalance_days, i)
        
        # Calculate volatility-weighted portfolio weights for the next period
        if i + rebalance_days < test_period_days:
            next_weights = volatility_weighted_portfolio(vol_forecasts_dict, rebalance_days, i + rebalance_days)
        else:
            next_weights = current_weights  # If it's the last period, use current weights
        
        # Average the current and next weights
        averaged_weights = {stock: (current_weights.get(stock, 0) + next_weights.get(stock, 0)) / 2 
                            for stock in returns.columns}
        
        # Calculate the portfolio return for the next rebalance period
        period_returns = returns.iloc[i:i + rebalance_days]
        
        # Convert averaged weights to a list in the same order as the returns DataFrame columns
        weights_list = np.array([averaged_weights[stock] if stock in averaged_weights else 0 for stock in returns.columns])
        
        # If previous weights exist, calculate the absolute change and calculate transaction costs
        if previous_weights is not None:
            weight_change = np.sum(np.abs(weights_list - previous_weights))  # Absolute change in weights
            transaction_cost = weight_change * transaction_cost_per_trade  # Calculate transaction cost
        else:
            transaction_cost = 0  # No transaction cost for the first period
        
        # Append daily transaction cost
        transaction_costs.append(transaction_cost)
        
        # Update previous_weights for the next iteration
        previous_weights = weights_list
        
        # Step 1: Calculate log returns over the rebalance period (sum of log returns)
        compounding_returns = period_returns.sum(axis=0)  # Sum of log returns over the period
        
        # Step 2: Apply portfolio weights to get the weighted portfolio return
        portfolio_return = np.dot(weights_list, compounding_returns)  # Directly use log returns
        
        # Update portfolio value
        portfolio_values.append(portfolio_values[-1] * np.exp(portfolio_return))
        
        # Apply transaction cost to the portfolio value (scaled by current portfolio value)
        post_transaction_value = post_transaction_portfolio_values[-1] * np.exp(portfolio_return)  # Portfolio value after return
        transaction_cost_scaled = transaction_cost * post_transaction_value  # Scale the transaction cost by current value
        post_transaction_portfolio_values.append(post_transaction_value - transaction_cost_scaled)  # Subtract transaction cost
    
    # Convert portfolio values and returns to arrays
    portfolio_values = np.array(portfolio_values)
    post_transaction_portfolio_values = np.array(post_transaction_portfolio_values)
    
    # Calculate daily returns
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    post_transaction_returns = np.diff(post_transaction_portfolio_values) / post_transaction_portfolio_values[:-1]
    
    return portfolio_values, portfolio_returns, transaction_costs, post_transaction_portfolio_values, post_transaction_returns






# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(portfolio_returns, rebalance_days, risk_free_rate=0.0256):
    avg_period_return = np.mean(portfolio_returns)
    
    # Annualize the portfolio's average return
    annualized_return = (1 + avg_period_return) ** (252 / rebalance_days) - 1
    
    # Calculate portfolio return volatility (standard deviation of returns for the rebalance period)
    portfolio_volatility_period = np.std(portfolio_returns)
    
    # Annualize the volatility based on the rebalance frequency
    annualized_volatility = portfolio_volatility_period * np.sqrt(252 / rebalance_days)
    print(annualized_volatility)
    # Calculate the annualized excess return (portfolio return minus the annualized risk-free rate)
    excess_return = annualized_return - risk_free_rate
    
    # Calculate Sharpe Ratio
    sharpe_ratio = excess_return / annualized_volatility
    
    return sharpe_ratio


# Parameters for test period
portfolio_size = 1
transaction_cost_per_trade = 0.001
rebalance_days = 1
test_period_days = len(test_returns)  # Test period is now the length of the test returns

#%%

rates = {
    "Q2 2021": 0.10,
    "Q3 2021": 0.10,
    "Q4 2021": 0.10,
    "Q1 2022": 0.20,
    "Q2 2022": 1.00,
    "Q3 2022": 2.50,
    "Q4 2022": 4.00,
    "Q1 2023": 4.50,
    "Q2 2023": 5.00,
    "Q3 2023": 5.25,
    "Q4 2023": 5.40,
}

# Calculate the average risk-free rate over the entire period
average_rate = sum(rates.values()) / len(rates)
average_rate

#%%

# Step 6: Run the simulation using test set data for 1-day rebalancing
LSTM_portfolio_values, LSTM_portfolio_returns, LSTM_transaction_costs, LSTM_post_transaction_portfolio_values, LSTM_post_transaction_returns = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=1, test_period_days=test_period_days)

Encoder_portfolio_values, Encoder_portfolio_returns, Encoder_transaction_costs, Encoder_post_transaction_portfolio_values, Encoder_post_transaction_returns = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=1, test_period_days=test_period_days)


# Calculate Sharpe ratios for the 1-day rebalancing
LSTM_sharpe = calculate_sharpe_ratio(LSTM_portfolio_returns, rebalance_days=1)
Encoder_sharpe = calculate_sharpe_ratio(Encoder_portfolio_returns, rebalance_days=1)

# Calculate adjusted Sharpe ratios for the 1-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted = calculate_sharpe_ratio(LSTM_post_transaction_returns, rebalance_days=1)
Encoder_sharpe_adjusted = calculate_sharpe_ratio(Encoder_post_transaction_returns, rebalance_days=1)

# Print results for 1-day rebalancing
print("LSTM Sharpe Ratio (1-day):", LSTM_sharpe)
print("Encoder Sharpe Ratio (1-day):", Encoder_sharpe)
print("LSTM Total Transaction Costs (1-day):", sum(LSTM_transaction_costs))
print("Encoder Total Transaction Costs (1-day):", sum(Encoder_transaction_costs))
print("LSTM Sharpe Ratio (Adjusted, 1-day):", LSTM_sharpe_adjusted)
print("Encoder Sharpe Ratio (Adjusted, 1-day):", Encoder_sharpe_adjusted)



#%%
# Step 7: Run the simulation using test set data for 2-day rebalancing
# Step 7: Run the simulation using test set data for 2-day rebalancing
LSTM_portfolio_values2, LSTM_portfolio_returns2, LSTM_transaction_costs2, LSTM_post_transaction_portfolio_values2, LSTM_post_transaction_returns2 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=2, test_period_days=test_period_days)
Encoder_portfolio_values2, Encoder_portfolio_returns2, Encoder_transaction_costs2, Encoder_post_transaction_portfolio_values2, Encoder_post_transaction_returns2 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=2, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 2-day rebalancing
LSTM_sharpe2 = calculate_sharpe_ratio(LSTM_portfolio_returns2, rebalance_days=2)
Encoder_sharpe2 = calculate_sharpe_ratio(Encoder_portfolio_returns2, rebalance_days=2)

# Calculate adjusted Sharpe ratios for the 2-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted2 = calculate_sharpe_ratio(LSTM_post_transaction_returns2, rebalance_days=2)
Encoder_sharpe_adjusted2 = calculate_sharpe_ratio(Encoder_post_transaction_returns2, rebalance_days=2)

# Print results for 2-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (2-day):", LSTM_sharpe2)
print("Encoder Sharpe Ratio (2-day):", Encoder_sharpe2)
print("LSTM Total Transaction Costs (2-day):", sum(LSTM_transaction_costs2))
print("Encoder Total Transaction Costs (2-day):", sum(Encoder_transaction_costs2))
print("LSTM Sharpe Ratio (Adjusted, 2-day):", LSTM_sharpe_adjusted2)
print("Encoder Sharpe Ratio (Adjusted, 2-day):", Encoder_sharpe_adjusted2)


# Step 7: Run the simulation using test set data for 3-day rebalancing
LSTM_portfolio_values3, LSTM_portfolio_returns3, LSTM_transaction_costs3, LSTM_post_transaction_portfolio_values3, LSTM_post_transaction_returns3 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=3, test_period_days=test_period_days)
Encoder_portfolio_values3, Encoder_portfolio_returns3, Encoder_transaction_costs3, Encoder_post_transaction_portfolio_values3, Encoder_post_transaction_returns3 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=3, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 3-day rebalancing
LSTM_sharpe3 = calculate_sharpe_ratio(LSTM_portfolio_returns3, rebalance_days=3)
Encoder_sharpe3 = calculate_sharpe_ratio(Encoder_portfolio_returns3, rebalance_days=3)

# Calculate adjusted Sharpe ratios for the 3-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted3 = calculate_sharpe_ratio(LSTM_post_transaction_returns3, rebalance_days=3)
Encoder_sharpe_adjusted3 = calculate_sharpe_ratio(Encoder_post_transaction_returns3, rebalance_days=3)

# Print results for 3-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (3-day):", LSTM_sharpe3)
print("Encoder Sharpe Ratio (3-day):", Encoder_sharpe3)
print("LSTM Total Transaction Costs (3-day):", sum(LSTM_transaction_costs3))
print("Encoder Total Transaction Costs (3-day):", sum(Encoder_transaction_costs3))
print("LSTM Sharpe Ratio (Adjusted, 3-day):", LSTM_sharpe_adjusted3)
print("Encoder Sharpe Ratio (Adjusted, 3-day):", Encoder_sharpe_adjusted3)


# Step 7: Run the simulation using test set data for 4-day rebalancing
LSTM_portfolio_values4, LSTM_portfolio_returns4, LSTM_transaction_costs4, LSTM_post_transaction_portfolio_values4, LSTM_post_transaction_returns4 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=4, test_period_days=test_period_days)
Encoder_portfolio_values4, Encoder_portfolio_returns4, Encoder_transaction_costs4, Encoder_post_transaction_portfolio_values4, Encoder_post_transaction_returns4 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=4, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 4-day rebalancing
LSTM_sharpe4 = calculate_sharpe_ratio(LSTM_portfolio_returns4, rebalance_days=4)
Encoder_sharpe4 = calculate_sharpe_ratio(Encoder_portfolio_returns4, rebalance_days=4)

# Calculate adjusted Sharpe ratios for the 4-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted4 = calculate_sharpe_ratio(LSTM_post_transaction_returns4, rebalance_days=4)
Encoder_sharpe_adjusted4 = calculate_sharpe_ratio(Encoder_post_transaction_returns4, rebalance_days=4)

# Print results for 4-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (4-day):", LSTM_sharpe4)
print("Encoder Sharpe Ratio (4-day):", Encoder_sharpe4)
print("LSTM Total Transaction Costs (4-day):", sum(LSTM_transaction_costs4))
print("Encoder Total Transaction Costs (4-day):", sum(Encoder_transaction_costs4))
print("LSTM Sharpe Ratio (Adjusted, 4-day):", LSTM_sharpe_adjusted4)
print("Encoder Sharpe Ratio (Adjusted, 4-day):", Encoder_sharpe_adjusted4)


# Step 7: Run the simulation using test set data for 5-day rebalancing
LSTM_portfolio_values5, LSTM_portfolio_returns5, LSTM_transaction_costs5, LSTM_post_transaction_portfolio_values5, LSTM_post_transaction_returns5 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=5, test_period_days=test_period_days)
Encoder_portfolio_values5, Encoder_portfolio_returns5, Encoder_transaction_costs5, Encoder_post_transaction_portfolio_values5, Encoder_post_transaction_returns5 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=5, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 5-day rebalancing
LSTM_sharpe5 = calculate_sharpe_ratio(LSTM_portfolio_returns5, rebalance_days=5)
Encoder_sharpe5 = calculate_sharpe_ratio(Encoder_portfolio_returns5, rebalance_days=5)

# Calculate adjusted Sharpe ratios for the 5-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted5 = calculate_sharpe_ratio(LSTM_post_transaction_returns5, rebalance_days=5)
Encoder_sharpe_adjusted5 = calculate_sharpe_ratio(Encoder_post_transaction_returns5, rebalance_days=5)

# Print results for 5-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (5-day):", LSTM_sharpe5)
print("Encoder Sharpe Ratio (5-day):", Encoder_sharpe5)
print("LSTM Total Transaction Costs (5-day):", sum(LSTM_transaction_costs5))
print("Encoder Total Transaction Costs (5-day):", sum(Encoder_transaction_costs5))
print("LSTM Sharpe Ratio (Adjusted, 5-day):", LSTM_sharpe_adjusted5)
print("Encoder Sharpe Ratio (Adjusted, 5-day):", Encoder_sharpe_adjusted5)


# Step 7: Run the simulation using test set data for 6-day rebalancing
LSTM_portfolio_values6, LSTM_portfolio_returns6, LSTM_transaction_costs6, LSTM_post_transaction_portfolio_values6, LSTM_post_transaction_returns6 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=6, test_period_days=test_period_days)
Encoder_portfolio_values6, Encoder_portfolio_returns6, Encoder_transaction_costs6, Encoder_post_transaction_portfolio_values6, Encoder_post_transaction_returns6 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=6, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 6-day rebalancing
LSTM_sharpe6 = calculate_sharpe_ratio(LSTM_portfolio_returns6, rebalance_days=6)
Encoder_sharpe6 = calculate_sharpe_ratio(Encoder_portfolio_returns6, rebalance_days=6)

# Calculate adjusted Sharpe ratios for the 6-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted6 = calculate_sharpe_ratio(LSTM_post_transaction_returns6, rebalance_days=6)
Encoder_sharpe_adjusted6 = calculate_sharpe_ratio(Encoder_post_transaction_returns6, rebalance_days=6)

# Print results for 6-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (6-day):", LSTM_sharpe6)
print("Encoder Sharpe Ratio (6-day):", Encoder_sharpe6)
print("LSTM Total Transaction Costs (6-day):", sum(LSTM_transaction_costs6))
print("Encoder Total Transaction Costs (6-day):", sum(Encoder_transaction_costs6))
print("LSTM Sharpe Ratio (Adjusted, 6-day):", LSTM_sharpe_adjusted6)
print("Encoder Sharpe Ratio (Adjusted, 6-day):", Encoder_sharpe_adjusted6)


# Step 7: Run the simulation using test set data for 7-day rebalancing
LSTM_portfolio_values7, LSTM_portfolio_returns7, LSTM_transaction_costs7, LSTM_post_transaction_portfolio_values7, LSTM_post_transaction_returns7 = simulate_volatility_weighted_portfolio(
    test_returns, LSTM_forecasts_dict, rebalance_days=7, test_period_days=test_period_days)
Encoder_portfolio_values7, Encoder_portfolio_returns7, Encoder_transaction_costs7, Encoder_post_transaction_portfolio_values7, Encoder_post_transaction_returns7 = simulate_volatility_weighted_portfolio(
    test_returns, Encoder_forecasts_dict, rebalance_days=7, test_period_days=test_period_days)

# Calculate Sharpe ratios for the 7-day rebalancing
LSTM_sharpe7 = calculate_sharpe_ratio(LSTM_portfolio_returns7, rebalance_days=7)
Encoder_sharpe7 = calculate_sharpe_ratio(Encoder_portfolio_returns7, rebalance_days=7)

# Calculate adjusted Sharpe ratios for the 7-day rebalancing (after transaction costs)
LSTM_sharpe_adjusted7 = calculate_sharpe_ratio(LSTM_post_transaction_returns7, rebalance_days=7)
Encoder_sharpe_adjusted7 = calculate_sharpe_ratio(Encoder_post_transaction_returns7, rebalance_days=7)

# Print results for 7-day rebalancing
print("$$")
print("LSTM Sharpe Ratio (7-day):", LSTM_sharpe7)
print("Encoder Sharpe Ratio (7-day):", Encoder_sharpe7)
print("LSTM Total Transaction Costs (7-day):", sum(LSTM_transaction_costs7))
print("Encoder Total Transaction Costs (7-day):", sum(Encoder_transaction_costs7))
print("LSTM Sharpe Ratio (Adjusted, 7-day):", LSTM_sharpe_adjusted7)
print("Encoder Sharpe Ratio (Adjusted, 7-day):", Encoder_sharpe_adjusted7)





#%%

# Load the S&P 500 data
snp500_data = pd.read_excel('SNP500.xlsx')
snp500_price = snp500_data.iloc[:, 1]

# Calculate log returns for different rebalance intervals
snp500_log_returns_daily = np.log(snp500_price / snp500_price.shift(1)).dropna()
snp500_log_returns_2d = np.log(snp500_price[::2] / snp500_price[::2].shift(1)).dropna()
snp500_log_returns_3d = np.log(snp500_price[::3] / snp500_price[::3].shift(1)).dropna()
snp500_log_returns_4d = np.log(snp500_price[::4] / snp500_price[::4].shift(1)).dropna()
snp500_log_returns_5d = np.log(snp500_price[::5] / snp500_price[::5].shift(1)).dropna()
snp500_log_returns_6d = np.log(snp500_price[::6] / snp500_price[::6].shift(1)).dropna()
snp500_log_returns_7d = np.log(snp500_price[::7] / snp500_price[::7].shift(1)).dropna()

# Calculate Sharpe ratios using log returns
snp500_sharpe_ratio = calculate_sharpe_ratio(snp500_log_returns_daily, rebalance_days=1)
snp500_sharpe_ratio_2d = calculate_sharpe_ratio(snp500_log_returns_2d, rebalance_days=2)
snp500_sharpe_ratio_3d = calculate_sharpe_ratio(snp500_log_returns_3d, rebalance_days=3)
snp500_sharpe_ratio_4d = calculate_sharpe_ratio(snp500_log_returns_4d, rebalance_days=4)
snp500_sharpe_ratio_5d = calculate_sharpe_ratio(snp500_log_returns_5d, rebalance_days=5)
snp500_sharpe_ratio_6d = calculate_sharpe_ratio(snp500_log_returns_6d, rebalance_days=6)
snp500_sharpe_ratio_7d = calculate_sharpe_ratio(snp500_log_returns_7d, rebalance_days=7)

# Print the results
print("S&P 500 Sharpe Ratio (Daily):         ", snp500_sharpe_ratio)
print("S&P 500 Sharpe Ratio (2-day interval):", snp500_sharpe_ratio_2d)
print("S&P 500 Sharpe Ratio (3-day interval):", snp500_sharpe_ratio_3d)
print("S&P 500 Sharpe Ratio (4-day interval):", snp500_sharpe_ratio_4d)
print("S&P 500 Sharpe Ratio (5-day interval):", snp500_sharpe_ratio_5d)
print("S&P 500 Sharpe Ratio (6-day interval):", snp500_sharpe_ratio_6d)
print("S&P 500 Sharpe Ratio (7-day interval):", snp500_sharpe_ratio_7d)

snp500_price_plot =snp500_price[:-5]
snp500_price_plot = snp500_price_plot / snp500_price_plot.iloc[0]


snp500_data['Dates'] = pd.to_datetime(snp500_data['Dates']).dt.date
date_list = snp500_data['Dates'].tolist()
date_list = date_list[:-5]


#%%

# Create figure with a larger size for clarity
plt.figure(figsize=(10, 6))

# Plot the daily portfolio values 
plt.plot(range(0, 700), LSTM_portfolio_values, label="Daily LSTM Portfolio Value", color='#1f77b4', linewidth=2)
plt.plot(range(0, 700), Encoder_portfolio_values, label="Daily Encoder Portfolio Value", color='#ff7f0e', linewidth=2)



#LSTM_post_transaction_portfolio_values
# Plot the S&P 500 price with a distinct color
plt.plot(range(0, 700), snp500_price_plot, label="S&P 500 Price", color='#2ca02c', linewidth=2)

# Add a horizontal line at y = 1
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

# Add grid for better readability
plt.grid(True)

# Add axis labels
plt.xlabel('Days')
plt.ylabel('Portfolio Value')

# Add a title
plt.title('Daily Portfolio Value vs S&P 500 Price')

# Add a legend to differentiate the two lines
plt.legend()

# Ensure the layout is tight
plt.tight_layout()

# Display the plot
plt.show()




#%%

# Create figure with a larger size for clarity
plt.figure(figsize=(10, 6))

# Plot the daily portfolio values 
plt.plot(range(0, 700), LSTM_post_transaction_portfolio_values, label="Daily LSTM Portfolio Value after transaction costs", color='#1f77b4', linewidth=2)
plt.plot(range(0, 700), Encoder_post_transaction_portfolio_values, label="Daily Encoder Portfolio Value after transaction costs", color='#ff7f0e', linewidth=2)

#LSTM_post_transaction_portfolio_values
# Plot the S&P 500 price with a distinct color
plt.plot(range(0, 700), snp500_price_plot, label="S&P 500 Price", color='#2ca02c', linewidth=2)

# Add a horizontal line at y = 1
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

# Add grid for better readability
plt.grid(True)

# Add axis labels
plt.xlabel('Days')
plt.ylabel('Portfolio Value')

# Add a title
plt.title('Daily Portfolio Value vs S&P 500 Price')

# Add a legend to differentiate the two lines
plt.legend()

# Ensure the layout is tight
plt.tight_layout()

# Display the plot
plt.show()


#%%

hybrid_dict = {}

# Iterate through the keys of one dictionary (they have the same keys)
for key in LSTM_forecasts_dict:
    # Perform elementwise average of the two DataFrames corresponding to the same key
    hybrid_dict[key] = (LSTM_forecasts_dict[key] + Encoder_forecasts_dict[key]) / 2

#%%

# Step 6: Run the simulation using test set data for 1-day rebalancing
Hybrid_portfolio_values, Hybrid_portfolio_returns, Hybrid_transaction_costs, Hybrid_post_transaction_portfolio_values, Hybrid_post_transaction_returns = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=1, test_period_days=test_period_days)

Hybrid_sharpe = calculate_sharpe_ratio(Hybrid_portfolio_returns, rebalance_days=1)
Hybrid_sharpe_adjusted = calculate_sharpe_ratio(Hybrid_post_transaction_returns, rebalance_days=1)

# Print results for 1-day rebalancing
print("Hybrid Sharpe Ratio (1-day):", Hybrid_sharpe)
print("Hybrid Total Transaction Costs (1-day):", sum(Hybrid_transaction_costs))
print("Hybrid Sharpe Ratio (Adjusted, 1-day):", Hybrid_sharpe_adjusted)

#%%

Hybrid_portfolio_values2, Hybrid_portfolio_returns2, Hybrid_transaction_costs2, Hybrid_post_transaction_portfolio_values2, Hybrid_post_transaction_returns2 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=2, test_period_days=test_period_days)

Hybrid_sharpe2 = calculate_sharpe_ratio(Hybrid_portfolio_returns2, rebalance_days=2)
Hybrid_sharpe_adjusted2 = calculate_sharpe_ratio(Hybrid_post_transaction_returns2, rebalance_days=2)

# Print results for 2-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (2-day):", Hybrid_sharpe2)
print("Hybrid Total Transaction Costs (2-day):", sum(Hybrid_transaction_costs2))
print("Hybrid Sharpe Ratio (Adjusted, 2-day):", Hybrid_sharpe_adjusted2)

# Step 7: Run the simulation using test set data for 3-day rebalancing
Hybrid_portfolio_values3, Hybrid_portfolio_returns3, Hybrid_transaction_costs3, Hybrid_post_transaction_portfolio_values3, Hybrid_post_transaction_returns3 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=3, test_period_days=test_period_days)

Hybrid_sharpe3 = calculate_sharpe_ratio(Hybrid_portfolio_returns3, rebalance_days=3)
Hybrid_sharpe_adjusted3 = calculate_sharpe_ratio(Hybrid_post_transaction_returns3, rebalance_days=3)

# Print results for 3-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (3-day):", Hybrid_sharpe3)
print("Hybrid Total Transaction Costs (3-day):", sum(Hybrid_transaction_costs3))
print("Hybrid Sharpe Ratio (Adjusted, 3-day):", Hybrid_sharpe_adjusted3)


# Step 7: Run the simulation using test set data for 4-day rebalancing
Hybrid_portfolio_values4, Hybrid_portfolio_returns4, Hybrid_transaction_costs4, Hybrid_post_transaction_portfolio_values4, Hybrid_post_transaction_returns4 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=4, test_period_days=test_period_days)

Hybrid_sharpe4 = calculate_sharpe_ratio(Hybrid_portfolio_returns4, rebalance_days=4)
Hybrid_sharpe_adjusted4 = calculate_sharpe_ratio(Hybrid_post_transaction_returns4, rebalance_days=4)

# Print results for 4-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (4-day):", Hybrid_sharpe4)
print("Hybrid Total Transaction Costs (4-day):", sum(Hybrid_transaction_costs4))
print("Hybrid Sharpe Ratio (Adjusted, 4-day):", Hybrid_sharpe_adjusted4)


# Step 7: Run the simulation using test set data for 5-day rebalancing
Hybrid_portfolio_values5, Hybrid_portfolio_returns5, Hybrid_transaction_costs5, Hybrid_post_transaction_portfolio_values5, Hybrid_post_transaction_returns5 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=5, test_period_days=test_period_days)

Hybrid_sharpe5 = calculate_sharpe_ratio(Hybrid_portfolio_returns5, rebalance_days=5)
Hybrid_sharpe_adjusted5 = calculate_sharpe_ratio(Hybrid_post_transaction_returns5, rebalance_days=5)

# Print results for 5-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (5-day):", Hybrid_sharpe5)
print("Hybrid Total Transaction Costs (5-day):", sum(Hybrid_transaction_costs5))
print("Hybrid Sharpe Ratio (Adjusted, 5-day):", Hybrid_sharpe_adjusted5)

# Step 7: Run the simulation using test set data for 6-day rebalancing
Hybrid_portfolio_values6, Hybrid_portfolio_returns6, Hybrid_transaction_costs6, Hybrid_post_transaction_portfolio_values6, Hybrid_post_transaction_returns6 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=6, test_period_days=test_period_days)

Hybrid_sharpe6 = calculate_sharpe_ratio(Hybrid_portfolio_returns6, rebalance_days=6)
Hybrid_sharpe_adjusted6 = calculate_sharpe_ratio(Hybrid_post_transaction_returns6, rebalance_days=6)

# Print results for 6-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (6-day):", Hybrid_sharpe6)
print("Hybrid Total Transaction Costs (6-day):", sum(Hybrid_transaction_costs6))
print("Hybrid Sharpe Ratio (Adjusted, 6-day):", Hybrid_sharpe_adjusted6)

# Step 7: Run the simulation using test set data for 7-day rebalancing
Hybrid_portfolio_values7, Hybrid_portfolio_returns7, Hybrid_transaction_costs7, Hybrid_post_transaction_portfolio_values7, Hybrid_post_transaction_returns7 = simulate_volatility_weighted_portfolio(
    test_returns, hybrid_dict, rebalance_days=7, test_period_days=test_period_days)

Hybrid_sharpe7 = calculate_sharpe_ratio(Hybrid_portfolio_returns7, rebalance_days=7)
Hybrid_sharpe_adjusted7 = calculate_sharpe_ratio(Hybrid_post_transaction_returns7, rebalance_days=7)

# Print results for 7-day rebalancing
print("$$")
print("Hybrid Sharpe Ratio (7-day):", Hybrid_sharpe7)
print("Hybrid Total Transaction Costs (7-day):", sum(Hybrid_transaction_costs7))
print("Hybrid Sharpe Ratio (Adjusted, 7-day):", Hybrid_sharpe_adjusted7)





#%%

# Create figure with a larger size for clarity
plt.figure(figsize=(10, 6))

# Plot the daily portfolio values 
plt.plot(date_list, LSTM_post_transaction_portfolio_values, label="Daily LSTM Portfolio Value", color='#1f77b4', linewidth=2)
plt.plot(date_list, Encoder_post_transaction_portfolio_values, label="Daily Encoder Portfolio Value", color='#ff7f0e', linewidth=2)
plt.plot(date_list, Hybrid_post_transaction_portfolio_values, label="Daily Hybrid Portfolio Value", color='red', linewidth=2)



#LSTM_post_transaction_portfolio_values
# Plot the S&P 500 price with a distinct color
plt.plot(date_list, snp500_price_plot, label="S&P 500 Price", color='#2ca02c', linewidth=2)

# Add a horizontal line at y = 1
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

# Add grid for better readability
plt.grid(True)

# Add axis labels
plt.xlabel('Days')
plt.ylabel('Scaled Portfolio Value')

# Add a title
#plt.title('Daily Portfolio Value vs S&P 500 Price')

# Add a legend to differentiate the two lines
plt.legend()

# Ensure the layout is tight
plt.tight_layout()

# Display the plot
plt.show()

























