"""
Stock Price Prediction model using Random Forest Regressor.
This script loads stock price data, preprocesses it, trains a Random Forest Regressor model,
and evaluates its performance on a test dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import yfinance as yf

#function to add features later
def add_features(df):
    """
    Add basic technical features to a multi-stock dataframe.
    
    Features added per ticker:
    - daily_return: Daily percentage return
    - volatility: 20-day rolling standard deviation of returns
    - gap: Overnight gap (today's open - yesterday's close)
    - SMA_5: 5-day simple moving average
    - SMA_20: 20-day simple moving average
    - EMA_12: 12-day exponential moving average
    - EMA_26: 26-day exponential moving average
    - volume_change: Daily volume percentage change
    - price_volume: Close price × Volume (money flow approximation)
    
    Parameters:
    df: DataFrame with MultiIndex columns (Price metrics, Tickers)
    
    Returns:
    DataFrame with additional feature columns
    """
    
    df_features = df.copy()
    
    # Get list of tickers
    tickers = df['Close'].columns.tolist()
    
    for ticker in tickers:
        # Extract data
        close = df[('Close', ticker)]
        open_price = df[('Open', ticker)]
        volume = df[('Volume', ticker)]
        
        #Daily return
        df_features[('daily_return', ticker)] = close.pct_change()
        #Volatility (20-day)
        df_features[('volatility', ticker)] = df_features[('daily_return', ticker)].rolling(window=20).std()
        #Gap
        df_features[('gap', ticker)] = open_price - close.shift(1)
        #SMA_5
        df_features[('SMA_5', ticker)] = close.rolling(window=5).mean()
        #EMA_12
        df_features[('EMA_12', ticker)] = close.ewm(span=12, adjust=False).mean()
        #Volume change
        df_features[('volume_change', ticker)] = volume.pct_change()
        #Price-volume approximation
        df_features[('price_volume', ticker)] = close * volume
    
    return df_features

#grab data
Tickers = ["^GSPC","^IXIC","^TNX","AAPL"]
og_data = yf.download(Tickers,start="2015-01-01", end="2026-01-01")
print(og_data)


#add features
data = add_features(og_data)


#splitting features and target
X = data.drop([("Close", "^IXIC")],axis = 1)
Y = data[("Close", "^IXIC")]


#split data into training and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=3)

# Data Cleaning for X_train, Y_train, X_test, Y_test

# 1. Identify and handle the column ('volume_change', '^TNX') which is entirely NaN in the original data.
#    It needs to be dropped from both training and testing features if it exists.
problematic_col = ("volume_change", "^TNX")
if problematic_col in X_train.columns:
    X_train = X_train.drop(columns=[problematic_col])
    X_test = X_test.drop(columns=[problematic_col])

# 2. Check for remaining NaNs and Infinities, and remove corresponding rows.
#    It's crucial to keep X_train/Y_train and X_test/Y_test synchronized.

# For training data
train_data_combined = pd.concat([X_train, Y_train], axis=1)
train_data_combined.dropna(inplace=True)

target_col_name = Y_train.name
X_train = train_data_combined.drop(columns=[target_col_name])
Y_train = train_data_combined[target_col_name]

# For testing data
test_data_combined = pd.concat([X_test, Y_test], axis=1)
test_data_combined.dropna(inplace=True)

X_test = test_data_combined.drop(columns=[target_col_name])
Y_test = test_data_combined[target_col_name]

# Additionally, ensure no infinite values remain and handle any newly introduced NaNs from replacement
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
Y_train = Y_train[X_train.index] # Align Y_train after potential inf removal from X_train

X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
Y_test = Y_test[X_test.index] # Align Y_test after potential inf removal from X_test

# Ensure data types are suitable for scikit-learn (e.g., float64)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
Y_train = Y_train.astype(np.float64)
Y_test = Y_test.astype(np.float64)


#Set Model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

#Train the model
regressor.fit(X_train, Y_train)

#prediction on Test Data
test_data_prediction = regressor.predict(X_test)

#R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


Y_test = list(Y_test)
#plot the prediction compared to actual data
plt.figure(figsize=(15, 7))

# Sort Y_test and test_data_prediction by date for proper plotting
sorted_indices = Y_test.index.sort_values()
Y_test_sorted = Y_test.loc[sorted_indices]
test_data_prediction_sorted = pd.Series(test_data_prediction, index=Y_test.index).loc[sorted_indices]

plt.plot(Y_test_sorted.index, Y_test_sorted, color= "blue", label = "Actual Value")
plt.plot(test_data_prediction_sorted.index, test_data_prediction_sorted, color="green", label="Predicted Value")
plt.title("Actual Price vs Predicted Price")
plt.xlabel("Date")
plt.ylabel("Nasdaq Index Price")
plt.legend()
plt.show()
