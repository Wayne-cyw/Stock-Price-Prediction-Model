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


#get data info
data.info()


#construct heatmap to understand correlation
correlation = data.corr()
plt.figure(figsize=(50,50))
sns.heatmap(correlation, cbar=True, square= False, fmt=".2f",annot=True, annot_kws={"size":8}, cmap="Blues")


#remove extreme redundancy
# Get pairs with correlation > 0.95
high_corr_pairs = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        if abs(correlation.iloc[i, j]) > 0.95:
            high_corr_pairs.append((
                correlation.columns[i], 
                correlation.columns[j],
            ))

print(high_corr_pairs)

#show correlation with close of nasdaq
print(correlation[("Close", "^IXIC")])
