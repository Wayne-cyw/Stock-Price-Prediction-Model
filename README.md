# Stock-Price-Prediction-Model
Random forrest regressor model used to predict stock prices.

The model uses data from 4 related stocks/indexs to predict the closing price of Nasdaq index.
- ^GSPC
- ^IXIC
- ^TNX
- AAPL

The data are sourced from yfinance

The original dataset includes the closing price, high price, low price, open price and volume of each ticker.

In order to achieve better results, the following features are calculated from the dataset and added for each stock/indexes:
- daily_return: Daily percentage return
- volatility: 20-day rolling standard deviation of returns
- gap: Overnight gap (today's open - yesterday's close)
- SMA_5: 5-day simple moving average
- EMA_12: 12-day exponential moving average
- volume_change: Daily volume percentage change
- price_volume: Close price × Volume (money flow approximation)


[Data Analytics.py](<Data Analytics.py>):
Then to optimize the model and minimize the noise, the data is analyized to identify features that have extreme correlation. Through the following heatmap, a direct correlation can be identified.

![Heatmap1.png](https://github.com/Wayne-cyw/Stock-Price-Prediction-Model/blob/main/Heatmap1.png)

The data that has extreme correlation were then removed. 

Here's a new heatmap:
![Heatmap2.png](https://github.com/Wayne-cyw/Stock-Price-Prediction-Model/blob/main/Heatmap2.png)

With reduced redundantcy, it can be proceeded to train the model.


[main.py](main.py):

Dataset that been through pre-processing are now split into features and target. Then it is cleaned to remove NaN and infinitie, and check to ensure data types are suitable for sklearn.

The radnom forrest regressor model achived a R^2 error of 0.999798...

Here is a comparison graph of the actual price vs predicted price:

![Actual price vs predicted price.png](https://github.com/Wayne-cyw/Stock-Price-Prediction-Model/blob/main/Actual%20price%20vs%20predicted%20price.png)
