# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:29:32 2024

@author: Divya Pahuja
"""


##Downloading required libraries
#pip install ta


#Importing required libraries
import pandas as pd
import numpy as ny
import yfinance as yf
import pandas_ta as pa
import mplfinance as mpf
import matplotlib.pyplot as ta
import matplotlib.pyplot as plt
import ta
import pandas_ta as ta


#%%

##Selection of four assests
tickers = ["C","MMYT","MRNA","PEP"]



start = '2023-09-01'
end = '2024-08-31'

data = yf.download(tickers, start = start, end = end)
data

data.columns
data_df = pd.DataFrame(data)
data_df.head()
data_df.isnull().sum()
data_df.describe()
test = data_df.describe()  ##here we are string the summary statistics to see it in variable explorer
print(test)

#%%
#Q1:a

data_df['Adj Close'].plot(xlabel = 'Tickers',ylabel = 'Adjusted Closing Price',
                          title = 'Assets Adjusted Closing Prices',
                          figsize = (10,5))

plt.show()


#%%
#Q1:b

##PLOTTING CANDELSTICK CHARTS
for ticker in tickers:
    # Fetch data for each ticker individually
    ticker_data = yf.download(ticker, start=start, end=end, interval='1d')
    
#Candlestick plots without indicators 
    mpf.plot(ticker_data, type='candle', volume=True,
             style='charles', title=f'Candlestick Plot for {ticker}', ylabel='Price')
    

#%%
#Q1:c
#Plotting SMA with Candlestick
for ticker in tickers:
    # Fetch data for each ticker individually
    ticker_data = yf.download(ticker, start=start, end=end, interval='1d')
    
    
    
    # ##Add two SMA trendlines that are different in time horizon to what we covered in class
    ticker_data['SMA_10'] = ticker_data['Adj Close'].rolling(window=10).mean()
    ticker_data['SMA_100'] = ticker_data['Adj Close'].rolling(window=100).mean()


    ## Plot SMA
    add_plot_SMA = [
    mpf.make_addplot(ticker_data['SMA_10'], color='blue', label='SMA 10'),
    mpf.make_addplot(ticker_data['SMA_100'], color='orange', label='SMA 100')  # Closing quotation mark added here
]

    # Plot the SMA in the candlestick chart
    mpf.plot(ticker_data, type='candle', volume=True, addplot=add_plot_SMA,
             style='charles', title=f'Candlestick Plot for {ticker} with SMA', ylabel='Price')
    
#%%
#Q1:d
#Plotting BB with Candlestick

for ticker in tickers:
    # Fetch data for each ticker individually
    ticker_data = yf.download(ticker, start=start, end=end, interval='1d')
    # Fetch data for each ticker individually
    ticker_data = yf.download(ticker, start=start, end=end, interval='1d')
    
#calculate SMA with last 10 and 100 rolling windows
    ticker_data['SMA_10'] = ticker_data['Adj Close'].rolling(window=10).mean()
    ticker_data['SMA_100'] = ticker_data['Adj Close'].rolling(window=100).mean()
    
    
    # Calculate Bollinger Bands
    ticker_data['Upper_BB_SMA_10'] = ticker_data['SMA_10'] + 2 * ticker_data['Adj Close'].rolling(10).std()
    ticker_data['Lower_BB_SMA_10'] = ticker_data['SMA_10'] - 2 * ticker_data['Adj Close'].rolling(10).std()
    ticker_data['Upper_BB_SMA_100'] = ticker_data['SMA_100'] + 2 * ticker_data['Adj Close'].rolling(100).std()
    ticker_data['Lower_BB_SMA_100'] = ticker_data['SMA_100'] - 2 * ticker_data['Adj Close'].rolling(100).std()
    
#check bollinger bands data
#print(ticker_data[['Upper_BB_SMA_10', 'Lower_BB_SMA_10','Upper_BB_SMA_100','Lower_BB_SMA_100']].head())

    # Create addplot for Bollinger Bands
    add_plot_BB = [
        mpf.make_addplot(ticker_data['SMA_10'], color='blue', label='SMA 10'),
        mpf.make_addplot(ticker_data['SMA_100'], color='orange', label='SMA 100'),
        mpf.make_addplot(ticker_data['Upper_BB_SMA_10'], color='red', linestyle='--', label='Upper BB 10'),
        mpf.make_addplot(ticker_data['Lower_BB_SMA_10'], color='red', linestyle='--', label='Lower BB 10'),
        #mpf.make_addplot(ticker_data['Upper_BB_SMA_100'], color='green', linestyle='--', label='Upper BB 100'),
        #mpf.make_addplot(ticker_data['Lower_BB_SMA_100'], color='green', linestyle='--', label='Lower BB 100'),
    ]

    # Plot the BB with candlestick chart
    mpf.plot(ticker_data, type='candle', volume=True, addplot=add_plot_BB,
             style='charles', title=f'Candlestick Plot for {ticker} with SMA & BB', ylabel='Price')


#%%
#Q1:e
#Plotting KAMA with Candlestick

for ticker in tickers:
    # Fetch data for each ticker individually
    ticker_data = yf.download(ticker, start=start, end=end, interval='1d')
    
    
    ##Add KAMA in the dataframe
    ticker_data['KAMA'] = ta.kama(ticker_data['Adj Close'], length=10)

    # Create addplot for KAMA
    add_plot_KAMA = mpf.make_addplot(ticker_data['KAMA'], color='purple', label='KAMA')

    # Plot the KAMA with candlestick chart
    mpf.plot(ticker_data, type='candle', volume=True, addplot=add_plot_KAMA,
             style='charles', title=f'Candlestick Plot for {ticker} with KAMA', ylabel='Price')
#%%
#Major Buy and Sell
#Q1:f


##CITI(C)
""""Major Buy Signal:
In the plot, just before November 13, 2023, we can see green candlesticks forming and
rising above the upper Bollinger Band. This is referred to as a "Bollinger Blast," which
indicates strong upward momentum. Such a movement suggests that the market is entering a bullish phase, 
signaling a good time to consider purchasing the stock.

Major Sell Signal:
In the plot, just before January 26, 2024 (or late December 2023),we can observe that the stock prices 
are falling and the red candlesticks are forming. The price drops below the upper Bollinger 
Band and approaches the 10-day Simple Moving Average (SMA). This downward movement is a signal that the 
bullish momentum may be weakening, indicating a good time to consider selling the stock or coming out of
the market position."""


##Make My Trip(MMYT)
""""Major Buy Signal:
In the plot, just after January 26, 2024 , we can see green candlesticks forming and
rising above the upper Bollinger Band. This is referred to as a "Bollinger Blast," which
indicates strong upward momentum. Such a movement suggests that the market is entering a bullish phase, 
signaling a good time to consider purchasing the stock.


Major Sell Signal:
In the plot,during late February 2024 (or early March 2024),we can observe that the stock prices 
are falling and the red candlesticks are forming.It is when the BB are contracting 
The price drops below the upper Bollinger Band and touches the 10-day Simple Moving Average (SMA). 
This downward movement indicates a good time to consider selling the stock back.
    
"""

##Moderna(MRNA)
""""Major Sell Signal:
In the plot,at the start of October, 2023 , we can see that the stock prices are going down and 
red candlesticks forming. This shows that the market is bearish.
The price goes below the lower Bollinger Band This downward movement indicates a good time to 
consider selling the stock.


Major Buy Signal:
In the plot, just at the beginning of November, 2023 , we can see green candlesticks forming and
touching the SMA 10 line. This signals a good time to consider purchasing the stock back.
"""


##Pepsico(PEP)
""""Major Buy Signal:
In the plot,during late March,2024 , we can see green candlesticks forming and
rising above the upper Bollinger Band. This is referred to as a "Bollinger Blast," which
indicates strong upward momentum. Such a movement suggests that the market is entering a bullish phase, 
signaling a good time to consider purchasing the stock.


Major Sell Signal:

In the plot,on April 9,2024,we can observe that the stock prices are falling and the red 
candlesticks are forming. The price drops below the upper Bollinger Band and approaches the 10-day 
Simple Moving Average (SMA). This downward movement is a signal that the bullish momentum may be 
weakening, indicating a good time to consider selling the stock or coming out of the market position.
    
"""



#%%
##Q2
##Fav ticker
good_ticker = ['C']

good_ticker_data = yf.download(good_ticker, start=start, end=end, interval='1d')

start = '2023-09-01'
end = '2024-08-31'


#%%
##RSI
good_ticker_data['RSI'] = ta.rsi(good_ticker_data['Adj Close'], length=14)
plt.plot(figsize = (10,5))
plt.plot(good_ticker_data.index, good_ticker_data['RSI'], color = 'blue')
plt.axhline(60, color = 'red' , linestyle = '--' , label = 'overbought')
plt.axhline(40, color = 'green' , linestyle = '--' , label = 'oversold')
plt.show()

##RSI CALCULATION
good_ticker_data['RSI'] = ta.rsi(good_ticker_data['Adj Close'], length = 14)
    
##CREATE PLOT FOR RSI
plt.plot(figsize = (10,5))
plt.plot(good_ticker_data.index, good_ticker_data['RSI'], color = 'blue')
plt.axhline(70, color = 'orange', linestyle = '--', label = 'overbought')
plt.axhline(30, color = 'blue', linestyle = '--', label = 'oversold')
plt.show()
    
    
#%%
##MACD
good_ticker_data['MACD'] = ta.macd(good_ticker_data['Adj Close'])['MACD_12_26_9']
good_ticker_data['MACD_Signal'] = ta.macd(good_ticker_data['Adj Close'])['MACDs_12_26_9']


plt.figure(figsize = (10,5))
plt.plot(good_ticker_data.index, good_ticker_data['MACD'], label = 'MACD', color = 'purple')
plt.plot(good_ticker_data.index, good_ticker_data['MACD_Signal'], label = 'Signal Line', color = 'red')
plt.legend()
plt.show()

#%%
##Q2:a

##Relative Strength Index(RSI)
##For CITI
""""Buy Signal:
In the plot,post November 2023, we observe the RSI crosses above 40 and stays above.This movement 
indicates that the asset is becoming less oversold and may be gaining upward momentum.
This means it is good time to buy the stock as the market is bullish 
Sell Signal:
In the plot, post January, 2024,we notice that RSI goes below 60  signaling a possible shift 
from bullish to bearish momentum.This indicates it may be prudent to sell the stock 
as the market appears to be weakening.""""

##Moving Average Convergence Divergence (MACD)
""""Buy Signal:
In the plot,post November 2023, we observe that MACD line  crosses above the signal line, which 
is a good indication of buying the stock.This crossover indicates a potential shift in momentum 
toward bullish sentiment.
Sell Signal:
In the plot, post January, 2024,we can see that the MACD line crosses below the signal line. 
This suggests a potential shift in momentum toward bearish sentiment, indicating that it\
may be a good time to sell or take profits.""""

#%%
##Q2:b
""""Yes, these observations line up with that of Q1 Citi stock. The above RSI and MACD plots indicates
the same buying and selling signal""""


#%%
##Q2:c
""""One additional performance indicator that pandas_ta can execute, which we did not cover in class, 
is the VWAP (Volume Weighted Average Price).

The VWAP is used to measure the average price a security has traded at throughout the day, 
based on both price and volume. It is particularly useful for traders as it helps to assess the
direction of the market and the strength of price movements. A price above the VWAP generally 
indicates a bullish sentiment, while a price below the VWAP can suggest a bearish trend. 

Function to Execute:
To calculate VWAP using the pandas_ta library, we would use the ta.vwap() function.

""""
##Calculate VWAP
vwap = ta.vwap(close=good_ticker_data['Close'], high=good_ticker_data['High'], 
low=good_ticker_data['Low'], volume=good_ticker_data['Volume'])


#%%

###Combine Candlestick plots with SMA,BB and RSI for all 4 tickers
good_ticker = ["C","MMYT","MRNA","PEP"]


start = '2023-09-01'
end = '2024-08-31'

# Loop over each ticker
for ticker in good_ticker:
    # Download data
    good_ticker_data = yf.download(ticker, start=start, end=end, interval='1d')

    # Calculate SMAs
    good_ticker_data['SMA_10'] = good_ticker_data['Adj Close'].rolling(window=10).mean()
    good_ticker_data['SMA_100'] = good_ticker_data['Adj Close'].rolling(window=100).mean()

    # Calculate Bollinger Bands
    good_ticker_data['Upper_BB_SMA_10'] = good_ticker_data['SMA_10'] + 2 * good_ticker_data['Adj Close'].rolling(10).std()
    good_ticker_data['Lower_BB_SMA_10'] = good_ticker_data['SMA_10'] - 2 * good_ticker_data['Adj Close'].rolling(10).std()
    good_ticker_data['Upper_BB_SMA_100'] = good_ticker_data['SMA_100'] + 2 * good_ticker_data['Adj Close'].rolling(100).std()
    good_ticker_data['Lower_BB_SMA_100'] = good_ticker_data['SMA_100'] - 2 * good_ticker_data['Adj Close'].rolling(100).std()

    # Calculate RSI
    good_ticker_data['RSI'] = ta.rsi(good_ticker_data['Adj Close'], length=14)

    # Calculate MACD
    macd = ta.macd(good_ticker_data['Adj Close'])
    good_ticker_data['MACD'] = macd['MACD_12_26_9']
    good_ticker_data['MACD_Signal'] = macd['MACDs_12_26_9']

    # Create addplot for BB, RSI, and MACD
    add_plot_BB_RSI_MACD = [
        # SMA plots
        mpf.make_addplot(good_ticker_data['SMA_10'], color='blue', label='SMA 10'),
        mpf.make_addplot(good_ticker_data['SMA_100'], color='orange', label='SMA 100'),
        
        # Bollinger Bands
        mpf.make_addplot(good_ticker_data['Upper_BB_SMA_10'], color='red', linestyle='--', label='Upper BB 10'),
        mpf.make_addplot(good_ticker_data['Lower_BB_SMA_10'], color='red', linestyle='--', label='Lower BB 10'),
        #mpf.make_addplot(good_ticker_data['Upper_BB_SMA_100'], color='green', linestyle='--', label='Upper BB 100'),
        #mpf.make_addplot(good_ticker_data['Lower_BB_SMA_100'], color='green', linestyle='--', label='Lower BB 100'),
        
        # RSI plot (on a separate panel)
        mpf.make_addplot(good_ticker_data['RSI'], panel=1, color='purple', ylabel='RSI', secondary_y=False),
        
        # MACD and Signal line plot (on another separate panel)
        mpf.make_addplot(good_ticker_data['MACD'], panel=2, color='blue', ylabel='MACD', secondary_y=False),
        mpf.make_addplot(good_ticker_data['MACD_Signal'], panel=2, color='orange', ylabel='Signal', secondary_y=False)
    ]

    # Plot the candlestick chart with BB, SMA, RSI, and MACD
    mpf.plot(good_ticker_data, type='candle', volume=False, addplot=add_plot_BB_RSI_MACD,
             style='charles', title=f'Candlestick Plot for {ticker} with BB, RSI, and MACD', ylabel='Price')


#%%
#Q3)
"""""WMA:Weighted Moving Average
It assigns different weights to each price in given period, where recent prices receives more weights
as compared to older prices,this means WMA helps traders understand the current trend of a stock by reflecting 
more accurately how recent prices. movements impact the average.

Usefulness- I believe the WMA is useful indicator for trading as it give more weights to the recent prices.
By focusing on how recent prices are reacting, traders can make more informed decision about buy or sell.opportunities
This responsiveness to recent price changes allows traders to capture trends and make timely decisions in the market """"""

""VWMA: Volume Weighted Moving Average
It a type of moving average which considers both price and volume, and assigns more weight to the 
prices during the periods of higher volume i.e., it adjust/gives weights based on the trading volume 
during the period. If the price is changing and the volume is low it will have less impact on the 
moving average.If the price is rising with the significant increase in the volume, that means it is 
a stronger market movement.
Usefulness- I believe that VWMA is a useful indicator at a time where the market is highly liquid.
It prevents the traders to  falsely judge the market as good when the prices are high 
and volumes are low.""""""
""""""

""KAMA: Kaufman’s Adaptive Moving Average
Unlike SMA, which treats every price same, KAMA reacts more and becomes more sensitive
when the market prices are trending as compared to when the prices are volatile or have too much
noise. Thus, it will slow down when there is too much noise in the prices but speeds up when 
they are trending.

Usefulness- I believe it is quite useful as it helps traders to identify the trend quicker and thus help
them to make buy or sell decision but when the market is volatile it will help to filter out unnecessary
noise and reduces false signals to the investor and thus preventing them to make any unnecessary 
investment.In short,it will help trader to react faster when the market is moving in clear direction 
and stay cautious when the market is uncertain or messy.""""


#%%

