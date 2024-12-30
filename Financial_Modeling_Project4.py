# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:31:29 2024

@author: Divya Pahuja
"""

# Importing required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as pa
import matplotlib.pyplot as plt
import riskfolio as rp

#%%  
etfs_ticker = ['XLP', 'XLV', 'XLY', 'XLB', 'CARZ', 'XLE', 'XLI', 'XLC']
    
# Date in YYYY-MM-DD
start = '2019-10-12'
end = '2024-10-12'

assets_data = yf.download(etfs_ticker, start = start , end = end, interval = '1mo')['Adj Close']
assets_data


##1a)
returns = assets_data.pct_change().dropna()
returns

port = rp.Portfolio(returns = returns)

rm = 'MSV'
##Risk Measure = Mean Semi-Deviation i.e, considering only downside risk instead of taking variance which takes both positive and negative risk.
mu_method = 'hist' # mu is calculted based on historical data
cov_method = 'hist' # covarience is calculated based on historical data
## if something is in varchar 


port.assets_stats(method_mu = mu_method , method_cov = cov_method)

mu = port.mu
cov = port.cov ## dot means we are accessing a attribute from the obj or dataframe created 

mu
cov

port

optimised_weights = port.optimization(model = 'Classic', rm = rm,
                                      obj ='Sharpe', rf =0, l=0, hist=True)

optimised_weights

#%%
#1b)
#Calculating GOAD
investment = 100000

##portfolio returns
optimized_portfolio_return = returns.dot(optimised_weights['weights'])
optimized_portfolio_return

etf_optimized_portfolio_goad = investment * (1 + optimized_portfolio_return).cumprod()
etf_optimized_portfolio_goad

##last assignment portfolio

unoptimized_weights = pd.Series([1/8] * 8, index=returns.columns)
unoptimized_weights

# ETFs
unoptimized_portfolio_return = (returns * unoptimized_weights).sum(axis=1)
etf_unoptimized_portfolio_goad = investment * (1 + unoptimized_portfolio_return).cumprod()
etf_unoptimized_portfolio_goad


#%%
##Plotting Optimized portfolio GOAD

plt.plot(etf_optimized_portfolio_goad, label="Portfolio 1: ETFs_Optimized_Portfolio_GOAD")
plt.title('Growth of $100,000 Investment in Optimized Portfolios')
plt.xlabel('Months')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.show()

#%%
#1c)
# Cumulative Returns Plot
plt.figure(figsize=(10,6))

# Calculate cumulative returns for each portfolio
cumulative_returns_etf_optimized = (1 + optimized_portfolio_return).cumprod() - 1
cumulative_returns_etf_unoptimized = (1 + unoptimized_portfolio_return).cumprod() - 1

# Plot cumulative returns
plt.plot(cumulative_returns_etf_optimized, label="Portfolio 1: ETFs_Optimized_Portfolio")
plt.plot(cumulative_returns_etf_unoptimized, label="Portfolio 2: ETFs_Unoptimized_Portfolio")

plt.title('Cumulative Returns for Different Portfolios')
plt.xlabel('Time (Months)')
plt.ylabel('Cumulative Returns')
plt.legend()

plt.grid(True)
plt.show()


""""After optimizing the portfolio by maximizing the Sharpe Ratio, there isn't a significant difference
between the performance of the optimized and unoptimized portfolios. In early 2020, the optimized portfolio 
performed better, but by the end of the year, the unoptimized portfolio started to outperform it. 
This trend continued until early 2022, when the optimized portfolio briefly outperformed again. 
However, by the beginning of 2023, the unoptimized portfolio regained the upper hand.

Given these fluctuating trends in portfolio growth, it's difficult to conclude that the optimized 
portfolio consistently outperformed the unoptimized one. The performance differences between the two 
portfolios seem to be insignificant, and neither portfolio consistently maintained 
an advantage over the other."""

#%%2)

#2a)
rp.plot_pie(w = optimised_weights, title = 'Portfolio',
                 height = 6, width = 10, ax = None)

    
##Change in the allocation of the weights is as follows:
"""After optimizing the ETFs portfolio, almost 75.8% of the weights are allocated to XLV which is the ETF that tracks
'HEALTH CARE' sector of the S&P500 index, 11.7% of the weights are allocated to XLE which tracks 
'ENERGY' sector of the S&P500 index, 7.1% of the weights are allocated to XLP which tracks 'CONSUMER STAPLES' 
sector of the S&P500 index.The remaining 5.3% of the weights are distributed among other ETFs(CARZ-Automobile sector) 
in the portfolio.
"""


#%%
#2b)
rp.plot_risk_con(w = optimised_weights, cov = cov,
                      returns = returns,
                      rm=rm, rf = 0,
                      alpha = 0.05, height = 6,
                      width = 10, t_factor = 252, ax = None)


##Analysing the risk in portfolio contributed by each asset
""""After analyzing the contributions to portfolio risk using plot_risk_con(), we found that:
    
XLV contributes the highest amount of risk in the portfolio, at 34%, which may be surprising given its 
typical defensive nature. As it provide essential products or services that people continue to purchase, 
such as food, healthcare, and utilities. 
 
XLE adds 10% of the risk, aligning with expectations due to its inherent sensitivity to oil prices and 
geopolitical factors.

XLP contributes 2.5% of risk, which is lower than expected for a typically stable sector, suggesting it is 
performing as a defensive asset.

CARZ contributes approximately 3.5%, which is consistent with expectations, given the industry's current 
stability amidst shifts towards electric vehicles.
""""


#%%Efficient Frontier
#2c)
ws = port.efficient_frontier(model = 'Classic', rm = rm, points = 20,
                             rf = 0, hist = True)


##Efficient frontier plot
label = 'Max Risk Adjusted Return Portfolio'
rp.plot_frontier(
    w_frontier = ws, mu = mu, cov = cov, returns = returns, rm = rm, rf = 0,
    alpha = 0.05, cmap = 'viridis', w = optimised_weights, label = label,
    marker = '*' , s= 16, c = 'r' , height = 6, width = 10, t_factor = 252,
    ax = None
    )

"""
Portfolios on the efficient frontier are considered to be the most desirable because they offer 
the highest possible expected return for a given level of risk. Investors should aim to allocate their 
assets to portfolios on this curve.


For our portfolio,the marker on the efficient frontier is created where Expected returns are 255% and Expected Risk is 
50% which is lot. This shows that high returns often come with higher risk.

The efficient frontier is steep which means with additional amount of risk, investor gets additional amount of 
returns from this portfolio.

This portfolio is an optimal choice for the investor according to Sharpe ratio, as it offers
highest risk adjusted return. However,it may not be suitable for all investors, particularly those who are risk averse, 
as it involves a relatively high level of risk. """"

#%%drawdowns
#2d)
rp.plot_drawdown(returns = returns,
                      w = optimised_weights,
                      alpha = 0.05,
                      height = 8,
                      width = 10,
                      ax = None)

#how the drawdown looks like for the portfolio
"""The maximum drawdown for our portfolio is -18.25%, which represents the largest decline observed 
over this five-year period. This drawdown was likely influenced by the pandemic in 2020. The average drawdown 
is -2.51%, indicating that, on average, the portfolio experiences a moderate dip of 2.51% from its peak values. 
The Ulcer Index, which measures the depth of declines over time, is -4.26%, suggesting that we experience 
moderate drawdowns and that the portfolio is able to recover from these dips. The Conditional Drawdown at a 
95% confidence level (CDaR) stands at -9.79%.

Overall, the drawdown analysis indicates that while the portfolio has experienced volatility, 
the declines have been moderate. Moreover, the portfolio demonstrates an ability to recover effectively over time. 
We can conclude that, although the portfolio may experience sharp declines during its performance, it tends to 
bounce back after these drops, making it suitable for long-term investors."""


#%%
#2e)
"""Overview of the Portfolio Performance: 
The efficient frontier illustrates that while the portfolio's expected returns are relatively high, 
they come with a substantial amount of risk. This analysis indicates that investors can earn additional 
returns by taking on extra risk, as the frontier is steep. Despite the portfolio's significant volatility, 
the drawdowns have been moderate, meaning that the declines are not severe. 
This is a positive sign regarding the portfolio's performance.

In conclusion, I would only consider investing in this portfolio if I had a high-risk tolerance and could 
manage the periods of drawdown. Although these drawdowns are moderate, they are accompanied by considerable volatility. 
This portfolio is suitable for risk-seeking investors looking for higher returns relative to the risks they 
can tolerate.

As a risk-averse investor, I would choose not to invest in this portfolio due to the volatility it entails, 
making it less appropriate for conservative or risk-averse investors."""