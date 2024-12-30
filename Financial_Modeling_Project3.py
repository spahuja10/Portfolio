# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:23:04 2024

@author: Divya Pahuja
"""

# Importing required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as pa
import mplfinance as mpf
import matplotlib.pyplot as plt
import riskfolio as rp
import statsmodels.api as sm
  
#%%  
ticker = ['NSRGY', 'JNJ', 'ADDYY', 'DOW', 'RACE', 'BP', 'DAL', 'VZ']
    
# Date in YYYY-MM-DD
start = '2019-10-12'
end = '2024-10-12'
 
#%% Q1  
#1a
# Pulling data for companies
data_pulling_companies = yf.download(ticker, start=start, end=end, interval='1mo')['Adj Close']
print(data_pulling_companies)
    
returns_companies = data_pulling_companies.pct_change().dropna()
print(returns_companies)


#1b
etfs = ['XLP', 'XLV', 'XLY', 'XLB', 'CARZ', 'XLE', 'XLI', 'XLC']
    
# Pulling data for ETFs
data_pulling_etfs = yf.download(etfs, start=start, end=end, interval='1mo')['Adj Close']
print(data_pulling_etfs)
    
returns_etfs = data_pulling_etfs.pct_change().dropna()
print(returns_etfs)


#1c
# Pulling data for SPY
data_pulling_third_variable = yf.download('SPY', start=start, end=end, interval='1mo')['Adj Close']
print(data_pulling_third_variable)
    
returns_spy = data_pulling_third_variable.pct_change().dropna()
print(returns_spy)

#%% Q2
# Weights
#2a
companies_weight = pd.Series([1/8] * 8, index=returns_companies.columns)
companies_weight
#2b
etf_weight = pd.Series([1/8] * 8, index=returns_etfs.columns)
etf_weight
#2c
spy_weight = pd.Series([1], index=['SPY'])
spy_weight 



#%% Q3
# GOAD

investment = 100000

print("Returns Companies Shape:", returns_companies.shape)
print("Returns ETFs Shape:", returns_etfs.shape)

# Companies
monthly_portfolio_returns_companies = (returns_companies * companies_weight).sum(axis=1)
portfolio1_goad = investment * (1 + monthly_portfolio_returns_companies).cumprod()
portfolio1_goad

# ETFs
monthly_portfolio_returns_etfs = (returns_etfs * etf_weight).sum(axis=1)
portfolio2_goad = investment * (1 + monthly_portfolio_returns_etfs).cumprod()
portfolio2_goad

# SPY
monthly_portfolio_returns_spy = returns_spy * spy_weight.values
portfolio3_goad = investment * (1 + monthly_portfolio_returns_spy).cumprod()
portfolio3_goad
    
portfolio = {
    "Portfolio 1 Companies": portfolio1_goad,
    "Portfolio 2 ETFs": portfolio2_goad,
    "Portfolio 3 SPY": portfolio3_goad
}

#%%
# Plotting
plt.figure(figsize=(12, 6))
for label, goad in portfolio.items():
    plt.plot(goad, label=label)

plt.title('Growth of $100,000 Investment in 3 Different Portfolios')
plt.xlabel('Months')  # Consider changing this if the x-axis is time/dates
plt.ylabel('Portfolio Value ($)')  # Changed for clarity
plt.grid(True)
plt.legend()
plt.show()

#%%
plt.figure(figsize=(12, 6))

# Histogram for Companies Portfolio
plt.subplot(1, 3, 1)
plt.hist(monthly_portfolio_returns_companies, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('Return Distribution - Companies Portfolio')
plt.xlabel('Monthly Returns')
plt.ylabel('Frequency')

# Histogram for ETFs Portfolio
plt.subplot(1, 3, 2)
plt.hist(monthly_portfolio_returns_etfs, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.title('Return Distribution - ETFs Portfolio')
plt.xlabel('Monthly Returns')
plt.ylabel('Frequency')

# Histogram for S&P 500 Portfolio
plt.subplot(1, 3, 3)
plt.hist(monthly_portfolio_returns_spy, bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.title('Return Distribution - S&P 500 Portfolio')
plt.xlabel('Monthly Returns')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


#%%Scattter Plot
# Calculating average returns and standard deviation
avg_returns = {
    'Companies Portfolio': monthly_portfolio_returns_companies.mean(),
    'ETFs Portfolio': monthly_portfolio_returns_etfs.mean(),
    'S&P 500 Portfolio': monthly_portfolio_returns_spy.mean()  # Assuming you have this
}

risk = {
    'Companies Portfolio': monthly_portfolio_returns_companies.std(),
    'ETFs Portfolio': monthly_portfolio_returns_etfs.std(),
    'S&P 500 Portfolio': monthly_portfolio_returns_spy.std()  # Assuming you have this
}

plt.figure(figsize=(8, 5))
plt.scatter(risk.values(), avg_returns.values(), color='purple')

# Adding labels to each point
for label, x, y in zip(risk.keys(), risk.values(), avg_returns.values()):
    plt.text(x, y, label, fontsize=10, ha='right', color='black')

# Plot details
plt.title('Risk-Return Scatter Plot')
plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Average Monthly Return')
plt.grid()
plt.show()

#%%
# Cumulative Returns Plot
plt.figure(figsize=(10,6))

# Calculate cumulative returns for each portfolio
cumulative_returns_companies = (1 + monthly_portfolio_returns_companies).cumprod() - 1
cumulative_returns_etfs = (1 + monthly_portfolio_returns_etfs).cumprod() - 1
cumulative_returns_spy = (1 + monthly_portfolio_returns_spy).cumprod() - 1

# Plot cumulative returns
plt.plot(cumulative_returns_companies, label="Portfolio 1: Companies")
plt.plot(cumulative_returns_etfs, label="Portfolio 2: ETFs")
plt.plot(cumulative_returns_spy, label="Portfolio 3: SPY")

plt.title('Cumulative Returns for Different Portfolios')
plt.xlabel('Time (Months)')
plt.ylabel('Cumulative Returns')
plt.legend()

plt.grid(True)
plt.show()


#%% 
##Correlation Heatmap for Companies
corr = returns_companies.corr()
companies = returns_companies.columns
plt.figure(figsize=(8,6))
plt.title("Companies Correlation Heatmap")
plt.imshow(corr, cmap="coolwarm", interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xticks(range(len(companies)), companies, rotation=45)
plt.yticks(range(len(companies)), companies)
plt.show()


##Correlation Heatmap for ETFs
corr = returns_etfs.corr()
plt.figure(figsize=(8,6))
plt.title("ETFs Correlation Heatmap")
plt.imshow(corr, cmap="coolwarm", interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xticks(range(len(etfs)), etfs, rotation=45)
plt.yticks(range(len(etfs)), etfs)
plt.show()


#%%Plotting Efficient Frontier_Companies
rm = 'MSV'
method_mu = 'hist'
method_cov = 'hist' 


companies_weight = pd.DataFrame([1/8] * 8, index=returns_companies.columns, columns=['Weight'])
companies_weight

#Companies
port_companies = rp.Portfolio(returns = returns_companies)##what does function mean and as P is capital it is capital

port_companies.assets_stats(method_mu = method_mu,
                           method_cov = method_cov)

mu = port_companies.mu
cov = port_companies.cov


ws_companies = port_companies.efficient_frontier(model = 'Classic', rm = rm, points = 20,
                             rf = 0, hist = True)

label = 'Companies:Max Risk Adjusted Return Portfolio'
rp.plot_frontier(
    w_frontier = ws_companies, mu = mu, cov = cov, returns = returns_companies, rm = rm, rf = 0,
    alpha = 0.05, cmap = 'viridis', w = companies_weight, label = label,
    marker = '*' , s= 16, c = 'r' , height = 6, width = 10, t_factor = 252,
    ax = None
    )


#%%Plotting Efficient Frontier_ETFs
etf_weight = pd.DataFrame([1/8] * 8, index=returns_etfs.columns, columns=['Weight'])
etf_weight


port_etfs = rp.Portfolio(returns = returns_etfs)

port_etfs.assets_stats(method_mu = method_mu,
                       method_cov = method_cov)

mu1 = port_etfs.mu
cov1 = port_etfs.cov

ws_etfs = port_etfs.efficient_frontier(model = 'Classic', rm = rm, points = 20,
                             rf = 0, hist = True)


label = 'ETFs:Max Risk Adjusted Return Portfolio'
rp.plot_frontier(
    w_frontier=ws_etfs, mu=mu1, cov=cov1, returns=returns_etfs, rm=rm, rf=0,
    alpha=0.05, cmap='viridis', w=etf_weight, label=label,
    marker='o', s=16, c='r', height=6, width=10, t_factor=252,
    ax=None
)


#%%Plotting Efficient Frontier_S&P500

if isinstance(returns_spy, pd.Series):
    returns_spy = returns_spy.to_frame()

spy_weight = pd.DataFrame([1], index=returns_spy.columns, columns=['Weight'])
spy_weight

port_spy = rp.Portfolio(returns = returns_spy)

port_spy.assets_stats(method_mu = method_mu,
                       method_cov = method_cov)

mu2 = port_spy.mu
cov2 = port_spy.cov

ws_spy = port_spy.efficient_frontier(model = 'Classic', rm = rm, points = 20,
                             rf = 0, hist = True)


label = 'SPY:Max Risk Adjusted Return Portfolio'
rp.plot_frontier(
    w_frontier=ws_spy, mu=mu2, cov=cov2, returns=returns_spy, rm=rm, rf=0,
    alpha=0.05, cmap='viridis', w=spy_weight, label=label,
    marker='^', s=16, c='r', height=6, width=10, t_factor=252,
    ax=None
)


#%% Comparing the plots of three Portfolios
"""
1)Insights from Investment Growth Plot:
If we invest $100000 in S&P500, it will grow to the highest value at the end of the 5 years as compared
to other two portfolios. Companies portfolio will give a lowest growth in the amount of investment at the end of 
5 years period.

2)Insights from Risk Return Scatter Plot:
Risk Return Scatter Plot shows that S&P500 gives highest average monthly return with lowest standard deviation.
Companies Portfolio shows low monthly return with a higher standard deviation. 
ETFs fall in between, offering moderate returns with slightly higher risk.
This suggests that S&P500 is well-diversified, reducing risk while maximizing returns, 
while Companies portfolio may experience more volatility.

3)Insights from Efficient Frontier:helps to determine the optimal or efficient portfolio.
Companies Portfolio: Provides the lowest risk but also the lowest return, making it the least aggressive option.
ETFs Portfolio: Offers a moderate balance of risk and return, making it more suitable 
for those seeking a middle ground between safety and growth.
SPY Portfolio: Gives the highest return for a moderate level of risk, indicating that 
it has the best potential for investors seeking aggressive growth.


S&P500 is the best-performing in terms of return for risk (low volatility, high return).
"""


#%% Q4
##4)
def calculate_beta_alpha(returns_portfolio, returns_benchmark):
    ##to align the data
    returns_portfolio = np.array(returns_portfolio).flatten()
    returns_benchmark = np.array(returns_benchmark).flatten()

    # adding constant to the benchmark returns
    X = sm.add_constant(returns_benchmark)
    
    # fiting the OLS model
    model = sm.OLS(returns_portfolio, X).fit()
    
    # extracting alpha and beta
    alpha, beta = model.params
    return alpha, beta

# Calculating for each portfolio
alpha1, beta1 = calculate_beta_alpha(monthly_portfolio_returns_companies, returns_spy.values)
alpha2, beta2 = calculate_beta_alpha(monthly_portfolio_returns_etfs, returns_spy.values)
# alpha3, beta3 = calculate_beta_alpha(returns_spy.values, returns_spy.values)  # For SPY vs SPY

#Printing the results
print(f"Portfolio 1 (Companies) - Alpha: {alpha1:.4f}, Beta: {beta1:.4f}")
print(f"Portfolio 2 (ETFs) - Alpha: {alpha2:.4f}, Beta: {beta2:.4f}")
# print(f"Portfolio 3 (S&P 500) - Alpha: {alpha3:.4f}, Beta: {beta3:.4f}")
print(f"Portfolio 3 (S&P 500) - Alpha: {alpha3:.4f}, Beta: {beta3:.4f}")


##4a
#Safest Portfolio
""""Portfolio 1 (Companies) has a beta of 0.8479, which indicates that it is less volatile than the market (less than 1). 
This suggests that it tends to move less than the market in response to market fluctuations, making it a safer choice in
terms of market risk.
Portfolio 2 (ETFs) has a beta of 1.0429, indicating it has similar volatility to the market, suggesting a higher risk 
compared to Portfolio 1.""""'""

##Conclusion: Portfolio 1 has lowest risk as it has lowest Beta, therefore it is considered as Safest Portfolio.

##4b
#highest excess return
""""Both the Portfolios are giving negative returns as compared to market returns

Portfolio 1 (Companies) has a alpha of -0.0043, which indicates that it gives less return than the market returns
(S&P500), but it is less negative.This indicates that the companies portfolio has underperformed the market by
about 0.43% after accounting for risk. 
Portfolio 2 (ETFs) has a alpha of -0.0016, indicating that is more negative than Portfolio 1 and gives less return
that the market returns.This indicates that the ETFs portfolio has underperformed the market by
about 0.16% after accounting for risk.""""'""

""""Conclusion: Portfolio 1 has highest return (less negative Alpha value) as compared to Portfolio2, despite both are
underperforming.This means companies portfolio is performing the least poorly relative to its risk profile.""""'""


#%%
#4a)safest portfolio
""""Beta Analysis:
Portfolio 1 (Companies) has a beta of 0.8336, which indicates that it is less volatile than 
the market (less than 1). This suggests that it tends to move less than the market in response 
to market fluctuations, making it a safer choice in terms of market risk.
Portfolio 2 (ETFs) has a beta of 1.0253, indicating it has similar volatility to the market, 
suggesting a higher risk compared to Portfolio 1.

Conclusion: Portfolio 1 (Companies) is the safest portfolio because it has the lowest beta, indicating 
lower volatility and less exposure to market risk."""

#4b)highest excess return
""""Alpha Analysis:
All portfolios have negative alpha values, which indicate that each portfolio has 
underperformed compared to the expected return based on their respective betas.

Portfolio 1 (Companies) has the highest alpha of -0.0041, meaning it has the least negative 
performance relative to its risk, but it is still negative.
Portfolio 2 (ETFs) follows with an alpha of -0.0014.

Conclusion: Portfolio 1 (Companies) has the highest excess return (least negative) as compared to ETFs
portfolio. It indicates that while none of the portfolios are providing positive excess returns, 
Portfolio 1 is performing the least poorly relative to its risk profile."""



#%% Q5
#Best Portfolio -- ETFs
etfs = ['XLP', 'XLV', 'XLY', 'XLB', 'CARZ', 'XLE', 'XLI', 'XLC']

# Pulling data for ETFs
data_pulling_etfs = yf.download(etfs, start=start, end=end, interval='1mo')['Adj Close']
print(data_pulling_etfs)

returns_etfs = data_pulling_etfs.pct_change().dropna()
print(returns_etfs)

etf_weight = pd.DataFrame([1/8] * 8, index=returns_etfs.columns, columns=['Weight'])
etf_weight

# Calculate portfolio returns using equal weights
portfolio_returns_eq = returns_etfs.dot(etf_weight)  # Use returns_etfs for portfolio returns
cumulative_returns_eq = (1 + portfolio_returns_eq).cumprod()  # Cumulative returns

# Portfolio performance using equal weights
rp.plot_series(returns=returns_etfs, w=etf_weight,
                    cmap='tab20', height=6, width=10,
                    ax=None)
plt.show()

rp.plot_table(returns=returns_etfs, w=etf_weight, MAR=0,
                   alpha=0.05, ax=None)
plt.show()

##5a)Discuss the mean return and CAGR for your portfolio
""""Mean Return (305.1509%): This high number likely reflects some strong positive months or periods in the portfolio.
However, the mean return might not be the best indicator of overall performance because it can be skewed by outliers.

CAGR (9.2%): This provides a better picture of how the portfolio has grown over time. A 9.2% CAGR indicates that the 
portfolio has had a solid performance, growing at a steady pace annually. Given the consistency of this number,
it can be considered a reliable measure for evaluating the portfolio's overall success.""""


""""A 9.2% CAGR is a good indicator of solid long-term performance,especially if we're comparing it to the market 
average such as the S&P 500, which historically has grown by around 7–10% per year.""""

##5b)
##1:Minimum Acceptable Return
"""" It is a minimum amount of return an investor expects to make from an investment. It is generally a threshold value
or benchmark below which we consider the portfolio performance to be unsatisfactory. It helps in determining whether
an investment worth the amount of risk associated with it.

ETF portfolio: We have set MAR at 0. This implies that we're simply aiming to break even or achieve any 
positive return, there is no specific threshold for our portfolio. If the portfolio earns anything above 0%, 
it’s considered acceptable.

Interpretation: Given that the portfolio has achieved a CAGR of 9.2%, it's certainly exceeding the MAR.
This tells that the portfolio is performing 'good' in terms of the baseline expectations, which is a positive sign""""

##2:Average Drawdown
"""" Drawdown basically refers to peak to trough decline during a particular duration for an investment. Average 
Drawdown refers to average of all these declines over time and thus measure the portfolio exposure to the risk.

ETF portfolio:We have an Average Drawdown of 3.3452%. This means that, on an average, 
the portfolio experiences a 3.35% decline from its peak value before recovering.
    
Interpretation: A lower average drawdown indicates lower volatility and risk. Given the value of 3.35% is 
relatively small, it suggests that the portfolio has not faced major drawdowns, 
indicating it is relatively stable. 
Also the fact that the Return-MAR/Risk is 91.22, it shows that for each unit of risk taken, 
your portfolio generates a significant return above MAR, which is quite favorable.
Therefore, the portfolio is considered good according to this risk measure.
""""

##3:Conditional Value at Risk
""""It is commonly use to assist tail risk, which is risk of extreme portfolio losses. 
It measures the average loss of a portfolio in a worst case scenerio, which is beyond certain confidence level.
It is different from VAR, where VAR just quantifY the amount of extreme losses but 
CVAR tells the expected loss when the worst- case threshold has ever crossed.   

ETF portfolio: With a CVaR of 190.77%, this indicates that in extreme adverse market conditions, 
the expected loss (given a specific confidence level) could be as high as 190.77%. 
This suggests that in a worst-case scenario, the portfolio could potentially lose almost twice its value.
    
Interpretation: The Return-MAR/Risk for CVaR is 1.59, meaning the portfolio is expected to generate 
1.59 times more return than the amount of risk associated with extreme losses. While this CVaR value seems high, 
having a positive return-to-risk ratio is a good sign, but it indicates that there is still substantial tail 
risk in the portfolio.

CVAR alone doesn't tell if the portfolio is 'good' or 'bad', if the objective is to achieve return with low risk
or if the investor is risk averse then this portfolio is not acceptable.
But iF the investor is risk seeking and can tolerate high amount of risk for high return, the portfolio 
might turned out to be good for him.
""""


##4:Tail GINI of Losses
""""It mainly focuses on tail risk and measures the inequality in the portfolio losses.
High value represents higher tells fewer but larger losses while small value indicates 
frequent but small losses.

ETF portfolio: A Tail GINI of 227.6% tells that the portfolio is prone to infrequent but significant losses when 
adverse market events occur.This means losses in extreme market conditions are highly concentrated.
    
Interpretation: The Return-MAR/Risk ratio of 1.34 means that for each unit of risk from extreme losses, 
the portfolio is generating returns 1.34 times higher than the Minimum Acceptable Return (MAR). While the 
Tail GINI shows high concentration of losses, it gives a positive indication that the portfolio generates a higher 
return than risk.

This metric suggests that while the portfolio is "good" in terms of managing risk-return trade-offs, 
it could be vulnerable to large, infrequent losses. Depending on the risk tolerance, this may or 
may not be acceptable.
""""

##5:Ulcer Index
""""The Ulcer Index is a measure of downside risk that quantifies the severe losses and 
duration of drawdowns in a portfolio. Unlike traditional volatility measures,it focuses 
specifically on periods when a portfolio's value is declining, providing a more targeted insight into 
the risk experienced by investors.

ETF portfolio: The value of 6.1% indicates that the average drawdown from peak to trough in the portfolio is 
approximately 6.1%. A lower Ulcer Index value suggests that the portfolio is less volatile in terms of drawdowns, 
while a higher value indicates greater volatility and potential stress during downturns.
    
Interpretation: The Return-MAR/Risk ratio of 50.02 suggests that the portfolio's returns significantly 
exceed the MAR relative to the risk of drawdowns. This is a strong positive indicator, indicating that your 
portfolio is generating robust returns compared to the level of downside risk experienced.

For the portfolio, the Ulcer Index value of 6.1006% indicates a relatively low level of drawdown risk. 
And with a high Return-MAR/Risk ratio of 50.019777, suggests that the portfolio has a favorable 
risk-return profile. Thus, based on this metric, the portfolio can be considered “good,” especially 
if investors prioritize downside protection and stability.
""""

##5c) NOTED!!

#%%

def calculate_beta_alpha(returns_portfolio, returns_benchmark):
    X= sm.add_constant(return_benchmark)
    model = sm.OLS(returns_portfolio, X).fit()
    alpha, beta = model.params
    return alpha,beta

def calculate_beta_alpha(returns_portfolio, returns_benchmark):
    # Flatten the input arrays
    returns_portfolio = np.array(returns_portfolio).flatten()
    returns_benchmark = np.array(returns_benchmark).flatten()

    # Ensure the lengths match
    if len(returns_portfolio) != len(returns_benchmark):
        raise ValueError("Portfolio and benchmark returns must have the same length!")

    # Add a constant (intercept) to the benchmark returns
    X = sm.add_constant(returns_benchmark)
    
    # Fit the OLS model
    model = sm.OLS(returns_portfolio, X).fit()
    
    # Extract alpha and beta
    alpha, beta = model.params
    return alpha, beta

# Calculate for each portfolio
alpha1, beta1 = calculate_beta_alpha(monthly_portfolio_returns_companies, returns_spy.values)
alpha2, beta2 = calculate_beta_alpha(monthly_portfolio_returns_etfs, returns_spy.values)
# alpha3, beta3 = calculate_beta_alpha(returns_spy.values, returns_spy.values)  # For SPY vs SPY

# Print results
print(f"Portfolio 1 (Companies) - Alpha: {alpha1:.4f}, Beta: {beta1:.4f}")
print(f"Portfolio 2 (ETFs) - Alpha: {alpha2:.4f}, Beta: {beta2:.4f}")
# print(f"Portfolio 3 (S&P 500) - Alpha: {alpha3:.4f}, Beta: {beta3:.4f}")