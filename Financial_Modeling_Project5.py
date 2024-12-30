    # -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:00:42 2024

@author: Divya Pahuja
"""

#Import Libraries
import requests
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot


api_key='m6RJac7MBthiPb2DczKZwqYkcGns4DPf'

#%%
#Q1)
def fetch_financial_ratios(ticker):
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit=4&apikey={api_key}"
    response = requests.get(url)
    ratios = response.json()
    
    if isinstance(ratios,list) and len(ratios) > 0:
        return pd.DataFrame([{
            'Company' : ticker,
            'Quarter' : entry['date'],
            'P/E Ratio' : entry.get('priceEarningsRatio' , None),
            'D/E Ratio' : entry.get('debtEquityRatio', None),
            'ROE' : entry.get('returnOnEquity', None)
            } for entry in ratios])
    
    else:
        return pd.DataFrame()
    

# Fetch data for each company
df_cmcsa = fetch_financial_ratios('CMCSA')
df_t = fetch_financial_ratios('T')
df_vz = fetch_financial_ratios('VZ')
df_chtr = fetch_financial_ratios('CHTR')

# Concatenate all company data
df_ratios = pd.concat([df_cmcsa, df_t, df_vz, df_chtr], ignore_index=True)
df_ratios['Quarter'] = pd.to_datetime(df_ratios['Quarter'])

# Create subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("P/E Ratio Over Quarters", "D/E Ratio Over Quarters", 'ROE Over Quarters'))
#1a)
# Add P/E Ratio traces
for company in df_ratios['Company'].unique():
    company_data = df_ratios[df_ratios['Company'] == company]
    fig.add_trace(go.Scatter(x=company_data['Quarter'], y=company_data['P/E Ratio'],
                             name=f'{company} P/E Ratio'), row=1, col=1)

# Add D/E Ratio traces
for company in df_ratios['Company'].unique():
    company_data = df_ratios[df_ratios['Company'] == company]
    fig.add_trace(go.Scatter(x=company_data['Quarter'], y=company_data['D/E Ratio'],
                             name=f'{company} D/E Ratio'), row=2, col=1)
#1b)
# Add ROE traces
for company in df_ratios['Company'].unique():
    company_data = df_ratios[df_ratios['Company'] == company]
    fig.add_trace(go.Scatter(x=company_data['Quarter'], y=company_data['ROE'],
                             name=f'{company} ROE'), row=3, col=1)
#1c)
# Update layout
fig.update_layout(height=800, title_text="P/E & D/E Ratios and ROE", showlegend=True)
fig.update_xaxes(title_text='Quarter')
fig.update_yaxes(title_text='P/E Ratio', row=1, col=1)
fig.update_yaxes(title_text='D/E Ratio', row=2, col=1)
fig.update_yaxes(title_text='ROE', row=3, col=1)

# Plot the figure
plot(fig)

##1d)
"""Performance analysis of CMCSA:
According to the plot, best performing asset is CMCSA(Comcast Corporation) because:
P/E Ratio: P/E ratio plot tells that CMCSA has increasing P/E ratio which tells that there are
positive sentiment for this asset in the market because good price to earning ratio tells that
investors expect higher future earnings growth from the company.
D/E Ratio : D/E ratio of CMCSA is lowest of all the assets and is quite stable which means that company has
very low amount of debt as compared to equity which tells that company may have conservative approach 
towards taking debt or maybe company is financially strong.
ROE Ratio : ROE plot suggest that though the ROE of CMCSA is lower as compared to CHTR and VZ
but ROE of CMCSA is stable and increasing as compared to other assets which means improved profitability.
This means CMCSA has good financial performing.

Thus,with good earning and profitability and low/stable debt CMCSA is performing better as
compared to other assets."""


#%%
#Q2
#2a)
#CHTR -- Charter Communication
ticker = 'CHTR'
url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=10&apikey={api_key}"
response = requests.get(url)
cash_flows = response.json()
#%%
# Extract FCF for the last 5 years

fcf_data = []
for entry in  cash_flows[:5]:
    fcf_data.append({
        'date' : entry['date'],
        'FCF' : entry['freeCashFlow']
        })
    

df_fcf = pd.DataFrame(fcf_data)

# Step 2: Estimate Growth Rate
# Calculate the historical growth rate of FCF
#%%
df_fcf = df_fcf.sort_values(by = 'date')
df_fcf['FCF Growth'] = df_fcf['FCF'].pct_change()

growth_rate = df_fcf['FCF Growth'].mean()
growth_rate
#%%
years = 5
projected_fcf = []

last_fcf = df_fcf.iloc[0]['FCF']
for i in range(1,years +1):
    projected_fcf.append(last_fcf * (1+growth_rate) ** i) ##** is exponent


projected_fcf
#%%
#2b)
#Calculation of WACC
rfr = 0.03
mr = 0.1
beta = 1.5

re = rfr + beta * (mr - rfr)
re

rd = 0.07

tax_rate = 0.21

#getting debt and equity values from balance sheet
balance_sheet_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=1&apikey={api_key}"
balance_sheet_response = requests.get(balance_sheet_url)
balance_sheet = balance_sheet_response.json()[0]


total_equity = balance_sheet['totalStockholdersEquity']
total_debt = balance_sheet['totalDebt']

wacc = ((total_equity/(total_equity+total_debt))*re) + ((total_debt/(total_equity+total_debt))*rd*(1-tax_rate))
wacc

#%%
#Terminal Value
terminal_growth_rate = .04
terminal_value = projected_fcf[-1]*(1+terminal_growth_rate) / (wacc - terminal_growth_rate)

terminal_value

#%%DCF
discounted_fcf = [cf/(1+wacc)**i for i, cf in enumerate(projected_fcf, start = 1)]
discounted_fcf
#%%TV
discounted_terminal_value = terminal_value / (1 + wacc)**years
discounted_terminal_value
#%%
enterprise_value = sum(discounted_fcf) + discounted_terminal_value
enterprise_value


#%%
#Net Debt
net_debt = balance_sheet['totalDebt'] - balance_sheet['cashAndCashEquivalents']
net_debt

#%%
equity_value = enterprise_value - net_debt
shares_outstanding = 145000000

price_per_share = equity_value / shares_outstanding
price_per_share


#%%
#2c)
# Print the results
print(f"Projected FCFs: {projected_fcf}")
print(f"Discounted FCFs: {discounted_fcf}")
print(f"Terminal Value: {terminal_value}")
print(f"Discounted Terminal Value: {discounted_terminal_value}")
print(f"Enterprise Value: {enterprise_value}")
print(f"Price per share: {price_per_share}")


"""The Enterprise Value is calculated to be $174.37 billion, representing the 
total company value.

Undervalued/Overvalued:
Given that CHTR's current price per share is $393.71.
The Price per Share of $533.14 according to the DCF analysis is significantly higher 
than the current market price of $393.71. This suggests that the stock is "Undervalued" 
according to the analysis,as investors are paying less than what the DCF model estimates 
the stock is worth.
Thus, there may be a buying opportunity if the assumptions hold."""















































