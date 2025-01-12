# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:52:41 2024

@author: Divya Pahuja
"""

import pandas_datareader as pdr
from darts import TimeSeries
from darts.models import LinearRegressionModel, AutoARIMA, Prophet, XGBModel
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
from darts.metrics import mae, rmse, mape

#%%
#Q1)
#S1:Producer Price Index for All Commodities
df_s1 = pdr.DataReader('PPIACO', 'fred', start = '2000-01-01').dropna()

##checking if null exists after dropping them
df_s1.isnull().sum()
df_s1.head()


#S2:DOGECOIN
df_s2 = yf.download('DOGE-USD', start='2014-01-01', end='2023-12-31', interval='1mo')
df_s2

##dropping null values
df_s2.dropna(inplace=True)

##checking if null exists after dropping them
df_s2.isnull().sum()


#%%
#Q2)-PPI
df_eco_ind = df_s1.resample('M').last()

tseries_eco_ind = TimeSeries.from_dataframe(df_eco_ind, value_cols = 'PPIACO',
                                            freq = 'M')

df_eco_ind_ppi = tseries_eco_ind.pd_dataframe()

#a)
df_eco_ind_ppi['lag_1'] = df_eco_ind_ppi['PPIACO'].shift(1)
df_eco_ind_ppi['lag_2'] = df_eco_ind_ppi['PPIACO'].shift(2)
df_eco_ind_ppi['lag_3'] = df_eco_ind_ppi['PPIACO'].shift(3)


##dropping null values
df_eco_ind_ppi.dropna(inplace=True)
df_eco_ind_ppi

#b)
##defining independent and dependent variable
x = df_eco_ind_ppi[['lag_1','lag_2','lag_3']]
y = df_eco_ind_ppi['PPIACO']


#add constant
x = sm.add_constant(x)

# Fit the OLS model
model = sm.OLS(y, x).fit()

# Print the summary of the OLS model
print(model.summary())


#c)
##output
"""The below results shows the regression model result where the independent variable are our 3 lags -
lag1,lag2,lag3 and dependent variable is PPIACO which is Producer price index.

The results interpretation is as follows:

R-squared: It is 0.996 which is close to 1. It tells that 99.6% of the variability in dependent variable is 
explained by our independent variable which are the lags. As the value is close to 1 it shows that the model
is good fit.
Adjusted R-squared : It priorities the number of input variable that we consider for the model prediction.
R-squared will always increase if we add new variables into the model, but Adjusted R-squared will decrease if the
variables added are not of any value to the model and increase if they improve the model.
As the value is same as R squared value, which means that model fits the data well and the three
variables are appropriate for the model.
F-statistics: It evaluates the overall significance of the independent variable towards contributing 
to the dependent variable. It has null hypothesis that if all the lags have coefficient which are 0
and opposite for alternative hyposthesis. As the value is high, it means model is statistically
significant.
P-value of F-statistics: As the p-value is less tha 5% it means that we will reject the null hypothesis
that no lags are contributing towards the dependent variable. This p-value tells that model is
statistically significant i.e., atleast one independent variable significantly contributes towards predicting
dependent varibale PPI.
AIC and BIC help you compare different models to choose the one that best balances fit and complexity.
AIC = 1298 means the model has a relatively good fit considering both the number of parameters and 
the fit to the data.
BIC = 1313 suggests that, while the model fits the data well, there may be some complexity that 
could lead to overfitting, considering the larger penalty for the number of parameters.

Coeffiecients:
Intercept: It is  0.7672, this is the value of y where all lags value is 0. As the p-value of the
y-intercept is more than 0.05 this means it is not statistically significant and does not
significantly differ from 0.


LAG1: It is  1.3994, this is the value of y where the intercept and other two lags value is 0. 
It means that with each increase in lag_1, PPI increases by  1.3994.
As the p-value is 0.000 and is less than 0.05 this means it is statistically significant
in explaining the dependent variable.

LAG2: It is -0.2907, this is the value of y where the intercept and other two lags value is 0.
It means that with each increase in lag_2, PPI decreases by  0.2907. 
As the p-value is 0.004 and is less than 0.05 this means it is statistically significant
in explaining the dependent variable.

LAG3: It is -0.1117, this is the value of y where the intercept and other two lags value is 0.
It means that with each increase in lag_3, PPI decreases by 0.1117. 
As the p-value is 0.057 and is litle more than 0.05 this means it this lag is marginally significant
in explaining the dependent variable.It means that this lag has very less impact in predicting the
dependent variable PPI.

Test:
Omnibus: The Omnibus test checks for the normality of the residuals. The result is 73.242 with 
a p-value of 0.000, indicating that the residuals are not normally distributed.

Jarque-Bera (JB): A similar test to Omnibus that also checks for normality of the residuals. 
The result is very high (294.880), and the p-value is very small (9.28e-65), confirming that the 
residuals deviate significantly from a normal distribution.

Skew: The skewness value is -0.983, which suggests that the residuals are negatively skewed, i.e., 
there are more large positive errors than negative errors.

Kurtosis: The kurtosis value is 7.486, which is much higher than the normal distribution value of 3, 
indicating heavy tails or outliers in the residuals.

Durbin-Watson: This statistic tests for autocorrelation in the residuals. A value near 2 suggests 
no autocorrelation. Your value is 1.996, which is close to 2, meaning there is no significant autocorrelation 
in the residuals."""
 
#%%
#Q3)
##train/test split
train,test = tseries_eco_ind.split_before(0.8)
forecast_horizon_future = 24

#a,b,c)

## i)Linear Regression model
lr_model = LinearRegressionModel(lags = 12)
lr_model.fit(train)
linear_reg_forecast = lr_model.predict(forecast_horizon_future)


# ii)Arima Model
arima_model = AutoARIMA()
arima_model.fit(train)
arima_forecast = arima_model.predict(forecast_horizon_future)

# iii)Prophet model
prophet_model = Prophet()
prophet_model.fit(train)
prophet_forecast = prophet_model.predict(forecast_horizon_future)

# iv)xgboost model
xgb_model = XGBModel(lags = 12)
xgb_model.fit(train)
xgb_forecast = xgb_model.predict(forecast_horizon_future)


#%%
#d)
# Plotting the results
plt.figure(figsize=(10, 6))
train.plot(label="Train")
test.plot(label="Test")
linear_reg_forecast.plot(label="Linear Regression", lw=2)
arima_forecast.plot(label="ARIMA", lw=2)
prophet_forecast.plot(label="Prophet", lw=2)
xgb_forecast.plot(label="XGBoost Mean Forecast", lw=2)
plt.legend()
plt.title("Forecasting Models Comparison of PPI")
plt.show()


##result interpretation of model and forecast
"""Here we have considered four models:
Linear Regression: This model is simple and interpretable but might miss complex patterns.
ARIMA: It works well for stationary time series but struggles with non-stationary or highly volatile data.
Prophet: It handles seasonality and irregular data well but might underperform with short time series.
XGBoost: It excels at capturing complex, non-linear relationships but may overfit without proper tuning.

After plotting the prediction/forecast of these four models,plot shows that predictions made through 
Linear Regression, ARIMA and XGboost are not able to give good predictions and are just forming a straight 
line but predictions made through Prophet model are better than other three models and
follows the trend of the actual data.

Thus, this shows that Prophet model is the best performing model for the economic indicator PPIACO.
"""


#%%
#Q4)DOGECOIN
df_s2 = df_s2.resample('M').last()

series_dc = TimeSeries.from_dataframe(df_s2, value_cols='Adj Close', freq='M')


##train/test split
train_dc,test_dc = series_dc.split_before(0.8)
forecast_horizon_future_dc = 24


##Linear Regression model
lr_model_dc = LinearRegressionModel(lags = 12)
lr_model_dc.fit(train_dc)
linear_reg_forecast_dc = lr_model_dc.predict(forecast_horizon_future_dc)


#Arima Model
arima_model_dc = AutoARIMA()
arima_model_dc.fit(train_dc)
arima_forecast_dc = arima_model_dc.predict(forecast_horizon_future_dc)

#Prophet model
prophet_model_dc = Prophet()
prophet_model_dc.fit(train_dc)
prophet_forecast_dc = prophet_model_dc.predict(forecast_horizon_future_dc)


##xgboost model
xgb_model_dc = XGBModel(lags = 12)
xgb_model_dc.fit(train_dc)
xgb_forecast_dc = xgb_model_dc.predict(forecast_horizon_future_dc)

#%%
# Plotting the results
plt.figure(figsize=(10, 6))
train_dc.plot(label="Train")
test_dc.plot(label="Test")
linear_reg_forecast_dc.plot(label="Linear Regression", lw=2)
arima_forecast_dc.plot(label="ARIMA", lw=2)
prophet_forecast_dc.plot(label="Prophet", lw=2)
xgb_forecast_dc.plot(label="XGBoost Mean Forecast", lw=2)
plt.legend()
plt.title("Forecasting Models Comparison of Dogecoin")
plt.show()


##result interpretation of Dogecoin model and forecast
"""This plot compares forecast models for Dogecoin prices, which exhibit higher volatility:

Prophet: Over-fits and reacts to volatility significantly, showing exaggerated peaks.
ARIMA: Provides stable forecasts but cannot capture high volatility effectively.
Linear Regression: Provides a flat forecast.
XGBoost: Struggles with sharp fluctuations, showing smoother lines.

By seeing the predictions, we can say that XGBoost is the best predicting model for Dogecoin
as it smooths out the volatility as compared to other three models. And Prophet is the worst performing, as it reacts 
excessively to volatility or over-fits to the data and produces forecasts that are less reliable.

Comparing this to above question, we can say that ARIMA and Linear Regression aren't good model for 
forecasting volatile assets as these are very simple and rigid models.
"""
    

#Performance metrics
print("Linear Regression:")
print("MAE:", mae(test, linear_reg_forecast_dc)) 
##Since Dogecoin prices are typically volatile, this error suggests that linear regression struggles to adapt to sudden price changes.
print("RMSE:", rmse(test, linear_reg_forecast_dc))
##A high RMSE relative to the price range highlights the limitations of the model in volatile datasets.
print("MAPE:", mape(test, linear_reg_forecast_dc))
##If Dogecoin prices have rapid swings, even a small percentage error might lead to predictions that miss important price peaks or drops.

##Thus, the linear regression model is not well-suited for forecasting highly volatile datasets like Dogecoin prices.

print("ARIMA:")
print("MAE:", mae(test, arima_forecast_dc))
print("RMSE:", rmse(test, arima_forecast_dc))
print("MAPE:", mape(test, arima_forecast_dc))

##Results for ARIMA ar same as that of Linear Regression, therefore ARIMA is not good model for the Dogecoin prices.
##Thus, these two models doesn't make good prediction for Dogecoin.

