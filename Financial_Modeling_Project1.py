# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:38:09 2024

@author: Divya Pahuja
"""

##download necessary libraries
#pip install nycflights13
#pip install pandas
#pip install numpy
#pip install matplotlib
#pip install plotly


#%%
#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nycflights13 import flights
import plotly.express as px 
import webbrowser


#%%

##Creating DataFrame
df = flights
df.columns


#%%
##1(a)
arrival_delay = df[df['arr_delay']>=120]
print(arrival_delay)
print(arrival_delay[['flight','carrier','arr_delay']])


#%%
##1(b)
flew_to_houstan = df[df['dest'].isin(['IAH','HOU'])]
print(flew_to_houstan)
print(flew_to_houstan[['flight','carrier','origin','dest']])


#%%
##1(c)
c=df['carrier'].unique()
print(c)


operations = df[df['carrier'].isin(['UA','AA','DL'])]
print(operations)
operations.head()
print(operations[['flight','carrier','origin','dest']])


#%%
##1(d)
month=df['month'].unique()
print(df['month'].head())


departed_in_sum = df[df['month'].isin([7,8,9])]
print(departed_in_sum)
departed_in_sum.head()
print(departed_in_sum[['month','flight']])


#%%
##1(e)
##Arrived more than two hours late, but didn’t leave late 
delay=df['dep_delay'].unique()
print(delay)


arr_delay = df[df['arr_delay'] >= 120] 
leave_on_time = arr_delay[arr_delay['dep_delay'] <= 0]
leave_on_time.head()
print(leave_on_time[['flight', 'carrier', 'arr_delay', 'dep_delay']].head())

##or

arr_delay_left_on_time = df[(df['arr_delay'] >= 120) & (df['dep_delay'] <= 0)]
print(arr_delay_left_on_time)
print(arr_delay_left_on_time[['flight', 'carrier', 'arr_delay', 'dep_delay']].head())


#%%
##1(f)
delayed_flight_cover_up =  df[(df['dep_delay'] >= 60) & ((df['dep_delay'] - df['arr_delay']) > 30)]

print(delayed_flight_cover_up[['year', 'month', 'day', 'carrier', 'flight', 'origin', 'dest', 'dep_delay', 'arr_delay']])


#%%
##1(g)
departure=df['dep_time'].unique()
print(departure)


departure_time = df[(df['dep_time'] >= 0) & (df['dep_time'] <= 600)] 
print(departure_time[['carrier', 'flight', 'origin', 'dep_time']])
departure_time.head()


#%%
##2
##most delayed flight
most_delayed_flight = df.sort_values(by = 'arr_delay' , ascending = False)
most_delayed_flight.head(10)
print(most_delayed_flight[['carrier', 'flight', 'origin', 'arr_delay']])

#earliest flight
earliest_flight = df.sort_values(by = 'dep_time' , ascending = True)
earliest_flight.head(10)
print(earliest_flight[['carrier', 'flight', 'origin', 'dep_time']])


#%%
##3
df['gain_in_air'] = df['dep_delay'] - df['arr_delay']
print(df[['dep_delay', 'arr_delay', 'gain_in_air']].head())

##positive value of gain in air means that the flight has arrived before the actual departure delay time. 
#For example, flight got delayed by 285 minutes but actual arrival delay was 246 minutes which was less than the expected arrival delay


#%%
##4
##calculating speed
df['flight_speed'] = df['distance'] / df['air_time'] * 60
print(df[['flight','distance','air_time','flight_speed']].head())


##sorting fastest flights
fastest_flight = df.sort_values(by = 'flight_speed' , ascending = False)
fastest_flight.head(10)

print(fastest_flight[['flight','distance','air_time','flight_speed']].head())

##rows with maximum distance
max_dis=df['distance'].idxmax()

##rows with minimum distance
min_dis=df['distance'].idxmin()

##4(a)
##longest flight
longest_flight = df.loc[max_dis, ['flight','distance','carrier']]
print(longest_flight)

##shortest flight
shortest_flight = df.loc[min_dis, ['flight','distance','carrier']]
print(shortest_flight)


#%%
##5
average_arrival_delay = df.groupby('carrier').agg({'arr_delay' : 'mean'}).reset_index()
average_arrival_delay

###highest average arrival
max_average_index = average_arrival_delay['arr_delay'].idxmax()
max_average_index

high_avg_arr = average_arrival_delay.loc[max_average_index]
print(high_avg_arr)


#%%
###6
fig = px.scatter(
    df, 
    x='dep_delay', 
    y='arr_delay', 
    labels={'dep_delay': 'Departure Delay (min)', 'arr_delay': 'Arrival Delay (min)'}, 
    title='Bivariate Analysis: Arrival Delay vs Departure Delay')

fig.write_html("plot.html")
webbrowser.open("plot.html")

plot(fig)

##The above scatter plot shows that both arrival delay and departure delay are positively correlated


#%%
##month on month count of flights
""""monthly_flights = df.groupby('month').size().reset_index(name = 'flight_count')

monthly_flights

fig = px.bar(monthly_flights, x = 'month' , y = 'flight_count', 
             labels = {'month' : 'MONTH' , 'flight_count' : 'NUMBER OF FLIGHTS'},
             title = 'Number of Flights Per Month')

fig.write_html("plot.html")
webbrowser.open("plot.html")

fig.show()""""

## above bar plot shows that lowest number of flights were departed in the month of February and highest number in July.










