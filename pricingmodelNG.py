import os 
import pandas as pd
import numpy as np
from datetime import date, timedelta
cws = os.getcwd()


'''
The input parameters that should be taken into account for pricing are:

Injection dates. 
Withdrawal dates.
The prices at which the commodity can be purchased/sold on those dates.
The rate at which the gas can be injected/withdrawn.
The maximum volume that can be stored.
Storage costs.

'''


def linear_model(current):

    df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
    prices = df['Prices'].values
    dates = df['Dates'].values

    # Simple linear regression model to predict future prices y = Ax + B
    A, B = 0, 0  # Placeholder for model coefficients
    A = np.cov(dates.astype(np.int64),prices)[0][1]/np.cov(dates.astype(np.int64),dates.astype(np.int64))[0][0]
    B = np.mean(prices) - A * np.mean(dates.astype(np.int64))
    # Implement your model logic here
    #predicted_price = 0  # Placeholder for predicted price
    


    # For example, you could use linear regression or any other forecasting method
    # to predict the price for the given 'current' date.

    return predicted_price




df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])

def  pricing(injection_dates, withdrawal_dates, prices, injection_rate, withdrawal_rate, max_volume, storage_costs):
    # Implement pricing logic here


    '''Kind of looks like a liquididity sweep;;; higher highs lowerlows then a breakout'''\
    while True:
        current = date(year, month, 1) + timedelta(days=+1)
        forecast = model(current)

