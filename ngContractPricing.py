'''
Inputs:

Injection dates → when you can buy and store gas

Withdrawal dates → when you can sell gas from storage

Prices on those dates → how much gas costs (buy/sell)

Injection/withdrawal rates → how much you can move per day

Max storage → total capacity of your storage

Storage costs → daily or per-unit costs for keeping gas in storage

'''

from datetime import date, timedelta
import csv

dates = [
]

prices = []

with open('Nat_Gas.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if (not row):
            continue
        dates.append(row[0])
        prices.append(float(row[1]))

date_to_index = {dates[i]: i for i in range(len(dates))}

def price_storage_contract(dates, prices, injection_dates, withdrawal_dates,
                           injection_rate, withdrawal_rate, max_volume, storage_cost_per_unit):
    storage=0 
    cash=0

    for i, date in enumerate(dates):
        price = prices[i]

        # Injection
        if date in injection_dates:
            inject_amount = min(injection_rate, max_volume - storage)
            storage += inject_amount
            cash -= inject_amount * price

        # Storage cost for current month
        cash -= storage * storage_cost_per_unit

        # Withdrawal
        if date in withdrawal_dates:
            withdraw_amount = min(withdrawal_rate, storage)
            storage -= withdraw_amount
            cash += withdraw_amount * price

    return cash

# Example injection and withdrawal months
injection_dates = ["11/30/20", "12/31/20"]
withdrawal_dates = ["2/28/21", "3/31/21"]

# Contract parameters
injection_rate = 100          # units per month
withdrawal_rate = 100         # units per month
max_volume = 200              # total storage capacity
storage_cost_per_unit = 0.05  # per unit per month

# Compute contract value
contract_value = price_storage_contract(dates, prices, injection_dates, withdrawal_dates,
                                        injection_rate, withdrawal_rate, max_volume, storage_cost_per_unit)

print("Net value of the storage contract:", contract_value)


    


