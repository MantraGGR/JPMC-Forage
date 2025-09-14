import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from datetime import date, timedelta
import csv 
#---- BAR CHART -----
x = []
y = []

def days_after(start_date, current_date):
    return (current_date - start_date).days


with open('Nat_Gas.csv') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')

    for row in plots:
        x.append(row[0])
        y.append(row[1])

plt.bar(x,y, color = 'g', width = 0.72, label = 'Price')
plt.xlabel('Days After 31-10-2020')

ax = plt.gca() # get current axis

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # show every month
'''
ax.xaxis gives us control over the x-axis specifically.

set_major_locator tells Matplotlib where to put major ticks (labels).

mdates.MonthLocator(interval=1) means: put a tick every 1 month.

If you use interval=2, it will show one tick every 2 months, etc.


'''
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # Format like "Jan 2021"


'''
set_major_formatter tells Matplotlib how to display those tick labels.

mdates.DateFormatter('%b %Y') formats the date:

%b → abbreviated month name (Jan, Feb, …).

%Y → full year (2020, 2021, …).

So instead of raw dates like 2020-11-01, it shows Nov 2020.


'''

plt.gcf().autofmt_xdate(rotation= 45) # Rotate date labels for better fit

'''

plt.gcf() = “get current figure.”

autofmt_xdate(rotation=45) automatically adjusts the x-axis date labels:

Rotates them 45° so they don’t overlap.

Adds spacing to keep them readable.



'''


plt.ylabel('Price')
plt.title('Natural Gas Prices Over Time')
plt.legend()
plt.show()


#---- LINE CHART -----

x = []
y = []

#monthly moving average
def moving_average(data): # 3 month moving average
    f = [None] * len(data)
    for i in range(2, len(data)):
        f[i] = (data[i] + data[i-1] + data[i-2]) / 3

    plt.plot(range(len(data)), f, color='g', marker='o',
                label='3-Month Moving Average', linestyle='dashed')
    return f



with open('Nat_Gas.csv') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        x.append(row[0])
        y.append(float(row[1]))
        
        
ma = moving_average(y)

plt.plot(x,y, color = 'g', marker = 'o', label = 'Price', linestyle = 'solid')
plt.plot(x, ma, color = 'b', marker = 'o', label='3-Month Moving Average', linestyle='dashed')

plt.xlabel('Days After 31-10-2020')
plt.ylabel('Price')
plt.title('Natural Gas Prices Over Time')
plt.legend()
plt.show()
'''

If the price is above the 3 month moving average, Long
If the price is below the 3 month moving average, Short




'''
