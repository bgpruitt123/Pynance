#!/usr/bin/env python3 #As written by Brandon Pruitt 8/25/19-2/12/2021
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance
from mplfinance.original_flavor import candlestick_ohlc
#importing candlestick unnecessary now
import matplotlib.dates as mdates 
import pandas as pd
import pandas_datareader.data as web


style.use('ggplot')

start = dt.datetime(2019,5,7)
end = dt.datetime.now()
df = web.DataReader('WMT', 'yahoo', start, end)

print('WMT Stock')
##print(df.tail(200))
    ###gives last 5
#print(df.head())
    ###gives first
##--------------------------------------------------------------------------------spreadsheat creation
df.to_csv('WMTcsv.csv')
    #makes a csv file of data
df = pd.read_csv('WMTcsv.csv', parse_dates = True, index_col=0)
##    #reads csv file 
df.to_excel('WMT.xlsx')
##    #makes a excel file of data
##-------------------------------------------------------------------------------------Plotting 
df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()
    ###moving average

print(df.tail())
print(df[['Open','High']].tail())

#df.plot()
    ###shows all
##df['Adj Close'].plot()
    #print(df['Adj Close'])
    #shows the adj close data only 
plt.show()

df_ohlc = df['Adj Close'].resample('10D').ohlc()
    ###resample('10D') is 10 days/alts are 6min etc
    ###ohlc-openhigh low close/ alts are sum() or mean()
df_volume = df['Volume'].resample('10D').sum()
##print(df_ohlc.tail())

df_ohlc.reset_index(inplace=True)
#print(df_ohlc.tail())

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date()
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price ($)')
##---------------------------------------------------------------------------------Pricing plots 
ax1.set_title('WMT ADJ CLOSE PRICE')
ax2.set_title('WMT VOLUME')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
ax1.plot(df.index, df['Adj Close'],color='blue')
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
plt.show()
##---------------------------------------------------------------------------------Candlestick plots 
ax3 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax4 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax3)
ax3.xaxis_date()
ax3.set_xlabel('Date')
ax4.set_ylabel('Price ($)')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price ($)')
ax3.set_title('WMT OHLC PRICE')
ax4.set_title('WMT VOLUME')
candlestick_ohlc(ax3, df_ohlc.values, width=5, colorup='g')
ax4.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
plt.show()