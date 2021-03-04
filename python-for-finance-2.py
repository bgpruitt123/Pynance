##dates are defined by running python-for-finance-1
DIR='C:/Users/Brandon Pruitt/Desktop/Research/Python Code/Finance/'
import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import yfinance as yf


#------------------------------------------------------------------------------Removes previous date's info (so it can be updated)
#Ideally in the future this would be changed to just update the most recent values, and not delete/download data
    #Would need to be retested on a day where the market data is uploaded already 

dir = 'C:/Users/Brandon Pruitt/Desktop/Research/Python Code/Finance/stock_dfs'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

DIR='C:/Users/Brandon Pruitt/Desktop/Research/Python Code/Finance/'

#os.remove("sp500_joined_closes.csv")
    #only needed if still in file inventory
os.rmdir("stock_dfs")
#----------------------------------------------------------------------------------

style.use('ggplot')
#there should be candlestick functionality now 
yf.pdr_override

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
  
    start = dt.datetime(2004,12,1)
    end = dt.datetime.now()
   
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

#-----------------------------#ONLY NEED TO DO THIS ONLY WHEN FILES DON'T ALREADY EXIST
save_sp500_tickers()
get_data_from_yahoo()
 
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        
        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
        if count % 10 == 0:
            print(count)
    print(main_df.tail())
    main_df.to_csv('sp500_joined_closes.csv')

compile_data()
##-------------------------------------Visualize stock price comparative data 
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
##    df['AAPL'].plot()
##    plt.show()
    df_corr =df.corr()
    ##generates correlational values
    
    print(df_corr.head())
    
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
##----------------------------------Visualize stock return comparative data 
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df.set_index('Date',inplace=True)
    df_corr =df.pct_change().corr()
    ##generates correlational values
    
    print(df_corr.head())
    
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
##------------------------------------------------------Heatmapping
    heatmap = ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) +0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) +0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
    
visualize_data()