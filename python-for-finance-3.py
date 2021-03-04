##ensure stock price option is selected from python-for-finance-2
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    
    hm_days = 7 ##how many days (of correlation) in future to make or lose
                ##NOTE: relationships between companies change frequently so this shouldn't exceed 2 years max   
    df =pd.read_csv('sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) -df[ticker]) / df[ticker]
            #percent change in option selected in python-for-finance-2
            #Shifts column up to get future value 
    df.fillna(0,inplace=True)
    return tickers, df
def buy_sell_hold(*args):
    cols = [c for c in args]
##-------------------------------------------------------------------------------requirements to buy or sell   
    requirement_buy = 0.025  ## percent change alerts system
    requirement_sell = 0.020 ## percent change alerts system
    for col in cols:
        if col > requirement_buy:   ##buy requirement 
            return 1
        if col < -requirement_sell: ##sell requirement
            return -1
    return 0
##---------------------------------------------------------------------------------------------------------------------------------
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    df['[]_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)],
                                                ))
    vals = df['[]_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
  
##--------------------------------------------Featuresets and label creation
    df_vals = df[[ticker for ticker in tickers]].pct_change()
        ## % change from one day for all companies
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0,inplace=True)
    
    X = df_vals.values
    y = df['[]_target'.format(ticker)].values
    
    return X, y, df

##-----------------------------------------------------------machine learning starts
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.25)
                ##test_size = 0.25 means 25% of sample data is used to test against for accuracy
##---------------------------------------------K nearest Neighbors classifier  
    ##clf = neighbors.KNeighborsClassifier()
##----------------------------------------------------Voting of 3 classifiers
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                           ('rfor',RandomForestClassifier())])
            ##LSVC = linear support vector classifier
            ## KNN = K  to the nearest neighbors
                ##NOTE: research into parameters of these methods will improve accuracy
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
        #NOTE: Accuracy is not perfect
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
        
    return confidence
    
do_ml('TSLA')
    ##stock to use machine learning to make predictions about
##------------------------------------------------------------------------------------How to read result
        ##Note: wanted Sample Spread to be as near equal for all three as possible
            ##run multiple times with near equal sample size to get result
##Data spread     : Counter({'HOLD': Sample Spread, 'SELL': Sample Spread, 'BUY': Sample Spread})
##Predicted spread: Counter({'0 is HOLD': Amount Favoring,
                         ##  '-1 is SELL': Amount Favoring,
                        ##   '1 is BUY': Amount Favoring})
    