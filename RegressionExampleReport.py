import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import math, datetime
import time 
import pickle
#mport matplotlib.pyplot as pplot
#from matplotlib import style
#style.use('ggplot')


# import data 
btc_daily_change = pd.read_csv('/home/joe/Downloads/btcdownloads/7-JULY-2016/BTC_DAILY_CHANGE.csv')
btc_daily_avg = pd.read_csv('/home/joe/Downloads/btcdownloads/7-JULY-2016/BTC_DAILY_AVG.csv')

df1 = btc_daily_avg[['Date','24h Average']]
df1.columns = ['date', 'dailyprice']

df2 = btc_daily_change[['Date','24h Average']]
df2.columns = ['date', 'dailychange']

#Merge syntax 
#merge(left, right, how='inner', on=None, left_on=None, right_on=None,
#      left_index=False, right_index=False, sort=True,
#      suffixes=('_x', '_y'), copy=True, indicator=False)

df = pd.merge(df1, df2, how='inner', on='date', left_index=True, right_index=True, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False)
      
df = df.sort(['date'],0)      
#import preprocerssing modules 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

forecast_col = 'dailyprice'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(.002 * len(df))) 

#pulls the label of the dailyprice in the amount of the number of days in the future
df['label'] = df[forecast_col].shift(-forecast_out)
dates = np.array(df['date'])
df = df.set_index(['date'])

#set x axis equal to an array of all the values except the label  and set y axist equal to label
x = np.array(df.drop(['label'],1))
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
#Scale X-Axis data and shape into two arrays. x_predict contaions all values of x without known labels
#x = preprocessing.scale(x)
df.dropna(inplace=True)
y = np.array(df['label'])


#partition training set 
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#set prediction factor 
clf = LinearRegression()

#fit and test the accuracy of predictor
clf.fit(x_train,y_train)

# pickle the classifier for later use.
with open (linearregression.pickle, 'wb') as 


accuracy = clf.score(x_test,y_test)

 # print(accuracy)
forecast_set = clf.predict(x_lately)
df['Forecast'] = np.nan


# last_date = df.iloc[-1].name
# last_unix = time.mktime(last_date.time())
# one_day = 86400
# next_unix = last_unix + one_day

# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [1]
    
    
    