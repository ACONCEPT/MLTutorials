import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

foldername = 'btcdownloads/18-JUNE-2016' 

onlyfiles = [f for f in listdir('/home/joe/Downloads/%s' % (foldername)) if isfile(join('/home/joe/Downloads/%s' % (foldername), f))]

for file in onlyfiles:
  end = file[len(file)-3:len(file)]
  if end == 'csv':
      filename = file[0:len(file)-4]
      filename = filename.lower()
      print (filename + ' = pd.read_csv(\'/home/joe/Downloads/%s/' % (foldername) + file + '\')')
      
      
btc_daily_change = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_DAILY_CHANGE.csv')
btc_daily_avg = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_DAILY_AVG.csv')
btc_monthly_avg = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_MONTHLY_AVG.csv')
btc_monthly_change = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_MONTHLY_CHANGE.csv')
btc_weekly_change = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_WEEKLY_CHANGE.csv')
btc_weekly_avg = pd.read_csv('/home/joe/Downloads/btcdownloads/18-JUNE-2016/BTC_WEEKLY_AVG.csv')