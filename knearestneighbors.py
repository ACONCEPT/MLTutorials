# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:00:08 2016

@author: joe
"""
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import math, datetime
from sklearn import preprocessing, cross_validation, neighbors , svm
import sklearn as sk 


df = pd.read_csv('/home/joe/anaconda3/bin/sentdex tutorials/breast_cancer_data')
df.replace('?',-99999, inplace=True)
df.drop(['Id'],1,inplace=True)

x= np.array(df.drop([' class '],1))
y = np.array(df[' class '])
 
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)
 
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
 
accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array ([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print (prediction)

#dir(neighbors)

#dir(svm)