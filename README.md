# google-stock-prediction
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:26:25 2017

@author: pandurang
"""

import pandas as pd
import math 
import quandl
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
df=quandl.get('WIKI/GOOGL')
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100.0
df['PC_CH']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df=df[['Adj. Close','HL_PCT','PC_CH','Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-9999, inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

x=np.array(df.drop(['label'],1))
y=np.array(df['label'])
X=preprocessing.scale(x)
Y=preprocessing.scale(y)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,Y,test_size=0.4)

clf=LinearRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)



