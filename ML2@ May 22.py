#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# In[2]:


data=pd.read_csv("D:/IMARTICUS WORK/Paper1/MonthWiseMarketArrivals_Clean.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.shape


# In[6]:


data.columns=['market', 'month', 'year', 'quantity', 'priceMin', 'priceMax',
       'priceMod', 'state', 'city', 'date']
data.info()


# In[10]:


data.drop('city', axis=1)


# In[12]:


data.hist()


# In[13]:


data.describe()


# In[14]:


data.value_counts()


# In[143]:


plt.plot(data.index)


# In[139]:


data.boxplot()


# In[35]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[40]:


from sklearn.model_selection import train_test_split
train=data.iloc[:132]
test=data.iloc[132:]


# In[41]:


trainlast=train.tail(1)


# In[42]:


print(trainlast)


# In[44]:


test["navie"]


# In[46]:


train["navie"]=405


# In[133]:


from sklearn.metrics import mean_squared_error
rmse1=mean_squared_error(test.priceMax,test.navie,squared=False)
rmse2=mean_squared_error(test.priceMax,test.navie,squared=False)
print(rmse1,rmse2)


# In[134]:


shift=data.shift(1)


# In[135]:


x=data.iloc[:,2:3].values
y=data.iloc[:,3:4].values


# In[78]:


from sklearn import datasets
from sklearn.model_selection import cross_val_score 


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)


# In[86]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[88]:


#predict the test result
y_pred=regressor.predict(x_test)


# In[132]:


plt.scatter(x_train,y_train,c='red')
plt.show()


# In[131]:


plt.plot(x_test,y_pred)   
plt.scatter(x_test,y_test,c='green')
plt.xlabel('month')
plt.ylabel('priceMax')


# In[94]:


import numpy as np


# In[95]:


rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))


# In[98]:


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[129]:


data = pd.read_csv('D:/IMARTICUS WORK/Paper1/MonthWiseMarketArrivals_Clean.csv',index_col='priceMax', parse_dates=True)


# In[104]:


data.plot(figsize=(10,5))


# In[128]:


# calculate acf
acf_values = acf(data['priceMax'])


# In[127]:


# keeping lag as 30
plot_acf(data['priceMax'], lags=20);


# In[126]:


# PACF
pacf_values = (data['priceMax'])


# In[124]:


# plot pacf
plot_pacf(data['priceMax'], lags=30)


# In[144]:


import pandas as pd


# In[145]:


#plotting data - matplotlib
from matplotlib import pyplot as plt


# In[146]:


# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose


# In[147]:


from statsmodels.tsa.seasonal import seasonal_decompose 


# In[148]:


# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   


# In[149]:


# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[166]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x_train,y_train)


# In[168]:


regressor.score(x_train,y_train)


# In[170]:


regressor.score(x_test,y_test)


# In[171]:


from sklearn.ensemble import GradientBoostingRegressor
regres = GradientBoostingRegressor(n_estimators=100,random_state=0)
regres.fit(x_train,y_train)



# In[172]:


regres.score(x_train,y_train)


# In[173]:


regres.score(x_test,y_test)


# In[174]:


from sklearn.ensemble import AdaBoostRegressor
regr= AdaBoostRegressor(n_estimators=100, random_state=0)
regr.fit(x_train,y_train)


# In[175]:


from sklearn.decomposition import PCA
model=PCA()


# In[176]:


pca=PCA()
pca.fit(x_train)


# In[178]:


ratio=pca.explained_variance_ratio_


# In[180]:


ratio.shape


# In[167]:


data = pd.read_csv('D:/IMARTICUS WORK/Paper1/MonthWiseMarketArrivals_Clean.csv')


# In[183]:


data.index.freq = 'MS'


# In[185]:


m=12
alpha = 1/(2*m)


# In[189]:


data['HWES1'] = SimpleExpSmoothing(data['priceMax']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
data[['priceMax','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');


# In[ ]:




