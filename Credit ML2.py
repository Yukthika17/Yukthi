#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as  sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LS
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


credit=pd.read_csv("D:/IMARTICUS WORK/ML Paper2 Sunday/credit_card.csv")


# In[3]:


credit.head()


# In[4]:


credit.info()


# In[5]:


credit.shape


# In[6]:


credit.isnull().sum()


# In[7]:


credit.CREDIT_LIMIT.fillna(0,inplace=True)


# In[8]:


credit.isnull().sum()


# In[9]:


credit.MINIMUM_PAYMENTS.fillna(0,inplace=True)


# In[10]:


credit.isnull().sum()


# In[11]:


credit['zscore'] = ( credit.PURCHASES - credit.PURCHASES.mean() ) / credit.PURCHASES.std()


# In[12]:


credit.head(5)


# In[13]:


credit[credit['zscore']>3]


# In[14]:


credit[credit['zscore']<-3]


# In[15]:


credit_new = credit[(credit.zscore>-3) & (credit.zscore<3)]


# In[16]:


from sklearn import datasets


# In[17]:


cor_matrix = credit.corr().abs()
    


# In[18]:


print(cor_matrix)


# In[19]:


upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))


# In[20]:


print(upper_tri)


# In[21]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]


# In[22]:


print(); print(to_drop)


# In[36]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# In[57]:


from sklearn.cluster import KMeans


# In[58]:


plt.style.use('ggplot')


# In[66]:


credit.boxplot()


# In[67]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[69]:


from sklearn.model_selection import train_test_split
train=credit.iloc[:132]
test=credit.iloc[132:]


# In[70]:


trainlast=train.tail(1)


# In[71]:


print(trainlast)


# In[74]:


train["navie"]=405


# In[75]:


test["navie"]=405


# In[76]:


from sklearn.metrics import mean_squared_error
rmse1=mean_squared_error(test.PURCHASES,test.navie,squared=False)
rmse2=mean_squared_error(test.PURCHASES,test.navie,squared=False)
print(rmse1,rmse2)


# In[78]:


shift=credit.shift(1)


# In[79]:


x=credit.iloc[:,2:3].values
y=credit.iloc[:,3:4].values


# In[80]:


from sklearn import datasets
from sklearn.model_selection import cross_val_score 


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)


# In[83]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[84]:


#predict the test result
y_pred=regressor.predict(x_test)


# In[85]:


plt.scatter(x_train,y_train,c='red')
plt.show()


# In[87]:


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[89]:


# calculate acf
acf_values = acf(credit['PURCHASES'])


# In[91]:


credit.plot(figsize=(10,5))


# In[93]:


plot_acf(credit['PURCHASES'], lags=20);


# In[95]:


# PACF
pacf_values = (credit['PURCHASES'])


# In[96]:


plot_pacf(credit['PURCHASES'], lags=30)


# In[ ]:




