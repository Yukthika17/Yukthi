#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data=pd.read_excel("D:\IMARTICUS WORK\Sample hyt wyt.xlsx")
data.head()
data.describe()
data.hist()


# In[3]:


from scipy import stats
stats.shapiro(data)


# In[8]:


from scipy import stats
stats.shapiro(data.WEIGHT)


# In[9]:


data.boxplot()


# In[10]:


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.boxplot(data.HEIGHT)
plt.subplot(1,2,2)
plt.boxplot(data.WEIGHT)


# In[11]:


train=data.iloc[:6]


# In[12]:


data.iloc[:6]


# In[13]:


test=data.iloc[6:]

data.iloc[6:]


# In[14]:


import seaborn as sns
sns.scatterplot(data=data,x="HEIGHT",y="WEIGHT")


# In[15]:


from scipy import stats   
stats.pearsonr(data.HEIGHT,data.WEIGHT)


# In[16]:


import statsmodels.api as sm
train_x=train.HEIGHT
train_y=train.WEIGHT
train_x=sm.add_constant(train_x)


# In[17]:


model=sm.OLS(train_y,train_x).fit()
model.summary()
model.params
model.predict(train_x)
test_x=test.HEIGHT
test_y=test.WEIGHT
test_x=sm.add_constant(test_x)
pre_test=model.predict(test_x)
model.predict(test_x)


# In[18]:


import seaborn as sns
sns.scatterplot(data=train,y="HEIGHT",x="WEIGHT")


# In[19]:


import seaborn as sns
sns.scatterplot(data=test,y="HEIGHT",x="WEIGHT")


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(train.HEIGHT,train.WEIGHT,color="blue")
plt.scatter(test.HEIGHT,test.WEIGHT,color="orange")
plt.plot(test.HEIGHT,pre_test, color="red")
plt.show()


# In[ ]:




