#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# In[39]:


data=pd.read_csv("C:/Users/DeLL cOrE I5/Downloads/ML Paper2 Sunday/Sentiment.csv")


# In[40]:


data.head()


# In[41]:


data.columns


# In[42]:


data.shape


# In[43]:


data.columns=['id', 'candidate', 'candidate_confidence', 'relevant_yn',
       'relevant_yn_confidence', 'sentiment', 'sentiment_confidence',
       'subject_matter', 'subject_matter_confidence', 'candidate_gold', 'name',
       'relevant_yn_gold', 'retweet_count', 'sentiment_gold',
       'subject_matter_gold', 'text', 'tweet_coord', 'tweet_created',
       'tweet_id', 'tweet_location', 'user_timezone']


# In[44]:


data.info()


# In[45]:


data.drop('retweet_count', axis=1)


# In[46]:


data.hist()


# In[47]:


data.describe()


# In[48]:


data.value_counts()


# In[49]:


plt.plot(data.index)


# In[50]:


data.boxplot()


# In[51]:


data = data[['text','sentiment']]


# In[52]:


import re


# In[53]:


data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[54]:


for idx,row in data.iterrows():
    row[0] = row[0].replace('re',' ')
    


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


from keras.datasets import mnist
from keras.models import Sequential


# In[ ]:


from keras.preprocessing.text import candidate_confidence
from keras.preprocessing.sequence import relevant_yn_confidence


# In[ ]:


max_fatures = 2000
candidate_confidence = candidate_confidence(num_words=max_fatures, split=' ')
candidate_confidence.fit_on_texts(data['text'].values)
X = candidate_confidence.texts_to_sequences(data['text'].values)
X = relevant_yn_confidence(X)


# In[ ]:




