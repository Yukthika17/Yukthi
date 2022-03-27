#!/usr/bin/env python
# coding: utf-8

# In[1]:


#calculate z-score for Dataframe in python
import numpy as np
from scipy.stats import zscore
import pandas as pd 
# Using seed function to generate the same random number every time with the given seed value 
np.random.seed(4)
#create a random 5*5 dataframe using pandas module
data = pd.DataFrame(np.random.randint(0, 10, size=(5, 5)), columns=['A', 'B', 'C','D','E'])
#calculate z-score of the above dataframe 
result = zscore(data)
#Print the dataframe
print("Created DataFrame is: \n",data)
#Print the Calculate Z-score
print("Z-score array: \n",result)


# In[ ]:




