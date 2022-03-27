# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:22:58 2022

@author: DeLL cOrE I5
"""

#Z-Score can be calculated for the one dim
from scipy.stats import zscore
#create one-dimensional array
data = np.array([7,4,8,9,6,11,16,17,19,12,11])
#calculate z-score
result = zscore(data)ensional array data using below python code.
import numpy as np
#Print the result
print("Z-score array: ",result)


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
