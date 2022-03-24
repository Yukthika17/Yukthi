# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 10:19:44 2022

@author: DeLL cOrE I5
"""

#PANDAS
import pandas as pd

path= r"C:\Users\DeLL cOrE I5\Desktop\Book1 Xcel Data wheather.xlsx"

data=pd.read_excel(path)

print(data)

data.Chennai.mean()
data.Chennai.mode()
data.Chennai.median()