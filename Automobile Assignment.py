#!/usr/bin/env python
# coding: utf-8

# In[198]:


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# In[199]:


data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",header=None,na_values="?")


# In[200]:


data.head()


# In[201]:


data.shape


# In[202]:


data.columns=["symbol","normalised_losses","make","fuel_type","aspiration","num_of_doors","body_style","drive_wheels","engine_location","wheel_base","length","width","height","curb_weight","engine_type","num_of_cylinders","engine_size","fuel_system","bore","strole","compress_ratio","hor_pow","peak_rpm","city_mpg","highway_mpg","price"]
data.info()


# In[204]:


data.info()


# In[205]:


data.wheel_base


# In[206]:


data.boxplot()


# In[207]:


data.drop("length", axis=1)


# In[208]:


data.isnull().sum()


# In[209]:


data.price.describe()


# In[210]:


data.price.value_counts()


# In[211]:


data.hist()


# In[212]:


data.describe()


# In[213]:


data.value_counts()


# In[214]:


sns.countplot(data.symbol)


# In[215]:


data.symbol.replace([-2,-1,0,1,2,3],['less than zero','less than zero','zero','1','2','3'],inplace=True)


# In[216]:


sns.countplot(data.symbol)
sns.countplot(data.symbol,order=['less than zero','zero','1','2','3'])


# In[217]:


# Column2:Normalised Losses
data.normalised_losses.isnull().sum()


# In[218]:


data.normalised_losses.describe()


# In[219]:


data.normalised_losses.fillna(122.0,inplace=True)


# In[220]:


data.normalised_losses.isnull().sum()


# In[221]:


plt.boxplot(data.normalised_losses)


# In[222]:


data.normalised_losses.unique()


# In[223]:


plt.hist(data.normalised_losses)


# In[224]:


plt.figure(figsize=(30,20))
sns.countplot(data.normalised_losses)


# In[225]:


#column:Make
data.make.describe()


# In[226]:


data.make.value_counts()


# In[227]:


data.make.isnull().sum()


# In[228]:


data.make.unique()


# In[229]:


plt.figure(figsize=(40,30))
sns.countplot(data.make)


# In[230]:


data.fuel_type.describe()


# In[231]:


data.symbol.value_counts()


# In[232]:


#Column4:Fuel types
data.fuel_type.describe()


# In[233]:


data.fuel_type.value_counts()


# In[234]:


data.fuel_type.isnull().sum()


# In[235]:


data.isnull().sum()


# In[236]:


data.fuel_type.unique()


# In[237]:


sns.countplot(data.fuel_type)


# In[238]:


#Column5:Aspiration
data.aspiration.describe()


# In[239]:


data.aspiration.value_counts()


# In[240]:


data.aspiration.isnull().sum()


# In[241]:


data.aspiration.unique()


# In[242]:


sns.countplot(data.aspiration)


# In[243]:


#Column6:No Of Doors
data.num_of_doors.describe()


# In[244]:


data.num_of_doors.value_counts()


# In[245]:


data.num_of_doors.isnull().sum()


# In[246]:


data.num_of_doors.fillna('four',inplace=True)


# In[247]:


data.num_of_doors.isnull().sum()


# In[248]:


data.num_of_doors.unique()


# In[249]:


sns.countplot(data.num_of_doors)


# In[250]:


#Column7:Body Style
data.body_style.describe()


# In[251]:


data.body_style.value_counts()


# In[252]:


data.body_style.isnull().sum()


# In[253]:


data.body_style.unique()


# In[254]:


sns.countplot(data.body_style)


# In[255]:


# Column8:Drive Wheels
data.drive_wheels.describe()


# In[256]:


data.drive_wheels.value_counts()


# In[257]:


data.drive_wheels.isnull().sum()


# In[258]:


data.drive_wheels.unique()


# In[259]:


sns.countplot(data.drive_wheels)


# In[260]:


data.drive_wheels.replace(['4wd','rwd'],['not fwd','not fwd'],inplace=True)


# In[261]:


sns.countplot(data.drive_wheels)


# In[262]:


# Column9:Engine Location
data.engine_location.describe()


# In[263]:


data.engine_location.value_counts()


# In[264]:


data.engine_location.isnull().sum()


# In[265]:


data.engine_location.unique()


# In[266]:


sns.countplot(data.engine_location)


# In[267]:


# Column10:Wheel Base
data.wheel_base.describe()


# In[268]:


data.wheel_base.isnull().sum()


# In[269]:


data.wheel_base.unique()


# In[270]:


plt.boxplot(data.wheel_base)


# In[271]:


stats.shapiro(data.wheel_base)
#The data is not normally distributed since the p value is less than alpha value


# In[272]:


plt.hist(data.wheel_base)


# In[273]:


#Column11:Length
data.length.describe()


# In[274]:


data.length.isnull().sum()


# In[275]:


data.length.unique()


# In[276]:


plt.boxplot(data.length)


# In[277]:


plt.hist(data.length)


# In[81]:


stats.shapiro(data.length)
# The data is not normally distributed since the p value is less than alpha value


# In[82]:


# Column12:Width
data.width.describe()


# In[83]:


data.width.isnull().sum()


# In[84]:


data.width.unique()


# In[85]:


plt.boxplot(data.width)


# In[86]:


plt.hist(data.width)


# In[87]:


stats.shapiro(data.width)
# The data is not normally distributed since the p value is less than the aplha value.


# In[88]:


# Column13:Height
data.height.describe()


# In[89]:


data.height.isnull().sum()


# In[90]:


data.height.unique()


# In[91]:


plt.boxplot(data.height)


# In[92]:


plt.hist(data.height)


# In[93]:


stats.shapiro(data.height)
# The data is not normally distributed since the p value is less than the alpha level.


# In[94]:


# Column14:Curb Weight
data.curb_weight.describe()


# In[95]:


data.curb_weight.isnull().sum()


# In[96]:


data.curb_weight.unique()


# In[97]:


plt.boxplot(data.curb_weight)


# In[98]:


plt.hist(data.curb_weight)


# In[99]:


stats.shapiro(data.curb_weight)
# The data is not normally distributed since the p value is less than the alpha value.


# In[100]:


# Column15:Engine Type
data.engine_type.describe()


# In[101]:


data.engine_type.value_counts()


# In[102]:


data.engine_type.isnull().sum()


# In[103]:


data.engine_type.unique()


# In[104]:


sns.countplot(data.engine_type)


# In[105]:


# Column16:Number Of Cylinders
data.num_of_cylinders.describe()


# In[106]:


data.num_of_cylinders.value_counts()


# In[107]:


data.num_of_cylinders.isnull().sum()


# In[108]:


data.num_of_cylinders.unique()


# In[109]:


sns.countplot(data.num_of_cylinders)
data.num_of_cylinders.replace(['three','two','five','four','six','eight','twelve'],['less than five','less than five','five','four','six','>=eight','>=eight'],inplace=True)


# In[110]:


sns.countplot(data.num_of_cylinders)


# In[111]:


# Column17:Engine Size
data.engine_size.describe()


# In[112]:


data.engine_size.value_counts()


# In[113]:


data.engine_size.isnull().sum()


# In[114]:


data.engine_size.unique()


# In[115]:


plt.boxplot(data.engine_size)


# In[116]:


plt.hist(data.engine_size)


# In[117]:


stats.shapiro(data.engine_size)
# The data is not distributed normally since the p value is less than the aplha value.


# In[118]:


# Column18:Fuel System
data.fuel_system.describe()


# In[119]:


data.fuel_system.isnull().sum()


# In[120]:


data.fuel_system.unique()


# In[121]:


sns.countplot(data.fuel_system)


# In[122]:


# Column19:Bore
data.bore.describe()


# In[123]:


data.bore.isnull().sum()


# In[124]:


data.bore.fillna(3.329751,inplace=True)


# In[125]:


data.bore.isnull().sum()


# In[126]:


data.bore.unique()


# In[127]:


plt.boxplot(data.bore)


# In[128]:


plt.hist(data.bore)


# In[129]:


stats.shapiro(data.bore)
# The data is not normally distributed since the pvalue is less than aplha value.


# In[130]:


# Column20:Stroke
data.strole.describe()


# In[131]:


data.strole.isnull().sum()


# In[132]:


data.strole.fillna(3.255423,inplace=True)


# In[133]:


data.strole.isnull().sum()


# In[134]:


data.strole.unique()


# In[135]:


plt.boxplot(data.strole)


# In[136]:


plt.hist(data.strole)


# In[137]:


stats.shapiro(data.strole)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[138]:


# Column21:Compression Ratio
data.compress_ratio.describe()


# In[139]:


data.compress_ratio.isnull().sum()


# In[140]:


data.compress_ratio.unique()


# In[141]:


data.compress_ratio.fillna(10.142537,inplace=True)


# In[142]:


data.compress_ratio.isnull().sum()


# In[143]:


plt.boxplot(data.compress_ratio)


# In[144]:


plt.hist(data.compress_ratio)


# In[145]:


stats.shapiro(data.compress_ratio)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[146]:


# Column22:Horse Power
data.hor_pow.describe()


# In[147]:


data.hor_pow.isnull().sum()


# In[148]:


data.hor_pow.fillna(104.256158,inplace=True)


# In[149]:


data.hor_pow.isnull().sum()


# In[150]:


data.hor_pow.unique()


# In[151]:


plt.boxplot(data.hor_pow)


# In[152]:


plt.hist(data.hor_pow)


# In[153]:


stats.shapiro(data.hor_pow)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[154]:


# Column23:Peak RPM
data.peak_rpm.describe()


# In[155]:


data.peak_rpm.isnull().sum()


# In[156]:


data.peak_rpm.fillna(5125.369458,inplace=True)


# In[157]:


data.peak_rpm.isnull().sum()


# In[158]:


data.peak_rpm.unique()


# In[159]:


plt.boxplot(data.peak_rpm)


# In[160]:


plt.hist(data.peak_rpm)


# In[161]:


stats.shapiro(data.peak_rpm)
# The data is not normally distributed since the pvalue is less than aplha value.


# In[162]:


# Column24:City MPG
data.city_mpg.describe()


# In[163]:


data.city_mpg.isnull().sum()


# In[164]:


data.city_mpg.unique()


# In[165]:


plt.boxplot(data.city_mpg)


# In[166]:


plt.hist(data.city_mpg)


# In[167]:


stats.shapiro(data.city_mpg)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[168]:


# Column25:Highway MPG
data.highway_mpg.describe()


# In[169]:


data.highway_mpg.isnull().sum()


# In[170]:


data.highway_mpg.unique()


# In[171]:


plt.boxplot(data.highway_mpg)


# In[172]:


plt.hist(data.highway_mpg)


# In[173]:


stats.shapiro(data.highway_mpg)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[174]:


# Column26:Price
data.price.describe()


# In[175]:


data.price.isnull().sum()


# In[176]:


data.price.isnull().sum()


# In[177]:


data.price.unique()


# In[197]:


plt.boxplot(data.price)


# In[195]:


plt.hist(data.price)


# In[194]:


stats.shapiro(data.price)
# The data is not normally distributed since the pvalue is less than alpha value.


# In[ ]:




