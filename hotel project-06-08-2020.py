#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[1]:


import pandas as pd
df = pd.read_excel (r'/Users/aoyuchen/Desktop/MGT 599/hotel original.xlsx',sheet_name = "Sheet1")
df.head(5)
print(df.columns)


# In[2]:


df['myDt']=pd.to_datetime(df.year*10000+df.month*100+df.day,format='%Y%m%d') # pd.to_datetime(df[['year','month','day']])

df.head(5)


# In[3]:


import pandas as pd

df['NewDate'] = pd.to_datetime(df.myDt) + pd.to_timedelta(pd.np.ceil(df.stays_in_weekend_nights), unit="D")
df


# In[4]:


import pandas as pd

df['NewDate1'] = pd.to_datetime(df.NewDate) + pd.to_timedelta(pd.np.ceil(df.stays_in_week_nights), unit="D")
df


# In[5]:


import pandas as pd
df['nmonth']= pd.DatetimeIndex(df['NewDate1']).month
df.head()


# In[6]:


df['month'].value_counts()


# In[7]:


cHotel = df.loc[df['hotel'] == 'City Hotel']
rHotel = df.loc[df['hotel'] == 'Resort Hotel']


# In[8]:


print(cHotel)


# In[9]:


print(rHotel)


# In[10]:


cHotelMon = cHotel['month'].value_counts()
print(cHotelMon)


# In[11]:


rHotelMon = rHotel['month'].value_counts()
print(rHotelMon)


‘’‘
# # Random Forest

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


# In[13]:


print(df.dtypes)


# In[28]:


df['stay_in_num_day'] = df['stay_in_num_day'].astype(float)
df['adults'] = df['adults'].astype(float)
df['children'] = df['children'].astype(float)
df['babies'] = df['babies'].astype(float)
df['meal_num'] = df['meal_num'].astype(float)
df['required_car_parking_spaces'] = df['required_car_parking_spaces'].astype(float)
df['adr'] = df['adr'].astype(float)


# In[29]:


X = df[['stay_in_num_day', 'adults', 'children', 'babies','meal_num','required_car_parking_spaces']]
y = df['adr']


# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[31]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[32]:


import numpy as np
X_test = np.nan_to_num(X_test)
X_train = np.nan_to_num(X_train)


# In[33]:


print(np.where(np.isnan(X_train)))


# In[34]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=12345)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[35]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[36]:


# Random Forest

X = df[['stay_in_num_day', 'adults', 'children', 'babies','meal_num','required_car_parking_spaces']]
y = df['adr']


# In[37]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123456)

X_test = np.nan_to_num(X_test)
X_train = np.nan_to_num(X_train)

print(X_train)


# In[38]:


## Debugged, from RandomForetClassifier to RandomForestRegressor as the test data are contiuous

from sklearn.ensemble import RandomForestRegressor 
#from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestRegressor(n_estimators = 100, random_state=123456) 
#model2 = RandomForestClassifier(n_estimators = 100, random_state=123456) 


model1.fit(X_train, y_train)
#model2.fit(X_train, y_train)
prediction_test = model1.predict(X_test)


# In[39]:


# Make predictions on test data
predictions = model1.predict(X_test)
# Performance metrics
errors = abs(predictions - y_test)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))
# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
’‘’
