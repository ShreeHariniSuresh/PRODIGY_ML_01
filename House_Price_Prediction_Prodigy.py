#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 


# In[2]:


df = pd.read_csv("train.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


# Handling missing data


# In[7]:


# filling missing value
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[8]:


df.drop(['Alley','GarageYrBlt','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


sns.heatmap(df.isnull(),cbar= False)


# In[12]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[13]:


sns.heatmap(df.isnull(),cbar= False)


# In[14]:


df.dropna(inplace=True)


# In[15]:


df.shape


# In[16]:


df.head()


# In[17]:


columns=['MSZoning','Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2','BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition','ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']


# In[18]:


len(columns)


# In[19]:


# handling categorical features


# In[20]:


def category_onehot_multcols(multcolumns):
    df_final = final_df
    i=0
    
    for fields in multcolumns:
        print(fields)
        df_1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        
        if i==0:
            df_final=df_1.copy()
        else: 
            df_final=pd.concat([df_final,df_1],axis=1)
            
        i=i+1
    
    df_final=pd.concat([final_df,df_final],axis=1)
    
    return df_final


# In[21]:


main_df=df.copy()


# In[22]:


# combine test data

test_df=pd.read_csv('test.csv')
test_df.shape


# In[23]:


test_df.head()


# In[24]:


# handling missing data in test.csv


# In[25]:


# filling missing value

test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])
test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])


# In[26]:


test_df.loc[:,test_df.isnull().any()].head()


# In[27]:


test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])


# In[28]:


test_df.shape


# In[29]:


test_df.to_csv('test.csv',index=False)


# In[30]:


test_df.head()


# In[31]:


extracted_col = df["Id"]
print(extracted_col)


# In[32]:


final_df=pd.concat([df,test_df],axis=0)


# In[33]:


final_df.shape


# In[34]:


final_df=category_onehot_multcols(columns)


# In[35]:


final_df.shape


# In[36]:


final_df=final_df.loc[:,~final_df.columns.duplicated()]


# In[37]:


final_df.shape


# In[38]:


final_df.head()


# In[39]:


# feature selection (to see the significance)


# In[40]:


# correlation between output columns and all other columns


# In[41]:


final_df[final_df.columns[1:]].corr()['SalePrice'][:]


# In[42]:


final_df.dropna(inplace=True)


# In[43]:


print("The important features are :\n" )
dfc = final_df[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd'
,'YearBuilt','LotArea','SalePrice']]  
dfc


# In[44]:


plt.figure(figsize = (16,5))
sns.heatmap(dfc.corr(),annot=True,cmap="YlGnBu")


# In[45]:


plt.hist(final_df['SalePrice'],bins=100)
print("Right Skewed Data: More houses with price between 1 million and 3 million ")


# In[46]:


# for outliers
# using box plot


# In[47]:


plt.figure(figsize=(16,5))
sns.boxplot(x='OverallQual',y='SalePrice',data=dfc)


# In[48]:


print(dfc.isnull().sum())


# In[49]:


print("No missing values")


# In[50]:


# Linear Regression


# In[51]:


#dividing the dataset into independent and dependent variable

X=dfc.iloc[:,:-1] 
y=dfc.iloc[:,-1] 


# In[52]:


X


# In[53]:


y


# In[54]:


# splitting the dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[55]:


#feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lin_reg.fit(X_train, y_train)


# In[56]:


# predicting the test set results

y_pred = lin_reg.predict(X_test)


# In[57]:


# Plotting Scatter graph to show the prediction

plt.scatter(y_test, y_pred)
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("Linear Regression = True value vs predicted value")
plt.show()


# In[58]:


import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("New RMSE: ", math.sqrt(mean_squared_error(y_pred, y_test)))


# In[59]:


y_pred.size


# In[60]:


residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual plot")
plt.show()


# In[ ]:




