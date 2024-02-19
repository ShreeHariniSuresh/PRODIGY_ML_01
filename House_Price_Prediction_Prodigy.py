#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 

#importing train file

df = pd.read_csv("train.csv")
df.head()

df.shape

df.info()

df.isnull().sum()

# Handling missing data
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

df.drop(['Alley','GarageYrBlt','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.shape
df.isnull().sum()
sns.heatmap(df.isnull(),cbar= False)
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
sns.heatmap(df.isnull(),cbar= False)
df.dropna(inplace=True)
df.shape
df.head()

columns=['MSZoning','Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2','BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition','ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
len(columns)

# handling categorical features

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

main_df=df.copy()

# combine test data

test_df=pd.read_csv('test.csv')
test_df.shape
test_df.head()

# handling missing data in test.csv
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
test_df.loc[:,test_df.isnull().any()].head()

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
test_df.shape

# importing test file

test_df.to_csv('test.csv',index=False)
test_df.head()
extracted_col = df["Id"]
print(extracted_col)
final_df=pd.concat([df,test_df],axis=0)
final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df=final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df.head()

# feature selection (to see the significance)
# correlation between output columns and all other columns

final_df[final_df.columns[1:]].corr()['SalePrice'][:]

final_df.dropna(inplace=True)

print("The important features are :\n" )
dfc = final_df[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd'
,'YearBuilt','LotArea','SalePrice']]  
dfc
plt.figure(figsize = (16,5))
sns.heatmap(dfc.corr(),annot=True,cmap="YlGnBu")
plt.hist(final_df['SalePrice'],bins=100)
print("Right Skewed Data: More houses with price between 1 million and 3 million ")

# for outliers
# using box plot

plt.figure(figsize=(16,5))
sns.boxplot(x='OverallQual',y='SalePrice',data=dfc)
print(dfc.isnull().sum())
print("No missing values")

# Linear Regression
#dividing the dataset into independent and dependent variable

X=dfc.iloc[:,:-1] 
y=dfc.iloc[:,-1] 
X
y

# splitting the dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lin_reg.fit(X_train, y_train)

# predicting the test set results

y_pred = lin_reg.predict(X_test)

# Plotting Scatter graph to show the prediction

plt.scatter(y_test, y_pred)
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("Linear Regression = True value vs predicted value")
plt.show()

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("New RMSE: ", math.sqrt(mean_squared_error(y_pred, y_test)))
y_pred.size
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual plot")
plt.show()
            
