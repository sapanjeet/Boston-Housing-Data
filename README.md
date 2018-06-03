# Boston-Housing-Data
Prediction of Boston Housing Prices using Advanced Regression in Python
# coding: utf-8

# Dataset has large number of features, approxiamtely 80 variables, we will proceed with following steps following SEMMA approach. SEMMA - Sample, Explore, Modify, Model and Assess. We can neglect Sample option here as the data is not that huge which needs sampling. Explore: We will do Multivariate Data Analysis to derive some potential inferences and insights. Modify: If required we will try modifying or transforming some variables to focus on the model selection process. Model: We will Model the data to search for a combination of features that reliably predicts a desired outcome. Assess: We will finally evaluate our model from the data mining process and estimate how well it performs. Let's import the datasets now.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.ticker as ticker


# In[2]:


train = pd.read_csv('train_house.csv')


# In[14]:


train.head(3)


# > Missing Data****

# In[15]:


#Let's check the missing values.Yellow ones are the missing columns
ax = plt.figure(figsize = (20,10))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


#We can even check the percentage of data missing
Total = train.isnull().sum().sort_values(ascending=False)
Percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
Missing_data = pd.concat([Total, Percent], axis=1, keys=['Total', 'Percent'])
Missing_data.head(5)


# In[17]:


#Seems like top 5 columns have more than 80% of the data missing, we can definitely remove them.
train.drop('Fence',axis=1,inplace=True)
train.drop('FireplaceQu',axis=1,inplace=True)
train.drop('PoolQC',axis=1,inplace=True)
train.drop('Alley',axis=1,inplace=True)
train.drop('MiscFeature',axis=1,inplace=True)


# In[18]:


#Let's view the heatmap now
#Yellow ones are the missing columns
ax = plt.figure(figsize = (20,10))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# There are many more columns which contains missing values, we will replace them accordingly with their mean values or zero.

# > **Correlation

# Let's check the correlation between any variables and remove them accordingly

# In[19]:


ax = plt.figure(figsize = (20,14))
sns.heatmap(train.corr(),cmap='plasma')


# From the heat map we can see that, we should check below variables for collinearity:
# 
# YearBuilt : GarageYrBlt
# 
# GarageCars : GarageArea
# 
# GrLivArea : TotRmsAbvGrd
# 
# But before that, since there are too many variables and we still can see lot of boxes in the heatmap close to yellow color. We can try reducing unimportant variables by visualizing them w.r.t the target variable

# **Target Variable Analysis

# In[9]:


sns.distplot(train['SalePrice'],bins = 50)


# In[10]:


train['SalePrice'].describe()


# In[9]:


train['SalePrice'].skew()


# In[10]:


train['SalePrice'].kurtosis()


# By all the above measures, we can say that data is slightly right skewed and and is not close to normal distribution. 

# ** Data Visualization

# Let's visualize each numerical columns and then each categorical columns to see how they impact SalePrice. Below each viz I have written some interpretation, for those I have not written, that means it was not giving any interesting insight.

# SalePrice Vs 1stFlrSF

# In[11]:


sns.jointplot(x='1stFlrSF', y = 'SalePrice',data=train, kind= 'reside')


# 1st Floor square feet is mostly concentrated in less then 2000 square feet.And we can also see that high square feet does not guarantee high sale price

# SalePrice Vs 2ndFlrSF

# In[14]:


sns.jointplot(x='2ndFlrSF', y = 'SalePrice',data=train, kind= 'reside')


# Lot many values are at 0, which means they must be missing, we will replace them with mean values which seems to be a better replacement.Here also, high prices does not mean large area on 2nd floor, under 1500 squarefeet , we can see pretty high sale prices. However, there are 4-5 exceptions which are more then 1500sq feet and have high SalePrices.We will consider that as of now and see how that helps in future analysis.

# SalePrice Vs ScreenPorch

# In[15]:


sns.jointplot(x='ScreenPorch', y = 'SalePrice',data=train, kind= 'reside')


# Seems like there are lot of missing values for ScreenPorch Area too and except few, most datapoints do not contribute to high SalePrice, infact it's basically low or negative for most of the data points.

# SalePrice Vs BedroomAbvGr

# In[16]:


sns.boxplot(x='BedroomAbvGr', y = 'SalePrice',data=train, palette = 'coolwarm')


# We can see that most of the bedrooms above grade between 2 to 4 , inclusive of both have all range of SalePrices. There are few high SalePrices case in 4 bedroom and few in 2 bedroom . However, bedrooms above grade consisting of 5 to 8 bedrroms have SalePrices mostly in negative.

# SalePrice Vs BsmtFinSF1

# In[17]:


sns.jointplot(x='BsmtFinSF1', y = 'SalePrice',data=train, kind= 'reside')


# Square feet for Basement doesn't make much difference,. We have both negative and positive SalePrices within 2000 square feet of basement area.

# SalePrice Vs BsmtFinSF2

# In[18]:


sns.jointplot(x='BsmtFinSF2', y = 'SalePrice',data=train, kind= 'reside')


# Even Type Basement Square Feet does not give clear values related to SalePrice, most of them are negative.We also see lot of values missing.

# In[19]:


sns.boxplot(x='BsmtFullBath', y = 'SalePrice',data=train,palette = 'coolwarm')


# In[20]:


sns.boxplot(x='BsmtHalfBath', y = 'SalePrice',data=train, palette = 'coolwarm')


# Most of the values are missing and those have 1 Basement half bathroom does not have much interesting SalePrice inferences. Even for 2 Basement half bathroom , prices are not high.

# In[21]:


sns.jointplot(x='BsmtUnfSF', y = 'SalePrice',data=train, kind= 'reside')


# For unfinished sqaure feet area in basement, we can see that most of the data points have low prices and few of them have very high, still nothing concrete but could be an interesting factor to explore.

# In[23]:


sns.jointplot(x='EnclosedPorch', y = 'SalePrice',data=train, kind= 'reside')


# In[24]:


sns.violinplot(x='Fireplaces', y = 'SalePrice',data=train, palette = 'rainbow')


# In[25]:


sns.swarmplot(x='FullBath', y = 'SalePrice',data=train)


# In[26]:


sns.jointplot(x='GarageArea', y = 'SalePrice',data=train, kind= 'reside')


# In[27]:


plt.figure(figsize = (20,15))
sns.boxplot(x='GarageYrBlt', y = 'SalePrice',data=train, palette="coolwarm")


# Although years are not that clear, we can clearly see that as the GarageYrBuilt increases , salePrice increases. Recent one built have high prices.

# In[28]:


sns.swarmplot(x='HalfBath', y = 'SalePrice',data=train)


# Most of them at zero, having 1 HalfBath doesn't show high price. Even 2 half bath room does not show high SalePrices.

# In[29]:


sns.swarmplot(x='KitchenAbvGr', y = 'SalePrice',data=train)


# Having 1 Kitchen Above grade is good enough to have good salePrices for the customers. 2 or 3 does not have good Saleprices

# In[30]:


sns.jointplot(x='LotFrontage', y = 'SalePrice',data=train, kind= 'reside')


# LotFrontage meaning linear feet of street connected to property do seem to have a good area of 50-100. More than that don't fetch high salePrices for the houses.

# In[31]:


sns.jointplot(x='LotArea', y = 'SalePrice',data=train, kind= 'reside')


# In[32]:


sns.jointplot(x='LowQualFinSF', y = 'SalePrice',data=train, kind= 'reside')


# Most of the values are missing, and rest available doesn't show much difference.

# In[33]:


plt.figure(figsize = (20,15))
sns.boxplot(x='MSSubClass', y = 'SalePrice',data=train, palette = 'coolwarm')


# Seems like MSSubclass which means the type of dwelling involved in the sale shows a good price value(more than average) for 60 , 20 and 120 which represents below styles: 60 - 2-STORY 1946 & NEWER 20 - 1-STORY 1946 & NEWER ALL STYLES 120 - 1-STORY PUD (Planned Unit Development) - 1946 & NEWER

# In[34]:


sns.jointplot(x='MasVnrArea', y = 'SalePrice',data=train, kind= 'reside')


# MasVnrArea i.e. Veneer area doesn't seem to make much difference.

# In[35]:


sns.jointplot(x='MiscVal', y = 'SalePrice',data=train, kind= 'reside')


# Most of the values are missing and rest are not making much difference.

# In[36]:


plt.figure(figsize = (20,10))
sns.swarmplot(x='MoSold', y = 'SalePrice',data=train, palette = 'viridis')


# Considering MonthSold, it's not showing much difference, however one sold in June and July have high salePrices comparatively.

# In[37]:


sns.jointplot(x='OpenPorchSF', y = 'SalePrice',data=train, kind= 'reside')


# Not showing much difference

# In[38]:


plt.figure(figsize = (20,10))
sns.boxplot(x='OverallCond', y = 'SalePrice',data=train, palette = 'viridis')


# OverallCond below 5 have low SalePrices . Overallcond of 5 which is average house rating have reasonable sale prices. Also, OverallCond of 9 which is excellent rating for House condition have high salePrices.

# In[39]:


plt.figure(figsize = (20,10))
sns.boxplot(x='OverallQual', y = 'SalePrice',data=train, palette ='plasma')


# Needless to say anything, we can clearly say that High rating for OverallQual means High SalePrices. Could be a significant factor.

# In[40]:


sns.jointplot(x='PoolArea', y = 'SalePrice',data=train, kind= 'reside')


# In[41]:


plt.figure(figsize = (20,10))
sns.boxplot(x='TotRmsAbvGrd', y = 'SalePrice',data=train)


# Could be a significant factor in our model

# In[42]:


sns.jointplot(x='TotalBsmtSF', y = 'SalePrice',data=train, kind= 'reside')


# In[43]:


sns.jointplot(x='WoodDeckSF', y = 'SalePrice',data=train, kind= 'reside')


# In[44]:


plt.figure(figsize = (30,10))
sns.factorplot(x='YearBuilt', y = 'SalePrice',data=train, palette = 'plasma')


# High YearBuilt means high SalePrices which means newly built houses have more prices. However ther's one blue line in the initial years which is high.

# In[45]:


ax = sns.pointplot(x='YearRemodAdd', y = 'SalePrice',data=train, kind= 'reside')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())


# As the YearRemodAdd increases that is recent remodel date leads to high SalePrices.

# In[46]:


sns.boxplot(x='YrSold', y = 'SalePrice',data=train,palette = 'coolwarm')


# YearSold is not making much difference in SalePrice

# After considering all the numeric columns, the variables which seems to be making difference in the SalePrices are: 
# 
# BsmtUnfSF
# 
# FullBath
# 
# GarageYrBlt
# 
# YearRemodAdd
# 
# YearBuilt
# 
# TotRmsAbvGrd
# 
# OverallQual
# 
# OverallCond
# 
# MoSold
# 
# MSSubClass
# 
# LotFrontage
# 
# KitchenAbvGr
# 
# HalfBath
# 
# BedroomAbvGr
# 
# 14 variables

# Let's now consider all the categorical columns

# In[52]:


non_numerics = [x for x in train.columns                 if not (train[x].dtype == np.float64                         or train[x].dtype == np.int64)]


# In[57]:


categorical = [x for x in train.columns if x in non_numerics]
categorical # list of categorical columns


# Let's evaluate all the categorical columns and see how we can encode if required and visualize them

# In[49]:


sns.boxplot(x= 'MSZoning', y = 'SalePrice', data = train)


# MSZoning basically means zoning classifcation of the sale. So RL is Residential Low Density, RM is Residential Medium Density C is Commercial FV is Floating Village Residential and RH is Residential High Density. Seems like Residential Low Density and and Floating Village Residential has high prices.

# In[50]:


sns.swarmplot(x= 'Street', y = 'SalePrice', data = train)


# We can see that Street type ,most of them which are Paved have high SalePrices also, For Gravel data points are not available

# In[51]:


sns.swarmplot(x= 'LotShape', y = 'SalePrice', data = train)


# Most of the data for LotShape i.e. Shape of property is for RegularShape and IR1 that is slightly irregular shape. SalePrices are high for IR2 which is moderately irregular although data points are less.

# In[52]:


sns.boxplot(x= 'LandContour', y = 'SalePrice', data = train)


# Except Bnk which is Banked- Quick and significant rise from street grade to building , rest all that is Lvl - Near Flat/Level Low - Depression HLS - Hillside - Significant slope from side to side have high prices.

# In[53]:


sns.boxplot(x= 'Utilities', y = 'SalePrice', data = train)


# Clearly all the data is available for AllPub which is All Public Utilities.

# In[54]:


sns.boxplot(x= 'LotConfig', y = 'SalePrice', data = train)


# In[55]:


sns.boxplot(x= 'LandSlope', y = 'SalePrice', data = train)


# In[56]:


plt.figure(figsize = (20,10))
sns.boxplot(x= 'Neighborhood', y = 'SalePrice', data = train, palette = 'coolwarm')


# Prices are high for NridgHt(Northridge Heights) , StoneBr(Stone Brook) and NoRidge(Northridge) Area and low for IDOTRR(Iowa DOT and Rail Road), MeadowV(Meadow Village) and BrDale (Briardale)

# In[57]:


sns.swarmplot(x= 'BldgType', y = 'SalePrice', data = train)


# Clearly the prices are high for 1Fam building Type which is Single-family Detached. It's also high for TwnhsE which Townhouse End Unit

# In[58]:


sns.swarmplot(x= 'Condition2', y = 'SalePrice', data = train)


# That's why swarmplot is good,we can see number of data points. For Norm we can see good data points, for others very less amount of data is available which is not good for deriving insights.

# In[59]:


sns.boxplot(x= 'HouseStyle', y = 'SalePrice', data = train)


# Prices are high for 2Story and 2.5Fin(2 and a half story) and low for 1.5 Unf(One and a half story)

# In[60]:


sns.boxplot(x= 'Condition1', y = 'SalePrice', data = train)


# In[61]:


sns.boxplot(x= 'RoofStyle', y = 'SalePrice', data = train)


# Good Price for Hip Roof style.

# In[62]:


plt.figure(figsize = (16,8))
sns.boxplot(x= 'RoofMatl', y = 'SalePrice', data = train)


# Prices are good for WdShngl which is Wood shingles and Tar&Grv(Gravel and Tar)

# In[63]:


plt.figure(figsize = (16,8))
sns.boxplot(x= 'Exterior1st', y = 'SalePrice', data = train)


# Prices are good for CementBd and VinylSd

# In[64]:


plt.figure(figsize = (16,8))
sns.boxplot(x= 'Exterior2nd', y = 'SalePrice', data = train)


# For exterior 2nd also we observe same rate, high for CmentBd, VinylSd and ImStucc

# In[65]:


sns.boxplot(x= 'MasVnrType', y = 'SalePrice', data = train)


# Very clear, prices are low for BrkCmn

# In[66]:


sns.boxplot(x= 'ExterQual', y = 'SalePrice', data = train)


# Prices are very low for Fa ExterQual which is Quality of the material on the exterior. Low for Fa - Fair, High for Ex - Excellent and Good for Gd- Good and average for TA which is Average/Typical

# In[67]:


sns.boxplot(x= 'ExterCond', y = 'SalePrice', data = train)


# ExterCond is Evaluates the present condition of the material on the exterior which is evident from the SalePrices

# In[68]:


sns.boxplot(x= 'Foundation', y = 'SalePrice', data = train)


# For Foundation type PConc(Poured Concrete) and Wood , prices are relatively high, compared to Slab and CBlock which are low

# In[69]:


sns.boxplot(x= 'BsmtQual', y = 'SalePrice', data = train)


# For BsmtQual which is basically height of the basement in inches, Prices are high for Ex - Excellent which is 100 inches in height, followed by Gd(90-99 inches), TA(80-89) and Fair(70-79 inches)

# In[70]:


sns.boxplot(x= 'BsmtCond', y = 'SalePrice', data = train)


# For BsmtCond, it is again obvious with the Condition rating, prices are justified.

# In[71]:


sns.boxplot(x= 'BsmtExposure', y = 'SalePrice', data = train)


# In[72]:


sns.boxplot(x= 'BsmtFinType1', y = 'SalePrice', data = train)


# In[73]:


sns.boxplot(x= 'BsmtFinType2', y = 'SalePrice', data = train)


# In[74]:


sns.boxplot(x= 'Heating', y = 'SalePrice', data = train)


# For Heating, we can see that it's good for GasA(Gas forced warm air furnace), followed by GasW(Gas hot water or steam heat) and then Grav(Gravity furnace)

# In[75]:


sns.boxplot(x= 'HeatingQC', y = 'SalePrice', data = train)


# It's clear that, HeatingQC are very deterministic when compared to SalePrices

# In[76]:


sns.boxplot(x= 'CentralAir', y = 'SalePrice', data = train)


# CentralAir costs much , true

# In[77]:


sns.boxplot(x= 'Electrical', y = 'SalePrice', data = train)


# For Electrical System, SBrke(Standard Circuit Breakers & Romex) have high prices.

# In[78]:


sns.boxplot(x= 'KitchenQual', y = 'SalePrice', data = train)


# In[79]:


sns.boxplot(x= 'Functional', y = 'SalePrice', data = train)


# For Home Functionality, Price range is similar except Maj2(major deductions) which has low prices

# In[80]:


sns.boxplot(x= 'GarageType', y = 'SalePrice', data = train)


# BuiltIn(Garage part of house - typically has room above garage) have high prices followed by Attchd(attached to home)

# In[81]:


sns.boxplot(x= 'GarageFinish', y = 'SalePrice', data = train)


# In[82]:


sns.boxplot(x= 'GarageQual', y = 'SalePrice', data = train)


# GarageQual is not making much difference, people even consider Average and Good GarageQuality as high saleprices

# In[83]:


sns.boxplot(x= 'GarageCond', y = 'SalePrice', data = train)


# Same with Garage Condition

# In[84]:


sns.boxplot(x= 'PavedDrive', y = 'SalePrice', data = train)


# In[85]:


sns.boxplot(x= 'SaleType', y = 'SalePrice', data = train)


# High for SaleType New(Home just constructed and sold), Con(Contract) and CWD(Warranty Deed - Cash)

# In[86]:


sns.boxplot(x= 'SaleCondition', y = 'SalePrice', data = train)


# High prices for Partial SaleCondition which means Home was not completed when last assessed (associated with New Homes) Followed by Alloca(Allocation - two linked properties with separate deeds, typically condo with a garage unit) and Normal.

# Let's now consider those categorical columns which seems to be connected with SalePrice : SaleCondition SaleType PavedDrive GarageFinish GarageType Functional KitchenQual Electrical CentralAir Heating HeatingQC BsmtExposure BsmtCond BsmtQual Foundation ExterCond ExterQual Exterior2nd Exterior1st RoofMatl HouseStyle BldgType Utilities Street Neighbourhood MSZoning 26 variables, we also have to convert them into numerical values by encoding them, we will do that next.

# In[20]:


#Replacing them with zero
train1 = train.copy()
#train = train.fillna(value='0', axis=1)


# **Feature Preprocessing and Normalization

# We will try to standardize our target variable and bring it to a uniform scale because that will help in predicting better model fit.We can also apply scalar object to training and test dataset but train the scalar object on the training dataset and not on the test data. If we don't apply the same scaling to training and test sets, you'll end up with more or less random data skew, which will invalidate your results. If we prepare the scaler or other normalization method by showing it the test data instead of the training data, this leads to a phenomenon called Data Leakage, where the training phase has information that is leaked from the test set.
# In [87]:
# 

# Let's concatenate training and test dataset first'

# In[15]:


train.columns


# In[16]:


test.columns


# In[21]:


train.head()


# In[3]:


#Let's consider only numeric columns for now
numcols = train.select_dtypes(include = ['number']).columns
train_1 = train[numcols]


# In[4]:


train_1.head()


# In[5]:


#Fill the NA with mean
train_1 = train_1.fillna(train_1.median())


# In[6]:


test_1 = pd.read_csv('test_house.csv')


# In[57]:


test_2 = test_1.copy()
test_2 = pd.get_dummies(test_2)


# In[58]:


test_2 = test_2.fillna(test_2.median())


# In[59]:


#Some variables exist in train but not in test
for col in numcols:
    if col not in test_2:
        test_2[col] = 0


# In[60]:


test_2 = test_2[numcols]


# # Modelling XG Boost

# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[15]:


train_1.shape


# In[17]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 37, oob_score=True, random_state=1234)
rfr.fit(train_1.drop('SalePrice', axis = 1), train_1['SalePrice'])


# In[18]:


print(rfr.feature_importances_)


# In[42]:


# Build a forest and compute the feature importances
forest = RandomForestRegressor(n_estimators=500,
                              random_state=1234)

forest.fit(train_1.drop('SalePrice', axis = 1), train_1['SalePrice'])
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_1.drop('SalePrice', axis = 1).shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_1.drop('SalePrice', axis = 1).shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(train_1.drop('SalePrice', axis = 1).shape[1]), indices)
plt.xlim([-1, train_1.drop('SalePrice', axis = 1).shape[1]])
plt.show()


# In[63]:


test_2.head()


# In[ ]:


rfr_pred = rfr.predict(test_2)


# In[44]:


train_1 = train_1.drop(train_1.index[2])


# In[52]:


train_1.to_csv('B:\Train.csv')


# In[45]:


from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(test_2['SalePrice'], rfr_pred)**0.5
RMSE


# In[54]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_2['SalePrice'],rfr_pred)


# In[53]:


score(test_2.drop('SalePrice'), test_2['SalePrice'], sample_weight=None)


# In[12]:


rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, random_state=1234)


# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


cv_score = cross_val_score(rf_test, train_1.drop('SalePrice', axis = 1), train_1['SalePrice'], cv = 5, n_jobs = -1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# In[89]:


train_n = train.shape[0]
test_n = test.shape[0]
y_train = train.SalePrice.values
Combined = pd.concat((train, test)).reset_index(drop=True)


# In[90]:


Combined.shape


# In[91]:


Combined1 = Combined.copy()


# In[92]:


Combined.drop(['SalePrice'], axis=1, inplace=True)


# Let's start feature encoding of categorical variables

# In[58]:


categorical


# In[93]:


#SaleCondition
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
Combined["SaleCondition_cat"] = lb_make.fit_transform(Combined["SaleCondition"])
Combined[["SaleCondition", "SaleCondition_cat"]].head(1)

# Normal - 4, Abnorml - 0, Partial - 5,AdjLand - 1, Alloca - 2, Family - 1


# In[94]:


#SaleType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["SaleType_cat"] = lb_make.fit_transform(Combined["SaleType"])
Combined[["SaleType", "SaleType_cat"]].head(1)

# WD - 8, New - 6, COD = 0


# In[95]:


#PavedDrive
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["PavedDrive_cat"] = lb_make.fit_transform(Combined["PavedDrive"])
Combined[["PavedDrive", "PavedDrive_cat"]].head(1)
# Y - 2, N- 0, P-1


# In[96]:


#GarageCond
from sklearn.preprocessing import LabelEncoder
#encoder = preprocessing.LabelEncoder()
lb_make = LabelEncoder()
#train["GarageFinish_cat"] = lb_make.fit_transform(train["GarageFinish"])
Combined["GarageCond_cat"] = lb_make.fit_transform(Combined["GarageCond"].fillna('0'))
Combined[["GarageCond", "GarageCond_cat"]].head(1)


# In[97]:


#GarageQual
from sklearn.preprocessing import LabelEncoder
#encoder = preprocessing.LabelEncoder()
lb_make = LabelEncoder()
#train["GarageFinish_cat"] = lb_make.fit_transform(train["GarageFinish"])
Combined["GarageQual_cat"] = lb_make.fit_transform(Combined["GarageQual"].fillna('0'))
Combined[["GarageQual", "GarageQual_cat"]].head(1)


# In[98]:


#GarageFinish
from sklearn.preprocessing import LabelEncoder
#encoder = preprocessing.LabelEncoder()
lb_make = LabelEncoder()
#train["GarageFinish_cat"] = lb_make.fit_transform(train["GarageFinish"])
Combined["GarageFinish_cat"] = lb_make.fit_transform(Combined["GarageFinish"].fillna('0'))
Combined[["GarageFinish", "GarageFinish_cat"]].head(1)


# In[99]:


#GarageType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["GarageType_cat"] = lb_make.fit_transform(Combined["GarageType"].fillna('0'))
Combined[["GarageType", "GarageType_cat"]].head(1)


# In[100]:


#Functional
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Functional_cat"] = lb_make.fit_transform(Combined["Functional"].fillna('0'))
Combined[["Functional", "Functional_cat"]].head(1)


# In[101]:


#KitchenQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["KitchenQual_cat"] = lb_make.fit_transform(Combined["KitchenQual"].fillna('0'))
Combined[["KitchenQual", "KitchenQual_cat"]].head(1)


# In[102]:


#Electrical
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Electrical_cat"] = lb_make.fit_transform(Combined["Electrical"].fillna('0'))
Combined[["Electrical", "Electrical_cat"]].head(1)


# In[103]:


#CentralAir
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["CentralAir_cat"] = lb_make.fit_transform(Combined["CentralAir"].fillna('0'))
Combined[["CentralAir", "CentralAir_cat"]].head(1)


# In[104]:


#Heating
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Heating_cat"] = lb_make.fit_transform(Combined["Heating"].fillna('0'))
Combined[["Heating", "Heating_cat"]].head(1)


# In[105]:


#HeatingQC
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["HeatingQC_cat"] = lb_make.fit_transform(Combined["HeatingQC"].fillna('0'))
Combined[["HeatingQC", "HeatingQC_cat"]].head(1)


# In[106]:


#BsmtExposure
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BsmtExposure_cat"] = lb_make.fit_transform(Combined["BsmtExposure"].fillna('0'))
Combined[["BsmtExposure", "BsmtExposure_cat"]].head(1)


# In[107]:


#BsmtExposure
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BsmtFinType1_cat"] = lb_make.fit_transform(Combined["BsmtFinType1"].fillna('0'))
Combined[["BsmtFinType1", "BsmtFinType1_cat"]].head(1)


# In[108]:


#BsmtExposure
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BsmtFinType2_cat"] = lb_make.fit_transform(Combined["BsmtFinType2"].fillna('0'))
Combined[["BsmtFinType2", "BsmtFinType2_cat"]].head(1)


# In[109]:


#BsmtCond
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BsmtCond_cat"] = lb_make.fit_transform(Combined["BsmtCond"].fillna('0'))
Combined[["BsmtCond", "BsmtCond_cat"]].head(1)


# In[110]:


#BsmtQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BsmtQual_cat"] = lb_make.fit_transform(Combined["BsmtQual"].fillna('0'))
Combined[["BsmtQual", "BsmtQual_cat"]].head(1)


# In[111]:


#Foundation
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Foundation_cat"] = lb_make.fit_transform(Combined["Foundation"].fillna('0'))
Combined[["Foundation", "Foundation_cat"]].head(1)


# In[112]:


#ExterCond
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["ExterCond_cat"] = lb_make.fit_transform(Combined["ExterCond"].fillna('0'))
Combined[["ExterCond", "ExterCond_cat"]].head(1)


# In[113]:


#ExterQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["ExterQual_cat"] = lb_make.fit_transform(Combined["ExterQual"].fillna('0'))
Combined[["ExterQual", "ExterQual_cat"]].head(1)


# In[114]:


#MasVnrType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["MasVnrType_cat"] = lb_make.fit_transform(Combined["MasVnrType"].fillna('0'))
Combined[["MasVnrType", "MasVnrType_cat"]].head(1)


# In[115]:


#Exterior2nd
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Exterior2nd_cat"] = lb_make.fit_transform(Combined["Exterior2nd"].fillna('0'))
Combined[["Exterior2nd", "Exterior2nd_cat"]].head(1)


# In[116]:


#Exterior1st
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Exterior1st_cat"] = lb_make.fit_transform(Combined["Exterior1st"].fillna('0'))
Combined[["Exterior1st", "Exterior1st_cat"]].head(1)


# In[117]:


#RoofMatl
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["RoofMatl_cat"] = lb_make.fit_transform(Combined["RoofMatl"].fillna('0'))
Combined[["RoofMatl", "RoofMatl_cat"]].head(1)


# In[118]:


#RoofStyle
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["RoofStyle_cat"] = lb_make.fit_transform(Combined["RoofStyle"].fillna('0'))
Combined[["RoofStyle", "RoofStyle_cat"]].head(1)


# In[119]:


#HouseStyle
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["HouseStyle_cat"] = lb_make.fit_transform(Combined["HouseStyle"].fillna('0'))
Combined[["HouseStyle", "HouseStyle_cat"]].head(1)


# In[120]:


#BldgType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["BldgType_cat"] = lb_make.fit_transform(Combined["BldgType"].fillna('0'))
Combined[["BldgType", "BldgType_cat"]].head(1)


# In[121]:


#Condition1
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Condition1_cat"] = lb_make.fit_transform(Combined["Condition1"].fillna('0'))
Combined[["Condition1", "Condition1_cat"]].head(1)


# In[122]:


#Condition2
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Condition2_cat"] = lb_make.fit_transform(Combined["Condition2"].fillna('0'))
Combined[["Condition2", "Condition2_cat"]].head(1)


# In[123]:


#LandSlope
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["LandSlope_cat"] = lb_make.fit_transform(Combined["LandSlope"].fillna('0'))
Combined[["LandSlope", "LandSlope_cat"]].head(1)


# In[124]:


#LotConfig
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["LotConfig_cat"] = lb_make.fit_transform(Combined["LotConfig"].fillna('0'))
Combined[["LotConfig", "LotConfig_cat"]].head(1)


# In[125]:


#LotShape
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["LotShape_cat"] = lb_make.fit_transform(Combined["LotShape"].fillna('0'))
Combined[["LotShape", "LotShape_cat"]].head(1)


# In[126]:


#LandContour
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["LandContour_cat"] = lb_make.fit_transform(Combined["LandContour"].fillna('0'))
Combined[["LandContour", "LandContour_cat"]].head(1)


# In[127]:


#Utilities
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Utilities_cat"] = lb_make.fit_transform(Combined["Utilities"].fillna('0'))
Combined[["Utilities", "Utilities_cat"]].head(1)


# In[128]:


#Street
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Street_cat"] = lb_make.fit_transform(Combined["Street"].fillna('0'))
Combined[["Street", "Street_cat"]].head(1)


# In[129]:


#Neighborhood
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["Neighborhood_cat"] = lb_make.fit_transform(Combined["Neighborhood"].fillna('0'))
Combined[["Neighborhood", "Neighborhood_cat"]].head(1)


# In[131]:


#MSZoning
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Combined["MSZoning_cat"] = lb_make.fit_transform(Combined["MSZoning"].fillna('0'))
Combined[["MSZoning", "MSZoning_cat"]].head(1)


# Now that we have 38 categorical variables encoded and rest numerical variables. Let's see if these variables can make a difference in predicting our model.

# In[132]:


Combined2 = Combined.copy()


# In[133]:


Combined.drop('MSZoning',axis=1,inplace=True)
Combined.drop('Neighborhood',axis=1,inplace=True)
Combined.drop('Street',axis=1,inplace=True)
Combined.drop('LotShape',axis=1,inplace=True)
Combined.drop('LandContour',axis=1,inplace=True)
Combined.drop('Utilities',axis=1,inplace=True)
Combined.drop('LotConfig',axis=1,inplace=True)
Combined.drop('LandSlope',axis=1,inplace=True)
Combined.drop('Condition1',axis=1,inplace=True)
Combined.drop('Condition2',axis=1,inplace=True)
Combined.drop('BldgType',axis=1,inplace=True)
Combined.drop('HouseStyle',axis=1,inplace=True)
Combined.drop('RoofStyle',axis=1,inplace=True)
Combined.drop('RoofMatl',axis=1,inplace=True)
Combined.drop('Exterior1st',axis=1,inplace=True)
Combined.drop('Exterior2nd',axis=1,inplace=True)
Combined.drop('MasVnrType',axis=1,inplace=True)
Combined.drop('ExterQual',axis=1,inplace=True)
Combined.drop('ExterCond',axis=1,inplace=True)
Combined.drop('Foundation',axis=1,inplace=True)
Combined.drop('BsmtQual',axis=1,inplace=True)
Combined.drop('BsmtCond',axis=1,inplace=True)
Combined.drop('BsmtExposure',axis=1,inplace=True)
Combined.drop('BsmtFinType1',axis=1,inplace=True)
Combined.drop('BsmtFinType2',axis=1,inplace=True)
Combined.drop('HeatingQC',axis=1,inplace=True)
Combined.drop('Heating',axis=1,inplace=True)
Combined.drop('CentralAir',axis=1,inplace=True)
Combined.drop('Electrical',axis=1,inplace=True)
Combined.drop('KitchenQual',axis=1,inplace=True)
Combined.drop('GarageType',axis=1,inplace=True)
Combined.drop('GarageFinish',axis=1,inplace=True)
Combined.drop('GarageQual',axis=1,inplace=True)
Combined.drop('GarageCond',axis=1,inplace=True)
Combined.drop('PavedDrive',axis=1,inplace=True)
Combined.drop('SaleType',axis=1,inplace=True)
Combined.drop('SaleCondition',axis=1,inplace=True)


# In[155]:


Combined.drop('Functional',axis=1,inplace=True)


# In[40]:


#Considering only 14 numerical columns and removing rest
train2.drop('1stFlrSF',axis=1,inplace=True)
train2.drop('2ndFlrSF',axis=1,inplace=True)
train2.drop('3SsnPorch',axis=1,inplace=True)
train2.drop('BsmtFinSF1',axis=1,inplace=True)
train2.drop('BsmtFinSF2',axis=1,inplace=True)
train2.drop('BsmtFullBath',axis=1,inplace=True)
train2.drop('BsmtHalfBath',axis=1,inplace=True)
train2.drop('EnclosedPorch',axis=1,inplace=True)
train2.drop('Fireplaces',axis=1,inplace=True)
train2.drop('GarageArea',axis=1,inplace=True)
train2.drop('HalfBath',axis=1,inplace=True)
train2.drop('LotArea',axis=1,inplace=True)
train2.drop('LowQualFinSF',axis=1,inplace=True)
train2.drop('MasVnrArea',axis=1,inplace=True)
train2.drop('MiscVal',axis=1,inplace=True)
train2.drop('TotalBsmtSF',axis=1,inplace=True)
train2.drop('Id',axis=1,inplace=True)
train2.drop('LotConfig',axis=1,inplace=True)
train2.drop('LandSlope',axis=1,inplace=True)
train2.drop('Condition1',axis=1,inplace=True)
train2.drop('Condition2',axis=1,inplace=True)
train2.drop('MasVnrType',axis=1,inplace=True)
train2.drop('BsmtFinType1',axis=1,inplace=True)
train2.drop('BsmtFinType2',axis=1,inplace=True)
train2.drop('Functional',axis=1,inplace=True)
train2.drop('GarageQual',axis=1,inplace=True)
train2.drop('GarageCond',axis=1,inplace=True)
train2.drop('WoodDeckSF',axis=1,inplace=True)
train2.drop('OpenPorchSF',axis=1,inplace=True)
train2.drop('PoolArea',axis=1,inplace=True)
train2.drop('ScreenPorch',axis=1,inplace=True)
train2.drop('GarageCars',axis=1,inplace=True) #correlated
train2.drop('GrLivArea',axis=1,inplace=True)
train2.drop('GarageYrBlt',axis=1,inplace=True)


# In[156]:


Combined3 = Combined.copy()


# Let's check missing values if any now:

# In[157]:


#Yellow ones are the missing columns
ax = plt.figure(figsize = (20,10))
sns.heatmap(Combined3.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[158]:


#Replacing them with zero and dropping LotFrontage as it contains lot of missing values
Combined3.drop('LotFrontage',axis=1,inplace=True)
Combined3 = Combined3.fillna(value=0, axis=1)


# Now for test data too. Let's import it and consider numerical columns.

# In[8]:


test = pd.read_csv('test_house.csv')


# In[9]:


#We can even check the percentage of data missing
Totaltest = test.isnull().sum().sort_values(ascending=False)
Percenttest = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
Missing_data = pd.concat([Totaltest, Percenttest], axis=1, keys=['Total', 'Percent'])
Missing_data.head(5)


# In[10]:


#Seems like top 5 columns have more than 80% of the data missing, we can definitely remove them.
test.drop('Fence',axis=1,inplace=True)
test.drop('FireplaceQu',axis=1,inplace=True)
test.drop('PoolQC',axis=1,inplace=True)
test.drop('Alley',axis=1,inplace=True)
test.drop('MiscFeature',axis=1,inplace=True)


# Let's do category encoding for these variables for this dataset too.

# In[120]:


#SaleCondition
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["SaleCondition_cat"] = lb_make.fit_transform(test["SaleCondition"])
test[["SaleCondition", "SaleCondition_cat"]].head(1)

# Normal - 4, Abnorml - 0, Partial - 5,AdjLand - 1, Alloca - 2, Family - 1


# In[121]:


#SaleType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["SaleType_cat"] = lb_make.fit_transform(test["SaleType"].fillna('0'))
test[["SaleType", "SaleType_cat"]].head(1)


# In[122]:


#PavedDrive
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["PavedDrive_cat"] = lb_make.fit_transform(test["PavedDrive"].fillna('0'))
test[["PavedDrive", "PavedDrive_cat"]].head(1)
# Y - 2, N- 0, P-1


# In[123]:


#GarageFinish
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["GarageFinish_cat"] = lb_make.fit_transform(test["GarageFinish"].fillna('0'))
test[["GarageFinish", "GarageFinish_cat"]].head(1)
# RFn - 2, Unf - 3, Fin - 1


# In[124]:


#GarageType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["GarageType_cat"] = lb_make.fit_transform(test["GarageType"].fillna('0'))
test[["GarageType", "GarageType_cat"]].head(1)


# In[125]:


#Functional
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Functional_cat"] = lb_make.fit_transform(test["Functional"].fillna('0'))
test[["Functional", "Functional_cat"]].head(1)


# In[126]:


#KitchenQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["KitchenQual_cat"] = lb_make.fit_transform(test["KitchenQual"].fillna('0'))
test[["KitchenQual", "KitchenQual_cat"]].head(1)


# In[127]:


#Electrical
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Electrical_cat"] = lb_make.fit_transform(test["Electrical"].fillna('0'))
test[["Electrical", "Electrical_cat"]].head(1)


# In[128]:


#CentralAir
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["CentralAir_cat"] = lb_make.fit_transform(test["CentralAir"].fillna('0'))
test[["CentralAir", "CentralAir_cat"]].head(1)


# In[129]:


#Heating
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Heating_cat"] = lb_make.fit_transform(test["Heating"].fillna('0'))
test[["Heating", "Heating_cat"]].head(1)


# In[130]:


#HeatingQC
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["HeatingQC_cat"] = lb_make.fit_transform(test["HeatingQC"].fillna('0'))
test[["HeatingQC", "HeatingQC_cat"]].head(1)


# In[131]:


#BsmtExposure
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["BsmtExposure_cat"] = lb_make.fit_transform(test["BsmtExposure"].fillna('0'))
test[["BsmtExposure", "BsmtExposure_cat"]].head(1)


# In[132]:


#BsmtCond
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["BsmtCond_cat"] = lb_make.fit_transform(test["BsmtCond"].fillna('0'))
test[["BsmtCond", "BsmtCond_cat"]].head(1)


# In[133]:


#BsmtQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["BsmtQual_cat"] = lb_make.fit_transform(test["BsmtQual"].fillna('0'))
test[["BsmtQual", "BsmtQual_cat"]].head(1)


# In[134]:


#Foundation
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Foundation_cat"] = lb_make.fit_transform(test["Foundation"].fillna('0'))
test[["Foundation", "Foundation_cat"]].head(1)


# In[135]:


#ExterCond
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["ExterCond_cat"] = lb_make.fit_transform(test["ExterCond"].fillna('0'))
test[["ExterCond", "ExterCond_cat"]].head(1)


# In[136]:


#ExterQual
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["ExterQual_cat"] = lb_make.fit_transform(test["ExterQual"].fillna('0'))
test[["ExterQual", "ExterQual_cat"]].head(1)


# In[137]:


#Exterior2nd
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Exterior2nd_cat"] = lb_make.fit_transform(test["Exterior2nd"].fillna('0'))
test[["Exterior2nd", "Exterior2nd_cat"]].head(1)


# In[138]:


#Exterior1st
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Exterior1st_cat"] = lb_make.fit_transform(test["Exterior1st"].fillna('0'))
test[["Exterior1st", "Exterior1st_cat"]].head(1)


# In[139]:


#RoofMatl
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["RoofMatl_cat"] = lb_make.fit_transform(test["RoofMatl"].fillna('0'))
test[["RoofMatl", "RoofMatl_cat"]].head(1)


# In[140]:


#HouseStyle
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["HouseStyle_cat"] = lb_make.fit_transform(test["HouseStyle"].fillna('0'))
test[["HouseStyle", "HouseStyle_cat"]].head(1)


# In[141]:


#BldgType
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["BldgType_cat"] = lb_make.fit_transform(test["BldgType"].fillna('0'))
test[["BldgType", "BldgType_cat"]].head(1)


# In[142]:


#Utilities
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Utilities_cat"] = lb_make.fit_transform(test["Utilities"].fillna('0'))
test[["Utilities", "Utilities_cat"]].head(1)


# In[143]:


#Street
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Street_cat"] = lb_make.fit_transform(test["Street"].fillna('0'))
test[["Street", "Street_cat"]].head(1)


# In[144]:


#Neighborhood
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["Neighborhood_cat"] = lb_make.fit_transform(test["Neighborhood"].fillna('0'))
test[["Neighborhood", "Neighborhood_cat"]].head(1)


# In[145]:


#MSZoning
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
test["MSZoning_cat"] = lb_make.fit_transform(test["MSZoning"].fillna('0'))
test[["MSZoning", "MSZoning_cat"]].head(1)


# In[146]:


test2 = test.copy()


# In[169]:


test.drop('MSZoning',axis = 1, inplace=True) 
test.drop('Neighborhood',axis=1,inplace=True) 
test.drop('Street',axis=1,inplace=True)
test.drop('Utilities',axis=1,inplace=True)
test.drop('BldgType',axis=1,inplace=True)
test.drop('HouseStyle',axis=1,inplace=True) 
test.drop('RoofMatl',axis=1,inplace=True) 
test.drop('Exterior1st',axis=1,inplace=True)
test.drop('Exterior2nd',axis=1,inplace=True) 
test.drop('ExterQual',axis=1,inplace=True) 
test.drop('ExterCond',axis=1,inplace=True) 
test.drop('Foundation',axis=1,inplace=True) 
test.drop('BsmtQual',axis=1,inplace=True) 
test.drop('BsmtCond',axis=1,inplace=True)
test.drop('BsmtExposure',axis=1,inplace=True) 
test.drop('HeatingQC',axis=1,inplace=True) 
test.drop('Heating',axis=1,inplace=True) 
test.drop('CentralAir',axis=1,inplace=True) 
test.drop('Electrical',axis=1,inplace=True) 
test.drop('KitchenQual',axis=1,inplace=True) 
test.drop('GarageType',axis=1,inplace=True) 
test.drop('GarageFinish',axis=1,inplace=True) 
test.drop('PavedDrive',axis=1,inplace=True) 
test.drop('SaleType',axis=1,inplace=True) 
test.drop('SaleCondition',axis=1,inplace=True) 
test.drop('RoofStyle',axis=1,inplace=True)


# In[189]:


test.drop('1stFlrSF',axis=1,inplace=True) 
test.drop('2ndFlrSF',axis=1,inplace=True)
test.drop('3SsnPorch',axis=1,inplace=True)
test.drop('BsmtFinSF1',axis=1,inplace=True)
test.drop('BsmtFinSF2',axis=1,inplace=True)
test.drop('BsmtFullBath',axis=1,inplace=True) 
test.drop('BsmtHalfBath',axis=1,inplace=True) 
test.drop('EnclosedPorch',axis=1,inplace=True)
test.drop('Fireplaces',axis=1,inplace=True) 
test.drop('GarageArea',axis=1,inplace=True)
test.drop('HalfBath',axis=1,inplace=True)
test.drop('LotArea',axis=1,inplace=True) 
test.drop('LowQualFinSF',axis=1,inplace=True)
test.drop('MasVnrArea',axis=1,inplace=True) 
test.drop('MiscVal',axis=1,inplace=True) 
test.drop('TotalBsmtSF',axis=1,inplace=True) 
test.drop('Id',axis=1,inplace=True) 
test.drop('LotShape',axis=1,inplace=True)
test.drop('LandContour',axis=1,inplace=True)
test.drop('LotConfig',axis=1,inplace=True) 
test.drop('LandSlope',axis=1,inplace=True) 
test.drop('Condition1',axis=1,inplace=True)
test.drop('Condition2',axis=1,inplace=True) 
test.drop('MasVnrType',axis=1,inplace=True) 
test.drop('BsmtFinType1',axis=1,inplace=True) 
test.drop('BsmtFinType2',axis=1,inplace=True)
test.drop('Functional',axis=1,inplace=True)
test.drop('GarageQual',axis=1,inplace=True) 
test.drop('GarageCond',axis=1,inplace=True) 
test.drop('WoodDeckSF',axis=1,inplace=True)
test.drop('OpenPorchSF',axis=1,inplace=True) 
test.drop('PoolArea',axis=1,inplace=True) 
test.drop('ScreenPorch',axis=1,inplace=True) 


# In[202]:


#Missing and Correlated values
#test.drop('Fence',axis=1,inplace=True)
#test.drop('FireplaceQu',axis=1,inplace=True)
#test.drop('PoolQC',axis=1,inplace=True)
#test.drop('Alley',axis=1,inplace=True)
test.drop('GarageCars',axis=1,inplace=True) #correlated
test.drop('GrLivArea',axis=1,inplace=True)
test.drop('GarageYrBlt',axis=1,inplace=True)


# In[213]:


#Replacing them with zero
test = test.fillna(value=0, axis=1)


# In[214]:


test3 = test.copy()


# In[215]:


train2.shape


# In[177]:


train2 = train2.drop(train.index[2])


# In[47]:


train_final = train2.copy()


# In[48]:


train2.drop('SalePrice',axis=1,inplace=True)


# In[216]:


test.shape


# In[206]:


test.columns


# In[201]:


train2.columns


# In[209]:


train2.head(3)


# Shape should be same otherwise there would be problem in Modelling, 41 is fine because 1 is target variable.To maintain number of records 1459 in both, we will remove one from training dataset.

# **Modelling and Evaluating

# Let's start with Ridge Regression because it is capable of handling large number of features. We will check feature importance and judge accordingly which features are contributing to the model.

# We should also scale our target variable as we mentioned intially

# In[77]:


Combined1["SalePrice"] = np.log1p(Combined1["SalePrice"])


# In[80]:


Combined1['SalePrice'].head(3)


# In[159]:


train1 = Combined3[:train_n]
test1 = Combined3[train_n:]


# In[190]:


train1.shape


# In[187]:


train1 = train1.drop(train1.index[2])


# In[188]:


pd.set_option('display.max_columns', None)
train1.head()


# In[193]:


y_train.shape


# In[194]:


from sklearn.linear_model import Ridge
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
linridge = Ridge(alpha = 20.0).fit(train1, y_train)
print('House Predictions')
print('ridge regression linear model intercept:{}'.format(linridge.intercept_))
print('Linear model coefficient : \n{}'. format(linridge.coef_))
print('R-squared score(training): {:.3f}'.format(linridge.score(train1,y_train)))
#print('R-squared score(test): {:.3f}'.format(linridge.score(X_test,y_test)))
print('Number of non-zero features: {}'.format(np.sum(linridge.coef_!= 0)))


# Let's see how alpha varies with R-squared for the training dataset

# In[195]:


print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(train1, y_train)
    r2_train = linridge.score(train1, y_train)
    #r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}'
         .format(this_alpha, num_coeff_bigger, r2_train))


# We can see basically it's not changing much, so we can take alpha 20 which we took intially

# In[196]:


linridge.coef_


# In[197]:


y_pred = linridge.predict(test1)
y_pred


# In[198]:


y_pred.shape


# In[192]:


#y_train.shape
#train = train.drop(train.index[2])
#train['SalePrice'].head(3)
y_train = train.SalePrice.values


# In[199]:


from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_train, y_pred)**0.5
RMSE


# In[200]:


def rmsle(predicted,real):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


# In[203]:


rmsle(y_pred,y_train)


# Let's check Random Forests too and see if that can improve our algorithm

# **Random Forest

# In[234]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1000)
rfr.fit(train2, train_final['SalePrice'])


# In[235]:


print(rfr.feature_importances_)


# In[236]:


# Build a forest and compute the feature importances
forest = RandomForestRegressor(n_estimators=250,
                              random_state=0)

forest.fit(train2, train_final['SalePrice'])
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train2.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train2.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(train2.shape[1]), indices)
plt.xlim([-1, train2.shape[1]])
plt.show()


# In[237]:


train2.columns


# Consider feature ranking. Below 10 variables are most important in determining the model:
# 
# OverallQual
# 
# FullBath
# 
# TotRmsAbvGrd
# 
# YearBlt
# 
# LotFrontage
# 
# YearRemodAdd
# 
# YearBuilt
# 
# Neighborhood
# 
# CentralAir
# 
# BsmtUnfSF
# 
# We can further improve this and these variables really make sense. Go back to the visualization and check, these variables were giving great picture of how SalePrices will vary.

# In[238]:


rfr_pred = rfr.predict(test)


# In[239]:


from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(train_final['SalePrice'], rfr_pred)**0.5
RMSE


# In[240]:


rmsle(rfr_pred,train_final['SalePrice'].as_matrix())


# We can use various other algorithms like Gradient Boosting to check this.
