import time
import pandas as pd 
import random
import numpy as np 
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
from scipy import misc
# from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import svm, preprocessing, tree
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
import csv
from sklearn import metrics


def rescale(X, X_test):
	scaler = preprocessing.StandardScaler().fit(X)
	return scaler.transform(X), scaler.transform(X_test)

def apply_pca(X, X_test):
	pca = PCA(n_components=n_pca, whiten = True)
	pca.fit(X)
	return  pca.transform(X), pca.transform(X_test)

def write_output(ids, y_pred):
	daten = {'Id': ids , 'SalePrice': y_pred}
	cols = ['Id', 'SalePrice']

	df = pd.DataFrame(daten, columns = cols)
	df.to_csv("sol.csv", index= False)

def search_grid(X, y):
	xx = []
	yy = []
	for k in np.arange(-7,-2, 0.5):
		est = RandomForestRegressor(max_depth = 15, min_impurity_split = np.power(10., k))
		sc = cross_val_score(est, X, y, cv=20, scoring = 'neg_mean_squared_error')
		print("CV = {}".format(np.sqrt(-np.mean(sc))))
		xx.append(k)
		yy.append(np.sqrt(-np.mean(sc)))
	plt.plot(xx,yy)
	plt.show()
def manual_t(y):
	A = [[np.power(12.2,3), np.power(12.2,2), 12.2, 1],[3*np.power(12.2,2), 2*12.2, 1, 0],[np.power(10.5,3), np.power(10.5,2), 10.5, 1],[np.power(13.5,3), np.power(13.5,2), 13.5, 1]]
	b = [12.2, 1, 10.65, 13.35]
	c = np.linalg.solve(A,b)
	return c[0]*np.power(y,3) + c[1]*np.power(y,2) + c[2]*y+c[3]
	
def predict_outcome(X, y, X_test):
	y = list(y)
	est = linear_model.Lasso(alpha = np.power(10.,-3.2))
	# for i in range(10):
	# 	est.fit(X,y)
	# 	y_test = est.predict(X)
	# 	li = [np.absolute(y[o]-y_test[o]) for o in range(len(y))]
	# 	ii = li.index(max(li))
	# 	del X[ii]
	# 	del y[ii]
	est.fit(X,y)
	sc = cross_val_score(est, X, y, cv=10, scoring = 'neg_mean_squared_error')
	print("CV = {}".format(np.sqrt(-np.mean(sc))))
	y_2 = est.predict(X_test)
	y_3 = []
	for ka in y_2:
		y_3.append(manual_t(ka))
	y_pred = np.exp(y_3)
	write_output(ids, y_pred)


def important_features_rfr(df_train, df_test, y):
	df_0 = df_train.copy()
	df_1 = df_test.copy()
	print(df_0.shape)
	for i in range(10):
		X = df_0.values.tolist()
		est = RandomForestRegressor(max_depth = 20, min_impurity_split = np.power(10., -3.5))
		est.fit(X,y)
		cols = df_0.columns
		feat_import = est.feature_importances_ 
		daten = {'feat_imp': feat_import, 'names': cols}
		cols = ['feat_imp', 'names']
		df = pd.DataFrame(daten, columns = cols)
		df = df.sort_values(by = 'feat_imp', ascending = False)
		dropped = df.loc[df['feat_imp'] <1e-06, 'names'].tolist()
		print(dropped)
		df_0 = df_0.drop(dropped, axis=1)
		df_1 = df_1.drop(dropped, axis=1)
		print("new dimensions", df_0.shape)
	return df_0, df_1

def lasso_bootstrap(df_train, y, df_test):
	print(len(df_train.columns.tolist()))
	for j in range(4):
		X = df_train.values.tolist()
		est = linear_model.Lasso(alpha = np.power(10.,-3.24))
		est.fit(X,y)
		feat_import = est.coef_
		print(len(feat_import))
		ind = [i for i in range(len(feat_import)) if np.absolute(feat_import[i])>2.1e-04]
		df_train = df_train.iloc[:, ind]
		df_test = df_test.iloc[:, ind]
	X = df_train.values.tolist()
	est = linear_model.Lasso(alpha = np.power(10.,-3.24))
	sc = cross_val_score(est, X, y, cv=6, scoring = 'neg_mean_squared_error')
	print("CV = {}".format(np.sqrt(-np.mean(sc))))
	est.fit(X,y)
	X_test = df_test.iloc[:, ind].values.tolist()
	y_pred = np.exp(est.predict(X_test))
	write_output(ids, y_pred)
# get data

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv('test.csv')
ids = df_test['Id']				# test ids for later



def prdict(X, y, X_test):
	est = GradientBoostingRegressor()
	est.fit(X, y)
	sc = cross_val_score(est, X, y, cv=10, scoring = 'neg_mean_squared_error')
	print("CV = {}".format(np.sqrt(-np.mean(sc))))
	y_0 = est.predict(X)
	y_pred = est.predict(X_test)
	plt.scatter(y, y_0, lw = 0)
	plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], color = 'red')
	x = np.arange(np.min(y), np.max(y), 0.1)
	z = []
	for k in x:
		z.append(manual_t(k))
	plt.plot(x,z,color = 'green')
	plt.show()
	y_3 = []
	for ka in y_pred:
		y_3.append(manual_t(ka))
	y_pred = np.exp(y_3)
	write_output(ids, y_pred)





##########################
# prep for learning pt 1 #
##########################

#merge the training and the test set to create common DUMMIES

y = np.log(df_train['SalePrice'].tolist())		#y is training prices # log is better for this problem
del df_train['SalePrice']
df = df_train.append(df_test)
df.loc[df['YearRemodAdd'] != df['YearBuilt'], 'YearRemodAdd'] = 0
df.loc[df['YearRemodAdd'] == df['YearBuilt'], 'YearRemodAdd'] = 1
conds = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
for i in range(len(conds)):
	df.loc[df['ExterQual']==conds[i], 'ExterQual'] = len(conds)-i-1
	df.loc[df['ExterCond']==conds[i], 'ExterCond'] = len(conds)-i-1
	df.loc[df['BsmtQual'] == conds[i], 'BsmtQual'] = len(conds)-i
	df.loc[df['BsmtCond'] == conds[i], 'BsmtCond'] = len(conds)-i
	df.loc[df['HeatingQC']==conds[i], 'HeatingQC'] = len(conds)-i-1
	df.loc[df['KitchenQual']==conds[i], 'KitchenQual'] = len(conds)-i-1
	df.loc[df['FireplaceQu']==conds[i], 'FireplaceQu'] = len(conds)-i
	df.loc[df['GarageQual']==conds[i], 'GarageQual'] = len(conds)-i-1
	df.loc[df['GarageCond']==conds[i], 'GarageCond'] = len(conds)-i-1


df.ExterCond.fillna(df['ExterCond'].mean(), inplace=True)
df.ExterQual.fillna(df['ExterQual'].mean(), inplace=True)
df.BsmtQual.fillna(0, inplace=True)
df.BsmtCond.fillna(0, inplace=True)
df.HeatingQC.fillna(df['HeatingQC'].mean(), inplace=True)
df.KitchenQual.fillna(df['KitchenQual'].mean(), inplace=True)
df.FireplaceQu.fillna(0, inplace=True)
df.GarageQual.fillna(0, inplace=True)
df.GarageCond.fillna(0, inplace=True)


df.loc[df['CentralAir'] == 'Y', 'CentralAir'] = 1
df.loc[df['CentralAir'] == 'N', 'CentralAir'] = 0
df.CentralAir.fillna(df['CentralAir'].mean(), inplace=True)

df.LotFrontage.fillna(0, inplace=True)	#nans here are most likely zeros (from no frontage)
# replace most categorical variables
columns_cat = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
df = pd.concat([df, pd.get_dummies(df[columns_cat])],axis = 1)
for c in columns_cat:	#delete categorical variables since we have dummies now
	del df[c]
#replace few missing values by mean.
df.BsmtFinSF1.fillna(df['BsmtFinSF1'].mean(), inplace=True)
df.BsmtFinSF2.fillna(df['BsmtFinSF2'].mean(), inplace=True)
df.BsmtUnfSF.fillna(df['BsmtUnfSF'].mean(), inplace=True)
df.TotalBsmtSF.fillna(df['TotalBsmtSF'].mean(), inplace=True)
df.GarageCars.fillna(df['GarageCars'].mean(), inplace=True)
df.BsmtFullBath.fillna(df['BsmtFullBath'].mean(), inplace=True)
df.BsmtHalfBath.fillna(df['BsmtHalfBath'].mean(), inplace=True)
df.GarageArea.fillna(df['GarageArea'].mean(), inplace=True)
df.loc[pd.isnull(df['GarageYrBlt']), 'GarageYrBlt'] = df.loc[pd.isnull(df['GarageYrBlt']), 'YearBuilt']
# del df['MasVnrType']
# del df['MasVnrArea']
df.loc[pd.isnull(df['MasVnrType']), 'MasVnrType']  = 'None'
df.MasVnrArea.fillna(df['MasVnrArea'].mean(), inplace=True) 
pd.concat([df, pd.get_dummies(df['MasVnrType'])],axis = 1)
del df['MasVnrType']

df = df.fillna(value=0)		#very few values are still nan's - just get rid of them (they are in 3 prediction datasets

#split everything back into training and testing set


df_train = df.iloc[0:1460,:]
df_test = df.iloc[1460:, :]
X = df_train.values.tolist()
X_test = df_test.values.tolist()


# df_0, df_1 = important_features_rfr(df_train, df_test, y)

# X = df_0.values.tolist()
# X_test = df_1.values.tolist()

############################
## prediction ##############
############################
# important_features_rfr(X,y, df_train.columns.tolist())
# important_features_lasso(X,y, df_train.columns.tolist())
# X, X_test = rescale(X, X_test)
# search_grid(X, y)
# predict_outcome(X, y, X_test)
prdict(X,y,X_test)
# lasso_bootstrap(df_train, y, df_test)