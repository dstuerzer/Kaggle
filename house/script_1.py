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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
	for k in np.arange(-3.4,-3,0.04):
		est = linear_model.Lasso(alpha = np.power(10.,k))
		sc = cross_val_score(est, X, y, cv=5, scoring = 'neg_mean_squared_error')
		print("CV = {}".format(np.sqrt(-np.mean(sc))))
		xx.append(k)
		yy.append(np.sqrt(-np.mean(sc)))
	plt.plot(xx,yy)
	plt.show()

def predict_outcome(X, y, X_test):
	y = list(y)
	est = linear_model.Lasso(alpha = np.power(10.,-3.24))
	for i in range(50):
		est.fit(X,y)
		y_test = est.predict(X)
		li = [np.absolute(y[o]-y_test[o]) for o in range(len(y))]
		ii = li.index(max(li))
		del X[ii]
		del y[ii]
	y_test = est.predict(X)
	plt.scatter(y, y_test)
	plt.show()
	est.fit(X,y)
	sc = cross_val_score(est, X, y, cv=10, scoring = 'neg_mean_squared_error')
	print("CV = {}".format(np.sqrt(-np.mean(sc))))
	y_pred = np.exp(est.predict(X_test))
	write_output(ids, y_pred)

def predict_outcome_2(X, y, X_test):
	est = linear_model.Lasso(alpha = np.power(10.,-3.24))
	# est = svm.SVR(kernel='linear', C = np.power(10.,-3.38))
	# est = LinearRegression()
	# est = svm.SVR(kernel = 'poly', degree = 2)
	# est = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,))
	# est = linear_model.Ridge(alpha = 0.1)
	# est = KernelRidge()
	# est = tree.DecisionTreeRegressor()
	# est = BayesianRidge(n_iter = 300)
	# est = RandomForestRegressor()

	sc = cross_val_score(est, X, y, cv=10, scoring = 'neg_mean_squared_error')
	print("CV = {}".format(np.sqrt(-np.mean(sc))))
	est.fit(X,y)
	y_pred = np.exp(est.predict(X_test))
	write_output(ids, y_pred)

def important_features_rfr(X,y,cols):
	est = RandomForestRegressor(400)
	est.fit(X,y)
	feat_import = est.feature_importances_ 
	daten = {'feat_imp': feat_import, 'names': cols}
	cols = ['feat_imp', 'names']

	df = pd.DataFrame(daten, columns = cols)
	df = df.sort_values(by = 'feat_imp', ascending = False)
	print(df.head(10))
	x = df['names'].tolist()
	y = df['feat_imp'].tolist()
	plt.xticks(rotation=-30)
	eo = 20
	plt.bar(range(len(x[:eo])), y[:eo], align='edge', width = -0.9, linewidth = 0)
	plt.xticks(range(len(x[:eo])), x[:eo])
	plt.title('The {} most important features w RandomForests'.format(eo))
	plt.show()

def important_features_lasso(X,y,cols):
	est = linear_model.Lasso(alpha = np.power(10.,-3.24))
	est.fit(X,y)

	feat_import = est.coef_ 
	daten = {'feat_imp': feat_import, 'names': cols}
	cols = ['feat_imp', 'names']

	df = pd.DataFrame(daten, columns = cols)
	df = df.sort_values(by = 'feat_imp', ascending = False)
	print(df.head(10))
	x = df['names'].tolist()
	y = df['feat_imp'].tolist()
	f, (ax1, ax2) = plt.subplots(1,2)#, sharey=True, sharex=True)
	eo = 20
	ax1.bar(range(len(x[:eo])), y[:eo], align='center', width = 0.9, linewidth = 0)
	ax1.set_xticks(range(len(x[:eo])))
	ax1.set_xticklabels(x[:eo], rotation = -90)
	ax1.set_title('The {} most important + features w Lasso'.format(eo))

	ax2.bar(range(len(x[-eo:])), y[-eo:], align='center', width = -0.9, linewidth = 0)
	ax2.set_xticks(range(len(x[-eo:])))
	ax2.set_xticklabels(x[-eo:], rotation = -90)
	ax2.set_title('The {} most important - features w Lasso'.format(eo))
	plt.show()


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

####################
# analysis of data #
####################


# cont_column = 'LotArea'
# df_analysis = (df_train.loc[:, [cont_column, 'SalePrice']]).dropna(how='any')

# df_analysis = df_analysis.sort_values(by = 'SalePrice', ascending = True)
# x_plt = df_analysis['SalePrice'].tolist()
# no_bins = 8
# bins = np.arange(np.min(x_plt)-0.1, np.max(x_plt)+0.1, (np.max(x_plt)-np.min(x_plt))/no_bins)
# group_names = range(1, len(bins))
# print(bins)
# df_analysis['cut'] = pd.cut(df_analysis['SalePrice'], bins, labels=group_names)
# print(df_analysis.head(4))
# x_plt = df_analysis['cut'].tolist()
# yps = []
# for g in group_names:
# 	yps.append(df_analysis.loc[df_analysis['cut'] == g, cont_column].tolist())
# # print(yps)
# y_plt = df_analysis[cont_column].tolist()
# y_plt = np.square(y_plt)
# # print(np.mean(y_plt))
# plt.boxplot(yps, labels=group_names, showfliers = False)

# # plt.scatter(x_plt, y_plt)
# plt.title(cont_column)
# plt.show()









##########################
# prep for learning pt 1 #
##########################

#merge the training and the test set to create common DUMMIES

y = np.log(df_train['SalePrice'].tolist())		#y is training prices # log is better for this problem
del df_train['SalePrice']
df = df_train.append(df_test)

df.LotFrontage.fillna(0, inplace=True)	#nans here are most likely zeros (from no frontage)
# replace most categorical variables
columns_cat = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
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
del df['GarageYrBlt']

df = df.fillna(value=0)		#very few values are still nan's - just get rid of them (they are in 3 prediction datasets

#split everything back into training and testing set
df_train = df.iloc[0:1460,:]
df_test = df.iloc[1460:, :]

X = df_train.values.tolist()
X_test = df_test.values.tolist()



############################
## prediction ##############
############################
# important_features_rfr(X,y, df_train.columns.tolist())
# important_features_lasso(X,y, df_train.columns.tolist())
# X, X_test = rescale(X, X_test)
# search_grid(X, y)
predict_outcome(X, y, X_test)
# lasso_bootstrap(df_train, y, df_test)