import time
import pandas as pd 
import random
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn import svm, preprocessing, tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor,RandomForestClassifier
import csv
import matplotlib.colors as colors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import RFECV


########################
# prepare training data -> X, y
########################





script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "train.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df =pd.read_csv(abs_file_path)

rel_path = "test.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df_test =pd.read_csv(abs_file_path)


def visualize_image(L):
# L is a row from dataframe
	A=[]
	for k in range(28):
		A.append(L[k*28:k*28+28])
	f, ax = plt.subplots(1,1)

	ax.imshow(A, cmap='gray')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.show()


y = df['label'].tolist()
del df['label']
X = df.values.tolist()

X_test = df_test.values.tolist()



# PCA

pca = PCA(n_components=50, whiten = True)
pca.fit(X)
X = pca.transform(X)
X_test_pca = pca.transform(X_test)

# just normalizing instead
# scaler = preprocessing.StandardScaler().fit(X)
# X = scaler.transform(X)
# X_test_pca = scaler.transform(X_test)

print(len(y))

def RFC_f(X, y):
	tri = RandomForestClassifier()
	sc = cross_val_score(tri , X, y, cv=5)
	tri.fit(X, y)
	print('CV', np.mean(sc),tri.score(X,y))
	#G=tri.feature_importances_
	return tri

def SUPP_VM_rbf(X,y):
	c = 100
	g = 0.02
	SVM = svm.SVC(kernel='rbf', C=c, gamma = g)
	sc = cross_val_score(SVM, X, y, cv=5)
	SVM.fit(X, y)
	print(np.mean(sc), SVM.score(X,y))
	return SVM

def NEURAL_NW(X,y):
	lay_size = (2000,)
	nnw=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=lay_size)
	sc = cross_val_score(nnw, X, y, cv=5)
	nnw.fit(X, y)
	print('CV', np.mean(sc), nnw.score(X,y))	
	return nnw

def check_manually(X_test, y_pred):
	for i in range(100):
		print("pred", y_pred[i])
		visualize_image(X_test[i])


estimator = SUPP_VM_rbf(X, y)
y_pred = estimator.predict(X_test_pca)

daten = {'ImageId': [i+1 for i in range(len(y_pred))] , 'Label': y_pred}
cols = ['ImageId', 'Label']

ddf = pd.DataFrame(daten, columns = cols)
ddf.to_csv('sol.csv', index= False)

# check_manually(X_test, y_pred)