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

name_train = "train.csv"
name_test = "test.csv"
output_name = 'sol_0.csv'

n_pca = 50		# 50 seems optimal
			# compression method:
use_pca = 1		# 0... rfc for reduction, 
				# 1... pca + whitening 
				# 2... for scaling without pca
name_estimator = "NEURAL_NW"
par_estimator = 200

# only for <rfc for reduction>:
par_rcf_reducer = 100	# only needed if use_pca = 0
epsilon = 0.002		#cutoff in <rfc for reduction>



def RFC_f(X, y, r):
	tri = RandomForestClassifier(r)
	sc = cross_val_score(tri , X, y, cv=5)
	tri.fit(X, y)
	print('CV', np.mean(sc),tri.score(X,y))
	G=tri.feature_importances_
	return G, tri

def SUPP_VM_rbf(X,y,c):
	g = 0.02
	SVM = svm.SVC(kernel='rbf', C=c, gamma = g)
	sc = cross_val_score(SVM, X, y, cv=5)
	SVM.fit(X, y)
	print(np.mean(sc), SVM.score(X,y))
	return SVM

def NEURAL_NW(X,y, c):
	lay_size = (c,)
	nnw=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=lay_size)
	sc = cross_val_score(nnw, X, y, cv=5)
	nnw.fit(X, y)
	print('CV', np.mean(sc), nnw.score(X,y))	
	return nnw

def check_manually(X_test, y_pred):
	for i in range(100):
		print("pred", y_pred[i])
		visualize_image(X_test[i])

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



########################
# prepare training data -> X, y
########################


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = name_train
abs_file_path = os.path.join(script_dir, rel_path)
df =pd.read_csv(abs_file_path)

rel_path = name_test
abs_file_path = os.path.join(script_dir, rel_path)
df_test =pd.read_csv(abs_file_path)

y = df['label'].tolist()
del df['label']
X = df.values.tolist()

X_test = df_test.values.tolist()

print(len(y))


if use_pca == 0:
# use rfc to reduce data dimension
	G, estimator = RFC_f(X, y, par_rcf_reducer)
	index_G = [i for i in range(len(G)) if G[i] > epsilon]
	print("length index_G", len(index_G))
	X_cpr = []
	for x in X:
		X_cpr.append([x[i] for i in index_G])
	scaler = preprocessing.StandardScaler().fit(X_cpr)
	X = scaler.transform(X_cpr)
	X_cpr_test = []
	for x in X_test:
		X_cpr_test.append([x[i] for i in index_G])
	X_test_pca = scaler.transform(X_cpr_test)
elif use_pca == 1:
	pca = PCA(n_components=n_pca, whiten = True)
	pca.fit(X)
	X = pca.transform(X)
	X_test_pca = pca.transform(X_test)
else:
	scaler = preprocessing.StandardScaler().fit(X)
	X = scaler.transform(X)
	X_test_pca = scaler.transform(X_test)


for r in [50,100,200,400,600,1000]:
	if name_estimator != "RFC_f":
		estimator = globals()[name_estimator](X, y, par_estimator)
	else:
		G, estimator = RFC_f(X,y, par_estimator)
y_pred = estimator.predict(X_test_pca)

daten = {'ImageId': [i+1 for i in range(len(y_pred))] , 'Label': y_pred}
cols = ['ImageId', 'Label']

ddf = pd.DataFrame(daten, columns = cols)
ddf.to_csv(output_name, index= False)

# check_manually(X_test, y_pred)