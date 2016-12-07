import time
import pandas as pd 
import random
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn import svm, preprocessing, tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.colors as colors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

name_train = "train_medium.csv"
name_test = "test.csv"
output_name = 'sol.csv'

n_pca = 50		# 50 seems optimal
			# compression method:
use_pca = 2		# 0... rfc for reduction, 
				# 1... pca + whitening 
				# 2... for linear discriminant analysis dim red
name_estimator = "SUPP_VM_rbf"
par_estimator = 0.027

def LDA(X, y, r):
	tri = LinearDiscriminantAnalysis()
	sc = cross_val_score(tri , X, y, cv=5)
	tri.fit(X, y)
	print('CV', np.mean(sc),tri.score(X,y))
	# G=tri.feature_importances_
	return np.mean(sc), tri

def RFC_f(X, y, r):
	tri = RandomForestClassifier(r)
	sc = cross_val_score(tri , X, y, cv=5)
	tri.fit(X, y)
	print('CV', np.mean(sc),tri.score(X,y))
	# G=tri.feature_importances_
	return np.mean(sc), tri

def SUPP_VM_rbf(X,y,g):
	c = 20
	g = 0.027
	SVM = svm.SVC(kernel='rbf', C=c, gamma = g)
	sc = cross_val_score(SVM, X, y, cv=5)
	SVM.fit(X, y)
	print(np.mean(sc), SVM.score(X,y))
	return np.mean(sc), SVM

def SUPP_VM_linear(X,y,c):
	SVM = svm.SVC(kernel='linear', C=np.power(10.,-0.8))
	sc = cross_val_score(SVM, X, y, cv=5)
	SVM.fit(X, y)
	print(np.mean(sc), SVM.score(X,y))
	return np.mean(sc), SVM

def NEURAL_NW(X,y, c):
	lay_size = (1000,)
	nnw=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=lay_size)
	sc = cross_val_score(nnw, X, y, cv=5)
	nnw.fit(X, y)
	print('CV', np.mean(sc), nnw.score(X,y))	
	return np.mean(sc), nnw

def check_manually(X_test, y_pred):

	for jj in range(100):
		i = random.choice(range(len(y_pred)))
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


if use_pca == 1:
	pca = PCA(n_components=n_pca, whiten = True)
	pca.fit(X)
	X = pca.transform(X)
	X_test_pca = pca.transform(X_test)
else:
	redux = LinearDiscriminantAnalysis( n_components=20)
	redux.fit(X, y)
	X = redux.transform(X)
	X_test = redux.transform(X_test)
	print(len(X[3]))
	scaler = preprocessing.StandardScaler().fit(X)
	X = scaler.transform(X)
	X_test_pca = scaler.transform(X_test)

# temporary loop
# par_x = np.arange(10, 100, 10)
# e_y = []
# for par_estimator in par_x:
E_cv, estimator = globals()[name_estimator](X, y, par_estimator)
	# e_y.append(E_cv)

# plt.plot(par_x, [np.log(1-p)/np.log(10) for p in e_y])	
# plt.show()


y_pred = estimator.predict(X_test_pca)

daten = {'ImageId': [i+1 for i in range(len(y_pred))] , 'Label': y_pred}
cols = ['ImageId', 'Label']

ddf = pd.DataFrame(daten, columns = cols)
ddf.to_csv(output_name, index= False)

# check_manually(X_test, y_pred)