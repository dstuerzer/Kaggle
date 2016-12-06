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
rel_path = "train_medium.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df =pd.read_csv(abs_file_path)

# df_test_later = df.iloc[40000:, :]
# df = df.iloc[0:40000, :]




y = df['label'].tolist()
del df['label']
X = (df.iloc[:, 1:]).values.tolist()

# PCA
pca = PCA(n_components=50, whiten = True)
pca.fit(X)
X = pca.transform(X)

print(len(y))


def SUPP_VM_rbf(X,y):
	man = 0
	for gam in np.arange(0.02, 0.1, 0.01):
		print(gam)
		for c in [5, 10, 100, 1000]:
			SVM = svm.SVC(kernel='rbf', C=c, gamma = gam, cache_size = 1000)
			sc = cross_val_score(SVM, X, y, cv=5)
			SVM.fit(X, y)
			if np.mean(sc)>man:
				man = np.mean(sc)
				print(gam,c)
				print(np.mean(sc), SVM.score(X,y))
				c_1 = c
				gam_1 = gam





SUPP_VM_rbf(X,y)
