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


def visualize_image(L):
# L is a row from dataframe
	A=[]
	for k in range(28):
		A.append(L[k*28:k*28+28])
	f, ax = plt.subplots(1,1)
	print(A)
	ax.imshow(A, cmap='gray')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.show()



y = df['label'].tolist()
del df['label']
X = df.values.tolist()

# PCA
pca = PCA(n_components=60, whiten = True)
pca.fit(X)
X = pca.transform(X)

print(len(y))

def NEURAL_NW(X,y):
	man = 0
	for layer in range(2,5):
		print(layer)
		for nodes in [50,100,500]:
			lay_size = tuple([nodes for i in range(layer)])
			nnw=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=lay_size)
			sc = cross_val_score(nnw, X, y, cv=5)
			nnw.fit(X, y)
			print(lay_size)
			print('CV', np.mean(sc), nnw.score(X,y))	
			if np.mean(sc)>man:
				man = np.mean(sc)
				laier = lay_size
	print('final layer', laier)
	return None


NEURAL_NW(X,y)
