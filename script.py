
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_val_score
import csv

########################
# prepare training data -> X, y
########################

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "Data/train.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df =pd.read_csv(abs_file_path)

df.loc[df.Sex == 'male', 'Sex'] = 1
df.loc[df.Sex == 'female', 'Sex'] = 0
df.Age.fillna(-1, inplace = True)
df.loc[df.Embarked == 'C', 'Embarked'] = 0
df.loc[df.Embarked == 'Q', 'Embarked'] = 1
df.loc[df.Embarked == 'S', 'Embarked'] = 2

df=df.dropna(how='any')

y = df.Survived.tolist()

df = df.loc[:, ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]



X=df.as_matrix()

###################
# read test-data
###################

rel_path = "Data/test.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df =pd.read_csv(abs_file_path)


df.loc[df.Sex == 'male', 'Sex'] = 1
df.loc[df.Sex == 'female', 'Sex'] = 0
df.Age.fillna(-1, inplace = True)
df.loc[df.Embarked == 'C', 'Embarked'] = 0
df.loc[df.Embarked == 'Q', 'Embarked'] = 1
df.loc[df.Embarked == 'S', 'Embarked'] = 2


df = df.loc[:, ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]

df.fillna(0, inplace = True)

X_test=df.as_matrix()



##############
# preprocess
##############

scaler = preprocessing.StandardScaler().fit(X)
X=scaler.transform(X)


################
# SVM
#################
mittel = 0
for g in np.arange(0.002,0.02, 0.001):
	for eC in [np.power(2.,k) for k in np.arange(-4,6)]:
		my_svm = svm.SVC(kernel='rbf', C=eC, gamma = g)
		my_svm.fit(X, y)
		sc = cross_val_score(my_svm, X, y, cv=5)
		if (np.mean(sc)/my_svm.score(X,y) > 0.95) and (np.mean(sc)/my_svm.score(X,y) <1.05) and (np.mean(sc)>0.77):
			if np.mean(sc) > mittel:
				mittel = np.mean(sc)
				cc = eC
				gg =g

			print("mu={}, std={},c={}, gama={}".format(np.mean(sc), np.std(sc),eC, g ))
			# print('             E_in = {}'.format(my_svm.score(X,y)))
		# print("gamma={}".format(g))

print(cc, gg)
my_svm = svm.SVC(kernel='rbf', C=cc, gamma = gg)
my_svm.fit(X, y)
X_test = scaler.transform(X_test)
y_pred = my_svm.predict(X_test)

##############
# write to csv
##############
passI  = df['PassengerId'].tolist()
daten = {'PassengerId': passI, 'Survived': y_pred}
cols = ['PassengerId', 'Survived']

ddf = pd.DataFrame(daten, columns = cols)
ddf.to_csv('sol.csv', index= False)