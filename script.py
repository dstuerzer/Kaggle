import time
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
df.loc[df.Embarked == 'C', 'Embarked'] = 0
df.loc[df.Embarked == 'Q', 'Embarked'] = 1
df.loc[df.Embarked == 'S', 'Embarked'] = 2

del df['Name']
del df['Ticket']
del df['Cabin']


df['FamilyMembers'] = df.SibSp.fillna(0) + df.Parch.fillna(0)
del df['SibSp']
del df['Parch']
del df['PassengerId']


#############
# svr to predict missing ages
###################


df_train_age = df[pd.isnull(df['Age']) == False].dropna(how='any')
y_age_tr = df_train_age.Age.tolist()
del df_train_age['Age']
X_train_age = df_train_age.as_matrix()
scaler_age = preprocessing.StandardScaler().fit(X_train_age)
X_train_age = scaler_age.transform(X_train_age)

svr_age = svm.SVR(kernel='rbf')
svr_age.fit(X_train_age, y_age_tr)
df_age_nan = df[pd.isnull(df['Age'])]
del df_age_nan['Age']
df_age_nan = df_age_nan.dropna(how='any')

Xpred = scaler_age.transform(df_age_nan.as_matrix())
y_age = svr_age.predict(Xpred)

df.loc[pd.isnull(df['Age']), 'Age'] = y_age

#############
# svm to predict missing ports
###################


df_train_embarked = df[pd.isnull(df['Embarked']) == False].dropna(how='any')

y_train_embarked = df_train_embarked.Embarked.tolist()
del df_train_embarked['Embarked']
X_train_embarked=df_train_embarked.as_matrix()

scaler_embarked = preprocessing.StandardScaler().fit(X_train_embarked)
X_train_embarked=scaler_embarked.transform(X_train_embarked)

svm_embark = svm.SVC(kernel='rbf')
svm_embark.fit(X_train_embarked, y_train_embarked)
dfe = df[pd.isnull(df['Embarked'])]
del dfe['Embarked']
dfe = dfe.dropna(how='any')

Xpred3 = scaler_embarked.transform(dfe.as_matrix())
y_embark = svm_embark.predict(Xpred3)

df.loc[pd.isnull(df['Embarked']), 'Embarked'] = y_embark

# TRAINNG DATA:

y_train = df['Survived'].tolist()
dfff = df
del dfff['Survived']
X_train = dfff.as_matrix()

###########
# scaler and age predictor: train -> test
###########
y_prp = df.Age.tolist()
df_prp=df.loc[:, ['Pclass', 'Sex', 'Fare', 'Embarked', 'FamilyMembers']]
X_prp = df_prp.as_matrix()
scaler_prp = preprocessing.StandardScaler().fit(X_prp)	# <<<<<
X_prp = scaler_prp.transform(X_prp)
svr_prp = svm.SVR(kernel='rbf')
svr_prp.fit(X_prp, y_prp)	# <<<<<


y_prp2 = df.Fare.tolist()
df_prp2=df.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked', 'FamilyMembers']]
X_prp2 = df_prp2.as_matrix()
scaler_prp2 = preprocessing.StandardScaler().fit(X_prp2)	# <<<<<
X_prp2 = scaler_prp2.transform(X_prp2)
svr_prp2 = svm.SVR(kernel='rbf')
svr_prp2.fit(X_prp2, y_prp2)	# <<<<<


###################
# prpare test-data
###################

rel_path = "Data/test.csv"
abs_file_path = os.path.join(script_dir, rel_path)
df_test =pd.read_csv(abs_file_path)

df_test.loc[df_test.Sex == 'male', 'Sex'] = 1
df_test.loc[df_test.Sex == 'female', 'Sex'] = 0
df_test.loc[df_test.Embarked == 'C', 'Embarked'] = 0
df_test.loc[df_test.Embarked == 'Q', 'Embarked'] = 1
df_test.loc[df_test.Embarked == 'S', 'Embarked'] = 2

del df_test['Name']
del df_test['Ticket']
del df_test['Cabin']
df_test['FamilyMembers'] = df_test.SibSp.fillna(0) + df_test.Parch.fillna(0)
del df_test['SibSp']
del df_test['Parch']


df_age_nan = df_test[pd.isnull(df_test['Age'])]
del df_age_nan['PassengerId']
del df_age_nan['Age']

X = scaler_prp.transform(df_age_nan.as_matrix())
y = svr_prp.predict(X)
df_test.loc[pd.isnull(df_test['Age']), 'Age'] = y


df_age_nan = df_test[pd.isnull(df_test['Fare'])]
del df_age_nan['PassengerId']
del df_age_nan['Fare']

X = scaler_prp2.transform(df_age_nan.as_matrix())
y = svr_prp2.predict(X)
df_test.loc[pd.isnull(df_test['Fare']), 'Fare'] = y





pass_id = df_test['PassengerId']	# <<<<<<<<<<
del df_test['PassengerId']

X_test = df_test.as_matrix()









# ##############
# # prediction based on X_train, y_train, and X_test
# ##############


scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)

del X, y


mittel = 0
for g in np.arange(0.01,0.05, 0.002):
	for eC in [np.power(2.,k) for k in np.arange(-2,7)]:
		my_svm = svm.SVC(kernel='rbf', C=eC, gamma = g)
		my_svm.fit(X_train, y_train)
		sc = cross_val_score(my_svm, X_train, y_train, cv=5)
		if (np.mean(sc)/my_svm.score(X_train, y_train) > 0.95) and (np.mean(sc)/my_svm.score(X_train, y_train) <1.05) and (np.mean(sc)>0.82):
			if np.mean(sc) > mittel:
				mittel = np.mean(sc)
				cc = eC
				gg =g
				print("    mu={}, std={},c={}, gama={}".format(np.mean(sc), np.std(sc),eC, g ))
				print('E_in = {}'.format(my_svm.score(X_train, y_train)))

print(cc, gg)
my_svm = svm.SVC(kernel='rbf', C=cc, gamma = gg)
my_svm.fit(X_train, y_train)
X_test = scaler.transform(X_test)
y_pred = my_svm.predict(X_test)

# ##############
# # write to csv
# ##############
daten = {'PassengerId': pass_id, 'Survived': y_pred}
cols = ['PassengerId', 'Survived']

ddf = pd.DataFrame(daten, columns = cols)
ddf.to_csv('sol.csv', index= False)