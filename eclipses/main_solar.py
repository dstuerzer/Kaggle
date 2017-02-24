import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#import count_days as cd

dic_months = {'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June': 6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

def mak_2_dig(x):
#transforms all integers bewteen 0 and 99 into a 2-digit string
    if x<10:
        return '0'+str(x)
    else:
        return str(x)


def transf_date(s):
    s = s.split()
    s[1] = str(dic_months[s[1]])
    return s[0]+':'+s[1]+':'+s[2]

def date_to_jd(date):
#transform date (either julian or gregorian) into a julian day number
    y = date[0]
    mo = date[1]
    d = date[2]
    h = date[3]
    mn = date[4]
    s = date[5]

    if date[:3] >= [1582,10,15]:
        #gregorian date
        return 367*y - (7*(y+int((mo+9)/12)))//4 - (3*(int((y+(mo-9)/7)/100)+1))//4+(275*mo)//9+d+1721028.5+h/24+mn/(24*60)+s/86400
    elif date[:3] <= [1582,10,4]:
        #julian date
        return 367*y - (7*(y+5001+int((mo-9)/7)))//4+(275*mo)//9+d+1729776.5+h/24+mn/(24*60)+s/86400

def jd_to_date(jd):
    Z = int(jd+0.5)
    F = (jd+0.5)%1
    if Z < 2299161:
        A = Z
    else:
        g = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + g - g//4 

    B = A + 1524
    C = int((B-122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B-D) / 30.6001)
 
    d = B - D - int(30.6001*E) + F
    if E<14:
        mo = E-1
    else:
        mo = E-13    

    if mo >2:
        y = C- 4716
    else:
        y = C - 4715
    
    return str(y)+'-'+mak_2_dig(mo)+'-'+mak_2_dig(int(d))
    
    
    
def create_training_set(x, L):
    # L... number of past eclipses used for prediction of next one
    X = []
    y = []
    for j in range(len(x)-L):
        X.append(x[j:j+L])
        y.append(x[j+L])
    return X, y
    
    
def soft(x):
    y = []
    for s in x:
        if np.absolute(s)>2:
            y.append(s)
        else:
            y.append(0)    
    return np.array(y)
    
    


#date separating observed data and future/predictions
date_sep = date_to_jd([2017, 3, 1, 0, 0,0])
df = pd.read_csv("data/solar.csv")

df = df.loc[:, ['Calendar Date', 'Eclipse Time', 'Eclipse Type']]
#combine Date and Time to Time JD
df['Calendar Date'] = df['Calendar Date'].apply(lambda x:transf_date(x))
df['Time'] = df.loc[:,['Calendar Date', 'Eclipse Time']].apply(lambda x: x[0]+':'+x[1], axis = 1) 
df = df.drop(['Calendar Date', 'Eclipse Time'], axis=1)
df['Time JD']=df['Time'].apply(lambda x : date_to_jd([int(j) for j in x.split(':')]))
del df['Time']

df = df[df['Eclipse Type'].str[0] != 'P']     #exclude partial eclipses
##possibly exclude data prior to date
date_start = date_to_jd([1800, 1, 1, 0, 0, 0])   #only use data after this date
df = df[df['Time JD'] > date_start]

t_between = df['Time JD'].diff().tolist()   #count days between consecutive eclipses
t_between = t_between[1:-1] #time (days) between consecutive eclipses

######################
##### PLOTS ##########
######################


#f = plt.figure(figsize = (50,10))
#plt.hist(t_between,int(max(t_between))+1)
#print("ratios are (in 5ths of the shortest): 6, 11, 12, 17, 23")

      
#############################################
##### Machine Learning Predictions ##########
#############################################


t_between = [int(j) for j in t_between]
# separate data in past and future       
df_before = df[df['Time JD']<=date_sep]
df_after = df[df['Time JD']>date_sep]
dates_before = df_before['Time JD'].tolist()
dates_after = df_after['Time JD'].tolist()
diff_before = [j for j in t_between[:len(dates_before)-1]]  #recorded differences between past ecl.
diff_after =  [j for j in t_between[len(dates_before)-1:]]   #differences betw. future ecl. 2b predicted

L = len(diff_before)//20    #every eclipse is predicted from the past L eclipses
print(L)

X, y = create_training_set(diff_before, L) #initialize training set
#split again into training and validation set:
p = 0.15 #fraction used for validation
X_val = X[int((1-p)*len(X)):]
y_val = y[int((1-p)*len(y)):]

X_train = X[:int((1-p)*len(X))]
y_train = y[:int((1-p)*len(y))]

print('training set created')
# learning algorithm
#lrn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,20), learning_rate = 'adaptive')
lrn = SVC(kernel = 'linear', C=1)
#lrn = RandomForestClassifier(100)
lrn.fit(X_train, y_train)
print('MLA fitted')

c0 = 0
c1 = 0

s = lrn.predict(X_val)

for i in range(len(y_val)):
    if np.absolute(y_val[i] - s[i]) >=1:
        c0 += 1
        if np.absolute(y_val[i] - s[i]) >1:
            c1+=1

#print("Validation Error at {}%".format((c0*100)//len(y_val)))
print("\nPredictions at most one day off: {}%".format((c1*100)//len(y_val)))
print("Tested on {} samples".format(len(y_val)))
print("\n\n")
##################################################
######## predictions of the next eclipses ########
##################################################


N_future = 5   #number of predictions into the future

# optionally train the model again on the 'entire' past 
lrn.fit(X, y)


xx = np.array(X[-1])
xx = np.roll(xx,-1)
xx[-1] = y[-1]
y_pred = []
d_last = dates_before[-1]

for i in range(N_future):
    yy = lrn.predict([xx])
    y_pred.append(yy[0])
    xx = np.roll(xx,-1)
    xx[-1] = yy


print(y_pred)
print(diff_after[:N_future])

days_pred = [d_last + i for i in np.cumsum(y_pred)]
print("Date (predicted)   Date (calculated/true)")
for i in range(len(days_pred)):
    print("  "+jd_to_date(days_pred[i])+"          "+jd_to_date(dates_after[i]))
print("avg. error in days: {}".format((np.linalg.norm(np.array(y_pred)-np.array(diff_after[:N_future]),1))/N_future))
print("number of wrong predictions: {}%".format((np.linalg.norm(soft(np.array(y_pred)-np.array(diff_after[:N_future])),0)*100)//N_future))
