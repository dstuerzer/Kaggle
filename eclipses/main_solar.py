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
    
    return y, mo, int(d)
    
    
    
def create_training_set(x):
    L = 150     #number of past eclipses used for prediction of next one
    X = []
    y = []
    for j in range(len(x)-L):
        X.append(x[j:j+L])
        y.append(x[j+L])
    return X, y
    
    
    
    
    


#date separating observed data and future/predictions
date_sep = date_to_jd([2022, 3, 1, 0, 0,0])

df = pd.read_csv("data/solar.csv")
df = df.loc[:, ['Calendar Date', 'Eclipse Time', 'Eclipse Type']]
#combine Date and Time to Time JD
df['Calendar Date'] = df['Calendar Date'].apply(lambda x:transf_date(x))
df['Time'] = df.loc[:,['Calendar Date', 'Eclipse Time']].apply(lambda x: x[0]+':'+x[1], axis = 1) 
df = df.drop(['Calendar Date', 'Eclipse Time'], axis=1)
df['Time JD']=df['Time'].apply(lambda x : date_to_jd([int(j) for j in x.split(':')]))
del df['Time']

######################
##### PLOTS ##########
######################

df = df[df['Eclipse Type'].str[0] != 'P']     #exclude partial eclipses
t_between = df['Time JD'].diff().tolist()   #count days between consecutive eclipses
t_between = t_between[1:-1] #time (days) between consecutive eclipses
#f = plt.figure(figsize = (50,10))
plt.hist(t_between,int(max(t_between))+1)
print("ratios are (in 5ths of the shortest): 6, 11, 12, 17, 23")


#############################################
##### Machine Learning Predictions ##########
#############################################


df_before = df[df['Time JD']<=date_sep]
df_after = df[df['Time JD']>date_sep]
dates_before = df_before['Time JD'].tolist()
dates_after = df_after['Time JD'].tolist()
diff_before = [int(j) for j in t_between[:len(dates_before)-1]]  #recorded differences between past ecl.
diff_after =  [int(j) for j in t_between[len(dates_before)-1:]]   #differences betw. future ecl. 2b predicted

N_future = 10   #number of predictions into the future

X, y = create_training_set(diff_before) #initialize training set

# learning algorithm
#lrn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20), learning_rate = 'adaptive')
#lrn = SVC(kernel = 'linear', C=1000)
lrn = RandomForestClassifier(1000)
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
print([jd_to_date(i) for i in days_pred])
print([jd_to_date(i) for i in dates_after[:N_future]])
