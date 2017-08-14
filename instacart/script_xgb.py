#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:28:27 2017

@author: dominik
"""
#import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import random as rd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# read data
df_opp = pd.read_csv("data/order_products__prior.csv")
df_tra = pd.read_csv("data/order_products__train.csv")
df_ord = pd.read_csv("data/orders.csv")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



def F1(user, predicted_products, df_opp_user):
# evaluates the F1 error for a given training user and a prediction 'predicted_products'
    training_order_id = df_ord[(df_ord['user_id'] == user) & (df_ord['eval_set'] == 'train')].order_id.tolist()[0]
    prior_products = set(df_opp_user.loc[:, 'product_id'].tolist())
    training_products = set(df_tra.loc[df_tra['order_id'] == training_order_id, 'product_id'].tolist())
    prior_reorder = prior_products & training_products  # all reordered products in the training order
    predi_reorder = set(predicted_products)   
    TP = len(prior_reorder & predi_reorder)   
    if len(prior_reorder) == 0:
        recall = 0
    else:
        recall = TP / len(prior_reorder)
    if len(predi_reorder) == 0:
        precision = 0
    else:
        precision = TP / len(predi_reorder)   
    if precision + recall > 0:
        return 2*precision*recall/(precision + recall)
    else:
        return 0




def predict(user, THRESH, xgboost_classifier, tr_te):
# takes user_id, prob. threshold for log. reg., and the classifier.
# out: predicted products for user.    
    predicted_products = []
    
    df_ord_user = df_ord[df_ord['user_id'] == user ].copy()
    del df_ord_user['user_id']  # not needed any more
    df_ord_user_copy = df_ord_user.copy()
          
    prior_orders = df_ord_user.loc[df_ord_user['eval_set'] == 'prior', 'order_id'].tolist()
    df_opp_user = df_opp[df_opp['order_id'].isin(prior_orders)].copy()
    n_orders = len(prior_orders)


    s = df_opp_user.product_id.value_counts()
    s = s[s>1]   # only keep products that were ordered more than once
    # we distinguish 3 cases: user has 3 or 4 prior orders, or >4 orders
    if n_orders <= 4:
        predicted_products = s.index.tolist()   # we're done already
    else:
        relevant_products = s.index.tolist()   # we loose a lot of garbage here!  
        for prod in relevant_products:
            df_ord_user = df_ord_user_copy.copy()

            # find all orders for 'prod'
            orders_product = df_opp_user.loc[df_opp_user['product_id'] == prod, 'order_id'].tolist()
            # add an indicator column
            df_ord_user['product'] = 0
            df_ord_user.loc[df_ord_user['order_id'].isin(orders_product), 'product']  = 1
        
            #we actually don't know much before the first order, so we drop all rows until
            # the first order (always remember the other way, I should try it too!!):
            df_ord_user = df_ord_user.loc[df_ord_user[df_ord_user['product'] == 1].index[0]: , : ].copy()
        
        
            #add cum_days since prior product order. 
            
            df_ord_user['cum_days'] = df_ord_user.groupby(df_ord_user['product'].cumsum().shift().fillna(0)).days_since_prior_order.cumsum()
            df_ord_user['cum_orders'] = df_ord_user.groupby(df_ord_user['product'].cumsum().shift().fillna(0)).days_since_prior_order.cumcount()
            df_ord_user['cum_orders'] = df_ord_user['cum_orders'].map(lambda x:x+1)
        
            # let's mirror some of the time-local activity for the product
            l_streak = min(5, n_orders)
            df_ord_user['ones'] = 1
            
            df_ord_user['sum_prod'] = df_ord_user['product'].rolling(l_streak, min_periods = 1).sum()
            df_ord_user['sum_days'] = df_ord_user['days_since_prior_order'].rolling(l_streak, min_periods = 1).sum()
            df_ord_user['sum_ones'] = df_ord_user['ones'].rolling(l_streak, min_periods = 1).sum()
            
            
            df_ord_user.sum_prod.fillna(df_ord_user.sum_prod.iloc[1], inplace = True)
            df_ord_user.sum_days.fillna(df_ord_user.sum_days.iloc[1], inplace = True)
            df_ord_user.sum_ones.fillna(2, inplace = True)
            df_ord_user['prod_per_day'] = round(df_ord_user.sum_prod / df_ord_user.sum_days,3)
            df_ord_user['prod_per_ord'] = round(df_ord_user.sum_prod / df_ord_user.sum_ones,2)
                       
            del df_ord_user['sum_days']
            del df_ord_user['sum_ones']
            del df_ord_user['ones']
            del df_ord_user['sum_prod']     
        
            # drop further irrelevant columns
            del df_ord_user['order_id']
        
        
            # I believe that only the most recent orders actually give information 
            # about the next order -> only keep those
            n_relevant = 10   # keep last 10 orders.
                           
            df_ord_user.drop(df_ord_user.index[0], inplace = True)   # first row starts with an order, remove that since we do not have prior info
            df_ord_user = df_ord_user.iloc[-n_relevant:].copy()
        
            if df_ord_user['product'].sum() >= 1:
                # otherwise product is discarded anyway
                if df_ord_user['product'].sum() == df_ord_user.shape[0]:
                    predicted_products.append(prod)
                    # product has been ordered every time, we can't do better
                    # no predictor needed for that insight
                else:
                    features = ['order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'cum_days', 'cum_orders', 'prod_per_day', 'prod_per_ord']
                    df_ord_user['order_number'] = df_ord_user.order_number + 1 - min(df_ord_user.order_number.tolist()) # shift order numbers so that they all start counting at 1 again
                    X =          df_ord_user.loc[df_ord_user['eval_set'] == 'prior', features].as_matrix()
                    y = np.array(df_ord_user.loc[df_ord_user['eval_set'] == 'prior', 'product'].tolist())
                    x_test = df_ord_user.loc[df_ord_user['eval_set'] != 'prior', features].as_matrix()
                    
                    xgboost_classifier.fit(X,y)

                    if xgboost_classifier.predict_proba(x_test)[0,1] > THRESH:
                        predicted_products.append(prod)
    if tr_te == 'train':
        return predicted_products, F1(user, predicted_products, df_opp_user)
    else:
        return predicted_products, 0




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#pick random training users for model development.
num_users = 50
training_users =  rd.sample(df_ord[df_ord['eval_set'] == 'train'].loc[:, 'user_id'].tolist(), num_users)

# Specify XGBoost:
learn_rate = 0.03
n_estim = 40
max_dep = 1
min_child = 1

lrn = xgb.XGBClassifier(max_depth = max_dep, min_child_weight = min_child, learning_rate =  learn_rate, n_estimators = n_estim, objective = 'binary:logistic')
    
THRESH = 0.37   # probab. for log. regression
list_F1 = []

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# run training round to evaluate the F1-score
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for user in training_users:
    predicted_products, f_1 = predict(user, THRESH, lrn, 'train')
    list_F1.append(f_1)
    
mu = round(np.array(list_F1).mean(),3)
sig = round(np.array(list_F1).std(),3)
print("F1_mu : "+str(mu)+"\nF1_sig: "+str(sig)+"\n\n")
fp = open("F1_xores", "w")
fp.write(str(str(list_F1)))
fp.close()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# now we make the prediction on the test set

test_users = df_ord.loc[df_ord['eval_set'] == 'test', 'user_id'].tolist()

user_test_order_id = pd.Series(df_ord.order_id.values, index = df_ord.user_id).to_dict()
dic_result = {}
for user in test_users:
    predicted_products, f1 = predict(user, THRESH, lrn, 'test')
    if len(predicted_products) == 0:
        pre_pro = 'None'
    else:
        pre_pro = ' '.join(str(e) for e in list(predicted_products))
    dic_result[user_test_order_id[user]] = pre_pro


ddf = pd.DataFrame(list(dic_result.items()), columns=['test_order_id', 'product_id'])
ddf.sort_values('test_order_id', inplace = True)
ddf.to_csv('results.csv', header = ['order_id', 'products'], index = False)
