#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:28:27 2017

@author: dominik
"""
#import matplotlib.pyplot as plt 
import numpy as np 
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
    FN = len(prior_reorder - predi_reorder)
    FP = len(predi_reorder - prior_reorder)
    if (len(prior_reorder) == 0) and (len(predi_reorder) == 0):
        return 1
    else:
        return 2*TP/(2*TP + FN + FP)




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
            
            prod_in_last_5 = df_ord_user.loc[df_ord_user['order_number'] .isin(range(1,6)), 'product'].sum()
                            
            # make guesses of orders before the 1. order:
            if prod_in_last_5 >= 2:
                df_ord_user.loc[-1] = [0, 'pre-prior', 0,0,0, np.NaN, 1]
                i_before = 1
            else:
                df_ord_user.loc[-1] = [0, 'pre-prior', -1, 0,0, np.NaN, 1]
                df_ord_user.loc[-2] = [0, 'pre-prior',  0, 0,0, np.NaN, 0]
                i_before = 2
            df_ord_user.sort_values('order_number', inplace = True)
            # fill NaN days since prior product:
            df_ord_user.loc[df_ord_user['order_number'].isin(range(2-i_before, 2)), 'days_since_prior_order'] = np.mean(df_ord_user.loc[df_ord_user['order_number'].isin(range(2,8)), 'days_since_prior_order'].tolist())


            #add cum_days since prior product order. 
            
            df_ord_user['cum_days'] = df_ord_user.groupby(df_ord_user['product'].cumsum().shift().fillna(0)).days_since_prior_order.cumsum()
            df_ord_user['cum_orders'] = df_ord_user.groupby(df_ord_user['product'].cumsum().shift().fillna(0)).days_since_prior_order.cumcount()
            df_ord_user['cum_orders'] = df_ord_user['cum_orders'].map(lambda x:x+1)
        
            # let's mirror some of the time-local activity for the product
            l_streak = min(5, n_orders)
            df_ord_user['ones'] = 1
            df_temp = df_ord_user.copy()
            df_ord_user = df_ord_user.iloc[:-1].copy()
            df_ord_user['sum_prod'] = df_ord_user['product'].rolling(l_streak, min_periods = 2).sum()
            df_ord_user['sum_days'] = df_ord_user['days_since_prior_order'].rolling(l_streak, min_periods = 1).sum()
            df_ord_user['sum_ones'] = df_ord_user['ones'].rolling(l_streak, min_periods = 2).sum()
            
            
            df_ord_user.sum_prod.fillna(df_ord_user.sum_prod.iloc[1], inplace = True)
            df_ord_user.sum_days.fillna(df_ord_user.sum_days.iloc[1], inplace = True)
            df_ord_user.sum_ones.fillna(2, inplace = True)
            df_ord_user['prod_per_day'] = round(df_ord_user.sum_prod / df_ord_user.sum_days,3)
            df_ord_user['prod_per_ord'] = round(df_ord_user.sum_prod / df_ord_user.sum_ones,2)
            df_temp.loc[df_temp['eval_set'] == 'prior', 'prod_per_day'] = df_ord_user.loc[:, 'prod_per_day']
            df_temp.loc[df_temp['eval_set'] == 'prior', 'prod_per_ord'] = df_ord_user.loc[:, 'prod_per_ord']
            df_temp.loc[df_temp['eval_set'] != 'prior', 'prod_per_day'] = df_ord_user.loc[df_ord_user['order_number'].idxmax() , 'prod_per_day']
            df_temp.loc[df_temp['eval_set'] != 'prior', 'prod_per_ord'] = df_ord_user.loc[df_ord_user['order_number'].idxmax() , 'prod_per_ord']
       
            df_ord_user = df_temp.loc[df_temp['order_number'] >= 1].copy()
            # drop further irrelevant columns
            del df_ord_user['order_id']
            # prod_per_day and prod_per_ord are biased for first entry
            df_ord_user.loc[df_ord_user['order_number'] == 1, ['prod_per_day', 'prod_per_ord']] = df_ord_user.loc[df_ord_user['order_number'] == 2, ['prod_per_day', 'prod_per_ord']].values.tolist()[0]
            
            del df_ord_user['ones']
        

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
# Specify XGBoost:
learn_rate = 0.03
n_estim = 40
max_dep = 1
min_child = 1

lrn = xgb.XGBClassifier(max_depth = max_dep, min_child_weight = min_child, learning_rate =  learn_rate, n_estimators = n_estim, objective = 'binary:logistic')
    
THRESH = 0.4   # probab. for log. regression

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# now we make the prediction on the test set

test_users = df_ord.loc[df_ord['eval_set'] == 'test', 'user_id'].tolist()

user_test_order_id = pd.Series(df_ord.order_id.values, index = df_ord.user_id).to_dict()
dic_result = {}
ct = 0
for user in test_users:
    ct += 1
    print(ct)
    predicted_products, f1 = predict(user, THRESH, lrn, 'test')
    if len(predicted_products) == 0:
        pre_pro = 'None'
    else:
        pre_pro = ' '.join(str(e) for e in list(predicted_products))
    dic_result[user_test_order_id[user]] = pre_pro
    if ct >10:
        break



ddf = pd.DataFrame(list(dic_result.items()), columns=['test_order_id', 'product_id'])
ddf.sort_values('test_order_id', inplace = True)
ddf.to_csv('results_new.csv', header = ['order_id', 'products'], index = False)
