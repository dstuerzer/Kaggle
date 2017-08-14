#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:28:27 2017

@author: dominik
"""
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import random as rd


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# read data
df_ais = pd.read_csv("data/aisles.csv")
df_dep = pd.read_csv("data/departments.csv")
df_opp = pd.read_csv("data/order_products__prior.csv")
df_tra = pd.read_csv("data/order_products__train.csv")
df_pro = pd.read_csv("data/products.csv")
df_ord = pd.read_csv("data/orders.csv")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#get user ids for test users:
users = df_ord[df_ord['eval_set'] == 'test'].loc[:, 'user_id'].tolist()
df_ord = df_ord[df_ord['user_id'].isin(users)].copy()
# now we only keep at most the last 10 orders
n_prior = 10
df_ord = df_ord.sort_values(['user_id', 'order_number'], ascending=[1,0]).groupby('user_id').head(n_prior + 1).copy()
# split dataframe into prior and test data.
df_ord_test = df_ord[df_ord['eval_set'] == 'test'].copy()
df_ord_prior = df_ord[df_ord['eval_set'] == 'prior'].copy()

prior_order_ids = df_ord_prior.order_id.tolist()
df_opp = df_opp[df_opp['order_id'].isin(prior_order_ids)].copy()
# prior order_ids per user
dic_temp = pd.Series(df_ord_prior.user_id.values, index = df_ord_prior.order_id).to_dict()
#d_user_prior_order = {k: list(v) for k,v in df_ord_prior.groupby('user_id')['order_id']}
df_opp['user_id'] = df_opp['order_id'].map(dic_temp)

# let's use df_opp to count the freq of products

s = df_opp.groupby('user_id')['product_id'].value_counts()
# contains products per customer, sorted by frequencies.
q = df_opp.groupby('user_id')['order_id'].nunique()
s /= q   # relative frequencies

thresh = 0.35    # threshold for reorder probability
s = s[s > thresh]

# dictionary user:list(pop products) : 
df = s.reset_index(name = 'ctt')
d_user_order = pd.Series(df_ord_test.order_id.values, index = df_ord_test.user_id).to_dict()
df['test_order_id'] = df.user_id.map(d_user_order)

d = df.groupby('test_order_id').product_id.apply(lambda x: ' '.join(str(e) for e in list(x))).reset_index()

# here, only users for which we predict at least one product are mentioned. 
# the 'none'-users are dropped in s[s>thresh]. We have to add them again:

users_non_none = set(df.user_id.tolist())
users_none = set(users) - users_non_none
dic_temp = {}
for u in users_none:
    dic_temp[d_user_order[u]] = 'None'

df_additional = pd.DataFrame(list(dic_temp.items()), columns=['test_order_id', 'product_id'])
d = d.append(df_additional).copy()
d.sort_values('test_order_id', inplace = True)
d.to_csv('results.csv', header = ['order_id', 'products'], index = False)
