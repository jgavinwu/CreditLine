
# coding: utf-8

# In[14]:

get_ipython().magic(u'matplotlib inline')
import os
import re
import json, codecs
import itertools
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime, date, time, timedelta
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import scale
from sklearn import (manifold, decomposition, ensemble, discriminant_analysis, random_projection)
from dateutil.parser import parse
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV
# from sklearn import model_selection


# In[3]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[4]:

os.chdir('/project/')
sam = pd.read_csv('./Data/cli_agg_v2.csv', dtype=object)
# sam.drop(['SCOREINDICATOR', 'R6195_AGE'], axis=1, inplace=True)
# sam.shape


# In[5]:

sam.drop(['SCOREINDICATOR', 'R6195_AGE'], axis=1, inplace=True)
sam.shape
# col = pd.to_numeric(sam['R6004_NEWESTTRADEAGE_MAX'])
# col = col[col<9999]
# sorted(col)[-50:]


# In[6]:

sam_sum = sam.describe()


# In[7]:

sam_prop = 0.2
sam2 = sam.ix[pd.to_numeric(sam['RANDOMNUM']) <= sam_prop, :]
sam2.shape


# In[8]:

indata = sam2
out_data = pd.DataFrame([])
columns_in = sam2.columns.tolist()
columns_excl = ['CUSTOMERSEGMENT', 'DWACCTID', 'CUSTOMERNUMBER', 'RANDOMNUM', 'FIRSTELIGIBLEDATE']
all_cat = []
all_con = []
for col_name in columns_in:
    print col_name
    col_out = indata[col_name]
    if col_name not in columns_excl:
        if (sam_sum.ix['unique',col_name] > 20):
            out_data[col_name] = pd.to_numeric(col_out)
            all_con.append(col_name)
        if (sam_sum.ix['unique',col_name] <= 20):
#             out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=col_out.unique()))
            out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=col_out[~pd.isnull(col_out)].unique()))
            all_cat.append(col_name)
    else: out_data[col_name] = col_out


# In[9]:

# out_data['CUSTOMERSEGMENT'] = pd.Series(pd.Categorical([v for v in out_data['CUSTOMERSEGMENT']],categories=(out_data['CUSTOMERSEGMENT']).unique()))
out_data['CUSTOMERSEGMENT'] = pd.Series(pd.Categorical([v for v in out_data['CUSTOMERSEGMENT']],categories=(out_data['CUSTOMERSEGMENT'])[~pd.isnull(out_data['CUSTOMERSEGMENT'])].unique()))


# In[10]:

target_raw = pd.read_csv('./Data/cli_maxcd.csv', dtype=object)
target_raw.drop('RANDOMNUM', axis=1, inplace=True)


# In[11]:

out_data_wtarget = out_data.set_index('DWACCTID').join(target_raw.set_index('DWACCTID'))
train_prop = 0.8
train = (out_data_wtarget.ix[pd.to_numeric(out_data_wtarget['RANDOMNUM']) <= sam_prop*train_prop,:]).copy()
test = (out_data_wtarget.ix[pd.to_numeric(out_data_wtarget['RANDOMNUM']) > sam_prop*train_prop,:]).copy()


# In[12]:

target = pd.to_numeric(train['MAX_CYCLEDUE'])
train.drop(['CUSTOMERNUMBER', 'RANDOMNUM', 'FIRSTELIGIBLEDATE', 'MAX_CYCLEDUE'], axis=1, inplace=True)
dmat = pd.get_dummies(train, prefix_sep='__', drop_first=True)


# In[21]:

xgdmat = xgb.DMatrix(dmat, target)
# xgb_model = xgb.XGBRegressor().fit(dmat, target)


# In[ ]:

targett = pd.to_numeric(test['MAX_CYCLEDUE'])
test.drop(['CUSTOMERNUMBER', 'RANDOMNUM', 'FIRSTELIGIBLEDATE', 'MAX_CYCLEDUE'], axis=1, inplace=True)
dmatt = pd.get_dummies(test, prefix_sep='__', drop_first=True)


# In[ ]:

pred = xgb_model.predict(dmatt)
len(pred)


# In[ ]:

scores = test.filter(regex=('SCORE.*')).copy()
test_out = pd.concat([targett, scores], axis=1)
test_out['PRED'] = pred


# In[ ]:

test_out.to_csv('./Results/fitted_notuning_20pct.txt', index=False)


# In[ ]:

wt = xgb_model.booster().get_score(importance_type='weight')
wt2 = pd.DataFrame(wt.items(), columns=['Feature', 'Weight'])
print_full(wt2.sort_values(['Weight'], ascending=False))


# In[ ]:

scores.columns


# In[15]:

grid = { 'eta' : [0.1, 0.5, 1], 'n_estimators' : [40, 80, 120], 'max_depth' : [10, 20, 40], 'min_child_weight' : [1, 5, 10], 'gamma' : [0, 0.5, 1, 5], 'subsample' : [0.5, 1], 'colsample_bytree': [0.5, 1]        
}

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

param_grid = expand_grid(grid)
grid_len = param_grid.shape[0]


# In[ ]:

seq = np.random.choice(range(grid_len), grid_len, replace=False)
for i in seq:
    params = {     'eta' : param_grid['eta'][i],     'n_estimators' : param_grid['n_estimators'][i],     'max_depth' : param_grid['max_depth'][i],     'min_child_weight' : param_grid['min_child_weight'][i],     'gamma' : param_grid['gamma'][i],     'subsample' : param_grid['subsample'][i],     'colsample_bytree' : param_grid['colsample_bytree'][i],     'objective' : 'reg:linear',     'seed' : 206
    }
    cv_xgb = xgb.cv(params = params, dtrain = xgdmat, num_boost_round = 200, nfold = 5,
                    metrics = ['rmse'],
                    early_stopping_rounds = 20)
    all_cv = cv_xgb['test-rmse-mean']
    best = np.argmin(all_cv)
    to_write = pd.DataFrame(param_grid.loc[i].tolist() + [len(all_cv), best, all_cv[best]]).transpose()
    to_write.to_csv('./Results/all_mse.txt', header=False, index=False, mode='a')

all_mse = pd.read_csv('./Results/all_mse.txt', names=['colsample_bytree', 'min_child_weight', 'n_estimators', 'subsample', 'eta', 'max_depth', 'gamma', 'path_len', 'best', 'best_cv'])
