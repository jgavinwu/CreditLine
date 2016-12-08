import os
import re
import json, codecs
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime, date, time, timedelta
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import scale
from sklearn import (manifold, decomposition, ensemble, discriminant_analysis, random_projection)
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

y=datetime.strptime(x, '%Y-%m-%d-%H.%M.%S.000000')..strftime('%Y-%m-%d')
y > parse('2016-11-01')
os.chdir('/Users/gavin.wu/Python/Project/Orchard/Rodeo/orchard_plcc')
# Read data
train = pd.read_csv('orchard_plcc_train.csv', dtype=object)
test = pd.read_csv('orchard_plcc_test.csv', dtype=object)
# Read dictionary
data = []
with codecs.open('data_dict_orchard_plcc.txt','rU','utf-8') as f:
    for line in f:
       data.append(json.loads(line))
# Read schema
schema = pd.read_csv('final_schema_orchard_plcc.txt', header=None)
schema = pd.Series.tolist(schema[0])

# Preprocess
indata = train
out_data = pd.DataFrame([])
columns_in = indata.columns.tolist()
all_cat = []
all_con = []
for i in range(len(data)):
    print i
    col_name = data[i]['col_name']
    # default = np.nan if data[i]['default_val']=='NA' else data[i]['default_val']
    default = np.nan
    if (col_name in columns_in):
        col_out = indata[col_name]
        if (data[i]['data_type']=='con'):
            out_data[col_name] = pd.to_numeric(col_out)
            all_con.append(i)
        if (data[i]['data_type']=='cat'):
            out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=data[i]['snapshot']))
            all_cat.append(i)
    else: out_data[col_name] = default

out_data[out_data.columns[all_cat]]=out_data[out_data.columns[all_cat]].apply(lambda x:x.fillna(x.value_counts().index[0]))
out_data[out_data.columns[all_con]]=out_data[out_data.columns[all_con]].apply(lambda x:x.fillna(x.median()))
# out_data_dum = pd.get_dummies(out_data, prefix_sep='__', sparse=True)
out_data_dum = pd.get_dummies(out_data, prefix_sep='__')
out_data_dum = out_data_dum.to_sparse(fill_value=0)
out_data_dum_sc = scale(out_data_dum)

plcc_train = pd.DataFrame(out_data_dum_sc)[schema]
plcc_train_target = pd.to_numeric(train['PLCC_OFR_CONV'])
plcc_test = pd.DataFrame(out_data_dum_sc)[schema] # reusing the code block above
plcc_test_target = pd.to_numeric(test['PLCC_OFR_CONV'])

# Non-linear dimension reduction
# X = out_data_dum_sca.copy()
# n_neighbors = 20
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
# plt.scatter(X_iso[:,0], X_iso[:,1])

# Fit a single model
xgb_setup = XGBClassifier(
     learning_rate=1,
     n_estimators=100,
     max_depth=20,
     min_child_weight=5,
     gamma=5,
     subsample=1,
     colsample_bytree=0.4,
     objective='binary:logistic',
     nthread=4,
#     scale_pos_weight=1,
     missing=np.nan,
     seed=206)
xgb_model = xgb_setup.fit(plcc_train, plcc_train_target, early_stopping_rounds=10, eval_metric="auc", eval_set=[(plcc_test, plcc_test_target)])
# dtrain = xgb.DMatrix(cc, cct, missing=np.nan)
# dtest = xgb.DMatrix(cc_test, cc_testt, missing=np.nan)
pred_test = xgb_model.predict_proba(plcc_test)
pred_test1 = [x[1] for x in pred_test]
metrics.roc_auc_score(plcc_test_target, pred_test1)

# Grid search
grid = {
    'learning_rate': [0.4, 0.8]
}

gs = GridSearchCV(
    estimator=xgb_setup,
    param_grid=grid,
    scoring='roc_auc',
    n_jobs=2,
    iid=False,
    cv=5)

gs.fit(cc, cct)
pred_test = gs.predict_proba(cc_test)
pred_test1 = [x[1] for x in pred_test]
metrics.roc_auc_score(cc_testt, pred_test1)
