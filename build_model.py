
# coding: utf-8

# In[1]:

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


# In[2]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[9]:

os.chdir('/project/')
sam = pd.read_csv('./Data/cli_cma_joined_1pct.csv', dtype=object)
# sam_cma = pd.read_csv('./Data/cli_cma_1pct.csv', dtype=object)
# sam_all = pd.merge(sam, sam_cma, on=['CREDITCYCLEFACTMKEY'])
sam.shape


# In[10]:

sam.columns


# In[11]:

sam_sum = sam.describe()
print_full(sam_sum.transpose())


# In[ ]:

# all_col = sam.columns.str.upper()


# In[12]:

# import math
# nan_col_ind = [math.isnan(val) if isinstance(val, float) else False for val in sam_sum.ix['top',]]
nan_col_ind = (sam_sum.ix['count',] == 0)
nan_col = sam.columns[nan_col_ind]
nan_col
# sam.drop(nan_col, axis=1, inplace=True)


# In[13]:

oneval_col_ind = (sam_sum.ix['unique',] == 1)
oneval_col = sam.columns[oneval_col_ind]
oneval_col


# In[14]:

# import re
# [re.match('.*date.*', col, re.IGNORECASE) for col in all_col]
# matches = all_col.str.findall('.*date.*', re.IGNORECASE)
# matches.str.get(0).tolist()
date_col_ind = sam.columns.str.contains('DATE')
date_col = sam.columns[date_col_ind]
date_col


# In[15]:

date_col2 = date_col.tolist()
date_col2.remove('CYCLEDATE')
date_col2


# In[16]:

to_drop = nan_col.tolist() + oneval_col.tolist() + date_col2
sam2 = sam.drop(to_drop, axis=1)
sam2.shape


# In[23]:

sam2['INCRTYPE'] = sam2['INCRTYPE'].fillna('Unknown')


# In[31]:

sam2['CURRENTDAYSPASTDUE'].unique()


# In[25]:

indata = sam2
out_data = pd.DataFrame([])
columns_in = sam2.columns.tolist()
columns_excl = ['CYCLEDATE', 'CUSTOMERSEGMENT', 'DWACCTID', 'CUSTOMERNUMBER', 'RANDOMNUM', 'CREDITCYCLEFACTMKEY']
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
            out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=col_out.unique()))
            all_cat.append(col_name)
    else: out_data[col_name] = col_out


# In[32]:

out_data['CUSTOMERSEGMENT'] = pd.Series(pd.Categorical([v for v in out_data['CUSTOMERSEGMENT']],categories=out_data['CUSTOMERSEGMENT'].unique()))


# In[49]:

out = out_data.sort_values(['CUSTOMERNUMBER', 'CYCLEDATE']).groupby(['CUSTOMERNUMBER']).last().reset_index()


# In[100]:

con_ft_raw = out_data[all_con + ['CUSTOMERNUMBER']]
min_ft = con_ft_raw.groupby('CUSTOMERNUMBER', group_keys=False).apply(lambda x:x.min())
max_ft = con_ft_raw.groupby('CUSTOMERNUMBER').apply(lambda x:x.max())
con_ft = pd.merge(min_ft, max_ft, on='CUSTOMERNUMBER', suffixes=('_min', '_max'))


# In[104]:

last_ft = out.drop(['DWACCTID', 'CYCLEDATE', 'RANDOMNUM', 'CREDITCYCLEFACTMKEY', 'CYCLEDUE'], axis=1)
last_ft['CUSTOMERNUMBER'] = last_ft['CUSTOMERNUMBER'].map(float)
all_ft = pd.merge(con_ft, last_ft, on='CUSTOMERNUMBER', suffixes=('', '_last'))


# In[108]:

target_raw = out_data[['CUSTOMERNUMBER', 'CYCLEDUE']]
target = target_raw.groupby('CUSTOMERNUMBER').apply(lambda x:x.max())


# In[111]:

dmat = pd.get_dummies(all_ft, prefix_sep='__')


# In[115]:

dmat.drop(['CUSTOMERNUMBER'], axis=1, inplace=True)
target.drop(['CUSTOMERNUMBER'], axis=1, inplace=True)


# In[116]:

xgb_model = xgb.XGBRegressor().fit(dmat, target)


# In[123]:

wt = xgb_model.booster().get_score(importance_type='weight')


# In[128]:

wt2 = pd.DataFrame(wt.items(), columns=['Feature', 'Weight'])
print_full(wt2.sort_values(['Weight'], ascending=False))


# In[131]:

dmat.shape


# In[8]:

one_cust_id = sam_all['DWACCTID_x'].sample(1)
one_cust = sam_all.ix[sam_all['DWACCTID_x'].isin(one_cust_id),:]
print_full(one_cust)

