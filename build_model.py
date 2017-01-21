
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import os
import re
import json, codecs
import pandas as pd
import numpy as np
import itertools
from dateutil.parser import parse
from datetime import datetime, date, time, timedelta
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import scale
from sklearn import (manifold, decomposition, ensemble, discriminant_analysis, random_projection)
from dateutil.parser import parse
from collections import Counter
import plotly.plotly as plt
import plotly.offline as plo
from plotly.graph_objs import *
import plotly
plo.init_notebook_mode()
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV
# from sklearn import model_selection


# In[2]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[3]:

os.chdir('/project/')
sam = pd.read_csv('./Data/cli_cma_joined_1pct.csv', dtype=object)
# sam_cma = pd.read_csv('./Data/cli_cma_1pct.csv', dtype=object)
# sam_all = pd.merge(sam, sam_cma, on=['CREDITCYCLEFACTMKEY'])
#sam.shape


# In[4]:

sam_sum = sam.describe()
# print_full(sam_sum.transpose())


# In[ ]:

# all_col = sam.columns.str.upper()


# In[5]:

# import math
# nan_col_ind = [math.isnan(val) if isinstance(val, float) else False for val in sam_sum.ix['top',]]
nan_col_ind = (sam_sum.ix['count',] == 0)
nan_col = sam.columns[nan_col_ind]
nan_col
# sam.drop(nan_col, axis=1, inplace=True)


# In[6]:

oneval_col_ind = (sam_sum.ix['unique',] == 1)
oneval_col = sam.columns[oneval_col_ind]
oneval_col


# In[7]:

# import re
# [re.match('.*date.*', col, re.IGNORECASE) for col in all_col]
# matches = all_col.str.findall('.*date.*', re.IGNORECASE)
# matches.str.get(0).tolist()
date_col_ind = sam.columns.str.contains('DATE')
date_col = sam.columns[date_col_ind]
date_col


# In[8]:

date_col2 = date_col.tolist()
date_col2.remove('CYCLEDATE')
date_col2


# In[9]:

to_drop = nan_col.tolist() + oneval_col.tolist() + date_col2
sam2 = sam.drop(to_drop, axis=1)
sam2.shape


# In[10]:

sam2['INCRTYPE'] = sam2['INCRTYPE'].fillna('Unknown')


# In[ ]:

#sam2['CURRENTDAYSPASTDUE'].value_counts()


# In[11]:

indata = sam2
out_data = pd.DataFrame([])
columns_in = sam2.columns.tolist()
columns_excl = ['CYCLEDATE', 'CUSTOMERSEGMENT', 'DWACCTID', 'CUSTOMERNUMBER', 'RANDOMNUM', 'CREDITCYCLEFACTMKEY']
all_cat = []
all_con = []
dd = dict()
with open('./Data/data_dict.txt', 'a') as outfile:
	for col_name in columns_in:
#		print col_name
		col_out = indata[col_name]
		if col_name not in columns_excl:
			if (sam_sum.ix['unique',col_name] > 20):
				out_data[col_name] = pd.to_numeric(col_out)
				dd[col_name] = {'nam':col_name, 'typ':'con', 'rng':[out_data[col_name].min(),out_data[col_name].max()], 'dft':out_data[col_name].median(), 'cov':(out_data[col_name].std())/(out_data[col_name].mean())}
				json.dump(dd[col_name], outfile)
				outfile.write('\n')
				all_con.append(col_name)
			if (sam_sum.ix['unique',col_name] <= 20):
				out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=col_out[~pd.isnull(col_out)].unique()))
				cat_map = {k:v for v,k in enumerate(out_data[col_name].cat.categories)}
				dd[col_name] = {'nam':col_name, 'typ':'cat', 'rng':cat_map, 'dft':out_data[col_name].value_counts().index[0]}
				_  = [out_data[col_name].replace(cat, cat_map[cat], inplace=True) for cat in cat_map]
				dd[col_name]['cov']=(pd.to_numeric(out_data[col_name]).std())/(pd.to_numeric(out_data[col_name]).mean())
				json.dump(dd[col_name], outfile)
				outfile.write('\n')
				all_cat.append(col_name)
		else: out_data[col_name] = col_out

def load_dict(path):
    data_dict = []
    with codecs.open(path,'rU','utf-8') as f:
	for line in f:
	   data_dict.append(json.loads(line))
    return data_dict

ddict = load_dict('./Data/data_dict.txt')

df = sam2
df = pd.DataFrame(df)
if df.shape[1] == 1:
    df = df.T
#indata = df.replace('-1', np.nan).replace('Discharge NA', np.nan).replace('', np.nan)
indata = df
out_data = pd.DataFrame([])
columns_in = indata.columns.tolist()
for i in range(len(ddict)):
    col_name = ddict[i]['nam']
    default = np.nan if ddict[i]['dft']=='NA' else ddict[i]['dft']
    if (col_name in columns_in):
	col_out = indata[col_name]
	if (ddict[i]['typ']=='con'):
	    try:
		col_out = pd.to_numeric(col_out)
		col_out[(~pd.isnull(col_out)) & (~((col_out>=float(ddict[i]['rng'][0])) & (col_out<=float(ddict[i]['rng'][1]))))]=float(default)
	    except:
		col_out = float(default)
	    out_data[col_name] = col_out
	if (ddict[i]['typ']=='cat'):
	    col_out = pd.Series([v if ((pd.isnull(v)) | (v in [key for key in ddict[i]['rng']])) else default for v in col_out])
	    out_data[col_name] = pd.Series(pd.Categorical([v for v in col_out],categories=ddict[i]['rng']))
            _  = [out_data[col_name].replace(cat, ddict[i]['rng'][cat], inplace=True) for cat in ddict[i]['rng']]
    else: out_data[col_name] = default

out_data2 = pd.get_dummies(out_data, prefix_sep='__')
out_data2 = out_data2[dschema]
dm = xgb.DMatrix(out_data2, missing=np.nan)

#con_cov = {dd[col]['nam']:dd[col]['cov'] for col in all_con if dd[col]['cov']>20}
#cat_cov = {dd[col]['nam']:dd[col]['cov'] for col in all_cat if dd[col]['cov']>18}

out_data['CYCLEYD'] = [x[:4]+x[5:7] for x in out_data['CYCLEDATE']]
all_cust = out_data['DWACCTID'].unique()
np.random.shuffle(all_cust)
train_num = int(len(all_cust)*0.75)
train_cust = all_cust[:train_num]
test_cust = all_cust[train_num:]
train_ind = pd.DataFrame({'TID': np.concatenate((np.ones(len(train_cust)), np.zeros(len(test_cust)))), 'DWACCTID': np.concatenate((train_cust, test_cust))})
#train_ind = train_ind.set_index('DWACCTID')
all_data = out_data.merge(train_ind, on='DWACCTID', how='left', suffixes=('', '_ind'))
bf_data = all_data[all_data['CYCLEYD']<='20140905']
af_data = all_data[all_data['CYCLEYD']>'20140905']

all_cov = ['CMA3236', 'CLIINELIGIBLEIND']
cov_ft = (bf_data[all_cov + ['CYCLEYD', 'DWACCTID']]).set_index(['CYCLEYD', 'DWACCTID']).unstack(0)
cov_ft.columns = ['_'.join(col).strip() for col in cov_ft.columns.values]

def most_freq(col):
#	pd.DataFrame([col.name]).to_csv('./Data/cat_col_dict.txt')
	return Counter(col).most_common()[0][0] 

all_cat = all_cat + ['TID']
cat_func = np.array(['last'])
cat_l = len(all_cat)
cat_op = dict(zip(all_cat, np.repeat(cat_func[np.newaxis,:], cat_l, 0)))
cat_ft_raw = bf_data[all_cat + ['DWACCTID']]
cat_ft = cat_ft_raw.groupby('DWACCTID').agg(cat_op)
cat_ft.columns = ['_'.join(col).strip() for col in cat_ft.columns.values]

con_func = np.array(['min', 'max', 'last'])
con_l = len(all_con)
con_op = dict(zip(all_con, np.repeat(con_func[np.newaxis,:], con_l, 0)))
con_ft_raw = bf_data[all_con + ['DWACCTID']]
con_ft = con_ft_raw.groupby('DWACCTID').agg(con_op)
con_ft.columns = ['_'.join(col).strip() for col in con_ft.columns.values]

all_ft = cov_ft.join(cat_ft).join(con_ft)
train_ft = all_ft[all_ft['TID_last']==1]
test_ft = all_ft[all_ft['TID_last']==0]

all_tg = af_data[['DWACCTID', 'CYCLEDUE', 'TID']].groupby(['DWACCTID']).agg({'CYCLEDUE':'max', 'TID':'last'})
#all_tg.columns = ['_'.join(col).strip() for col in all_tg.columns.values]
train_tg = all_tg[all_tg['TID']==1]
test_tg = all_tg[all_tg['TID']==0]

train = train_ft.join(train_tg, how='inner')
test = test_ft.join(test_tg, how='inner')

#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import numpy as np
#import plotly.plotly as py
#from plotly.graph_objs import *

from plotly import tools

one_cust_id = out_data['DWACCTID'].sample(1)
one_cust = out_data.ix[out_data['DWACCTID'].isin(one_cust_id),:]
trace0 = Scatter(
        x=train['SCORE01_last'],
        y=train['SCORE02_last'],
        mode='markers'
                )
trace1 = Histogram(
    x=out_data['CMA3236'][out_data['CMA3236']<=9999991],
    histnorm='probability',
    opacity=0.75
                )
trace2 = Histogram(
    x=train['SCORE02_last'],
    histnorm='probability',
    opacity=0.75
                 )
data = [trace1]
layout = Layout(
    title='CMA3236',
#    barmode='overlay'
)
fig = Figure(data=data, layout=layout)
plo.plot(fig, filename = './Results/cma3236')
fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=600, width=600, title='Histograms Side-by-Side')
plo.plot(fig, filename='make-subplots')

trace0 = Scatter(
		x=one_cust['CYCLEYD'],
		y=one_cust['CMA3236'],
#		mode='markers',
                mode='lines+markers',
                marker = dict(
##                    size = 10,
                    color = 'rgba(152, 0, 0, .8)',
#                    color=train['CYCLEDUE'],
                    line = dict(
                        width = 2,
                        color = 'rgb(0, 0, 0)'
                        )
                    )
		)
layout = dict(title = 'CMA3236',
            yaxis = dict(zeroline = False),
            xaxis = dict(zeroline = False)
             )
data = [trace0]
fig = Figure(data=data, layout=layout)
plo.plot(fig, filename='./Results/cli_explore.html')


trace1 = Scatter(
    # x=[1, 2, 3, 4],
    # y=[16, 5, 11, 9]
		)

trace0 = Box(
        x=train['SCORE01_last'],
        name='SCORE01',
        marker=dict(
            color='rgb(0, 128, 128)'
            )
        )
trace1 = Box(
        x=train['SCORE02_last'],
        name='SCORE02',
        marker=dict(
            color='rgb(214, 12, 140)'
            )
        )
data = [trace0, trace1]
layout = Layout(
        title='Score Comparo',
        xaxis=dict(
        title='Score Value',
        zeroline=False
        ),
        boxmode='group'
        )
fig = Figure(data=data, layout=layout)
plo.plot(fig, filename='./Results/cli_explore.html')
#py.plot(data, filename = 'cli-explore')

dmat = xgb.DMatrix(train.ix[:, train.columns!='CYCLEDUE'], train.ix[:, 'CYCLEDUE'])
dmatt = xgb.DMatrix(test.ix[:, test.columns!='CYCLEDUE'], test.ix[:, 'CYCLEDUE'])
grid = { \
		'eta' : [0.1, 0.5, 1], \
		'n_estimators' : [40, 80, 120], \
		'max_depth' : [10, 20, 40], \
		'min_child_weight' : [1, 5, 10], \
		'gamma' : [0, 0.5, 1, 5], \
		'subsample' : [0.5, 1], \
		'colsample_bytree': [0.5, 1]        
		}
def expand_grid(data_dict):
	    rows = itertools.product(*data_dict.values())
	    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

param_grid = expand_grid(grid)
param_grid_list = param_grid.T.to_dict().values()
grid_len = param_grid.shape[0]
i = 0

params = { \
'eta' : param_grid['eta'][0], \
'n_estimators' : param_grid['n_estimators'][0], \
'max_depth' : param_grid['max_depth'][0], \
'min_child_weight' : param_grid['min_child_weight'][2], \
'gamma' : param_grid['gamma'][0], \
'subsample' : param_grid['subsample'][1], \
'colsample_bytree' : param_grid['colsample_bytree'][0], \
'objective' : 'reg:linear', \
'seed' : 206
}

params['nthread'] = 1
bst = xgb.train(params, dtrain=dmat)
pred = bst.predict(dmatt)

wt = bst.get_score(importance_type='weight')
wt2 = pd.DataFrame(wt.items(), columns=['Feature', 'Weight'])
print_full(wt2.sort_values(['Weight'], ascending=False))

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
kf = KFold(n_splits=5)

def cv_run(params):
    params['nthread'] = 1
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    for train_idx, test_idx in kf.split(train):
        train_cv_train, train_cv_test = train.ix[train_idx,:], train.ix[test_idx,:]
        dmat = xgb.DMatrix(train_cv_train.ix[:, train.columns!='CYCLEDUE'], train_cv_train.ix[:, 'CYCLEDUE'])
        dmatt = xgb.DMatrix(train_cv_test.ix[:, test.columns!='CYCLEDUE'], train_cv_test.ix[:, 'CYCLEDUE'])
        bst = xgb.train(params, dtrain=dmat)
        pred = bst.predict(dmatt)
        mse = mean_squared_error(train_cv_test.ix[:, 'CYCLEDUE'], pred)
        print(mse)

from multiprocessing import Pool
pool = Pool(10)
pool.map(cv_run, param_grid_list)

out_data['GINDEX'] = out_data.sort_values(['DWACCTID', 'CYCLEDATE']).groupby(['DWACCTID']).cumcount()

out_data['CYCLEDT'] = [datetime.strptime(cd[:7], '%Y-%m') for cd in out_data['CYCLEDATE']]

out_data.columns[0]


col = all_cat[0]
Counter(out_data[col]).most_common()[0][0]
out_data[col].value_counts().index[0]
out_data[col].unique()

dd = dict()

dd[col] = {'nam':col, 'typ':'con', 'rng':[out_data[col].min(),out_data[col].max()], 'dft':out_data[col].median(), 'cov':(out_data[col].std())/(out_data[col].mean())}

with open('data_dict.txt', 'a') as outfile:
    json.dump(dd[col], outfile)

col = all_con[0]
out_data[col].max()

CYCLEYD = [x[:7] for x in out_data['CYCLEDATE']]
out_data['CYCLEYD'] = CYCLEYD


# In[54]:

(out_data[['SCORE01', 'SCORE02', 'CYCLEYD', 'DWACCTID']]).set_index(['CYCLEYD', 'DWACCTID']).unstack(0)


# In[12]:
out_data['CUSTOMERSEGMENT'] = [v.strip() for v in out_data['CUSTOMERSEGMENT']]
out_data['CUSTOMERSEGMENT'] = pd.Categorical([v for v in out_data['CUSTOMERSEGMENT']],categories=out_data['CUSTOMERSEGMENT'].unique())
cat_col = out_data['CUSTOMERSEGMENT']
cat_map = {k:v for v,k in enumerate(cat_col.cat.categories)}
dd['custseg'] = {'nam':'custseg', 'typ':'cat', 'rng':cat_map, 'dft':cat_col.value_counts().index[0]}
_  = [cat_col.replace(cat, cat_map[cat], inplace=True) for cat in cat_map]
dd['custseg']['cov']=(pd.to_numeric(cat_col).std())/(pd.to_numeric(cat_col).mean())
# In[ ]:

out_data.pivot_table('CURRENTBAL', index=['CUSTOMERSEGMENT', 'INCRTYPE'], columns='HADINCREASE')


# In[57]:

r_scores = out_data.filter(regex=('R[0-9]{4}_.*'))
r_scores['CUSTOMERNUMBER'] = out_data['CUSTOMERNUMBER']
cus_max = r_scores.groupby('CUSTOMERNUMBER').max()
overall_max = cus_max.max()
overall_max


# In[59]:

cma_scores = out_data.filter(regex=('CMA[0-9]{4}'))
cma_scores['CUSTOMERNUMBER'] = out_data['CUSTOMERNUMBER']
cus_max = cma_scores.groupby('CUSTOMERNUMBER').max()
overall_max = cus_max.max()
overall_max


# In[58]:

scores = out_data.filter(regex=('SCORE[0-9]{2}'))
scores['CUSTOMERNUMBER'] = out_data['CUSTOMERNUMBER']
cus_min = scores.groupby('CUSTOMERNUMBER').min()
overall_min = cus_min.min()
overall_min


# In[ ]:

x = np.array(['min', 'max', 'last'])
l = len(all_con)
con_op = dict(zip(all_con, np.repeat(x[np.newaxis,:], l, 0)))
con_op['CMA3208']


# In[ ]:

con_ft_raw = out_data[all_con + ['CUSTOMERNUMBER']]
con_ft = con_ft_raw.groupby('CUSTOMERNUMBER').agg(con_op)


# In[ ]:

con_ft.columns = ['_'.join(col).strip() for col in con_ft.columns.values]


# In[ ]:

pd.Series([1,2,3]).pct_change()


# In[15]:

out = out_data.sort_values(['CUSTOMERNUMBER', 'CYCLEDATE']).groupby(['CUSTOMERNUMBER']).last().reset_index()


# In[13]:

con_ft_raw = out_data[all_con + ['CUSTOMERNUMBER']]
min_ft = con_ft_raw.groupby('CUSTOMERNUMBER').apply(lambda x:x.min())
max_ft = con_ft_raw.groupby('CUSTOMERNUMBER').apply(lambda x:x.max())
con_ft = pd.merge(min_ft, max_ft, on='CUSTOMERNUMBER', suffixes=('_min', '_max'))


# In[16]:

last_ft = out.drop(['DWACCTID', 'CYCLEDATE', 'RANDOMNUM', 'CREDITCYCLEFACTMKEY', 'CYCLEDUE'], axis=1)
last_ft['CUSTOMERNUMBER'] = last_ft['CUSTOMERNUMBER'].map(float)
all_ft = pd.merge(con_ft, last_ft, on='CUSTOMERNUMBER', suffixes=('', '_last'))


# In[17]:

target_raw = out_data[['CUSTOMERNUMBER', 'CYCLEDUE']]
target = target_raw.groupby('CUSTOMERNUMBER').max()


# In[18]:

dmat = pd.get_dummies(all_ft, prefix_sep='__')


# In[ ]:

out_data.columns.tolist()


# In[ ]:

# dmat.drop(['CUSTOMERNUMBER'], axis=1, inplace=True)
# target.drop(['CUSTOMERNUMBER'], axis=1, inplace=True)


# In[19]:

xgb_model = xgb.XGBRegressor().fit(dmat, target)


# In[20]:

wt = xgb_model.booster().get_score(importance_type='weight')


# In[21]:

wt2 = pd.DataFrame(wt.items(), columns=['Feature', 'Weight'])
print_full(wt2.sort_values(['Weight'], ascending=False))


# In[22]:

gavin = wt2['Feature'].str.split('_').map(lambda x:x[0]).unique()
gavin_derived = wt2['Feature']


# In[23]:

peng_ltw = [
'SCORE01', 
'SCORE05', 
'NOPAYDEFERREDBAL', 
'R6196_BALHIGHCREDITPERCENT', 
'R6011_OPENTRADEBALTOTAL', 
'CTCDBSBPURCHAMOUNT', 
'CYCLEDUE'
]
peng_latest = [
"SCORE04","SCORE06","SCORE07","SCORE08","SCORE09","OPENMORTGAGETRADECOUNT","R6008_MO24TRADEOPENCOUNT",
"FRAUDALERTCODE","OPENBANKRTYPEHIGHCREDITTOTAL","R6015_CURRENTPASTDUEAMOUNT","R6035_OPENTRADEBAL50PERHIGHCOUNT",
"NONFHBALTOTAL","R6027_MO24TRADENEVER30DCOUNT","OTHERFEEBAL","LTDNETPAYMENTTALLY","DONOTSOLICITFLAG",
"DAYSSINCELASTSALE","EXTERNALSTATUSREASONCODE","FINANCECHARGEBAL","CTCDLATECHARGERVSLAMOUNT","SCOREINDICATOR",
"RMS5002","OVERLIMITAMOUNT","LTCDBSBPURCHRVSLTALLY","BANKRUPCYFLAG","ALLTRADEPAYMENTTOTAL",
"R7000_DEROGATORYPUBLICRECORDCOUNT","MO12TRADE30DLATECOUNT","CTCDLATECHARGEAMOUNT","CTCDBSBPURCHTALLY",
"R6009_TRADECOUNT","R6115_DEPTTRADEBAL75HIGHCREDITCOUNT","R5006_MO06PROMOTIONINQUIRYCOUNT","MOSINCEMAJORDEROG",
"R6021_BADDEBTTRADECOUNT","LTCDPAYMENTRVSLAMOUNT","MARGINRATE","CTCDPAYMENTAMOUNT","CREDITLIMITAMOUNT","CURRENTBAL",
"CURRENTDAYSPASTDUEDELINQUENT","CTCDPAYMENTTALLY","R6042_TOTALOPENBALANCE","R6040_OPENDEPTTRADECOUNT",
"CTCDPAYMENTRVSLAMOUNT","CRITERIACODEANDDNSCODE","NUMBEROFMTHSINACTIVE","NONFHOPENTRADECOUNT","R6010_OPENTRADECOUNT",
"R6018_D30TRADECOUNT","R5003_MO06PROMOTIONINQUIRYCOUNT2","R6003_OLDESTTRADEAGE","ACCUMDEFERREDINTEREST","CUSTOMERAPR",
"CTCDNSFFEEAMOUNT","ADDRESSDISCREPANCYFLAG","BRANDXID","R6117_REVTRADE50PERHIGHCOUNT","R8000_COLLECTIONCOUNT",
"CUSTOMERSEGMENT","R5001_MOSINCELASTINQUIRYCOUNT","HOMEPHONEBADIND","R6073_D90DEPTTRADECOUNT","R6066_D30REVTRADECOUNT",
"SCORE","LTCDBSBPURCHRVSLAMOUNT","MO12OPENRETAILCOUNT","MO03TRADECLOSEDCOUNT","ADDRESSBADIND","GOODRTYPETRADECOUNT",
"R6199_CONSUMERBALHIGHCREDITPERCENT","R8003_PUBLICRECPLUSBADDEBTNOMEDCOUNT","MOSINCELASTDELINQUENCYCOUNT",
"MO03PROMOTIONINQUIRYCOUNT","LTCDPAYMENTAMOUNT","R8002_MOSINCELASTACTIVITYCOUNT","R6195_AGE",
"R6054_REVTRADEPASTDUECOUNT","LTDMAXDAYSDELQ","R6022_TRADENEVER30DCOUNT","PLACEDORDERFLAG","MO12BANKINSTALLCOUNT",
"R5007_MO06BANKPROMOTIONINQUIRYCOUNT"
]


# In[55]:

peng_all = peng_ltw + peng_latest
all = set(gavin_derived) | set(peng_all)
gavin_only = set(all) - set(peng_all)
list(gavin_only)


# In[56]:

gavin


# In[ ]:

all_all = set(gavin_derived) | set(all)


# In[ ]:

one_cust_id = sam2['DWACCTID'].sample(1)
one_cust = sam2.ix[sam2['DWACCTID'].isin(one_cust_id),:]
print_full(one_cust)

