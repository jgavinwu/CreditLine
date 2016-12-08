# Build dictionary
data_type = as.vector(sapply(df, get_type))
uniq_level = as.vector(sapply(df, get_uniq))
na_count = as.vector(sapply(df, get_na))
col_name = names(df)

# alldate = prop.table(table(df$ORDER_DATE_TIME))
# table(rank(as.vector(alldate), ties.method=c("min")))
# alldate[rank(as.vector(alldate), ties.method=c("min")) %in% 6320396]*nrow(df)

snapshot=list()
df_df = as.data.frame(df)
for (name in names(df_df)) {
  if (is.numeric(df_df[,name])) snapshot[[name]] = c(min(df_df[,name], na.rm=T), max(df_df[,name], na.rm=T), median(df_df[,name], na.rm=T), mean(df_df[,name], na.rm=T), sd(df_df[,name], na.rm=T))
  else snapshot[[name]] = Mode(df_df[,name])
}

# with(df[df$PLCC_OFR_CONV==1,], table(INSTANT_CREDIT_OFFER, BATCH_CREDIT_OFFER, REALTIME_CREDIT_OFFER))

# Univariate signal strength analysis
aucsub = sample(nrow(df), 100000)
target = df$PLCC_OFR_CONV[aucsub]
df2 = df[aucsub, ]
con_col = col_name[data_type == 'con'] # Subset to eliminate columns with too big values
str_col = col_name[data_type == 'str'] # Subset to eliminate columns with too many unique values

no_vet = c('COMPANY', 'CUSTOMER_NUM', 'ORDER_NUM', 'ORDER_DATE_TIME', 'ORDER_DATE')

auc_val = c()
for (con_coli in con_col[-which(con_col %in% no_vet)]) {
  print(con_coli)
  auc_val = c(auc_val, auc(target, df2[,get(con_coli)]))
}

for (str_coli in str_col[-which(str_col %in% no_vet)]) {
  print(str_coli)
  barplot(prop.table(table(df2[,get(str_coli)])), main=str_coli)
  cat ("Press [enter] to continue")
  line <- readline()
}

# Collapsing categories not in the top 10 most frequent into 'Other' to reduce the number of categories
most_common = names(sort(summary(as.factor(df$STATE)), decreasing=T)[1:10])
df[, STATE := ifelse((STATE %in% most_common),as.character(STATE),"Other")]
# df[, ORDER_DATE := as.Date(ORDER_DATE_TIME)]
df[, COOR_OFFER_PREV_ACC := COOR_OFFCOOR_OFFER_PREV_ACC]

# Convert factors (except those whose values are already 0/1s)
names(df)[uniq_level > 2 & uniq_level < 20]
df$MSTR_COMP_CODE = as.factor(df$MSTR_COMP_CODE)
df$STATE = as.factor(df$STATE)

# Identify and rough-fix NA

# sqlDF feature engineering
# query = 'select * from df where PLCC_OFR_MADE = 1 and MSTR_COMP_CODE in ("APL", "TOG", "WTS", "LNS")'
# plcc = sqldf(query)
# df3[45:50, rollmean(MERCH_AMOUNT,k=3,fill=0,align="right"), by = eval('CUSTOMER_NUM')]
# rollapplyr(df3[,get('MERCH_AMOUNT')],list(-(3:1)),mean,fill=NA)
df = df[order(CUSTOMER_NUM, ORDER_DATE_TIME)]
df = df[, HIST_LEN := length(MERCH_AMOUNT), by = .(CUSTOMER_NUM)]
df = df[HIST_LEN > 1, ]
# df = df[, PRIOR_MERCH := rollapplyr(MERCH_AMOUNT,list(-1),mean,fill=NA), by = eval('CUSTOMER_NUM')]
df = df[, PRIOR_MERCH := rollapplyr(MERCH_AMOUNT,list(-(nrow(.SD):1)),mean,fill=NA,partial=T), by = eval('CUSTOMER_NUM')]
df = df[!is.na(PRIOR_MERCH), ]

df2 = df[1:100]
df2 = df2[, PRIOR_MERCH := rollapplyr(MERCH_AMOUNT,list(-(nrow(.SD):1)),mean,fill=NA,partial=T), by = eval('CUSTOMER_NUM')]

# One-hot encode (identify factors first)
fvec = c('HIST_LEN', 'PRIOR_MERCH',
         'WAS_VIP', 'PRIOR_DECL_VIP', 'EXISTING_VIP', 'VIP_NEVER', 
         'PRIOR_AC_PURCH', 'PRIOR_AC_OFFRS', 'CUST_CHOICE_PREV_ACC', 'COOR_OFFER_PREV_ACC', 'PROT_PLUS_PREV_ACC', 
         'MEDIA', 'NEW_CUST_FLG', 'STATE', 'MSTR_COMP_CODE', 
         'APL_CC', 'LNC_CC', 'TOG_CC', 'WTS_CC', 'AMO_CC', 'BFA_CC', 'BLR_CC', 'GDV_CC', 'NTO_CC', 'SAH_CC', 'SOL_CC', 'DND_CC')
label = c('PLCC_OFR_MADE', 'VIP_OFFER_MADE', 'AC_OFFER_MADE', 'COOR_OFFER_OFFRD', 'CUST_CHOICE_OFFRD', 'PROTECT_PLUS_OFFRD', 
          'PLCC_OFR_APPROVED', 'VIP_OFFER_CONV', 'AC_OFFER_CONV', 'COOR_OFFER_ACCPT', 'CUST_CHOICE_ACCPT', 'PROTECT_PLUS_ACCPT')
fmat = df[, mget(fvec)]
tmat = df[, mget(label)]

# Drop zero variance and highly correlated columns
nzv = nearZeroVar(fmat, saveMetrics=T)
keep = c(1:ncol(fmat))[!(nzv$freqRatio == 0)]
# keep = c(1:ncol(fmat))[!nzv$nzv]
fmat = fmat[,keep, with=F]
dmat = dummy.data.frame(fmat, sep="__", dummy.classes=c("factor"))
highcor = findCorrelation(dmat, cutoff=0.75, exact=F)
dmat = dmat[,-which(names(dmat) %in% names(dmat[,highcor]))]

# Output columns to keep and final input file
nam = names(dmat)
namr = c()
for (inam in nam) {
  namr = c(namr, gsub('__[[:alpha:]]*$', '', inam))
}
namr = unique(namr)

df_out = df[, mget(namr)]
data_type = as.vector(sapply(df_out, get_type))
uniq_level = as.vector(sapply(df_out, get_uniq))
na_count = as.vector(sapply(df_out, get_na))
col_name = names(df_out)

snapshot=list()
df_df = as.data.frame(df_out)
for (name in names(df_df)) {
  if (is.numeric(df_df[,name])) snapshot[[name]] = c(min(df_df[,name], na.rm=T), max(df_df[,name], na.rm=T), median(df_df[,name], na.rm=T), mean(df_df[,name], na.rm=T), sd(df_df[,name], na.rm=T))
  else if (is.factor(df_df[,name])) snapshot[[name]] = levels(df_df[,name])
  else snapshot[[name]] = Mode(df_df[,name])
}

# Split into train and test sets
all_cust = unique(df$CUSTOMER_NUM)
all_data = cbind(CUSTOMER_NUM=df$CUSTOMER_NUM, df_out, tmat)
train_cust = sample(all_cust, floor(length(all_cust)*0.75))
test_cust = setdiff(all_cust, train_cust)
train_cust2 = data.table(CUSTOMER_NUM=train_cust)
train_data = merge(all_data, train_cust2, by="CUSTOMER_NUM")
test_cust2 = data.table(CUSTOMER_NUM=test_cust)
test_data = merge(all_data, test_cust2, by="CUSTOMER_NUM")


full2 = full
full2$OFFER_CT = full2$PLCC_OFR_MADE + full2$VIP_OFFER_MADE + full2$AC_OFFER_MADE + full2$COOR_OFFER_OFFRD + full2$CUST_CHOICE_OFFRD + full2$PROTECT_PLUS_OFFRD
offer_dist = prop.table(table(full2[,get('OFFER_CT')]))
barplot(offer_dist, main='Number of Offers')
offer_dist2 = apply(full2[,mget(c('PLCC_OFR_MADE', 'VIP_OFFER_MADE', 'AC_OFFER_MADE', 'COOR_OFFER_OFFRD', 'CUST_CHOICE_OFFRD', 'PROTECT_PLUS_OFFRD'))], 2, sum)
barplot(offer_dist2)
full2$ACCPT_CT = full2$PLCC_OFR_CONV + full2$VIP_OFFER_CONV + full2$AC_OFFER_CONV + full2$COOR_OFFER_ACCPT + full2$CUST_CHOICE_ACCPT + full2$PROTECT_PLUS_ACCPT
offer_dist3 = prop.table(table(full2[,get('ACCPT_CT')]))
barplot(offer_dist3, main='Number of Acceptances')
full2$ACCPT_RATE = full2$ACCPT_CT / full2$OFFER_CT
accept_dist = prop.table(table(full2[,get('ACCPT_RATE')]))
barplot(accept_dist, main='Acceptance Rate')
