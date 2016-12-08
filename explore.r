df$ind = as.numeric(row.names(df))
all_col = names(full)
date_col = all_col[grepl('date', all_col, ignore.case=T)]
payment_col = all_col[grepl('pay', all_col, ignore.case=T)]
count_col = all_col[grepl('count', all_col, ignore.case=T)]
due_col = all_col[grepl('due', all_col, ignore.case=T)]
amount_col = all_col[grepl('amount', all_col, ignore.case=T)]
limit_col = all_col[grepl('limit', all_col, ignore.case=T)]
balance_col = all_col[grepl('bal', all_col, ignore.case=T)]

df2 = data.table(df)
all_len = tapply(df2$CUSTOMERNUMBER, df2$CUSTOMERNUMBER, length)
all_cust = unique(df2$CUSTOMERNUMBER)

cust_pick = sample(all_cust, 1)
cust = df2[df2$CUSTOMERNUMBER %in% cust_pick,]
cycle_date = as.Date(cust$CYCLEDATE)
to_plot1 = xts(-cust$CTCDPAYMENTAMOUNT, order.by=cycle_date)
to_plot2 = xts(cust$CREDITLIMITAMOUNT, order.by=cycle_date)
to_plot3 = xts(cust$CURRENTBAL, order.by=cycle_date)
plot.zoo(cbind(to_plot1,to_plot2,to_plot3), ylab=c('payment','creditlimit','balance'))
