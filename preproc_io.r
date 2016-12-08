Mode <- function(x) {
  ux <- unique(x)
  as.character(ux[which.max(tabulate(match(x, ux)))])
}
get_type = function(x) {
  ifelse(is.numeric(x), "con", ifelse(is.factor(x), "cat", "str"))
}
get_uniq = function(x) length(unique(x))
get_na = function(x) sum(is.na(x))

# returns string w/o leading whitespace
trim.leading <- function (x)  sub("^\\s+", "", x)

# returns string w/o trailing whitespace
trim.trailing <- function (x) sub("\\s+$", "", x)

# returns string w/o leading or trailing whitespace
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

trimws()

library(data.table)
library(sqldf)
library(dummies)
library(caret)
library(rjson)
library(pROC)
library(zoo)
library(xts)
library(hash)

# Input
setwd('~/Projects/CreditLine/')
system('gunzip -c cli_Pull_20161115.csv.gz > cli_Pull_20161115.csv')
# full = fread('cli_Pull_20161115.csv')
full = read.csv('cli_Pull_20161115.csv')
# full = readRDS('orchard_train_100516.Rds')
test_cust = read.csv('Test_Customer_Number.csv')
full = full[!(full$CUSTOMER_NUM %in% test_cust[,1]),]
# Dedupe
full2 = unique(full)

callData = full[full$PLCC_OFR_MADE %in% 1, ]
callData$ORDER_DATE = base::as.Date(callData$ORDER_DATE_TIME)
plcc1 = callData[callData$ORDER_DATE >= "2013-02-05" & callData$MSTR_COMP_CODE %in% c("APL", "TOG", "WTS", "LNS"), ]
plcc2 = callData[callData$ORDER_DATE >= "2013-07-09" & callData$MSTR_COMP_CODE %in% c("NTO", "SAH", "SOL", "GDV"), ]
plcc3 = callData[callData$ORDER_DATE >= "2013-09-19" & callData$MSTR_COMP_CODE %in% "DND", ]
plcc4 = callData[callData$MSTR_COMP_CODE %in% c("BLR", "AMO", "BFA"), ]
plcc = rbind(plcc1, plcc2, plcc3, plcc4)

callData = full[full$VIP_OFFER_MADE %in% 1, ]
callData$ORDER_DATE = base::as.Date(callData$ORDER_DATE_TIME)
vip1 = callData[callData$ORDER_DATE >= "2012-08-02"
                & callData$EXISTING_VIP %in% 0
                & callData$VIP_NEVER %in% 0
                & callData$AC_OFFER_MADE %in% 0 
                & callData$MSTR_COMP_CODE %in% c("NTO", "SAH"),
                ]
vip2 = callData[callData$ORDER_DATE >= "2013-04-01"
                & callData$EXISTING_VIP %in% 0
                & callData$VIP_NEVER %in% 0
                & callData$AC_OFFER_MADE %in% 0
                & callData$MSTR_COMP_CODE %in% c("SOL", "GDV", "BFA", "AMO"),
                ]
vip3 = callData[callData$ORDER_DATE >= "2013-05-01"
                & callData$EXISTING_VIP %in% 0
                & callData$VIP_NEVER %in% 0
                & callData$AC_OFFER_MADE %in% 0
                & callData$MSTR_COMP_CODE %in% c("APL", "WTS", "LNS", "DND", "TOG"),
                ]
vip4 = callData[callData$ORDER_DATE >= "2013-06-29"
                & callData$EXISTING_VIP %in% 0
                & callData$VIP_NEVER %in% 0
                & callData$AC_OFFER_MADE %in% 0
                & callData$MSTR_COMP_CODE %in% c("BLR"),
                ]
vip <- rbind(vip1, vip2, vip3, vip4)

df = plcc

# Output
for (i in 1:length(col_name)) {
  out = list(
    col_name=col_name[i],
    data_type=data_type[i],
    snapshot=snapshot[i][[col_name[i]]]
  )
  write.table(toJSON(out), "data_dict_orchard_plcc.txt", append=T, quote=F, sep="", row.names=F, col.names=F)
}

write.table(nam,"final_schema_orchard_plcc.txt", append=F, quote=F, row.names=F, col.names=F)

# require(feather)
require(readr)
system.time(readr::write_csv(train_data, 'orchard_plcc_train.csv'))
system.time(readr::write_csv(test_data, 'orchard_plcc_test.csv'))

#system.time(fwrite(DT,”fwrite.csv”))
# system.time(write_feather(df, 'orchard_input_feather.bin'))
# system.time(write.csv(df, 'orchard_input_writecsv_noquote.csv', row.names=F, quote=F))

