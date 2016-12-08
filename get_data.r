setwd('/home/gavinwu/Projects/CreditLine/Script')
options(java.parameters = "-Xmx8g")
library(RJDBC)
options(scipen = 999)


jarFileLocation = "nzjdbc3.jar "


drvNetezza <- JDBC(driverClass="org.netezza.Driver", classPath = jarFileLocation, "NZConnection")
conNetezza <- dbConnect(drvNetezza, "jdbc:netezza://bisprod01:5480//sandbox", "gavin.wu",.rs.askForPassword(prompt = "Enter your Password in the box below:"))

# I use the function below to send my queries to the server

sendTheQuery = function(myConnectionObject, queryString,verbose=F){
  verboseFunction = function(message,verboseVal){
    if(verboseVal){
      print(message)
    }
  }
  
  query.fn = dbSendQuery(myConnectionObject,queryString)
  verboseFunction("Done with dbSendQuery.",verbose)
  myResults.fn=dbFetch(query.fn,n=-1)
  verboseFunction("Done with Fetch.",verbose)
  dbClearResult(query.fn)
  verboseFunction("Done with clear Result.",verbose)
  
  myResults.fn
  
}

# Define the string
RandomSample = "
with onesRandomSample as
(
  select customerNumber,dwacctid, max(randomNum) as randomNum
  from sandbox..fullCLIdata
  where hadIncrease = 1
  group by customerNumber,dwacctid
  order by randomNum 
  limit 1000
),
  zerosRandomSample as
  (
  select customerNumber,dwacctid, max(randomNum) as randomNum
  from sandbox..fullCLIdata
  where hadIncrease = 0
  group by customerNumber,dwacctid
  order by randomNum 
  limit 50
  ),
  fullSample as
  (
  select * from onesRandomSample
  union
  select * from zerosRandomSample
  )
  
  select 20140905 as SnapDate,  b.*
  from fullSample a
  inner join sandbox..fullCLIdata b on a.customerNumber=b.customerNumber and a.dwacctid=b.dwacctid
  order by b.customerNumber,b.cycleDate
  "
  
  # send the string to the server to be processed.
  fullCLI=sendTheQuery(myConnectionObject = conNetezza,queryString = RandomSample)
 
# Format the review & cycle dates 
fullCLI$REVIEWDATE = as.numeric(format(as.Date(fullCLI$REVIEWDATE,format = "%Y-%m-%d"),"%Y%m%d"))
fullCLI$CYCLEDATE = as.numeric(format(as.Date(fullCLI$CYCLEDATE,format = "%Y-%m-%d"),"%Y%m%d"))


# Eliminate the external attributes after the snap date
fullCLI[fullCLI$REVIEWDATE>20140905,(which(names(fullCLI)%in%"REVIEWDATE")+1):ncol(fullCLI)]=NA