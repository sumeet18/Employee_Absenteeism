#remove all data
rm(list=ls(all=T))

#set working directory
setwd("G:/edwisor")

#Current working directory
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

##Load data in R
#reading Excel sheet
library(xlsx)
data = read.xlsx("data.xlsx", sheetIndex = 1, header = T)

#view data
View(data)

#check datatype
class(data)

#summary of data
summary(data)

#column names
colnames(data)

#number of variables
length(unique(data))

#change class from numerical to factor
data$ID = as.factor(data$ID)
data$Month.of.absence = as.factor(data$Month.of.absence)
data$Seasons = as.factor(data$Seasons)
data$Reason.for.absence = as.factor(data$Reason.for.absence)
data$Day.of.the.week = as.factor(data$Day.of.the.week)
data$Disciplinary.failure = as.factor(data$Disciplinary.failure)
data$Education = as.factor(data$Education)
data$Son = as.factor(data$Son)
data$Social.drinker = as.factor(data$Social.drinker)
data$Social.smoker = as.factor(data$Social.smoker)
data$Pet = as.factor(data$Pet)

missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

# Checking for missing value
sum(is.na(new))


new = data[complete.cases(data[ , 21]),]

#Mean Method
new$Age[is.na(new$Age)] = mean(new$Age, na.rm = T)

#Median Method
new$Age[is.na(new$Age)] = median(new$Age, na.rm = T)

# kNN Imputation
new = knnImputation(new, k = 3)

new[71,9]

#Actual=28
#mean = 36.48815
#median = 37
#knn = 28

new[71,9] = NA


#Box plot
ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Age, 
                        fill = new$Age)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("Age") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Distance.from.Residence.to.Work, 
                       fill = new$Distance.from.Residence.to.Work)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("Distance") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Service.time, 
                       fill = new$Service.time)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("time") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Work.load.Average.day., 
                       fill = new$Work.load.Average.day.)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("workload") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Weight, 
                       fill = new$Weight)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("weight") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Height, 
                       fill = new$Height)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("height") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Body.mass.index, 
                       fill = new$Body.mass.index)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("bmi") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Transportation.expense, 
                       fill = new$Transportation.expense)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("Expense") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Absenteeism.time.in.hours, y = new$Hit.target, 
                       fill = new$Hit.target)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("hours") + ylab("Target") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

quantile(new$Height, c(.01))
quantile(new$Height, c(.85))
new[which(new$Height<165),("Height")] = 165
new[which(new$Height>175),("Height")] = 175

quantile(new$Work.load.Average.day., c(.96))
new[which(new$Work.load.Average.day.>343253),("Work.load.Average.day.")] = 343253

quantile(new$Age, c(.99))
new[which(new$Age>50),("Age")] = 50

quantile(new$Service.time, c(.99))
new[which(new$Service.time>18),("Service.time")] = 18

quantile(new$Transportation.expense, c(.99))
new[which(new$Transportation.expense>378),("Transportation.expense")] = 378

quantile(new$Hit.target, c(.04))
new[which(new$Hit.target<87),("Hit.target")] = 87



#correlation
#Load Libraries
library(corrgram)
numeric_index = sapply(new,is.numeric)

numeric_data = new[,numeric_index]

corrgram(new[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


## ANOVA test for Categprical variable
summary(aov(formula = Absenteeism.time.in.hours~ID,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Reason.for.absence,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Month.of.absence,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Day.of.the.week,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Seasons,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Disciplinary.failure,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Education,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Social.drinker,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Social.smoker,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Son,data = new))
summary(aov(formula = Absenteeism.time.in.hours~Pet,data = new))




#Histogram 
ggplot(new, aes_string(x = new$Transportation.expense)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("expense") + ylab("Frequency") + ggtitle('transportation expense') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Distance.from.Residence.to.Work)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("distance") + ylab("Frequency") + ggtitle('distance from home to office') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Work.load.Average.day.)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("workload") + ylab("Frequency") + ggtitle('workload') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Hit.target)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("target") + ylab("Frequency") + ggtitle('Hit target') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Weight)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("weight") + ylab("Frequency") + ggtitle('weight') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Height)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("Height") + ylab("Frequency") + ggtitle('Height') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Body.mass.index)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("BMI") + ylab("Frequency") + ggtitle('BMI') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Service.time)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("service time") + ylab("Frequency") + ggtitle('service time') +
  theme(text=element_text(size=20))

ggplot(new, aes_string(x = new$Age)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("Age") + ylab("Frequency") + ggtitle('Age') +
  theme(text=element_text(size=20))

#Normalisation
cnames = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Height', 'Weight',
                    'Body.mass.index')

for(i in cnames){
  print(i)
  new[,i] = (new[,i] - min(new[,i]))/
    (max(new[,i] - min(new[,i])))
}





copy=new
new=copy

# dummify the data
# Creating dummy variables for categorical variables
library(mlr)
catagorical = c('ID', 'Reason.for.absence', 'Month.of.absence',
                'Seasons', 'Day.of.the.week',
                'Disciplinary.failure', 'Education', 'Son',
                'Social.drinker','Social.smoker','Pet')
new = dummy.data.frame(new, catagorical)

new = subset(new, 
             select = -c(ID36,Reason.for.absence0,Month.of.absence0,Day.of.the.week2,Seasons4,Education4,Disciplinary.failure1,Son4,Social.drinker0,Social.smoker0,Pet8))

#check multicollearity
library(usdm)
vif(new[,-106])

vifcor(new[,-106], th = 0.9)

new = subset(new, 
             select = -c(Son3,Pet5,Weight)


#Decision treemodel
#Load Libraries
library(rpart)

#Divide the data into train and test
#set.seed(123)
train_index = sample(1:nrow(new), 0.8 * nrow(new))
train = new[train_index,]
test = new[-train_index,]

# ##rpart for regression
fit = rpart(Absenteeism.time.in.hours ~ ., data = train, method = "anova")

#Predict for new test cases
predictions_DT = predict(fit, test[,-106])

#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test[,106], predictions_DT)


#calculate MSE
MSE = function(m, o){
  (mean((m - o)^2))
}
MSE(test[,106], predictions_DT)

#calculate RMSE
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

RMSE(test[,106], predictions_DT)

#Linear Regression

#run regression model
lm_model = lm(Absenteeism.time.in.hours ~., data = train)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,1:105])

#Calculate MAPE,MSE,RMSE
MAPE(test[,106], predictions_LR)
MSE(test[,106], predictions_LR)
RMSE(test[,106], predictions_LR)


#Randomforest
RF_model = randomForest(Absenteeism.time.in.hours ~ ., train, importance = TRUE, ntree = 100)

#transform rf object to an inTrees' format
treeList = RF2List(RF_model)

#Extract rules
exec = extractRules(treeList, train[,-106])  # R-executable conditions

exec[1:2,]

# #Make rules more readable:
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]

ruleMetric = getRuleMetric(exec, train[,-106], train$cnt)  # get rule metrics
# 
# #evaulate few rules
ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-106])


#Calculate MAPE,MSE,RMSE
MAPE(test[,106], RF_Predictions)
MSE(test[,106], RF_Predictions)
RMSE(test[,106], RF_Predictions)





