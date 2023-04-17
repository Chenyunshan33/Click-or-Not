#package install and library
install.packages("BiocManager")
install.packages('e1071')
install.packages('caTools')
install.packages(rpart)
install.packages("rattle")
install.packages("adabag")
install.packages("ROSE")
install.packages('pROC')
install.packages('ROCit')
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("klaR")
install.packages("corrplot")
library(corrplot)
library(adabag)
library('e1071')
library('caTools')
library('caret')
library(ISLR)
library(MASS)
library(leaps)
library(rpart)
library(rattle)
library(adabag)
library(ROSE)
library('pROC')
library('ROCit')
library(dplyr)
library("klaR")

#import raw data
user_profile=read.csv("D:/3.CUHK/课程/3.Data mining/2.Project/archive (5)/user_profile.csv",header=T,na.strings="?")
raw_sample=read.csv("D:/3.CUHK/课程/3.Data mining/2.Project/archive (5)/raw_sample.csv",header=T,na.strings="?")
feature=read.csv("D:/3.CUHK/课程/3.Data mining/2.Project/archive (5)/ad_feature.csv",header=T,na.strings="?")

#merge data
names(user_profile)[1]<-"uid"
names(raw_sample)[1]<-"uid"
data_t<-merge(raw_sample,user_profile,by="uid",all.user_profile=T)
data_tt <-merge(data_t,feature,by="adgroup_id",all.data_t=T)

#########data preprocessing#############
#delete NA 
data_tt <-na.omit(data_tt)
skimr::skim(data_tt)

#data cleanning_age
data_tt = subset(data_tt,data_tt$age_level!="0")
data_tt$age[data_tt$age_level==1] <-1
data_tt$age[data_tt$age_level==2] <-2
data_tt$age[data_tt$age_level>=3 & data_tt$age_level<=4] <-3
data_tt$age[data_tt$age_level>=5 & data_tt$age_level<=6] <-4

#data cleaning_price
summary(data_tt$price)
sd(data_tt$price)
boxplot(data_tt$price)
boxplot.stats(data_tt$price)$out
QL <- quantile(data_tt$price, probs = 0.25)
QU <- quantile(data_tt$price, probs = 0.75)
QU_QL <- QU-QL
data_tt=subset(data_tt,data_tt$price > QL-1.5*QU_QL)
data_tt=subset(data_tt,data_tt$price < QU+1.5*QU_QL)
summary(data_tt$price)
data_tt $price[data_tt$price<=78] <-1
data_tt $price[data_tt$price>78 &data_tt$price<=147.5 ] <-2
data_tt $price[data_tt$price>147.5 &data_tt$price<=268 ] <-3
data_tt $price[data_tt$price>268]  <-4
data_tt $price=as.factor(data_tt $price)

#randomly pick 20000 rows 
set.seed(2)
data_sub <- data_tt[sample(nrow(data_tt), size=20000), ]
skimr::skim(data_sub)

#data cleaning_time period
data_sub$time=as.POSIXlt(data_sub$time_stamp, origin="1970-01-01")
data_sub$hour=strftime(data_sub$time,"%H","GMT")
data_sub$hour=as.integer(data_sub$hour)
data_sub$time_period[data_sub$hour>=0 & data_sub$hour<=8 ] <- 0
data_sub$time_period[data_sub$hour>=9 & data_sub$hour<=11 ] <- 1
data_sub$time_period[data_sub$hour>=12 & data_sub$hour<=14 ] <- 2
data_sub$time_period[data_sub$hour>=15 & data_sub$hour<=18 ] <- 3
data_sub$time_period[data_sub$hour>=19 & data_sub$hour<=23 ] <- 4
data_sub$cate_id <-as.factor(data_sub$cate_id)

#data cleaning_data type 
table(data_sub$cate_id)
table(data_sub$brand)
data_sub$occupation=as.factor(data_sub$occupation)
data_sub$cms_group_id=as.factor(data_sub$cms_group_id)
data_sub$cms_segid=as.factor(data_sub$cms_segid)
data_sub$final_gender_code=as.factor(data_sub$final_gender_code)
data_sub$age=as.factor(data_sub$age)
data_sub$pvalue_level=as.factor(data_sub$pvalue_level)
data_sub$shopping_level=as.factor(data_sub$shopping_level)
data_sub$cate_id=as.factor(data_sub$cate_id)
data_sub$time_period = as.factor(data_sub$time_period)
data_sub$new_user_class_level = as.factor(data_sub$new_user_class_level)
data_sub <- data_sub[,-1:-5]
data_sub <- data_sub[,-2:-3]
data_sub <- data_sub[,-8:-11]
data_sub <- data_sub[,-10:-11]
data_sub <- data_sub[,-3]

#split train_set and test_set (0.7&0.3)
set.seed(2)
index <-  sort(sample(nrow(data_sub), nrow(data_sub)*.7))
train <- data_sub[index,]
test <-  data_sub[-index,]

#########logit regression###########
#logistic regression
logistic<-glm(clk~.,data=train,family = "binomial"(link = "logit"))
summary(logistic)
logit.step <- step(logistic,direction= "both")
summary(logit.step)
#Predict
pred <- predict.glm(logit.step,newdata = test,type = "response")
pred_1<-ifelse(test = pred>0.5,yes = 1,no = 0)
#MSE
mean((as.numeric(pred_1)-as.numeric(test$clk))^2)
#Confusion Matrix
cm <- table(pred_1,test$clk)
cm
confusionMatrix(cm,positive = "1")

#over_sample -- handle the problem that only predict 0
over_sample_traindata<-ovun.sample(clk~.,data=train,method="over",N=25000,seed=5)$data
table(over_sample_traindata$clk)
logistic2<-glm(clk~.,data=over_sample_traindata,family = "binomial"(link = "logit"))
summary(logistic2)
logit.step2 <- step(logistic2,direction= "both")
summary(logit.step2)
pred2 <- predict.glm(logit.step2,newdata = test,type = "response")
pred_2<-ifelse(test = pred2>0.5,yes = 1,no = 0)
#MSE
mean((as.numeric(pred_2)-as.numeric(test$clk))^2)
#Confusion Matrix
cm <- table(pred_2,test$clk)
confusionMatrix(cm,positive = "1")
accuracy.meas(test$clk, test$over_y_pred)
roc.curve(test$clk, test$over_y_pred, plotit = F)

#########Naive Bayes Classifier###########
#assumption fulfillment: attribute needs to be independent
data_subb=cbind(data_sub$final_gender_code,data_sub$age,data_sub$pvalue_level,data_sub$shopping_level,data_sub$occupation,data_sub$new_user_class_level,data_sub$price,data_sub$time_period)
colnames(data_subb) <- c("gender","age","pvalue","shopping_level","occupation","user_class","price","time_period")
corrplot(cor(data_subb),order = "AOE",method = "number")

##Bayes classifier
classifier_cl <- naiveBayes(clk ~final_gender_code+shopping_level+pvalue_level+time_period+new_user_class_level+age+price, data = train)
classifier_cl
#predict
y_pred1 <- predict(classifier_cl, newdata = test)
y_pred1 <- as.character(y_pred1)
#MSE
mean((as.numeric(y_pred1)-as.numeric(test$clk))^2)
#Confusion Matrix
cm <- table(test$clk, y_pred1)
cm
confusionMatrix(cm,positive = "1")

#over_sample -- handle the problem that only predict 0
table(train$clk)
over_sample_traindata <- ovun.sample(clk ~final_gender_code+shopping_level+pvalue_level+time_period+new_user_class_level+age+price, data =train, method = "over",N =25000,seed=5)$data
table(over_sample_traindata$clk)
over_classifier<- naiveBayes(clk ~final_gender_code+shopping_level+pvalue_level+time_period+new_user_class_level+age+price, data = over_sample_traindata)
test$over_y_pred <- predict(over_classifier, newdata = test)
test$over_y_pred=as.character(test$over_y_pred)
#MSE
mean((as.numeric(test$over_y_pred)-as.numeric(test$clk))^2)
#Confusion Matrix and evaluation
cm4 <- table(test$clk, test$over_y_pred)
cm4
confusionMatrix(cm4,positive = "1")


