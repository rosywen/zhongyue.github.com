if(!require("MASS"))install.packages("MASS")
if(!require("dplyr"))install.packages("dplyr")
if(!require("class"))install.packages("class")
if(!require("glmnet"))install.packages("glmnet")
if(!require("tree"))install.packages("tree")
if(!require("randomForest"))install.packages("randomForest")
if(!require("gbm"))install.packages("gbm")
if(!require("e1071"))install.packages("e1071")
if(!require("e1071"))install.packages("DMwR")
############Read data
read.csv('rain1.csv')
Rain_origin=read.csv('rain1.csv',header = TRUE)
Rain=read.csv('rain1.csv',header = TRUE)
names(Rain)
dim(Rain)

###########standardize and remove the outliers
library(MASS)
ind=sapply(Rain, is.numeric)
Rain[ind]=lapply(Rain[ind], scale)
attach(Rain)
library(dplyr)
Rain_clean=filter(Rain,MinTemp<3 & MaxTemp<3 & Rainfall<3 & Evaporation<3 & Sunshine<3 & WindGustSpeed<3 & WindSpeed9am<3 & WindSpeed3pm<3 & Humidity9am<3 & Humidity3pm<3 & Pressure9am<3 & Pressure3pm<3 & Cloud9am<3 &  
                    Cloud3pm<3 & Temp9am<3 & Temp3pm<3)

#################set variables and split data into training dataset and test dataset
set.seed(3)
train <-sample(1:nrow(Rain_clean), 26760,replace=FALSE)
# Create training data set
Rain_clean.train <- Rain_clean[train,]
# Create testing data set
Rain_clean.test <- Rain_clean[-train,]

###############using diffrent classificaiton methods to predict whether or not it will rain tomorrow

############ Method1: fit K-nearest neighbours on the traning dataset and predict it using the test dataset
library(class)
names(Rain_clean.test)
# Create training data for X
x.train=cbind(Rain_clean.train$MinTemp,Rain_clean.train$MaxTemp,Rain_clean.train$Rainfall,Rain_clean.train$Evaporation,Rain_clean.train$Sunshine,Rain_clean.train$WindGustDir,Rain_clean.train$WindGustSpeed,Rain_clean.train$WindDir9am,Rain_clean.train$WindDir3pm,Rain_clean.train$WindSpeed9am,Rain_clean.train$WindSpeed3pm,Rain_clean.train$Humidity9am,Rain_clean.train$Humidity3pm,Rain_clean.train$Pressure9am,Rain_clean.train$Pressure3pm,Rain_clean.train$Cloud9am,Rain_clean.train$Cloud3pm,Rain_clean.train$Temp9am,Rain_clean.train$Temp3pm,Rain_clean.train$RainToday)
# Create testing data for X
y.train=Rain_clean.train$RainTomorrow
# Create training data for Y
x.test=cbind(Rain_clean.test$MinTemp,Rain_clean.test$MaxTemp,Rain_clean.test$Rainfall,Rain_clean.test$Evaporation,Rain_clean.test$Sunshine,Rain_clean.test$WindGustDir,Rain_clean.test$WindGustSpeed,Rain_clean.test$WindDir9am,Rain_clean.test$WindDir3pm,Rain_clean.test$WindSpeed9am,Rain_clean.test$WindSpeed3pm,Rain_clean.test$Humidity9am,Rain_clean.test$Humidity3pm,Rain_clean.test$Pressure9am,Rain_clean.test$Pressure3pm,Rain_clean.test$Cloud9am,Rain_clean.test$Cloud3pm,Rain_clean.test$Temp9am,Rain_clean.test$Temp3pm,Rain_clean.test$RainToday)
# Create testing data for Y
y.test=Rain_clean.test$RainTomorrow
############## using cross-validation to selsect the best k for KNN
error.rate.knn=rep(0,40)
i=1
for(k in 1:40){
  knn.pred=knn(x.train,x.test,y.train,k=k)
  error.rate.knn[i]=mean(knn.pred!=y.test)
  i=i+1
}
# Misclassfication error rate
mean(error.rate.knn)
plot(error.rate.knn,xlab="number of neighbors",ylab="MER",type = "b")
# Confusion matrix
# knn_pred is the predicted Y for testing data and y.test is the true Y for testing data
knn.pred_14 <- knn(x.train,x.test,y.train,k=14)
table(knn.pred_14, y.test)
# Proportation of making correct classification
suc.rate.knn=mean(knn.pred_14==y.test);suc.rate.knn
error.rate.knn[14]

######################fit Logistic regression on the traning dataset and predict it using the test dataset
################# fit a gernalized linear regression using a logit link function, set distribution of the response variable to be binomial
#################first run the full model
# logistic regression on the training data set
glm_fit = glm(RainTomorrow ~ .,data = Rain_clean.train,family = binomial)
# Predicted probabilities for the testing data set;use"type = "response"" to predict the probability rather than log-odds
glm_probs = predict(glm_fit, Rain_clean.test, type = "response")
# Sample size for the testing data
dim(Rain_clean.test)
contrasts(y.test)
# For predicted probabilities greater than 0.5, assign Y to be "Yes"; otherwise assign Y to be "No"
glm_pred = rep("No",26760)
glm_pred[glm_probs > .5] = "Yes"
length(y.test)
# Confusion matrix
# glm_pred is the predicted Y for testing data and y.test is the true Y for testing data
table(glm_pred, y.test)
# Proportation of making correct classification
suc.rate.logistic=mean(glm_pred==y.test);suc.rate.logistic
# Misclassfication error rate
error.rate.logistic=mean(glm_pred!=y.test);error.rate.logistic

#threshold = 0.4
glm_pred2 = rep("No",26760)
glm_pred2[glm_probs > .4] = "Yes"
table(glm_pred2, y.test)
suc.rate.logistic2=mean(glm_pred2==y.test);suc.rate.logistic2
error.rate.logistic2=mean(glm_pred2!=y.test);error.rate.logistic2

#threshold = 0.3
glm_pred3 = rep("No",26760)
glm_pred3[glm_probs > .3] = "Yes"
table(glm_pred3, y.test)
suc.rate.logistic3=mean(glm_pred3==y.test);suc.rate.logistic3
error.rate.logistic3=mean(glm_pred3!=y.test);error.rate.logistic3

#threshold = 0.2
glm_pred4 = rep("No",26760)
glm_pred4[glm_probs > .2] = "Yes"
table(glm_pred4, y.test)
suc.rate.logistic4=mean(glm_pred4==y.test);suc.rate.logistic4
error.rate.logistic4=mean(glm_pred4!=y.test);error.rate.logistic4

#threshold = 0.1
glm_pred5 = rep("No",26760)
glm_pred5[glm_probs > .1] = "Yes"
table(glm_pred5, y.test)
suc.rate.logistic5=mean(glm_pred5==y.test);suc.rate.logistic5
error.rate.logistic5=mean(glm_pred5!=y.test);error.rate.logistic5

####################### Method2: ridge Approach
x.train1=model.matrix(RainTomorrow~.,data=Rain_clean.train)[,-1]
y.train1=(Rain_clean.train$RainTomorrow)
x.test1=model.matrix(RainTomorrow~.,data=Rain_clean.test)[,-1]
y.test1=(Rain_clean.test$RainTomorrow)
x=model.matrix(RainTomorrow~.,data=Rain_clean)[,-1]
y=(Rain_clean$RainTomorrow)
library(glmnet)
#####################use"family="binomial"" because wo are doing binary classification, use"alpha = 0"  to do ridge penalty
ridge.fit=glmnet(x.train1,y.train1,alpha = 0,family="binomial")
## Plot of CV mse vs log (lambda)
plot(ridge.fit, xvar="lambda", label= TRUE)
plot(ridge.fit, xvar="dev", label= TRUE)
###################### use cross-validation to select the best lambda for ridge,by default in k-fold cross validation k is set to be 10
ridge.cv <-cv.glmnet(x.train1, y.train1,family="binomial",alpha = 0)
plot(ridge.cv)
coef(ridge.cv)
##################### select the best lambda
#################### select a model using the one-standard-error rule. We first calculate the one standard error rule
###################select the smallest model for which the estimated test error is within one standard error rule.
ridge.cv$lambda.min
n1_ridge<-which(ridge.cv$lambda==ridge.cv$lambda.min);n1_ridge
ridge.cv$lambda.1se
n2_ridge<-which(ridge.cv$lambda==ridge.cv$lambda.1se);n2_ridge
################### use the best lambda to predict the response,use"type = "response"" to predict the probability rather than log-odds
ridge_prob_hat=predict(object = ridge.fit,newx = x.test1,type = "response",s=ridge.cv$lambda.1se)
# For predicted probabilities greater than 0.4, assign Y to be "Yes"; otherwise assign Y to be "No"
glm_predl = rep("No",26760)
glm_predl[ridge_prob_hat > .4] = "Yes"
# Confusion matrix
table(glm_predl, y.test)
#proporation of making correct classification
suc.rate.ridge=mean(glm_predl==y.test);suc.rate.ridge
#Misclassfication error rate
error.rate.ridge=mean(glm_predl!=y.test);error.rate.ridge
#####################Finally, we refit our ridge regression model on the full data set,
######using the value of optimal lambda chosen by cross-validation, and examine the coefficient estimates.
ridge_out=glmnet(x,y,alpha=0,family="binomial")
predict (ridge_out,type="coefficients",s=ridge.cv$lambda.1se)[1:63,]

####################### Method3: use logistic-lasso model to select variables
###################### use"family="binomial"" because wo are doing binary classification, use"alpha = 1"  to do lasso penalty
lasso.fit=glmnet(x.train1,y.train1,alpha = 1,family="binomial")
plot(lasso.fit,xvar="lambda", label= TRUE)
plot(lasso.fit,xvar="dev", label= TRUE)
###################### use cross-validation to select the best lambda for lasso,by default in k-fold cross validation k is set to be 10
lasso.cv <-cv.glmnet(x.train1, y.train1,alpha = 1,family="binomial")
## Plot of CV mse vs log (lambda)
plot(lasso.cv)
coef(lasso.cv)
##################### select the best lambda
#################### select a model using the one-standard-error rule. We first calculate the one standard error rule
###################select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve.
###################The rationale here is that if a set of models appear to be more or less equally good, 
################## then we might as well choose the simplest model, that is, the model with the smallest number of predictors
lasso.cv$lambda.min
n1_lasso<-which(lasso.cv$lambda==lasso.cv$lambda.min);n1_lasso
lasso.cv$lambda.1se
n2_lasso<-which(lasso.cv$lambda==lasso.cv$lambda.1se);n2_lasso
###############use the best lambda to refit lasso and predict the response,
####use"type = "response"" to predict the probability rather than log-odds
lasso_prob_hat<-predict(object = lasso.fit,newx = x.test1,type = "response",s=lasso.cv$lambda.1se)
# For predicted probabilities greater than 0.4, assign Y to be "Yes"; otherwise assign Y to be "No"
glm_pred2 = rep("No",26760)
glm_pred2[lasso_prob_hat > .4] = "Yes"
# Confusion matrix
table(glm_pred2, y.test)
#proporation of making correct classification
suc.rate.lasso=mean(glm_pred2==y.test);suc.rate.lasso
#Misclassfication error rate
error.rate.lasso=mean(glm_pred2!=y.test);error.rate.lasso
#####################Finally, we refit our ridge regression model on the full data set,using the value of optimal lambda
#########chosen by cross-validation, and examine the coefficient estimates.
lasso_out=glmnet(x,y,alpha=1,family="binomial")
a=predict (lasso_out,type="coefficients",s=lasso.cv$lambda.1se)[1:63,];a
##################### remove the coefficients which is 0
remove_lasso<-which(a==0)
a[-c(remove_lasso)]


############################### Method4:LDA
#############remove the categorical predictors because they obviously break the normal distribution assumption

library(MASS)
Rain_clean.train_remove=subset(Rain_clean.train, select = -c(WindGustDir,WindDir9am,WindDir3pm,RainToday) )
Rain_clean.test_remove=subset(Rain_clean.test, select = -c(WindGustDir,WindDir9am,WindDir3pm,RainToday) )
####################using the training data to fit lda
lda.fit=lda(RainTomorrow~.,data =Rain_clean.train_remove)
lda.fit
plot(lda.fit)
################### predict the result for test data set
lda.pred=predict (lda.fit , Rain_clean.test_remove)
names(lda.pred)
lda.pred.posterior=predict(lda.fit,Rain_clean.test_remove)$posterior
head(lda.pred.posterior)
lda.pred.class=predict(lda.fit,Rain_clean.test_remove)$class
head(lda.pred.class)
#confusion matrix using lda
table(lda.pred.class,y.test)
#proporation of making correct classification
suc.rate.lda=mean(lda.pred.class==y.test);suc.rate.lda
#Misclassfication error rate
error.rate.lda=mean(lda.pred.class!=y.test);error.rate.lda


############################### Method5: QDA
############# As in lda, remove the categorical predictors because they obviously break the normal distribution assumption
####################using the training data to fit QDA
qda.fit = qda(RainTomorrow ~ ., data =Rain_clean.train_remove)
qda.fit
################### predict the result for test data set
qda.pred = predict(qda.fit, Rain_clean.test_remove)
qda.pred.class = predict(qda.fit, Rain_clean.test_remove)$class
# Confusion matrix
table(qda.pred.class, y.test)
#proporation of making correct classification
suc.rate.qda=mean(qda.pred.class==y.test);suc.rate.qda
#Misclassfication error rate
error.rate.qda=mean(qda.pred.class!=y.test);error.rate.qda

########################### Method6: Classification Tree
library(tree)
####### use the tree() function to fit a classification tree in order to predict whether it will rain using all variables
tree.Rain_clean <- tree(RainTomorrow ~ ., data = Rain_clean)
#####The summary() function lists the variables that are used as internal nodes in the tree, number of terminal nodes and the traning error rate
summary(tree.Rain_clean)
################ Show the tree plot graphically
plot(tree.Rain_clean)
mean(Rain_origin$Humidity3pm)
sd(Rain_origin$Humidity3pm)
################ use text() function to display nodel labels; "pretty=0" to include the category names 
######## for any qualitative predictors, rather than simply displaying a letter for each category
text(tree.Rain_clean, pretty = 0, cex = 1)
###############output corresponding to each branch of the tree
tree.Rain_clean

################### We estimate the misclassification rate using test data
################### Run classification tree on the traning data
tree.Rain_clean <-tree(RainTomorrow~., Rain_clean.train)
## Predict the class on the test data
tree.pred <-predict(tree.Rain_clean, Rain_clean.test, type="class")
## Confusion matrix
table(tree.pred, y.test)
#proporation of making correct classification
suc.rate.tree=mean(tree.pred==y.test);suc.rate.tree
#Misclassfication error rate
error.rate.tree=mean(tree.pred!=y.test);error.rate.tree

## We next consider whether prunning the tree might leadt to improved results
## The function cv.tree() performs cross-validation in order to determine the optimal level of tree complexity
## Cost complexity pruning is used to select a sequence of trees for consideration
## We use the argument FUN=prune.misclass to indicate that we want the classification error rate to guild 
##the cross-validation and pruning process,rather than the default for the cv.tree() which is deviance
set.seed(3)
cv.Rain_clean <-cv.tree(tree.Rain_clean, FUN = prune.misclass)
## size: the number of terminal nodes of each tree considered
## dev: corresponding error rate
## k: value of the cost-complexity parameter, which corresponds to alpha used in our slides
names(cv.Rain_clean)
cv.Rain_clean
par(mfrow=c(1,2))
plot(cv.Rain_clean$size, cv.Rain_clean$dev, type="b")
plot(cv.Rain_clean$k, cv.Rain_clean$dev, type="b")

## The optimal number of terminal node is 3 and we display the pruned tree graphically
par(mfrow=c(1,1))
prune.Rain_clean <-prune.misclass(tree.Rain_clean, best=4)
plot(prune.Rain_clean)
text(prune.Rain_clean, pretty=0)
## Compute the test error rate using the pruned tree 
tree.pred <-predict(prune.Rain_clean, Rain_clean.test, type="class")
## Confusion matrix
table(tree.pred, y.test)
#proporation of making correct classification
suc.rate.tree=mean(tree.pred==y.test);suc.rate.tree
#Misclassfication error rate
error.rate.tree=mean(tree.pred!=y.test);error.rate.tree

#################### Method7: Bagging 
library(randomForest)
set.seed(3)
## Recall that bagging is simply a special case of a random forest with m=p, here we use mtry=20
bag.Rain_clean <- randomForest(RainTomorrow~., data=Rain_clean.train, mtry=20, importance=TRUE)
bag.Rain_clean
## We estimate the test error using test data
## Predict the class on the test data
bag.pred <-predict(bag.Rain_clean, Rain_clean.test, type="class")
## Confusion matrix
table(bag.pred, y.test)
#proporation of making correct classification
suc.rate.bag=mean(bag.pred==y.test);suc.rate.bag
#Misclassfication error rate
error.rate.bag=mean(bag.pred!=y.test);error.rate.bag
## We can view the importance of each variable
importance(bag.Rain_clean)
varImpPlot(bag.Rain_clean)

###################### Method8: Random Forest
################ Recall that random forest usually choose m=p^(1/2), p=20 so here we use mtry=4
forest.Rain_clean <- randomForest(RainTomorrow~., data=Rain_clean.train, mtry=4, importance=TRUE)
forest.Rain_clean
## We estimate the test error using test data
## Predict the class on the test data
forest.pred <-predict(forest.Rain_clean, Rain_clean.test, type="class")
## Confusion matrix
table(forest.pred, y.test)
#proporation of making correct classification
suc.rate.forest=mean(forest.pred==y.test);suc.rate.forest
#Misclassfication error rate
error.rate.forest=mean(forest.pred!=y.test);error.rate.forest
## We can view the importance of each variable
importance(forest.Rain_clean)
varImpPlot(forest.Rain_clean)

########################## Method9: Boosting
library(gbm)
set.seed (3)
a=Rain_clean
a$RainTomorrow1=(a$RainTomorrow=="Yes")*1
a=a[,-21]
boost.Rain_clean = gbm( RainTomorrow1 ~ ., data = a[train , ],n.trees = 5000, distribution="bernoulli",interaction.depth = 3)
print(boost.Rain_clean)
summary(boost.Rain_clean)
# We see that "Humidity3pm" "WindDir9am" "WindDir3pm" "WindGustDir""sunshine"and"Pressure3pm"are by far the most important variables
boost.Rain_clean_cv=gbm(RainTomorrow1 ~ ., data = a[train , ],n.trees = 1500, distribution="bernoulli",interaction.depth = 2,shrinkage=0.01,cv.folds = 5)
ntree_opt_cv=gbm.perf(boost.Rain_clean_cv, method = "cv")
ntree_opt_oob=gbm.perf(boost.Rain_clean_cv, method = "OOB")
print(ntree_opt_cv) #optimum number of trees  error on test (green line) and train data set (black line)
print(ntree_opt_oob)
boost_predict=predict.gbm(boost.Rain_clean_cv,data=a[-train,],n.trees=ntree_opt_oob,type ="response" )
PredictionsBinaries=as.factor(ifelse(boost_predict>0.4,1,0))
boost_predictl = rep(0,26760)
boost_predictl[boost_predict > .4] = 1
yy.test = rep(0,26760)
yy.test=(y.test=="Yes")*1
# Confusion matrix
table(boost_predictl, yy.test)
#proporation of making correct classification
suc.rate.boost=mean(boost_predictl== yy.test);suc.rate.boost
# Misclassfication error rate
error.rate.boost=mean(boost_predictl!= yy.test);error.rate.boost



############################ Method 11:stepwise methods
########################## stepwise backward using AIC
full.model = glm(RainTomorrow~ .,data =Rain_clean.train,family = binomial)

AICstep.model1 <- stepAIC(full.model, direction = "backward")
coef(AICstep.model1)
glm_sum_AIC1=summary(AICstep.model1)
names(glm_sum_AIC1)
glm_sum_AIC1$terms
AICstep.model1$anova
probabilities_AIC1=predict(AICstep.model1,Rain_clean.test, type = "response")
AIC1_pred = rep(0, 26760)
AIC1_pred[probabilities_AIC1 > .4] = 1
yy.test = rep(0,26760)
yy.test=(y.test=="Yes")*1
# Confusion matrix
table(AIC1_pred, yy.test)
#proporation of making correct classification
suc.rate.AIC1=mean(AIC1_pred==yy.test);suc.rate.AIC1
# Misclassfication error rate
error.rate.AIC1=mean(AIC1_pred!=yy.test);error.rate.AIC1

########################## stepwise backward using BIC
BICstep.model1<-stepAIC(full.model,direction = "backward",k=log(26760))
coef(BICstep.model1)
glm_sum_BIC1=summary(BICstep.model1)
names(glm_sum_BIC1)
glm_sum_BIC1$terms
BICstep.model1$anova

probabilities_BIC1=predict(BICstep.model1,Rain_clean.test, type = "response")
BIC1_pred = rep(0, 26760)
BIC1_pred[probabilities_BIC1 > .4] = 1
yy.test = rep(0,26760)
yy.test=(y.test=="Yes")*1
# Confusion matrix
table(BIC1_pred, yy.test)
#proporation of making correct classification
suc.rate.BIC1=mean(BIC1_pred==yy.test);suc.rate.BIC1
# Misclassfication error rate
error.rate.BIC1=mean(BIC1_pred!=yy.test);error.rate.BIC1


########################## stepwise forward using AIC
AICstep.model2 <- stepAIC(full.model, direction = "forward")
coef(AICstep.model2)
glm_sum_AIC2=summary(AICstep.model2)
names(glm_sum_AIC2)
glm_sum_AIC2$terms
AICstep.model2$anova
probabilities_AIC2=predict(AICstep.model2,Rain_clean.test, type = "response")
AIC2_pred = rep(0, 26760)
AIC2_pred[probabilities_AIC2 > .4] = 1
yy.test = rep(0,26760)
yy.test=(y.test=="Yes")*1
# Confusion matrix
table(AIC2_pred, yy.test)
#proporation of making correct classification
suc.rate.AIC2=mean(AIC2_pred==yy.test);suc.rate.AIC2
# Misclassfication error rate
error.rate.AIC2=mean(AIC2_pred!=yy.test);error.rate.AIC2

########################## stepwise forward using BIC
BICstep.model2<-stepAIC(full.model,direction = "forward",k=log(26760))
coef(BICstep.model2)
glm_sum_BIC2=summary(BICstep.model2)
names(glm_sum_BIC2)
glm_sum_BIC2$terms
BICstep.model2$anova

probabilities_BIC2=predict(BICstep.model1,Rain_clean.test, type = "response")
BIC2_pred = rep(0, 26760)
BIC2_pred[probabilities_BIC2 > .4] = 1
yy.test = rep(0,26760)
yy.test=(y.test=="Yes")*1
# Confusion matrix
table(BIC2_pred, yy.test)
#proporation of making correct classification
suc.rate.BIC2=mean(BIC2_pred==yy.test);suc.rate.BIC2
# Misclassfication error rate
error.rate.BIC2=mean(BIC2_pred!=yy.test);error.rate.BIC2






























