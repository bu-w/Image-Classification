#PACKAGES-----------------------------------------------------------------------
library(jpeg)
library(e1071)
library(scatterplot3d)
#IMPORT DATA SET----------------------------------------------------------------
pm <- read.csv("~/Desktop/2021WINTER/MATH3333/project/photoMetaData.csv")
n <- nrow(pm)
set.seed(3333)

# partitioning the dataset into 50% training set and 50% testing set.
# in logic form
trainFlag <- (runif(n) > 0.5)  
y <- as.numeric(pm$category == "outdoor-day")

# read variable "name" for all images
X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/2021WINTER/MATH3333/project/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median) #extract 3 median intensities of each color channel
  print(sprintf("%03d / %03d", j, n))
}

#OVERVIEW OF DATA---------------------------------------------------------------------------------------
scatterplot3d(x = X[,1], y = X[,2], z = X[,3], color = c("blue", "red")[as.factor(y)], pch = 19)

#BUILD GLM MODEL---------------------------------------------------------------------------------------
out <- glm(y ~ X, family=binomial, subset=trainFlag)
out$iter
summary(out)

#CLASSIFICATION---------------------------------------------------------------------------------------
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
y[order(pred)] #prediction outcome in training data set
y[!trainFlag][order(pred[!trainFlag])] #prediction outcome in testing data set

mean((as.numeric(pred > 0.5) == y)[trainFlag])
#[1] 0.729798
# 77% of photo belongs to y in training data set

mean((as.numeric(pred > 0.5) == y)[!trainFlag]) 
#[1] 0.7747525
# 73% of photo belongs to y in testing data set

#ANALYSIS---------------------------------------------------------------------------------------
#misclassification table
tt<-table(y[!trainFlag],y[!trainFlag][order(pred[!trainFlag])],dnn=c("actual group","predicted group"))
tt
#misclassification rate
1-sum(diag(tt))/sum(tt)
#[1] 0.3663366

## ROC curve (see lecture 12)
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
# y[!trainFlag] means the true value of y in testing data set
# pred[!trainFlag] means the predicted value of y in testing data set
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc
#[1] 0.817901

#SVM MODEL--------------------------------------------------------------------------------
svm_model <- svm(X,y,subset=trainFlag,type="C-classification",kernel = "radial")
summary(svm_model)
pred_svm <- predict(svm_model,X[!trainFlag,])
pred_svm
#ANALYSIS------------------------------------------------------------------------------------
tt_svm <- table(y[!trainFlag],pred_svm,dnn=c("actual group","predicted group"))
tt_svm
#misclassification rate
1-sum(diag(tt_svm))/sum(tt_svm)
#[1] 0.220297

#roc
r_svm <- roc(y[!trainFlag],order(pred_svm))
# y[!trainFlag] means the true value of y in testing data set
# pred[!trainFlag] means the predicted value of y in testing data set
plot(r_svm$fpr, r_svm$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")
auc(r_svm)
# [1] 0.5238525

#EXTRACT MORE INFORMATION FROM THE IMAGES----------------------------------------------------------------
X <- matrix(NA, ncol=3*(length(seq(0,1,0.01))), nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/2021WINTER/MATH3333/project/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,quantile,probs=seq(0,1,0.01)) 
  print(sprintf("%03d / %03d", j, n))
}

#logistic regression
out <- glm(y ~ X, family=binomial, subset=trainFlag)
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
mean((as.numeric(pred > 0.5) == y)[trainFlag])
#[1] 1
mean((as.numeric(pred > 0.5) == y)[!trainFlag]) 
#[1] 0.6089109
#misclassification table
tt<-table(y[!trainFlag],y[!trainFlag][order(pred[!trainFlag])],dnn=c("actual group","predicted group"))
tt
#misclassification rate
1-sum(diag(tt))/sum(tt)
#[1] 0.4059406

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")
glmAuc <- auc(r)
glmAuc
#[1] 0.6105962

# svm
svm_model <- svm(X,y,subset=trainFlag,type="C-classification",kernel = "radial")
summary(svm_model)
pred_svm <- predict(svm_model,X[!trainFlag,])
pred_svm
tt_svm <- table(y[!trainFlag],pred_svm,dnn=c("actual group","predicted group"))
tt_svm
#misclassification rate
1-sum(diag(tt_svm))/sum(tt_svm)
#[1] 0.1980198
#roc
r_svm <- roc(y[!trainFlag],order(pred_svm))
# y[!trainFlag] means the true value of y in testing data set
# pred[!trainFlag] means the predicted value of y in testing data set
plot(r_svm$fpr, r_svm$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")
auc(r_svm)
#[1] 0.492113

#CHANGE THE PROPORTION OF TRAINING SET FOR UPDATED VARIABLE X---------------------------------------------------------
trainFlag <- (runif(n) > 0.2)
out <- glm(y ~ X, family=binomial, subset=trainFlag)
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
mean((as.numeric(pred > 0.5) == y)[trainFlag])
#[1] 1
mean((as.numeric(pred > 0.5) == y)[!trainFlag]) 
#[1] 0.6424242
#misclassification table
tt<-table(y[!trainFlag],y[!trainFlag][order(pred[!trainFlag])],dnn=c("actual group","predicted group"))
tt
#misclassification rate
1-sum(diag(tt))/sum(tt)
#[1] 0.4242424
r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")
glmAuc <- auc(r)
glmAuc
#[1] 0.7479146

# svm
svm_model <- svm(X,y,subset=trainFlag,type="C-classification",kernel = "radial")
summary(svm_model)
pred_svm <- predict(svm_model,X[!trainFlag,])
pred_svm
tt_svm <- table(y[!trainFlag],pred_svm,dnn=c("actual group","predicted group"))
tt_svm
#misclassification rate
1-sum(diag(tt_svm))/sum(tt_svm)
#[1] 0.2
#roc
r_svm <- roc(y[!trainFlag],order(pred_svm))
plot(r_svm$fpr, r_svm$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")
auc(r_svm)
#[1] 0.5523857


