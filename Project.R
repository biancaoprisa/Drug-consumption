data <- read.csv("student_2497327data.csv")

library(ggplot2)
library(GGally)
library(class)
library(e1071)
library(rpart);
library(rpart.plot)
library(randomForest)
library(gridExtra)

str(data)
#Class in coded as integer, not factor.
data$Class <- as.factor(data$Class)

#Splitting the data into training, validation and test sets. The first column X is removed.
set.seed(267)
n <- nrow(data)
intrain<-sample(c(1:n),n/2)
invalid<-sample((c(1:n)[-intrain]),n/4)
train.data<-data[intrain, -1]
valid.data<-data[invalid, -1]
test.data<- data[-c(intrain,invalid), -1]

dim(train.data)
dim(valid.data)
dim(test.data)

#Performing exploratory analysis on the train.data set.
str(train.data)
summary(train.data)

library(skimr)
skim(train.data)

ggpairs(train.data,
        upper=list(continuous=wrap("points", alpha=0.4)),
        lower="blank", axisLabels="none")

#Agreeablesness (Ascore) seems to be negatively associated with neuroticism.
#It is difficult to tell from the plots, but there might be a couple of outliers present.
#Conscientiousness seems to have a positive relationship with Education and is negatively associated with Neuroticism.
p1 <- ggplot(train.data, aes(x=SS, y=Impulsive, colour = factor(Class))) +
  geom_point(cex = 1);p1

p2 <-  ggplot(train.data) +
  geom_boxplot(aes(x=Class, y=Age)); p2

#We can see from p2 that the median line for the age of people who have consumed drugs lies below the median 
#line for age of the people who haven't consumed drugs.

p3 <- ggplot(train.data) +
  geom_boxplot(aes(x=Class, y= Education)); p3


#The plot is indicating that the level of education of those who have never consumed drugs is higher
#than that of those who have consumed drugs.

grid.arrange(p2, p3, p4, nrow=1)

p4 <- ggplot(train.data) +
  geom_boxplot(aes(x=Class, y= SS)); p4
#The median line for the sensation seeking score of people who have never consumed drugs lies below the median
#line of those who have scored lower on this variable, indicating that higher scores on sensation seeking are associated
#with drug use.

#Similarly, higher scores on the Impulsivity variables seem to be positively associated with drug use.

p5 <- ggplot(train.data, aes(x=Escore, y=Nscore, colour = factor(Class))) +
  geom_point(cex = 2, show.legend = FALSE);p5

#It appears that there is a negative relationship between Neuroticism (Nscore) and Extraversion (Escore). 
#There is a lot of overlap between drug users / non-users.

p6 <- ggplot(train.data, aes(x=SS, y=Oscore, colour = factor(Class))) +
  geom_point(cex = 2, show.legend = FALSE);p6
#There is a positive association between Openess (Oscore) and sensation seeking (SS). Also, the users vs non-users
#are fairly divided on the plot, indicating that the probability of drug use increases with higher values scored on
#on the Oscore and SS variables.


p7 <- ggplot(train.data, aes(x=Ascore, y=Cscore, colour = factor(Class))) +
  geom_point(cex = 1);p7

#Lower levels of Agreeableness (Ascore) are associated with drug use, as well as lower levels of Conscienciousness.

grid.arrange(p5, p6, p7, nrow = 1)

#K Nearest Neighbours

#Looking at the algorithm's performance on the validation data for a range of k
corr.class.rate.knn <- numeric(50)
for(k in 1:50)
{
  pred.class.knn <- knn(train.data[,-12], valid.data[, -12], train.data[,12], k=k)
  corr.class.rate.knn[k] <- sum((pred.class.knn == valid.data$Class)) / length(pred.class.knn)
}

corr.class.rate.knn

plot(c(1:50), corr.class.rate.knn, type = "l",
     main = "Correct Classification Rates for a range of k",
     xlab = "k", ylab = "Correct Classification Rate")

which.max(corr.class.rate.knn)
# we can see from the plot that the best performance is at k=31 on the validation data. 
#We will carry on and try k=31 on the test data set.

pred.valid.knn <- knn(train.data[,-12], valid.data[, -12], train.data[,12], k=31)


#computing the error rate on the validation data for k=31
error.rate.knn <- 1 - sum(diag(table(pred.valid.knn, valid.data$Class))) / length(pred.valid.knn)
error.rate.knn



#Support Vector Machines
#First trying a linear kernel
linear.tune <- tune(svm, Class ~., data=train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.1, 1, 5, 10)))
summary(linear.tune)$best.parameters

#I will try different cost values since the cost value = 0.1 and is at a boundary
linear.tune <- tune(svm, Class ~., data=train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.01, 0.05, 0.1, 0.5)))
summary(linear.tune)$best.parameters

#Again, the cost value = 0.01 is at a boundary so will try once more
linear.tune <- tune(svm, Class ~., data=train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.001, 0.005, 0.01, 0.05)))
summary(linear.tune)$best.parameters 

#the cost value is 0.01 and it is no longer at a boundary
summary(linear.tune)

linear.tune$best.model

#there are 201 support vectors for this model on the training data

valid.pred.linear.svm <- predict(linear.tune$best.model, valid.data)
error.linear.svm <- 1-sum(diag(table(valid.pred.linear.svm, valid.data$Class))) / length(valid.pred.linear.svm)
error.linear.svm


#Now I will try a radial svm
radial.tune <- tune(svm, Class ~., data = train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.1, 1, 5, 10), gamma = c(0.1, 1, 2, 3, 4)))
summary(radial.tune)$best.parameters
#Both cost = 0.1 and gamma = 0.1 are at a boundary so I will try again with smaller values

radial.tune <- tune(svm, Class ~., data = train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.05, 0.1, 0.5, 1), gamma = c(0.001, 0.005, 0.01, 0.05, 0.1)))
summary(radial.tune)$best.parameters

#Cost = 0.01 is no longer at a boundary, but gamma still is at a boundary
radial.tune <- tune(svm, Class ~., data = train.data, type = "C-classification", kernel = "linear",
                    ranges = list(cost = c(0.01, 0.05, 0.1, 0.5), gamma = c(0.0001, 0.0005, 0.001, 0.005, 0.01)))
summary(radial.tune)$best.parameters

radial.tune$best.model

valid.pred.radial.svm <- predict(radial.tune$best.model, valid.data)
error.radial.svm <- 1-sum(diag(table(valid.pred.radial.svm, valid.data$Class))) / length(valid.pred.radial.svm)
error.radial.svm

#Now we will try a polynomial svm with degree = 2
poly.tune<-tune(svm,Class~.,data=train.data,type="C-classification",kernel="polynomial",
                degree=2,ranges=list(cost=c(0.1,1,5),gamma=c(0.01,0.05,0.1),coef0=c(0,1,2,3)))
summary(poly.tune)$best.parameters

#Cost=0.1 and is at a boundary

poly.tune<-tune(svm,Class~.,data=train.data,type="C-classification",kernel="polynomial",
                degree=2,ranges=list(cost=c(0.01,0.05,0.1),gamma=c(0.005,0.01,0.1),coef0=c(0,1,2,3)))
summary(poly.tune)$best.parameters

poly.tune$best.model

#the parameters were chosen to me cost = 0.1, gamma = 0.01 and coef = 2
valid.pred.poly.svm <- predict(poly.tune$best.model, valid.data)
error.poly.svm <- 1-sum(diag(table(valid.pred.poly.svm, valid.data$Class))) / length(valid.pred.poly.svm)
error.poly.svm

#computing the error matrix to compare the performance between the three SVMs
error<-matrix(c(error.linear.svm, error.radial.svm, error.poly.svm),1,3)
colnames(error)<-c("Linear","Radial","Polynomial.2")
round(error,4)

#We can see that the polynomial SVM performs best.


#Trees
#we will first grow a full tree and then prune it
fully.grown.tree <- rpart(Class ~., data = train.data, method = "class", parms = list(split = "information"),
                         cp = -1, minsplit = 2, minbucket = 1)

rpart.plot(fully.grown.tree)

fully.grown.tree$variable.importance
#We can see the most important variables are Oscore, Nscore, SS, Age and Education. The least important variables 
#are Ethnicity and Country.

printcp(fully.grown.tree)
#The lowest xerror is 0.52667 with standard deviation of 0.050858
0.52667 + 0.050858

#We want the largest tree with xerror less than 0.577528, which is the tree with cp = 0.0044444

pruned.tree <- prune(fully.grown.tree, cp = 0.0044444)
rpart.plot(pruned.tree)

pruned.tree

#we now apply the pruned tree to the validation data set and compute the error rate
valid.pred.tree <- predict(pruned.tree, valid.data, type = "class")
error.tree <- 1-sum(diag(table(valid.pred.tree, valid.data$Class))) / length(valid.pred.tree)
error.tree

#we will now try a bagged model
bag.tree <- randomForest(Class ~., data = train.data, method = "class", mtry = 11, ntree = 250)
bag.tree

#applying the bagged tree model to the validation data set and computing the error rate
valid.pred.bag <- predict(bag.tree, valid.data, type = "class")
error.bag <- 1-sum(diag(table(valid.pred.bag, valid.data$Class))) / length(valid.pred.bag)
error.bag  
  
#The bagged model is performing better than the tree

rate.test.bag[2,2]  
#sensitivity

rate.test.bag[1,1]
#specificity
  
#We can see from the table that the KNN method performs best, because it has the lowest error rate 
error.matrix <- matrix(c(error.rate.knn, error.poly.svm, error.bag), 1, 3)
colnames(error.matrix) <- c("KNN", "SVM", "Tree(Bag)")
round(error.matrix, 4)
  

#we now apply the KNN algorithm to the test data and compute the correct classification rate  
pred.test.knn <- knn(train.data[,-12], test.data[, -12], train.data[,12], k=31)
corr.class.rate.test.knn <- sum(pred.test.knn == test.data$Class) / length(pred.test.knn)
corr.class.rate.test.knn

#the future performance of the model with k=31 is 0.76.


tab.test.knn <- table(test.data$Class,pred.test.knn)
tab.test.knn

rate.test.knn <-sweep(tab.test.knn,1,apply(tab.test.knn,1,sum),"/")
rate.test.knn

rate.test.knn[2,2]  
#sensitivity

rate.test.knn[1,1]
#specificity

cross.class.rates<-sweep(tab.test.knn,2,apply(tab.test.knn,2,sum),"/")
cross.class.rates  

cross.class.rates[1,2]
#false discovery rate

