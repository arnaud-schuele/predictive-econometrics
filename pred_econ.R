### title: "Data Analytics I - Individual Assignment"
### author: "Arnaud Schuele"
### date: "08.01.2020"

### "This code may take a bit of time to be run. 
###  Thank you for your patience."
 
############################
##### 0. Preparation ####### 
############################
### Setting my working directory
# Define the working directory
wdAS = "~/Desktop/Predictive_Econometrics/Individual_Assignment"

# Set the specific work directory
setwd(wdAS)

### Install and load the R packages
# Install all necessary packages
#install.packages("corrplot")
#etc

# Load all necessary packages
library(Matrix)
library(data.table) 
library(dplyr)    
library(corrplot)
library(ggplot2)
library(fastDummies) #for dummy treatment
library(leaps)
library(glmnet) 
library(lattice)
library(psych)
library(caret)

### Load the data
# Import the training and test dataset
train <- fread("juice.csv")
test <- fread("new_grocery.csv")

############################
##### 1. Data cleaning #####
############################ 
### We check to see what class of data we are dealing with.
sapply(train, class)
sapply(test, class)

### Brands
# As the brands were populated in character class, we want to ensure that there was no typo.
as.data.frame(table(test$brand))
as.data.frame(table(train$brand)) #there are exactly three similar brands in both datasets.

### We get a summary
summary(train) 
# We note in the data summary that we have 465 NA's in the train price column.
summary(test)
# We see in the data summary that we have 178 NA's in the test price column.

### We drop all lines with NA's
# In both datasets
complete_train<-na.omit(train)  
complete_test<-na.omit(test) 
# We double check that we indeed dropped all NAs in train
any(is.na(complete_train)) #confirmed
any(is.na(complete_test)) #confirmed
# We eventually get
summary(complete_train) 
# We see that we have reasonably realistic min and max values for id and price (e.g. no negative values).
summary(complete_test)
# We see that we have reasonably realistic min and max values for id and price (e.g. no negative values).

### We drop the column V1, as it is irrelevant.
complete_train$V1 <- NULL
complete_test$V1 <- NULL

### We plot a histogram of the sales.
hist(complete_train$sales,
     main = "Train (complete data)",
     xlab = "Train sales")
# We note how skewed to the right the data is. Thus, we will store log sales for more convenience.
log_sales <- log(complete_train$sales)

### We plot a histogram of the log sales.
hist(log_sales,
     main = "Train log sales (complete data)",
     xlab = "Train log sales")
# The distribution is now closer to normal.

### We plot a histogram of the price.
hist(complete_train$price,
     main = "Train (complete data)",
     xlab = "Train price")
# We note that the distribution is less skewed. Hence, computing log prices might be less useful.
# The same happens with test price.
hist(complete_test$price,
     main = "Test (complete data)",
     xlab = "Test price")
 
### Clean data visualization
# We plot the boxplots of the sales by brands, in order to spot any informative pattern or potential issue.
ggplot(complete_train, aes(x=brand,y=log_sales, fill=brand)) +
  geom_boxplot()+ 
  xlab(label = "brand") +
  ylab(label = "sales") +
  theme(axis.text.x = element_text(angle=30, hjust=1, vjust=1))+
  theme(legend.position="none")+
  ggtitle("GGplot by brand")
#There is none.
  
### Treatment of dummies
# For more convenience in the next steps, we will split the categorical variable "brand" into separate
# dummies.
complete_train <- fastDummies::dummy_cols(complete_train) 
complete_test <- fastDummies::dummy_cols(complete_test)
# We then need to remove the first dummy from the category, in order to avoid perfect multicollinearity 
# issues in the models. This is something that R would have done automatically when running regressions.
complete_train$brand_minute.maid <- NULL #drop variable brand
complete_test$brand_minute.maid <- NULL #drop variable brand
complete_train$brand <- NULL #drop variable brand
complete_test$brand <- NULL #drop variable brand

### Columns re-ordering
# For more convenience in the future steps, we reorder the columns to put sales on the right side.
complete_train<- complete_train[,c(3, 4, 5, 6, 2, 1)]

### Remarks
# a. Non-linearity: using GG-plot2, we see that a model allowing for a non-linerar relationship
# between price and log sales might lead to more accurate predictions in our case.
ggplot(complete_train, aes(x=price, y=log_sales))+
  geom_point()+
  geom_smooth()+
  geom_smooth(method="lm", linetype="dashed",
              color="darkred") 
# Given the nature of our price data, and based on aforementionned remarks, we will build, if possible, 
# log-level models which will allow to fit polynomial regressions. Note that this non-linear relationship 
# would disappear if using log-log models.

# b. Training-test split:
num_obs = nrow(complete_train) #checks the size of the sample.
# We split our juice data set into a train subsample train and test subsample.
set.seed(42)
idx = createDataPartition(log_sales, p = 0.66, list = FALSE)
trn = complete_train[idx, ]
tst = complete_train[-idx, ]
# Storing log sales for each subsample.
log_sales.trn<-log(trn$sales)
log_sales.tst<-log(tst$sales)

### We are now set to go.

############################
######## 2. Ridge ########## 
############################
### a. Cross Validation.
# Perform 10-fold cross-validation to select lambda.
lambdas_to_try <- 10^seq(-5, 5, length.out = 100)
set.seed(28000) 

#We specify the whole path for Ridge.
ridge.cv <- cv.glmnet(data.matrix(trn[,c(1:4)]), log_sales.trn,
                      type.measure = "mse", lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 10, family= "gaussian", alpha = 0)
plot(ridge.cv)
# Best cross-validated lambda.
lambda_cv <- ridge.cv$lambda.min

# The model with cross-validation.
model_cv <- glmnet(data.matrix(trn[,c(1:4)]), log_sales.trn, alpha = 0, 
                   lambda = lambda_cv, standardize = TRUE)
# For comparability with other methods, we predict the sales for the training sample
tst$pred_ridge_cv <- exp(predict(ridge.cv, newx = data.matrix(tst[,c(1:4)]), 
                                            s = ridge.cv$lambda.min))
predMSE_ridge_cv <- mean((tst$sales-tst$pred_ridge_cv)^2)
R2.ridge_cv <- R2(tst$pred_ridge_cv, tst$sales)

### b. Information Criterion: AIC and BIC
# Use information criteria to select lambda. Note that this step might take few minutes.
X_scaled <- scale(data.matrix(trn[,c(1:4)]))
aic <- c() 
bic <- c()

for (lambda in seq(lambdas_to_try)) {
  # We run the model.
  model <- glmnet(data.matrix(trn[,c(1:4)]), 
                  log_sales.trn, alpha = 0, lambda = lambdas_to_try[lambda], standardize = TRUE)
  # We extract the coefficients and residuals (remove first row for the intercept).
  betas <- as.vector((as.matrix(coef(model))[-1, ]))
  resid <- log_sales.trn - (X_scaled %*% betas)
  # We compute the hat-matrix and degrees of freedom.
  ld <- lambdas_to_try[lambda] * diag(ncol(X_scaled))
  H <- X_scaled %*% solve(t(X_scaled) %*% X_scaled + ld) %*% t(X_scaled)
  df <- tr(H)
  # Finally, we compute the information criteria.
  aic[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df
  bic[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df * log(nrow(X_scaled))
}
# Optimal lambdas according to both criteria
lambda_aic <- lambdas_to_try[which.min(aic)]
lambda_bic <- lambdas_to_try[which.min(bic)]

# The model with AIC
model_aic <- glmnet(data.matrix(trn[,c(1:4)]), log_sales.trn, alpha = 0, 
                    lambda = lambda_aic, standardize = TRUE)
# For comparability with other methods, we predict the sales for the training sample.
tst$pred_ridge_aic <- exp(predict(model_aic, newx = data.matrix(tst[,c(1:4)])))
predMSE_ridge_aic <- mean((tst$sales-tst$pred_ridge_aic)^2)
R2.ridge_aic <- R2(tst$pred_ridge_aic, tst$sales)

# The model with BIC
model_bic <- glmnet(data.matrix(trn[,c(1:4)]), log_sales.trn, 
                    alpha = 0, lambda = lambda_bic, standardize = TRUE)
# For comparability with other methods, we predict the sales for the training sample.
tst$pred_ridge_bic <- exp(predict(model_bic, newx = data.matrix(tst[,c(1:4)])))
predMSE_ridge_bic <- mean((tst$sales-tst$pred_ridge_bic)^2)
R2.ridge_bic <- R2(tst$pred_ridge_bic, tst$sales)
# Unsurprisingly, the R2 of the cross-validated model is the highest amongst the three models.

### See how increasing lambda shrinks the coefficients
# Each line shows coefficients for one variables, for different lambdas.
# The higher the lambda, the more the coefficients are shrinked towards zero.
res <- glmnet(data.matrix(trn[,c(1:4)]), log_sales.trn, alpha = 0, 
              lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(data.matrix(trn[,c(1:4)])), cex = .7)

############################
######### 3. OLS ########### 
############################
### Best subset selection approach
# Model accuracy: to select a fit, we will have a look at the R-squared, adjusted R-squred, BIC and Cp.
# As we will create different OLS fits, we will first of all code functions for all fits.
# We create a function for retrieving the adjusted R squared of our fits.
adj.rsq = function(model) {
  summary(model)$adj.r.sq
}
 
# Model complexity
# With nested OLS models, we define that the more predictors that a model has,the more complex it is. 
# This step partly explains why we treated the k-categorical variable into k-1 different dummies.
complexity = function(model) {
  length(coef(model)) - 1
}

# First of all, we need to know how many covariates to include in our model.
# We will start with the simplest possible linear model, that is, a model with no predictors.
# Each successive model we fit will be increasingly flexible using both interactions and polynomial terms.
# Mathematically, the R-squared of these model fit will increase as we add variables, while the adjusted 
# R-squared might eventually change behaviour. Indeed, we expect the test error to decrease a number of times, 
# then eventually start going up, as a result of overfitting (if any). 
ols_0 = lm(log_sales.trn ~ 1, data = trn)
complexity(ols_0) 
adj.rsq(ols_0)
 
#From ols_0, we will add variables, interations and polynomial terms until the full model (below).
ols_12 = lm(log_sales.trn ~ 
            feat+price+brand_dominicks+brand_tropicana
            +I(price^2)
            +feat*price+feat*brand_dominicks+feat*brand_tropicana
            +price*brand_dominicks+price*brand_tropicana+
            +price*brand_dominicks*feat+price*brand_tropicana*feat,
            data = trn) 
complexity(ols_12)
adj.rsq(ols_12) 
# We do not include any interaction term between brand dummies, as we want to 
# avoid perfect collinearity issues.

#On the way to model ols_12, a natural candidate for regression would be ols_4.
ols_4 = lm(log_sales.trn ~ feat+price+brand_dominicks+brand_tropicana,
           data = trn)
complexity(ols_4) 
adj.rsq(ols_4)

### Accuracy check with our mode
#The following command helps us to vizaualize the accuracy of some candidate fits.
plot(ols_4) #hit <Return> to the next plot
# Graph 1: a typical issue of that occurs whoen we fit a linear regression model to a particular data set
# is the non-linearity of the response-predictor relationships, as linear regression model assumes 
# that there is a straight-line relationship between the predictors and the response.

# Graph 2: the normal Q-Q plot is used to examine whether the residuals are normally distributed 
# (i.e. they perfectly lie on the straight dashed line.)

# Grpah 3: the scale-Location used to check the homogeneity of variance of the residuals (homoscedasticity). 
# A horizontal line with equally spread points is a good indication of homoscedasticity. 
# This is not the case in our example, where we have a heteroscedasticity problem.
 
# Graph 4: the residuals vs leverage plut is used to identify influential cases, that is extreme 
# values that might influence the regression results when included or excluded from the analysis.

### In order to select the best model, we would need to to have a look at every model 
# between ols_0 and ols_12. Here we could imagine using the forward selection.
regfit.fwd=regsubsets (log_sales.trn ~
                        feat+price+brand_dominicks+brand_tropicana
                       +I(price^2)
                       +feat*price+feat*brand_dominicks+feat*brand_tropicana
                       +price*brand_dominicks+price*brand_tropicana+
                       +price*brand_dominicks*feat+price*brand_tropicana*feat, 
                        data = trn, nvmax =12, method ="forward")
summary (regfit.fwd) #the asterisks mark the data selection in an optimized model.

# This would allow us to plot the evolution of different metrics and to chose the 
# best level of model complexity.
par(mfrow=c(2,2))
reg.summary =summary (regfit.fwd)

which.min (reg.summary$rss) #RSS to be minimized
plot(reg.summary$rss ,xlab=" Number of Variables ",ylab=" RSS",
     type="l")
points (12, reg.summary$rss[12], col ="red",cex =2, pch =20)

which.max (reg.summary$adjr2) #adjusted R2 to be maximized
plot(reg.summary$adjr2 ,xlab =" Number of Variables ",
     ylab=" Adjusted R2",type="l")
points (12, reg.summary$adjr2[12], col ="red",cex =2, pch =20)

which.min (reg.summary$cp ) #Cp (test error) to be minimized
plot(reg.summary$cp ,xlab =" Number of Variables ",ylab="Cp",
     type='l')
points (11, reg.summary$cp [11], col ="red",cex =2, pch =20)

which.min (reg.summary$bic ) #BIC (test error) to be minimized
plot(reg.summary$bic ,xlab =" Number of Variables ",ylab="BIC",
     type='l')
points (10, reg.summary$bic [10], col =" red",cex =2, pch =20)
par(mfrow=c(1,1))

# While the R square indicates model 12, other metrics suggest some model with a lower number of 
# variables, in order to avoid overfitting. Here, after comparing the evolution of the R-squared, 
# adjusted R-squred, bic and Cp of the models, it could be tempting to directly select the best one 
# (best being here defined by RSS).

# However, since we are in the specific case were we added interaction coefficients, we must respect
# the following hierarchy principle: we should include any coefficient whose interaction is included 
# in the model. Therefore, in our case, we cannot use the method presented above for the full model.

# Taking this into account, we will "manually" reduce the variance at the cost of introducing some bias.
# In other words, we will reduce the complexity of the model to 11 covariables. In order to do so, we will 
# drop the interaction term with the highest p-value in the full model. Here, we acknowledge that we somehow 
# take a risk (overfitting-wise) not to reduce the variance further, as some metrics suggest to further 
# reduce the complexity of the model. Note that we chose to use 11 covariates exogenously, given the evolution 
# of the metrics we plotted.
summary(ols_12)
# feat*price*brand_tropicana has the highest p-value amongst the interaction terms.

### Conclusion
# The choice of our OLS model might not be optimal, due to the fact that could not compare the 
# evolution of the predicted R squared amongst the optimal model at each complexity level.
# It nevertheless remain a fairly good candidate to be compared with other methods.

### log-level
# Chosen ols model.
ols_11.loglev = lm(log_sales.trn ~ 
             feat+price+brand_dominicks+brand_tropicana
            +I(price^2)
            +feat*price+feat*brand_dominicks+feat*brand_tropicana
            +price*brand_dominicks+price*brand_tropicana+
            +price*brand_dominicks*feat,
             data = trn)
complexity(ols_11.loglev)
adj.rsq(ols_11.loglev) 
# For comparability with other methods, we predict the sales for the training sample.
tst$pred_ols_loglev <- exp(predict(ols_11.loglev, newdata=tst))
predMSE_ols_loglev <- mean((tst$sales - tst$pred_ols_loglev)^2)
R2.ols_loglev <- R2(tst$pred_ols_loglev, tst$sales)

### Accuracy check with our model.
plot(ols_11.loglev) #hit <Return> to the next plot
# From the plots from ols_4, we see that:
# - the residuals and fitted values now have a more linear relationship
# - more points now lie on the QQ plot
# All in all, our model has become more accurate.
  
### log-log
# Having a quick look at the log-log level, we plot the same model, in a log-log dimension. 
ols_11.loglog = lm(log_sales.trn ~ 
                    feat+log(price)+brand_dominicks+brand_tropicana
                   +I(log(price)^2)
                   +feat*log(price)+feat*brand_dominicks+feat*brand_tropicana
                   +log(price)*brand_dominicks+log(price)*brand_tropicana+
                   +log(price)*brand_dominicks*feat,
                   data = trn)
complexity(ols_11.loglog)
adj.rsq(ols_11.loglog) 
# For comparability with other methods, we predict the sales for the training sample
tst$pred_ols_loglog <- exp(predict(ols_11.loglog, newdata=tst))
predMSE_ols_loglog <- mean((tst$sales - tst$pred_ols_loglog)^2)
R2.ols_loglog <- R2(tst$pred_ols_loglog, tst$sales)
# We see that the model performs less well than on the log-level plan. This confirms our 
# earlier hypothesis on the non-linear relationship of prices and log-sales. 
print(adj.rsq(ols_11.loglev))
print(adj.rsq(ols_11.loglog))
# We will just stick to the log-lev OLS model.
 
##################################
######## 4. Elastic net ########## 
##################################
# The elastic net is a combination of Lasso and Ridge.
# It fine-tunes both alpha and lambda.

### a. Simple elastic net
# We specify the variables.
y_trn <-log_sales.trn
X_trn <-data.matrix(trn[,c(1:4)])
X_tst<-data.matrix(tst[,c(1:4)])
 
set.seed(421)
# We set the training control for the model.
rcv_5 = trainControl(method = "repeatedcv", number = 5, repeats = 5)

# We train the model.
hit_elnet_int = train(
  y_trn ~ ., data = cbind(y_trn, X_trn),
  method = "glmnet",
  trControl = rcv_5,
)
#The model provides the following.
hit_elnet_int
 
#We create a function to easily extract the best results for the tuning parameters.
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result}
#We obtain the following.
get_best_result(hit_elnet_int)

# For comparability with other methods, we predict the sales for the training sample.
tst$pred_elnet_simple <- exp(predict(hit_elnet_int, X_tst))
predMSE_elnet_simple <- mean((tst$sales-tst$pred_elnet_simple)^2)
R2.elnet_simple <- R2(tst$pred_elnet_simple, tst$sales)

### b. Expanded elastic net
# We train it again, with an expanded feature space and a larger tuning grid.
def_elnet_int = train(
  y_trn ~ . ^ 2, data = cbind(y_trn, X_trn),
  method = "glmnet",
  trControl = rcv_5, 
  tuneLength = 10)
#The model provides the following.
def_elnet_int
#We extract the best results out of it. 
get_best_result(def_elnet_int)
 
# For comparability with other methods, we predict the sales for the training sample.
tst$pred_elnet_expand <- exp(predict(def_elnet_int, X_tst))
predMSE_elnet_expand <- mean((tst$sales-tst$pred_elnet_expand)^2)
R2.elnet_expand <- R2(tst$pred_elnet_expand, tst$sales)

############################################
######## 6. Final choice of model ########## 
############################################ 
### Print the R squared and the MSE of all models.
all_models <- cbind("R-squared" = c(R2.ridge_aic, R2.ridge_bic, 
                                    R2.ridge_cv, R2.elnet_simple, R2.elnet_expand, 
                                    R2.ols_loglev, R2.ols_loglog), 
                "MSE" = c(predMSE_ridge_aic, predMSE_ridge_bic, predMSE_ridge_cv, 
                          predMSE_elnet_simple, predMSE_elnet_expand, predMSE_ols_loglev,
                          predMSE_ols_loglog))
rownames(all_models) <- c("ridge AIC", "ridge BIC", "ridge CV", "simple elastic net", 
                          "expanded elastic net", "ols log-lev", "ols log-log")
print(all_models)
# By comparing the R squared and predicted RSE of our models, we see that the expanded elastic net 
# has the highest R squared, and that the OLS has the lowest predicted MSE (the expanded elastic 
# net comes second). 

# Therefore, we want to have a look at how similar the new grocery data is compared to the juice 
# data, in order to choose between the expanded elastic net and the OLS model.
summary(complete_test) 
sd(complete_train$price)
sd(trn$price)
sd(complete_test$price)
# From the summary, we see that the means of the different predictors are relatively close, which confirms that
# we could use a model with high complexity which results in high variance and low bias (following the variance-
# bias trade-off).
 
# Hence, we predict the sales in the test sample with our expanded elastic net model.
X.complete_juice<-data.matrix(complete_test[,c(2:5)]) #setting the new data
complete_test$pred_elnet.expand <- exp(predict(def_elnet_int, X.complete_juice))

# Note that in our log-level models, we use the exponential of the ols predictions to transform the data back.
# Nevertheless, we acknowledge that this is an approximation to backtransform ou data, which will slightly 
# bias the predictions. As we do not want to get involved into complex normal correction here, we will 
# assume perfect normality. Moreover, as the bias is similar for all models, it does not interfere with 
# our model selection.
 
# We store the results.
complete_test<- complete_test[,c(2, 3, 4, 5, 1, 6)]
results <- complete_test[,c(5:6)]
   
### Activate the following line in order to write the predictions in a csv with semi-column separator. 
write.csv2(results, "final_results.csv")
   
########################## 
##########################