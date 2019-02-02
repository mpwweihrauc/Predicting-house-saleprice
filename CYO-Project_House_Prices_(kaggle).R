# Please download the datasets from the provided link or find them in my provided Github repository, links below, or in the report.
# Link to Kaggle page: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link to my Github page: https://github.com/mpwweihrauc/ML_House_Prices.git
# You will need the test.csv and the train.csv files. There is also a data_description.txt with descriptions for all the different parameters in the dataset.

# We begin by loading/installing all required libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")

# We import the training and testing data subsets (files from Kaggle)
train <- read.csv("train.csv", stringsAsFactors = TRUE)
test <- read.csv("test.csv", stringsAsFactors = TRUE)




###
# Exploratory data analysis
###

# We inspect some basic properties of the dataset. The desired outcome column is named "SalePrice" and denotes a houses' sale price in dollars.
# We are dealing with 1460 different houses and 81 different features of them (including their id),
# such as the year they were built in or their overall condition.

head(train)
dim(train)
str(train)

# Taking a look at the summary, we see that there are some missing values in the following columns/features:
# LotFrontage (Linear feet of street connected to property), MasVnrArea (Masonry veneer type), and GarageYrBlt (Year garage was built).
summary(train)

# We inspect the desired outcome "SalePrice" and can see that house prices range from 34900 to 755000 in the training subset. The median house price is 163000.
summary(train$SalePrice)



#########################
# Dataset manipulations #
#########################

# Are certain features skewed? We visualize SalePrice with a histogram and take note of its skewness.
# House sale prices are right skewed, as a relatively small number of houses has a much larger than average price.
# Only a very small number of houses actually costs more than ~400000 dollars.
train %>%
  ggplot(aes(x = SalePrice)) +
  geom_histogram(bins = 30, binwidth = 5000) +
  scale_x_continuous(breaks = seq(0, 800000, 80000)) +
  ggtitle("Sale price distribution") +
  xlab("Sale price in dollars") +
  ylab("Number of houses")

# What does the distribution look like if we log-transform SalesPrice? Monetary amounts are often
# lognormally distributed, i.e. the log of the data is normally distributed.
# Indeed, the log-transformed SalesPrice looks much more normally distributed.
# Note: log1p(x) computes log(1+x), which enables log-transformation even for 0 values.
train %>%
  ggplot(aes(x = log1p(SalePrice))) +
  geom_histogram(bins = 30) +
  ggtitle("Sale price distribution") +
  xlab("Sale price in dollars") +
  ylab("Number of houses")


###
# Feature engineering
###

# To facilitate feature engineering we temporarily merge train and tet into "dataset"
test$SalePrice <- 0
dataset <- rbind(train, test)

# We fix the column encoding, as sometimes "NA" was put for the absence of a feature (e.g. when there is no pool).
dataset$Alley <- str_replace_na(dataset$Alley, replacement = "NoAccess") # Replace NA in the Alley column with "None"

dataset$PoolQC <- str_replace_na(dataset$PoolQC, replacement = "NoPool") # Replace NA in the PoolQC column with "None"


dataset$Fence <- str_replace_na(dataset$Fence, replacement = "NoFence") # Replace NA in the Fence column with "None"


dataset$MiscFeature <- str_replace_na(dataset$MiscFeature, replacement = "None") # Replace NA in the MiscFeature column with "None"


dataset$FireplaceQu <- str_replace_na(dataset$FireplaceQu, replacement = "NoFireplace") # Replace NA in the FirePlaceQu column with "None"


dataset$GarageCond <- str_replace_na(dataset$GarageCond, replacement = "NoGarage") # Replace NA in the GarageCond column with "None"


dataset$GarageQual <- str_replace_na(dataset$GarageQual, replacement = "NoGarage") # Replace NA in the GarageQual column with "None"


dataset$GarageFinish <- str_replace_na(dataset$GarageFinish, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"


dataset$GarageType <- str_replace_na(dataset$GarageType, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"


dataset$BsmtFinType1 <- str_replace_na(dataset$BsmtFinType1, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"


dataset$BsmtFinType2 <- str_replace_na(dataset$BsmtFinType2, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"


dataset$Electrical <- str_replace_na(dataset$Electrical, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"


dataset$GarageYrBlt <- str_replace_na(dataset$GarageYrBlt, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"


dataset$Functional <- str_replace_na(dataset$Functional, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"


dataset$SaleType <- str_replace_na(dataset$SaleType, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"



###
# Dealing with the missing values
###

# Dealing with missing values in the MSZoning column, which identifies the general zoning classification
# of the sale. We plot the zoning classifications to discern any patterns.
plot(dataset$MSZoning,
     col = "orange",
     xlab = "Zoning Classification",
     ylab = "Count",
     main = "MSZoning classifications")

# Clearly, the most common zoning classification is RL (Residential low density).
# By imputing "RL" for missing MSZoning values, we have the highest chance to be correct.
dataset$MSZoning[is.na(dataset$MSZoning)] <- "RL" # We impute "RL", the mode, for missing values in MSZoning


# Dealing with missing values concerning the basement. We take a look at all basement-related columns that have missing values. Is there a pattern?
Bsmt_missing_vals = c("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2")
dataset[!complete.cases(dataset[, names(dataset) %in% Bsmt_missing_vals]), names(dataset) %in% names(dataset)[which(grepl("Bsmt", names(dataset)))]]

# All missing values related to basement can be explained by the fact that there either is no basement, or it is yet unfinished.
# We can impute a "NoBasemet" for BsmtQual, BsmtCond, and BsmtExposure missing values.

# We convert to a character, add the NA replacement and change back to factor
dataset$BsmtQual <- as.character(dataset$BsmtQual)
dataset$BsmtQual[is.na(dataset$BsmtQual)] <- "NoBasement"
dataset$BsmtQual <- as.factor(dataset$BsmtQual)

dataset$BsmtExposure <- as.character(dataset$BsmtExposure)
dataset$BsmtExposure[is.na(dataset$BsmtExposure)] <- "NoBasement"
dataset$BsmtExposure <- as.factor(dataset$BsmtExposure)

dataset$BsmtCond <- as.character(dataset$BsmtCond)
dataset$BsmtCond[is.na(dataset$BsmtCond)] <- "NoBasement"
dataset$BsmtCond <- as.factor(dataset$BsmtCond)

# We replace the remaining NAs in BsmtFullBath and BsmtHalfBath with 0 as they have "NoBasement"
dataset$BsmtFullBath[is.na(dataset$BsmtFullBath)] <- 0
dataset$BsmtHalfBath[is.na(dataset$BsmtHalfBath)] <- 0



# Dealing with missing values in MasVnrType and MasVnrArea. We observe that MasVnrType equal to "None" can still have an area.
# We take a look at a summary of the features without missing values to get an idea about their mode.
plot(dataset[, c("MasVnrType", "MasVnrArea")],
     col = "orange",
     main = "MasVnrType vs MasVnrArea"
     )

summary(dataset$MasVnrType[!is.na(dataset$MasVnrType)]) # Looking at MasVnrType without missing values
summary(dataset$MasVnrArea[!is.na(dataset$MasVnrArea)]) # Looking at MasVnrArea without missing values

# Since "None" is the most common value of type, we impute "None" for MasVnrType and 0 for the area.
dataset$MasVnrType[is.na(dataset$MasVnrType)] <- "None"
dataset$MasVnrArea[is.na(dataset$MasVnrArea)] <- 0




#########################################################################################################
# Dealing with missing values in LotFrontage, which are the linear feet of street connected to property.
# LotFrontage might be closely correlated to the LotArea, the lot size in square feet.

# We plot log-transformed LotArea against LotFrontage. Indeed, there seems to be a positive correlation
# between LotFrontage and LotArea as shown by the fitted general additive model explaining 
# LotFrontage as a smooth function of LotArea.
dataset %>%
  ggplot(aes(x = log(LotArea), y = log(LotFrontage))) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"))

# We can assume that other features might also be informative about LotFrontage, so we include
# them in a random forest regression model below.

library(xgboost)
library(vtreat)
library(magrittr)

# We separate "dataset" into test and train by filtering out the rows with missing LotFrontage
LF_index <- which(is.na(dataset$LotFrontage))
LF_train <- dataset[-LF_index, ]
LF_test <- dataset[LF_index, ]



# We select all variables that might influence LotFrontage
variables <- c("LotArea", "Street", "LotShape", "LandContour", "LotConfig", "LandSlope",
               "Neighborhood", "BldgType")

# The vtreat function designTreatmentsZ helps encode all variables numerically
treatment_plan <- designTreatmentsZ(LF_train, variables) # Devise a treatment plan for the variables
(newvars <- treatment_plan %>%
    use_series(scoreFrame) %>%        
    filter(code %in% c("clean", "lev")) %>%  # get the rows you care about
    use_series(varName))           # get the varName column

train_treated <- prepare(treatment_plan, LF_train,  varRestriction = newvars)
test_treated <- prepare(treatment_plan, LF_test,  varRestriction = newvars)

str(train_treated)
str(test_treated)


cv <- xgb.cv(data = as.matrix(train_treated), 
             label = LF_train$LotFrontage,
             nrounds = 100,
             nfold = 5,
             objective = "reg:linear",
             eta = 0.3,
             max_depth = 6,
             early_stopping_rounds = 10,
             verbose = 0)    # silent

elog <- cv$evaluation_log # Get the evaluation log of the cross-validation so we can find the number of trees to use to minimize RMSE without overfitting the training data

elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.min(test_rmse_mean))   # find the index of min(test_rmse_mean)

# Next we run xgboost with the information gained by running xgboost cross-validation
LotFrontage_model_xgb <- xgboost(data = as.matrix(train_treated), # Treated training data as a matrix
                          label = LF_train$LotFrontage,  # Column of outcomes from original data
                          nrounds = 10,       # number of trees to build, which we determined via cross-validation
                          objective = "reg:linear", # objective
                          eta = 0.3, # The learning rate; Closer to 0 is slower, but less prone to overfitting; Closer to 1 is faster, but more likely to overfit
                          max_depth = 6,
                          verbose = 0)  # silent

# Now we can predict LotFrontage
LF_test$LotFrontagePred <- predict(LotFrontage_model_xgb, newdata = as.matrix(test_treated)) # We predict LotFrontage; newdata has to be a matrix

# We use the predicted values for imputation
dataset$LotFrontage[LF_index] <- LF_test$LotFrontagePred
summary(dataset)

# The missing values for LotFrontage were imputed with xgboost-predicted values based on `variables`









### Other remaining missing values ###

# There is one missing value in GarageCars and GarageArea - maybe this comes from the same house which doesn't have a garage?
garage_NA_ind <- which(is.na(dataset$GarageArea)) # Find the row with the missing GarageArea
print(dataset[garage_NA_ind, ]) # Look at the entry with the id 2577 and realize that there is "NoGarage" in multiple garage-related columns

# We can replace the missing values in GarageCars and GarageArea with "NoGarage"
dataset$GarageCars[garage_NA_ind] <- "NoGarage"
dataset$GarageArea[garage_NA_ind] <- "NoGarage"


# There is one NA remaining in KitchenQual, but is not immediately evident why it is missing
dataset[which(is.na(dataset$KitchenQual)), ]

# From the plot it appears that a "TA" = "Typical" kitchen is the most common (mode); will will impute that for the single missing value
plot(dataset$KitchenQual,
     col = "orange",
     xlab = "Kitchen quality",
     ylab = "Count",
     main = "Kitchen quality")

dataset$KitchenQual[which(is.na(dataset$KitchenQual))] <- "TA" # We impute "TA" as in "typical" kitchen quality due to it being most likely



# There is one value missing in TotalBsmtSF - does this house even have a basement?
dataset[which(is.na(dataset$TotalBsmtSF)), ] #We take a look at the entry with the missing value; it doesnt have a Basement, we can impute a 0
dataset$TotalBsmtSF[which(is.na(dataset$TotalBsmtSF))] <- 0 # This house has no basement and thus 0 total basement square feet area


# There is one value missing in BsmtFinSF1 - does this house even have a basement?
dataset[which(is.na(dataset$BsmtFinSF1)), ] #We take a look at the entry with the missing value; it doesnt have a Basement, we can impute a 0
dataset$BsmtFinSF1[which(is.na(dataset$BsmtFinSF1))] <- 0 # This house has no basement and thus 0 total basement square feet area

# There is one value missing in BsmtFinSF2 - does this house even have a basement?
dataset[which(is.na(dataset$BsmtFinSF2)), ] #We take a look at the entry with the missing value; it doesnt have a Basement, we can impute a 0
dataset$BsmtFinSF2[which(is.na(dataset$BsmtFinSF2))] <- 0 # This house has no basement and thus 0 total basement square feet area

# There is one value missing in BsmtUnfSF - does this house even have a basement?
dataset[which(is.na(dataset$BsmtUnfSF)), ] #We take a look at the entry with the missing value; it doesnt have a Basement, we can impute a 0
dataset$BsmtUnfSF[which(is.na(dataset$BsmtUnfSF))] <- 0 # This house has no basement and thus 0 total basement square feet area





# There is one value missing in Exterior1st and Exterior2nd
dataset[which(is.na(dataset$Exterior1st)), ] #We take a look at the entry with the missing value; it has typical exterior quality
dataset %>% select(Exterior1st) %>% group_by(Exterior1st) %>% tally() # VinylSd is the mode
dataset %>% select(Exterior2nd) %>% group_by(Exterior2nd) %>% tally() # VinylSd is the mode

# We impute the most common type of exterior
dataset$Exterior1st[which(is.na(dataset$Exterior1st))] <- "VinylSd"
dataset$Exterior2nd[which(is.na(dataset$Exterior2nd))] <- "VinylSd"



# There are two missing values in Utilities
dataset[which(is.na(dataset$Utilities)), ] # Both houses appear pretty typical, so we will go with the most common value for Utilities: AllPub

dataset %>% select(Utilities) %>% group_by(Utilities) %>% count()

dataset$Utilities[which(is.na(dataset$Utilities))] <- "AllPub"




############################################################################################################
### All missing values have been dealt with and we can once again separate `dataset` into train and test ###
train <- dataset[train$Id, ]                                                                             ###
test <- dataset[test$Id, ]                                                                               ###
############################################################################################################











# We compute the skewness of all numerical features and identify features with a skewness > 0.75 for train and test.
# Before we can identify all numerical features, we make sure that no actual factors are encoded as numerics.

# We change all character columns into factors
train <- train %>% mutate_if(is.character, factor) # Change character columns to factors
test <- test %>% mutate_if(is.character, factor)

# Additionally, some numeric/integer columns which are actually categorical have to be changed into a factor.
# We define feature_vector, containing all features to be changed into factors.
feature_vector <- c("MSSubClass", "MSZoning", "Street", "OverallQual", "OverallCond", "YearBuilt",
                    "YearRemodAdd", "BsmtFullBath", "BsmtHalfBath", "BedroomAbvGr", "KitchenAbvGr",
                    "TotRmsAbvGrd", "GarageYrBlt", "GarageCars", "MoSold", "YrSold", "FullBath",
                    "HalfBath", "Fireplaces", "GarageType", "GarageFinish")
train[, feature_vector] <- map(train[, feature_vector], factor)
test[, feature_vector] <- map(test[, feature_vector], factor)

# We take a look at the changed structure of the dataset
str(train)









# Dealing with feature skewedness

nums <- unlist(lapply(train, is.numeric)) # Selecting all numerical columns from train

skewed_vals <- sapply(train[, nums], skewness) # Determining skewedness of numerical train features
(large_skew_index <- skewed_vals > 0.75 | skewed_vals < -0.75) #Building an index for skewed features in train


# As an example, we visualize of LotArea and log-transformed LotArea.
train %>%
  ggplot(aes(x = LotArea)) +
  geom_histogram(bins = 30) +
  ggtitle("Linear feet of street connected to property") +
  xlab("Linear feet of street connected to property") +
  ylab("Number of houses")

train %>%
  ggplot(aes(x = log1p(LotArea))) +
  geom_histogram(bins = 30) +
  ggtitle("Linear feet of street connected to property") +
  xlab("Linear feet of street connected to property") +
  ylab("Number of houses")

# We log-transform all features with a skewness > 0.75 with the function log1p(x), which computes log(1+x). We do the same for test to keep modifications the same.
train[, names(large_skew_index[large_skew_index == TRUE])] <- train[, names(large_skew_index[large_skew_index == TRUE])] %>% log1p()

# Note: `test` does not contain a SalesPrice column.
test[, names(large_skew_index[large_skew_index == TRUE])] <- test[, names(large_skew_index[large_skew_index == TRUE])] %>% log1p()


# We plot LotArea and SalePrice after the log-transformation
train %>%
  ggplot(aes(x = LotArea)) +
  geom_histogram(bins = 30) +
  ggtitle("Linear feet of street connected to property") +
  xlab("Linear feet of street connected to property") +
  ylab("Number of houses")

train %>%
  ggplot(aes(x = SalePrice)) +
  geom_histogram(bins = 30) +
  ggtitle("Sale price distribution") +
  xlab("Sale price in dollars") +
  ylab("Number of houses")




###
# Modelling approaches
###

# First we write a loss-function to determine the residual mean squared error, or RMSE of the model.
# The function calculates the residuals and then takes the root mean square of them.
RMSE <- function(predicted_prices, true_prices) {
  residuals <- predicted_prices - true_prices
  sqrt(mean(residuals^2))
}

# Next, we split `train` into separate train_set and test_set for algorithm evaluation purposes (we won't use the real `test` subset for this and treat it as completely new data for final evaluations only)
# test_set will receive 20% of the data, train_set will receive 80%.
set.seed(1)
test_index <- createDataPartition(train$SalePrice, p = 0.2, list = FALSE)
train_set <- train[-test_index, ]
test_set <- train[test_index, ]

# Model 1: Simple linear regression as a baseline. Id is removed as it has no predictive value.
model_1_lm <- lm(SalePrice ~ ., data = train[, -Id])
model_1_pred <- predict(model_1_lm, newdata = test[, -Id])