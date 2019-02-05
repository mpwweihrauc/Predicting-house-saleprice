# Please download the datasets from the provided link or find them in my provided Github repository, links below, or in the report.
# Link to Kaggle page: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link to my Github page: https://github.com/mpwweihrauc/ML_House_Prices.git
# You will need the test.csv and the train.csv files. There is also a data_description.txt with descriptions for all the different parameters in the dataset.

# We begin by loading/installing all required libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(vtreat)) install.packages("vtreat", repos = "http://cran.us.r-project.org")


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


dataset$GarageYrBlt <- str_replace_na(dataset$GarageYrBlt, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"



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


# There is one missing value in Electrical
which(is.na(dataset$Electrical))
dataset[1380, ] # The house with the missing Electrical value seems pretty normal

# From the plot we can see that SBrkr or standard circuit breakers are the most common value
plot(dataset[, "Electrical"],
     col = "orange",
     main = "Electrical"
)

# We impute SBrkr as it is the most common, standard Electrical entry
dataset$Electrical[which(is.na(dataset$Electrical))] <- "SBrkr"


# There is one missing value in GarageCars and GarageArea - maybe this comes from the same house which doesn't have a garage?
garage_NA_ind <- which(is.na(dataset$GarageArea)) # Find the row with the missing GarageArea
print(dataset[garage_NA_ind, ]) # Look at the entry with the id 2577 and realize that there is "NoGarage" in multiple garage-related columns

# We can replace the missing values in GarageCars and GarageArea with 0.
dataset$GarageCars[garage_NA_ind] <- 0
dataset$GarageArea[garage_NA_ind] <- 0


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


# There is a missing value in SaleType
dataset[which(is.na(dataset$SaleType)), ] # We take a look at the house with the missing SaleType; its SaleCOndition is normal
plot(dataset[, "SaleType"],
     col = "orange",
     ylab = "Number of houses",
     xlab = "Sale type",
     main = "Sale type"
)

# From the plot we can see that WD or "Warranty deal conventional" is the most common value by far
# We will impute WD as it is the most likely value

dataset$SaleType[which(is.na(dataset$SaleType))] <- "WD"


# There are still two missing values in Functional: Home functionality (Assume typical unless deductions are warranted) (from data description)
dataset[which(is.na(dataset$Functional)), ] # The OverallCond of one house and the OverallQual of the other are at 1!

# What kind of Functional values do houses with an OverallCond of only 1 and OverallQual 4 have most likely?
dataset[dataset$OverallCond == 1 & dataset$OverallQual == 4, ]$Functional %>% plot() # There is only one other house like this, with similar YearBuilt

# As the houses are very similar from various attributes, we will impute a Functional of "Typ" typical
dataset$Functional[2474] <- "Typ"

# The other house with missing Functional...
dataset[dataset$OverallCond == 5 & dataset$OverallQual == 1, ]$Functional %>% plot() # There is no house like it
dataset[dataset$OverallQual == 1, ]$Functional %>% plot() # Low-quality houses tend to be "Maj1", "Mod", or "Typ"
dataset[dataset$OverallCond == 5, ]$Functional %>% plot() # Houses with average condition tend to be "Typ", typical

dataset$Functional %>% plot() # The mode is "Typ", which we will therefore impute in this case

dataset$Functional[2217] <- "Typ"


#########################################################################################################
# Dealing with missing values in LotFrontage, which are the linear feet of street connected to property.
# LotFrontage might be closely correlated to the LotArea, the lot size in square feet.

# We plot log-transformed LotArea against LotFrontage. Indeed, there seems to be a positive correlation
# between LotFrontage and LotArea as shown by the fitted general additive model explaining 
# LotFrontage as a smooth function of LotArea. NAs are automatically removed from the plot.
dataset %>%
  ggplot(aes(x = log(LotArea), y = log(LotFrontage))) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  ggtitle("Linear feet of street connected to property as a function of the lot size in square feet")

# We can assume that other features might also be informative about LotFrontage, so we include
# them in a gradient boosting machine model below (xgboost package; and vtreat package for data preparation)


# We separate "dataset" into "LF" test and train by filtering out the rows with missing LotFrontage
LF_index <- which(is.na(dataset$LotFrontage))
LF_train <- dataset[-LF_index, ]
LF_test <- dataset[LF_index, ]



# We select all variables that might influence LotFrontage according to our intuition and the data description
variables <- c("LotArea", "Street", "LotShape", "LandContour", "LotConfig", "LandSlope",
               "Neighborhood", "BldgType", "Condition1", "Condition2")

# The vtreat function designTreatmentsZ helps encode all variables numerically
treatment_plan <- designTreatmentsZ(LF_train, variables) # Devise a treatment plan for the variables
(newvars <- treatment_plan %>%
    use_series(scoreFrame) %>%  # use_series() works like a $, but within pipes, so we can access scoreFrame      
    filter(code %in% c("clean", "lev")) %>%  # We select only the rows we care about
    use_series(varName))           # We get the varName column

# The prepare() function prepares our data subsets according to the treatment plan
# we devised above and encodes all relevant variables "newvars" numerically
train_treated <- prepare(treatment_plan, LF_train,  varRestriction = newvars)
test_treated <- prepare(treatment_plan, LF_test,  varRestriction = newvars)

# We can now see the numerical encoding of all variables thanks to treatment and preparation above
str(train_treated)
str(test_treated)

# We conduct gradient boosting cross-validation; as the outcome variable was removed from the treated data we have to get it from the original data
cv <- xgb.cv(data = as.matrix(train_treated),  # xgb.cv only takes a matrix of the treated, all-numerical input data
             label = LF_train$LotFrontage, # Outcome from untreated data
             nrounds = 100, # We go up to 100 rounds of fitting models on the remaining residuals
             nfold = 5, # We use 5 folds for cross-validation
             objective = "reg:linear",
             eta = 0.3, # The learning rate; Closer to 0 is slower, but less prone to overfitting; Closer to 1 is faster, but more likely to overfit
             max_depth = 6,
             early_stopping_rounds = 10,
             verbose = 0)    # silent

# While the RMSE may continue to decrease on more and more rounds if iteration, the test RMSE usualyl doesn't.
# We choose the number of rounds that minimize RMSE for test
elog <- cv$evaluation_log # Get the evaluation log of the cross-validation so we can find the number of trees to use to minimize RMSE without overfitting the training data

elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.min(test_rmse_mean))   # find the index of min(test_rmse_mean)

# We save the number of trees that minimize test RMSE in ntrees
ntrees <- elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean),
            ntrees.test  = which.min(test_rmse_mean)) %>%
  use_series(ntrees.test)

# Next we run the actual modelling process with the information gained by running xgboost cross-validation above
LotFrontage_model_xgb <- xgboost(data = as.matrix(train_treated), # Treated training data as a matrix
                          label = LF_train$LotFrontage,  # Column of outcomes from original data
                          nrounds = ntrees,       # number of trees to build, which we determined via cross-validation
                          objective = "reg:linear", # objective
                          eta = 0.3, # The learning rate; Closer to 0 is slower, but less prone to overfitting; Closer to 1 is faster, but more likely to overfit
                          max_depth = 6,
                          verbose = 0)  # silent

# Now we can predict LotFrontage in the test set that contains all the missing LotFrontage values
LF_test$LotFrontagePred <- predict(LotFrontage_model_xgb, newdata = as.matrix(test_treated)) # We predict LotFrontage; newdata has to be a matrix
LF_test

# We can plot the predicted LotFrontage values against the LotArea values in LF_test to see if we observe the
# correlation between the two variables.
# Indeed, the predicted LotFrontage values seem to behave in similar fashion to the actual ones we observed above in the whole dataset
LF_test %>%
  ggplot(aes(x = log(LotArea), y = log(LotFrontagePred))) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  ggtitle("(Predicted) linear feet of street connected to property as a function of the lot size in square feet")

# We use the predicted values for imputation of LotFrontage in our dataset
dataset$LotFrontage[LF_index] <- LF_test$LotFrontagePred
summary(dataset)

# The missing values for LotFrontage were imputed with xgboost-predicted values based on `variables` related to it


# All missing values have been dealt with.
summary(dataset)







#########################################################################
### Variable encoding & Skewness ########################################
#########################################################################
# We compute the skewness of all numerical features and identify features with a skewness > 0.75 for train and test.
# Before we can identify all numerical features, we make sure that no actual factors are encoded as numerics.
# Additionally, some numeric/integer columns which are actually categorical have to be changed into a factor.
# We define feature_vector, containing all features to be changed into factors.
dataset$GarageArea <- as.numeric(dataset$GarageArea) # We fix GarageArea to be a numeric instead of a character variable
dataset$GarageArea[2577] <- 0 # An NA was introduced due to the prior conversion. This house simply has no garage and therefore 0 GarageArea

feature_vector <- c("MSSubClass", "Alley", "GarageQual", "GarageYrBlt","GarageType", "MoSold", "YrSold",
                    "BsmtFinType1", "BsmtFinType2", "Electrical", "GarageFinish",
                    "GarageCond", "Functional", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "SaleType")
dataset[, feature_vector] <- map(dataset[, feature_vector], factor) # We use map() from purrr to turn all features above into factors

# We take a look at the structure after the modifications
str(dataset)



############################################################################################################
### All missing values have been dealt with and we can once again separate `dataset` into train and test ###
train <- dataset[train$Id, ]                                                                             ###
test <- dataset[test$Id, -81]   # We remove the temporary SalePrice column again                                                                            ###
############################################################################################################





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
temp <- train[test_index, ] # temporary test set

# We make sure there are no entries in test_set that aren't in train_set
test_set <- temp %>% 
  semi_join(train_set, by = c("YearBuilt", "RoofMatl", "Exterior1st", "Exterior2nd", "Electrical",
                              "MiscFeature")) # Variables were determined by trial and error with the lm() models below

# We return the removed entries from test to train
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

summary(train_set)
summary(test_set)

# There is not a single house with "NoSeWa" value in "Utilities" in the train_set and only a single in test_set. This predictor will not help in training an algorithm and is now removed.
train_set <- subset(train_set, select = -Utilities)
test_set <- subset(test_set, select = -Utilities)

# Monetary values like SalePrice are often log-normal and can be skewed considerably, we determine whether SalePrice is skewed in our dataset
train_set %>%
  ggplot(aes(SalePrice)) +
  geom_histogram(bins = 30, binwidth = 5000)

# Indeed, SalePrice is right-skewed. We try a log-transformation. SalePrice looks much more normal now.
train_set %>%
  ggplot(aes(log1p(SalePrice))) +
  geom_histogram(bins = 30)

# We log-transform SalePrice.
train_set$SalePrice <- log1p(train_set$SalePrice)
test_set$SalePrice <- log1p(test_set$SalePrice)




# 1. Model 1: Simple linear regression as a baseline.
# As our first, very simple model we predict house sale price via linear regression of the LotArea.
# We generate a table to keep track of the RMSEs our various models generate.

model_1_lm <- lm(SalePrice ~ LotArea, data = train_set) # Linear regression with a single predictor
model_1_pred <- predict(model_1_lm, newdata = test_set) # Predict on test_set

model_1_lm_RMSE <- RMSE(model_1_pred, test_set$SalePrice) # Calculate RMSE

model_rmses <- data_frame(Model = "Model_1_lm", RMSE = model_1_lm_RMSE) # Record RMSE of Model 1

model_rmses %>% knitr::kable()

# 2. Model 2: Multivariate linear regression with all predictors.
# In our second model, we use all available predictors.
model_2_lm <- lm(SalePrice ~ ., data = train_set[, -1]) # We remove Id and predict with all
model_2_pred <- predict(model_2_lm, newdata = test_set[, -1])
model_2_lm_RMSE <- RMSE(model_2_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_2_Multivariate_lm", RMSE = model_2_lm_RMSE))

model_rmses %>% knitr::kable()

# RMSE improved substantially, to below 0.14.
# This linear regression RMSE will serve as the baseline RMSE.

# We examine the multivariate lm model 2. It seems that most predictors are not significant as per p-value.
summary(model_2_lm)

