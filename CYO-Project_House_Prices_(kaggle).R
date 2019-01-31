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

# Fix column encoding, as sometimes "NA" was put for the absence of a feature (e.g. when there is no pool).
train$Alley <- str_replace_na(train$Alley, replacement = "NoAccess") # Replace NA in the Alley column with "None"
test$Alley <- str_replace_na(test$Alley, replacement = "NoAccess") # Replace NA in the Alley column with "None"

train$PoolQC <- str_replace_na(train$PoolQC, replacement = "NoPool") # Replace NA in the PoolQC column with "None"
test$PoolQC <- str_replace_na(test$PoolQC, replacement = "NoPool") # Replace NA in the PoolQC column with "None"

train$Fence <- str_replace_na(train$Fence, replacement = "NoFence") # Replace NA in the Fence column with "None"
test$Fence <- str_replace_na(test$Fence, replacement = "NoFence") # Replace NA in the Fence column with "None"

train$MiscFeature <- str_replace_na(train$MiscFeature, replacement = "None") # Replace NA in the MiscFeature column with "None"
test$MiscFeature <- str_replace_na(test$MiscFeature, replacement = "None") # Replace NA in the MiscFeature column with "None"

train$FireplaceQu <- str_replace_na(train$FireplaceQu, replacement = "NoFireplace") # Replace NA in the FirePlaceQu column with "None"
test$FireplaceQu <- str_replace_na(test$FireplaceQu, replacement = "NoFireplace") # Replace NA in the FirePlaceQu column with "None"

train$GarageCond <- str_replace_na(train$GarageCond, replacement = "NoGarage") # Replace NA in the GarageCond column with "None"
test$GarageCond <- str_replace_na(test$GarageCond, replacement = "NoGarage") # Replace NA in the GarageCond column with "None"

train$GarageQual <- str_replace_na(train$GarageQual, replacement = "NoGarage") # Replace NA in the GarageQual column with "None"
test$GarageQual <- str_replace_na(test$GarageQual, replacement = "NoGarage") # Replace NA in the GarageQual column with "None"

train$GarageFinish <- str_replace_na(train$GarageFinish, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"
test$GarageFinish <- str_replace_na(test$GarageFinish, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"

train$GarageType <- str_replace_na(train$GarageType, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"
test$GarageType <- str_replace_na(test$GarageType, replacement = "NoGarage") # Replace NA in the GarageFinish column with "None"

train$BsmtFinType1 <- str_replace_na(train$BsmtFinType1, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"
test$BsmtFinType1 <- str_replace_na(test$BsmtFinType1, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"

train$BsmtFinType2 <- str_replace_na(train$BsmtFinType2, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"
test$BsmtFinType2 <- str_replace_na(test$BsmtFinType2, replacement = "NoBasement") # Replace NA in the GarageFinish column with "None"

train$Electrical <- str_replace_na(train$Electrical, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"
test$Electrical <- str_replace_na(test$Electrical, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"

train$GarageYrBlt <- str_replace_na(train$GarageYrBlt, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"
test$GarageYrBlt <- str_replace_na(test$GarageYrBlt, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"

train$Functional <- str_replace_na(train$Functional, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"
test$Functional <- str_replace_na(test$Functional, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"

train$SaleType <- str_replace_na(train$SaleType, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"
test$SaleType <- str_replace_na(test$SaleType, replacement = "Unknown") # Replace NA in the GarageFinish column with "None"




###
# Dealing with the missing values
###

# Dealing with missing values in the MSZoning column, which identifies the general zoning classification
# of the sale. We plot the zoning classifications to discern any patterns.
plot(train$MSZoning,
     col = "orange",
     xlab = "Zoning Classification",
     ylab = "Count",
     main = "MSZoning classifications")

# Clearly, the most common zoning classification is RL (Residential low density).
# By imputing "RL" for missing MSZoning values, we have the highest chance to be correct.
train$MSZoning[is.na(train$MSZoning)] <- "RL" # We impute "RL", the mode, for missing values in MSZoning


# Dealing with missing values concerning the basement. We take a look at all basement-related columns that have missing values. Is there a pattern?
Bsmt_missing_vals = c("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2")
train[!complete.cases(train[, names(train) %in% Bsmt_missing_vals]), names(train) %in% names(train)[which(grepl("Bsmt", names(train)))]]

# All missing values related to basement can be explained by the fact that there either is no basement, or it is yet unfinished.
# We can impute a "NoBasemet" for BsmtQual, BsmtCond, and BsmtExposure missing values. We ignore entry #949 for now.

train$BsmtQual[is.na(train$BsmtQual)] <- "NoBasement"
test$BsmtQual[is.na(test$BsmtQual)] <- "NoBasement"

train$BsmtExposure[is.na(train$BsmtExposure)] <- "NoBasement"
test$BsmtExposure[is.na(test$BsmtExposure)] <- "NoBasement"

train$BsmtCond[is.na(train$BsmtCond)] <- "NoBasement"
test$BsmtCond[is.na(test$BsmtCond)] <- "NoBasement"


# Dealing with missing values in MasVnrType and MasVnrArea. We observe that MasVnrType equal to "None" can still have an area.
# We take a look at a summary of the features without missing values to get an idea about their mode.
plot(train[, c("MasVnrType", "MasVnrArea")],
     col = "orange",
     main = "MasVnrType vs MasVnrArea"
     )

summary(train$MasVnrType[!is.na(train$MasVnrType)]) # Looking at MasVnrType without missing values
summary(train$MasVnrArea[!is.na(train$MasVnrArea)]) # Looking at MasVnrArea without missing values

# Since "None" is the most common value of type, we impute "None" for MasVnrType and 0 for the area.
train$MasVnrType[is.na(train$MasVnrType)] <- "None"
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0

test$MasVnrType[is.na(test$MasVnrType)] <- "None"
test$MasVnrArea[is.na(test$MasVnrArea)] <- 0



# Dealing with missing values in LotFrontage, which are the linear feet of street connected to property.
# LotFrontage might be closely correlated to the LotArea, the lot size in square feet.

# We plot log-transformed LotArea against LotFrontage. Indeed, there seems to be a positive correlation
# between LotFrontage and LotArea as shown by the fitted general additive model explaining 
# LotFrontage as a smooth function of LotArea.
train %>%
  ggplot(aes(x = log(LotArea), y = log(LotFrontage))) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"))

# We can assume that other features might also be informative about LotFrontage, so we include
# them in a random forest regression model below.

library(ranger)
index <- which(is.na(train$LotFrontage))
train_subset <- train[-index, ]
LotFrontage_model <- ranger(LotFrontage ~ LotArea + Street + LotShape + LandContour + LotConfig + LandSlope + Neighborhood + BldgType, data = train_subset)
LotFrontage_pred <-predict(LotFrontage_model, data = train[index, ])
LotFrontage_pred$predictions

# We use the predicted values for imputation
train$LotFrontage[index] <- LotFrontage_pred$predictions
summary(train)



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