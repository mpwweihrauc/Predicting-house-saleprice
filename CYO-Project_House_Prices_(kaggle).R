# Please download the datasets from the provided link or find them in my provided Github repository, links below, or in the report.
# Link to Kaggle page: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link to my Github page: https://github.com/mpwweihrauc/ML_House_Prices.git
# You will need the test.csv and the train.csv files. There is also a data_description.txt with descriptions for all the different parameters in the dataset.

# We begin by loading/installing all required libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

# We import the training and testing data subsets (files from Kaggle)
train <- read_csv("train.csv")
test <- read_csv("test.csv")




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

# We deal with missing values by imputing column means
impute_index_LotFrontage <- which(is.na(train$LotFrontage))
train$LotFrontage[impute_index_LotFrontage] <- mean(train$LotFrontage[-impute_index_LotFrontage])

impute_index_MasVnrArea <- which(is.na(train$MasVnrArea))
train$MasVnrArea[impute_index_MasVnrArea] <- mean(train$MasVnrArea[-impute_index_MasVnrArea])

impute_index_GarageYrBlt <- which(is.na(train$GarageYrBlt))
train$GarageYrBlt[impute_index_GarageYrBlt] <- mean(train$GarageYrBlt[-impute_index_GarageYrBlt])

summary(train) # The missing values are now replaced with respective column means.



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

# What does the distribution look like if we log10-transform SalesPrice? Monetary amounts are often
# lognormally distributed, i.e. the log of the data is normally distributed.
train %>%
  ggplot(aes(x = SalePrice)) +
  geom_histogram(bins = 30) +
  scale_x_log10(breaks = c(35000, 100000, 150000, 200000, 300000, 500000, 750000)) +
  ggtitle("Sale price distribution") +
  xlab("Sale price in dollars") +
  ylab("Number of houses")

# We compute the skewness of all numerical features and log-transform any features
# with a skewness > 0.75.
nums <- unlist(lapply(train, is.numeric)) # Selecting all numerical columns

skewed_vals <- sapply(train[, nums], skewness)
skewed_vals > 0.75 | skewed_vals< -0.75







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

# Model 1: Simple linear regression as a baseline
model_1_lm <- lm(SalePrice ~ YearBuilt, data = train)
model_1_pred <- predict(model_1_lm, newdata = test)