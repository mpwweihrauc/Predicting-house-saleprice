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
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")


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

#######################################################################
# Feature engineering, data wrangling and dealing with missing values #
#######################################################################



# To facilitate feature engineering and data cleaning we temporarily merge train and tet into "dataset"
test$SalePrice <- 0 # Temporarily add SalePrice column with 0s to test
dataset <- rbind(train, test) # Merge train and test
summary(dataset) # There's a lot of NA values in many columns

# We fix the encoding of certain columns, as sometimes "NA" was put for the absence of a feature
# (e.g. when there is no pool). This information comes from the data description text.
# We can replace these "fake" NAs with "None"
dataset$Alley <- str_replace_na(dataset$Alley, replacement = "None") # Replace NA in the Alley column with "None"
dataset$PoolQC <- str_replace_na(dataset$PoolQC, replacement = "None")
dataset$Fence <- str_replace_na(dataset$Fence, replacement = "None")
dataset$MiscFeature <- str_replace_na(dataset$MiscFeature, replacement = "None")
dataset$FireplaceQu <- str_replace_na(dataset$FireplaceQu, replacement = "None")
dataset$GarageCond <- str_replace_na(dataset$GarageCond, replacement = "None")
dataset$GarageQual <- str_replace_na(dataset$GarageQual, replacement = "None")
dataset$GarageFinish <- str_replace_na(dataset$GarageFinish, replacement = "None")
dataset$GarageType <- str_replace_na(dataset$GarageType, replacement = "None")
dataset$BsmtFinType1 <- str_replace_na(dataset$BsmtFinType1, replacement = "None") 
dataset$BsmtFinType2 <- str_replace_na(dataset$BsmtFinType2, replacement = "None") 


#####
#####

# MSSubclass is encoded as an integer, even though the magnitude of the integer is meaningless for the variable.
# We will change encoding into factor.
dataset$MSSubClass <- as.factor(dataset$MSSubClass)


#####
#####

# Dealing with missing values in the MSZoning column, which identifies the general zoning classification
# of the sale. We plot the zoning classifications to discern any patterns.
dataset[is.na(dataset$MSZoning), ] # There are four houses with a missing MSZoning value

# RL, or residential low-density is clearly the most common value
plot(dataset$MSZoning,
     col = "orange",
     xlab = "Zoning Classification",
     ylab = "Count",
     main = "MSZoning classifications")

# Does MSZoning depend on MSSubclass? Below we can see that for MSSubClass of 20,
# residential low-density is clearly the most common value, while it is less clear for MSSubClass of 30 or 70.
dataset %>% select(MSSubClass, MSZoning) %>%
  group_by(MSSubClass, MSZoning) %>%
  filter(MSSubClass %in% c(20, 30, 70)) %>%
  count()

# We will impute the most common value, "RL", for the houses
dataset$MSZoning[is.na(dataset$MSZoning)] <- "RL"# We impute "RL", the mode, for missing values in MSZoning

#####
#####

# LotArea is encoded as an integer, but is clearly numeric. We change it accordingly.
dataset$LotArea <- as.numeric(dataset$LotArea)


#####
#####

# Alley is encoded as character, but should clearly be a facor variable. We change it accordingly.
dataset$Alley <- as.factor(dataset$Alley)

#####
#####



#####
#####

# OverallQual, OverallCond are ordinal, but encoded as integers. We change them into factor variables
dataset$OverallCond <- as.factor(dataset$OverallCond)
dataset$OverallQual <- as.factor(dataset$OverallQual)

#####
#####


# GarageYrBuilt has some missing values, are these due to there being no Garage?
# Indeed, most houses with a missing GarageYrBlt simply have no garage, but two houses have a detached Garage.
dataset[which(is.na(dataset$GarageYrBlt)), ] %>% select(GarageYrBlt, GarageType, YearBuilt)

# As there is already a large number of garage-related variables and information about the age of the house itself,
# it seems best to simply remove GarageYrBlt from the dataset, as imputing YearBuilt for the missing GarageYrBlt might
# lead to more harm than good.
# GarageYrBlt will be removed at the end of this section, alongside other variables.


#####
#####


# There is a missing value in Exterior1st and Exterior2nd, is it the same house?
dataset[which(is.na(dataset$Exterior1st)), ]
dataset[which(is.na(dataset$Exterior2nd)), ]

# Indeed, the info is missing for the same house. Which exterior material is most common?
dataset %>% select(Exterior1st) %>% group_by(Exterior1st) %>% tally() # VinylSd is the mode
dataset %>% select(Exterior2nd) %>% group_by(Exterior2nd) %>% tally() # VinylSd is the mode

# Which exterior materials do similar houses use? We look at several attributes and the year the houses were built.
dataset %>% select(Exterior1st, Exterior2nd, RoofMatl, HouseStyle, BldgType, YearBuilt) %>%
  filter(BldgType == "1Fam", HouseStyle == "1Story", RoofMatl == "Tar&Grv") %>%
  arrange(YearBuilt)

# Similar houses mostly use either Plywood or BrkComm, but it looks like VinylSd was only used at later YearBuilt.
# Which is the most common exterior material used overall?
# It looks like Plywood is most common, but Wood sliding might also be possible.
dataset %>% filter(YearBuilt < 0.7 & YearBuilt > 0.30, BldgType == "1Fam", RoofMatl == "Tar&Grv") %>%
  select(Exterior1st) %>%
  group_by(Exterior1st) %>%
  summarize(count = n())

# There is not a very clear most common value, but both Plywood and Wd Sdng (Wood sliding) might be similar in their contribution to the value of a house.
# It appears that Plwood might imply a higher average SalePrice compared to Wood sliding.
dataset %>% filter(Exterior1st %in% c("Plywood", "Wd Sdng"), SalePrice > 0) %>%
  select(Exterior1st, SalePrice) %>%
  group_by(Exterior1st) %>%
  summarize(avg_price = mean(SalePrice))

# This is not an ideal imputation, but based on our observations we will impute Plywood as Exterior1st and 2nd, due to it being most common
# for similar houses in terms of building tpye, roof material and year built.
dataset$Exterior1st[which(is.na(dataset$Exterior1st))] <- "Plywood"
dataset$Exterior2nd[which(is.na(dataset$Exterior2nd))] <- "Plywood"


#####
#####


# Dealing with missing values concerning the basement. We take a look at all basement-related columns that have missing values. Is there a pattern?
Bsmt_missing_vals = c("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "BsmtFinSF1", "BsmtFinSF2")
dataset[!complete.cases(dataset[, names(dataset) %in% Bsmt_missing_vals]), names(dataset) %in% names(dataset)[which(grepl("Bsmt", names(dataset)))]]

# All missing values related to basement can be explained by the fact that there either is no basement, or it is yet unfinished.
# We can impute a "None" for BsmtQual, BsmtCond, and BsmtExposure missing values.

# We convert to a character, add the NA replacement and change back to factor
dataset$BsmtQual <- as.character(dataset$BsmtQual)
dataset$BsmtQual[is.na(dataset$BsmtQual)] <- "None"
dataset$BsmtQual <- as.factor(dataset$BsmtQual)

dataset$BsmtExposure <- as.character(dataset$BsmtExposure)
dataset$BsmtExposure[is.na(dataset$BsmtExposure)] <- "None"
dataset$BsmtExposure <- as.factor(dataset$BsmtExposure)

dataset$BsmtCond <- as.character(dataset$BsmtCond)
dataset$BsmtCond[is.na(dataset$BsmtCond)] <- "None"
dataset$BsmtCond <- as.factor(dataset$BsmtCond)

# We replace the remaining NAs in BsmtFullBath and BsmtHalfBath with 0 as they have no basement.
dataset$BsmtFullBath[is.na(dataset$BsmtFullBath)] <- 0
dataset$BsmtHalfBath[is.na(dataset$BsmtHalfBath)] <- 0

# We replace the missing Bsmt-values of house 2121 with 0, as this house has no basement
dataset[2121, ] # This house has no basement
dataset$BsmtFinSF1[2121] <- 0
dataset$BsmtFinSF2[2121] <- 0
dataset$BsmtUnfSF[2121] <- 0
dataset$TotalBsmtSF[2121] <- 0

# We change the encoding of BsmtFinType1 and type 2 from character to factor
dataset$BsmtFinType1 <- as.factor(dataset$BsmtFinType1)
dataset$BsmtFinType2 <- as.factor(dataset$BsmtFinType2)

# We change the encoding of BsmtFinSF1 and SF2, as well as BsmtUnfSF and TotalBsmtSF from integer to numeric
dataset$BsmtFinSF1 <- as.numeric(dataset$BsmtFinSF1)
dataset$BsmtFinSF2 <- as.numeric(dataset$BsmtFinSF2)
dataset$BsmtUnfSF <- as.numeric(dataset$BsmtUnfSF)
dataset$TotalBsmtSF <- as.numeric(dataset$TotalBsmtSF)



###





#####
#####


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


#####
#####


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


#####
#####


# There is one missing value in GarageCars and GarageArea - maybe this comes from the same house which doesn't have a garage?
garage_NA_ind <- which(is.na(dataset$GarageArea)) # Find the row with the missing GarageArea
print(dataset[garage_NA_ind, ]) # Look at the entry with the id 2577 and realize that there is "NoGarage" in multiple garage-related columns

# We can replace the missing values in GarageCars and GarageArea with 0.
dataset$GarageCars[garage_NA_ind] <- 0
dataset$GarageArea[garage_NA_ind] <- 0


#####
#####

# There is one NA remaining in KitchenQual, but is not immediately evident why it is missing
dataset[which(is.na(dataset$KitchenQual)), ]

# From the plot it appears that a "TA" = "Typical" kitchen is the most common (mode); will will impute that for the single missing value
plot(dataset$KitchenQual,
     col = "orange",
     xlab = "Kitchen quality",
     ylab = "Count",
     main = "Kitchen quality")

dataset$KitchenQual[which(is.na(dataset$KitchenQual))] <- "TA" # We impute "TA" as in "typical" kitchen quality due to it being most likely


#####
#####


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


#####
#####


# There are still two missing values in Functional: Home functionality (Assume typical unless deductions are warranted) (from data description)
dataset[which(is.na(dataset$Functional)), ] # The OverallCond of one house and the OverallQual of the other are at 1!

# What kind of Functional values do houses with an OverallCond of only 1 and OverallQual 4 have most likely?
dataset[dataset$OverallCond == 1 & dataset$OverallQual == 4, ]$Functional %>% plot(col = "orange") # There is only one other house like this, with similar YearBuilt

# As the houses are very similar from various attributes, we will impute a Functional of "Typ" typical
dataset$Functional[2474] <- "Typ"

# The other house with missing Functional...
dataset[dataset$OverallCond == 5 & dataset$OverallQual == 1, ]$Functional %>% plot(col = "orange") # There is no house like it
dataset[dataset$OverallQual == 1, ]$Functional %>% plot(col = "orange") # Low-quality houses tend to be "Maj1", "Mod", or "Typ"
dataset[dataset$OverallCond == 5, ]$Functional %>% plot(col = "orange") # Houses with average condition tend to be "Typ", typical

dataset$Functional %>% plot(col = "orange") # The mode is "Typ", which we will therefore imputed in this case

dataset$Functional[2217] <- "Typ"



###

#####
#####


# We change the encoding of FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces,
# GarageType, FireplaceQu, GarageFinish, GarageQual, GarageCond, PoolQc, Fence, MiscFeature, MiscVal
dataset$FullBath <- as.numeric(dataset$FullBath)
dataset$HalfBath <- as.numeric(dataset$HalfBath)
dataset$BedroomAbvGr <- as.numeric(dataset$BedroomAbvGr)
dataset$KitchenAbvGr <- as.numeric(dataset$KitchenAbvGr)
dataset$TotRmsAbvGrd <- as.numeric(dataset$TotRmsAbvGrd)
dataset$Fireplaces <- as.numeric(dataset$Fireplaces)
dataset$MiscVal <- as.numeric(dataset$MiscVal)
dataset$YearBuilt <- as.numeric(dataset$YearBuilt)
dataset$MoSold <- as.numeric(dataset$MoSold)
dataset$YrSold <- as.numeric(dataset$YrSold)


dataset$GarageType <- as.factor(dataset$GarageType)
dataset$GarageFinish <- as.factor(dataset$GarageFinish)
dataset$Fence <- as.factor(dataset$Fence)
dataset$MiscFeature <- as.factor(dataset$MiscFeature)


# We change the categorical encoding of quality variables to numeric
dataset$ExterQual <- as.numeric(factor(dataset$ExterQual, levels=c("None","Po","Fa", "TA", "Gd", "Ex")))
dataset$ExterCond <- as.numeric(factor(dataset$ExterCond, levels=c("None","Po","Fa", "TA", "Gd", "Ex")))
dataset$BsmtQual <- as.numeric(factor(dataset$BsmtQual, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$BsmtCond <- as.numeric(factor(dataset$BsmtCond, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$BsmtExposure <- as.numeric(factor(dataset$BsmtExposure, levels=c("None","No", "Mn", "Av", "Gd")))
dataset$BsmtFinType1 <- as.numeric(factor(dataset$BsmtFinType1, levels=c("None","Unf","LwQ","Rec","BLQ","ALQ","GLQ")))
dataset$BsmtFinType2 <- as.numeric(factor(dataset$BsmtFinType2, levels=c("None","Unf","LwQ","Rec","BLQ","ALQ","GLQ")))
dataset$HeatingQC <- as.numeric(factor(dataset$HeatingQC, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$KitchenQual <- as.numeric(factor(dataset$KitchenQual, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$FireplaceQu <- as.numeric(factor(dataset$FireplaceQu, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$GarageQual <- as.numeric(factor(dataset$GarageQual, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$GarageCond <- as.numeric(factor(dataset$GarageCond, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$PoolQC <- as.numeric(factor(dataset$PoolQC, levels=c("None", "Fa", "TA", "Gd", "Ex")))


#####
#####


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
  ggtitle("Linear feet of street connected to property as a function of the lot size in square feet") +
  theme_bw()

# We will apply a gradient boosting machine model to predict the missing LotFrontage values.
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


########### Tuning for LotFrontage ###########

# We define a function to help us plot the tuning effects
plot_tunings <- function(model, probs = .90) {
  ggplot(data = model) +
    coord_cartesian(ylim = c(quantile(model$results$RMSE, probs = probs), min(model$results$RMSE))) +
    theme_bw()
}

# Define a grid of tuning parameters for the caret::train() function
grid_default <- expand.grid(
  nrounds = seq(from = 400, to = 1000, 50), # We search for the best number of rounds
  max_depth = 3,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 1, # We don't change column sampling
  min_child_weight = 1, # We don't change min child weight; the larger the more conservative the algorithm
  subsample = 1 # We don't change subsampling
)

# Train control for caret train() function; we use cross-validation to estimate out-of-sample error
train_control <- caret::trainControl(
  method = "cv", # We use 5-fold cross-validation
  number = 5,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE
)

# Train the xgboost model for LotFrontage
xgb_LF_tuned <- train(
  x = train_treated,
  y = LF_train$LotFrontage, # The outcome variable comes from the pre-treated data
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verboseIter = TRUE
)

xgb_LF_tuned$bestTune # We take a look at the best tuning values

plot_tunings(xgb_LF_tuned) # We use our defined function to visualize the tuning effects


# We predict LotFrontage
LF_test$LotFrontagePred<- predict(xgb_LF_tuned, newdata = as.matrix(test_treated))

# Missing value imputation with the predicted values for LotFrontage
dataset$LotFrontage[LF_index] <- LF_test$LotFrontagePred
summary(dataset)

# We can plot the predicted LotFrontage values against the LotArea values in LF_test to see if we observe the
# correlation between the two variables.
# Indeed, the predicted LotFrontage values seem to behave in similar fashion to the actual ones we observed above in the whole dataset
LF_test %>%
  ggplot(aes(x = log(LotArea), y = log(LotFrontagePred))) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  ggtitle("(Predicted) linear feet of street connected to property as a function of the lot size in square feet")



#######
# Dealing with variable correlation
#######

# Certain characteristics such as garage have many different variables within the dataset.
# Some might be strongly correlated. We will visualize correlations between SalePrice and all numeric variables.
# This will identify the most important numeric variables that influence SalePrice the most.
# We have to restrict dataset to the Ids that only occur in train, as test has no SalePrice (or we set it to 0)

numVars_ind <- sapply(dataset[train$Id, ], is.numeric) # Select all numeric variables
dataset_numVars <- dataset[train$Id, ][, numVars_ind] # Select numeric data
 
cor_numVar <- cor(dataset_numVars, use = "pairwise.complete.obs") # Correlations of all numeric variables

cor_sorted <- as.matrix(sort(cor_numVar[, "SalePrice"], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.4)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col = "black", tl.pos = "lt")





# What about other Garage-related variables? Plotting quality and condition
# reveals a clear relationship between the two variables.
# "TA" GarageQual is basically overlapping with GarageCond "TA", or in other words, a typical/average garage quality is in typtical/average condition.

p_qual <- dataset %>% 
  ggplot(aes(x = GarageQual)) +
  geom_bar(color = "black", fill = "orange") +
  theme_bw()

p_cond <- dataset %>% 
  ggplot(aes(x = GarageCond)) +
  geom_bar(color = "black", fill = "orange") +
  theme_bw()

grid.arrange(p_qual, p_cond, nrow = 1)

###
# We can remove one of these variables as they are redundant
dataset <- subset(dataset, select = -GarageCond)
###




###
# Total number of bathrooms
###

# While each bathroom variable individually has little influence on SalePrice, combined they might become a stronger predictor.
# We will value different types of bath according to their SalePrice correlation ratio compared to FullBaths.
cor(dataset[train$Id, ]$FullBath, dataset[train$Id, ]$SalePrice) # Full bathrooms have the strongest correlation to SalePrice
cor(dataset[train$Id, ]$HalfBath, dataset[train$Id, ]$SalePrice) # Half baths are much less valued, about half as much
(FullToHalfBathRatio <- cor(dataset[train$Id, ]$HalfBath, dataset[train$Id, ]$SalePrice) / cor(dataset[train$Id, ]$FullBath, dataset[train$Id, ]$SalePrice)) # We calculate the ratio of the correlations of Full and Half baths to SalePrice to use as a weighting factor below

# Basement full and especially half baths have weaker correlation with SalePrice
cor(dataset[train$Id, ]$BsmtFullBath, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtHalfBath, dataset[train$Id, ]$SalePrice)

# Weighting factor for FullBath to BsmtFullBath. A basement full bath receives a weighting factor of 0.3971685 as calculated below.
(FullToBsmtFullBathRatio <- cor(dataset[train$Id, ]$BsmtFullBath, dataset[train$Id, ]$SalePrice) / cor(dataset[train$Id, ]$FullBath, dataset[train$Id, ]$SalePrice))

# Weighting factor for FullBath to BsmtHalfBath. This actualy returns a negative value! We rather remove basement half baths from the calculations.
(FullToBsmtHalfBathRatio <- cor(dataset[train$Id, ]$BsmtHalfBath, dataset[train$Id, ]$SalePrice) / cor(dataset[train$Id, ]$FullBath, dataset[train$Id, ]$SalePrice))


# Only very few houses even have BsmtHalfBaths
summary(dataset$BsmtHalfBath)

# We will createa variable "TotalBaths", that sums up the various types of baths into one variable, taking into consideration the different correlations to SalePrice for full and half baths.
# We don't add BsmtHalfBaths, as their correlation with SalePrice is very low/negative.
dataset$TotalBaths <- dataset$FullBath + (dataset$BsmtFullBath * FullToBsmtFullBathRatio) + (dataset$HalfBath * FullToHalfBathRatio)
cor(dataset[train$Id, ]$TotalBaths, dataset[train$Id, ]$SalePrice) # Correlation of TotalBaths is higher than of just FullBaths

# The new TotalBaths variable has a correlation of almost 70% with SalePrice.

# We can remove the previous bath variables
dataset <- subset(dataset, select = -c(FullBath, HalfBath, BsmtFullBath, BsmtHalfBath))


###
# Skewedness of the Outcome variable SalePrice
###

# Monetary values are often log-normally distributed. How does SalePrice look like?
# The histogram below indicates a certain skewedness.
dataset[train$Id, ] %>%
  ggplot(aes(SalePrice)) +
  geom_histogram(bins = 30, color = "black", fill = "orange")

# We can check for skewedness with a function. The skew is considerably higher than 0.8.
# A log-transformation could potentially lead to SalePrice being more normal.
e1071::skewness(dataset[train$Id, ]$SalePrice) # Determine skew of SalePrice

dataset[train$Id, ] %>%
  ggplot(aes(log1p(SalePrice))) +
  geom_histogram(bins = 30, color = "black", fill = "orange")

# We log-transform SalePrice
dataset$SalePrice <- log1p(dataset$SalePrice)


#####
# Removal of redundant/colinear variables
#####


# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# YearBuilt and YearRemodAdd are nearly identically correlated with SalePrice
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$YearBuilt)
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$YearRemodAdd)

# We can remove YearRemodAdd
dataset <- subset(dataset, select = -YearRemodAdd)


#####
#####


# We will remove GarageYrBlt as described further above.
dataset <- subset(dataset, select = -GarageYrBlt)


#####
#####


# There are two missing values in Utilities.
dataset[which(is.na(dataset$Utilities)), ] # Both houses appear pretty typical

dataset %>% select(Utilities) %>% group_by(Utilities) %>% count()

# As there is only a single house with "NoSeWa" and all others have the same value, the Utilities column is not very informative.
# We will therefore remove Utilities from the dataset.
dataset[which(dataset$Utilities == "NoSeWa"), ] # THe only house with "NoSeWa" doesn't seem particulary special

dataset <- subset(dataset, select = -Utilities) # We remove the Utilities column from the dataset, as it contains little information


#####
#####


###
# Basement square feet variables
###
# What is the correlation between TotalBsmtSF and the individual measurements of basement square feet?
cor(dataset$TotalBsmtSF, (dataset$BsmtFinSF1 + dataset$BsmtFinSF2 + dataset$BsmtUnfSF)) # It is exactly 1.

# Correlation between the variables and SalePrice: They are all mostly weak individually., while TotalBsmtSF is highly correlated.
cor(dataset[train$Id, ]$TotalBsmtSF, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtFinSF1, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtFinSF2, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtUnfSF, dataset[train$Id, ]$SalePrice)

# We remove these unnecessary, redundant values, as they are all contained within TotalBsmtSF.
dataset <- subset(dataset, select = -c(BsmtFinSF1, BsmtFinSF2, BsmtUnfSF))


#####
#####


# It looks like GarageCars and GarageArea are highly correlated.
# We take a look at the correlation between GarageArea and GarageCars, as a bigger garage might contain more cars.
cor(dataset[train$Id, ]$GarageArea, dataset[train$Id, ]$GarageCars) # It's almost 0.89!

# As having highly correlated predictors is detrimental for modelling, we choose to keep only one of them.
# Which one has a higher correlation with SalePrice?
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$GarageArea)
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$GarageCars) # The number of cars is slightly stronger correlated with SalePrice!


###
# We can remove one of these variables as they are redundant
dataset <- subset(dataset, select = -GarageArea)
###


#####
#####


# The above ground living area is composed of 1StFlrSF, 2ndFlrSF and LowQualFinSF.
# The correlation is 1.
cor(dataset$GrLivArea, (dataset$X1stFlrSF + dataset$X2ndFlrSF + dataset$LowQualFinSF))

# We can drop the redundant variables
dataset <- subset(dataset, select = -c(X1stFlrSF, X2ndFlrSF, LowQualFinSF))


#####
#####




### All missing values have been dealt with and we can once again separate `dataset` into train and test ###
train <- dataset[train$Id, ]                                                                             ###
test <- subset(dataset[test$Id, ], select = -SalePrice)   # We remove the temporary SalePrice column again                         ###                                                        ###
############################################################################################################



##################################

###
# Modelling approaches
###

###################################
# First we write a loss-function to determine the residual mean squared error, or RMSE of the model.
# The function calculates the residuals/error and then takes the root mean square of them.
RMSE <- function(predicted_prices, true_prices) {
  error <- predicted_prices - true_prices
  sqrt(mean(error^2))
}

# Next, we split `train` into separate train_set and test_set for algorithm evaluation purposes (we won't use the real `test` subset for this and treat it as completely new data for final evaluations only)
# test_set will receive 10% of the data, train_set will receive 90%.
set.seed(1)
test_index <- createDataPartition(train$SalePrice, p = 0.1, list = FALSE)
train_set <- train[-test_index, ]
temp <- train[test_index, ] # temporary test set

# We make sure there are no entries in test_set that aren't in train_set
test_set <- temp %>%
  semi_join(train_set, by = c("RoofMatl", "Condition1", "OverallQual", "Exterior1st", "Exterior2nd",
                              "Electrical", "Functional", "MiscFeature")) # Variables were determined by trial and error with the lm() models below
# We return the removed entries from test to train
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

summary(train_set)
summary(test_set)


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
model_2_lm <- lm(SalePrice ~ ., data = subset(train_set, select = -c(Id))) # We remove Id and predict with all
model_2_pred <- predict(model_2_lm, newdata = subset(test_set, select = -c(Id)))
model_2_lm_RMSE <- RMSE(model_2_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_2_Multivariate_lm", RMSE = model_2_lm_RMSE))

model_rmses %>% knitr::kable()

# RMSE improved substantially, to below 0.14.
# This linear regression RMSE will serve as the baseline RMSE.

# We examine the multivariate lm model 2. It seems that most predictors are not significant as per p-value.
summary(model_2_lm)















# 3. Model 3: Gradient Boosting Machine - XGBOOST

# We select all relevant predictors
variables <- names(subset(train_set, select = -c(Id, SalePrice)))

# The vtreat function designTreatmentsZ helps encode all variables numerically via one-hot-encoding
treatment_plan <- designTreatmentsZ(train_set, variables) # Devise a treatment plan for the variables
(newvars <- treatment_plan %>%
    use_series(scoreFrame) %>%  # use_series() works like a $, but within pipes, so we can access scoreFrame      
    filter(code %in% c("clean", "lev")) %>%  # We select only the rows we care about: catP is a "prevalence fact" and tells whether the original level was rare or common and not really useful in the model
    use_series(varName))           # We get the varName column

# The prepare() function prepares our data subsets according to the treatment plan
# we devised above and encodes all relevant variables "newvars" numerically
train_set_treated <- prepare(treatment_plan, train_set,  varRestriction = newvars)
test_set_treated <- prepare(treatment_plan, test_set,  varRestriction = newvars)

# We can now see the numerical encoding of all variables thanks to treatment and preparation above
str(train_set_treated)
str(test_set_treated)



cv <- xgb.cv(data = as.matrix(train_set_treated),  # xgb.cv only takes a matrix of the treated, all-numerical input data
             label = train_set$SalePrice, # Outcome from untreated data
             nrounds = 200, # We go up to 100 rounds of fitting models on the remaining residuals
             nfold = 3, # We use 5 folds for cross-validation
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
SalePrice_model_xgb <- xgboost(data = as.matrix(train_set_treated), # Treated training data as a matrix
                                 label = train_set$SalePrice,  # Column of outcomes from original data
                                 nrounds = ntrees,       # number of trees to build, which we determined via cross-validation
                                 objective = "reg:linear", # objective
                                 eta = 0.3, # The learning rate; Closer to 0 is slower, but less prone to overfitting; Closer to 1 is faster, but more likely to overfit
                                 max_depth = 6,
                                 verbose = 0)  # silent

# Now we can predict SalePrice in the test_set with the xgb-model
SalePrice_model_xbg_pred <- predict(SalePrice_model_xgb, newdata = as.matrix(test_set_treated)) # We predict LotFrontage; newdata has to be a matrix
SalePrice_model_xbg_pred

model_3_xgb_RMSE <- RMSE(SalePrice_model_xbg_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_3_xgb_linreg", RMSE = model_3_xgb_RMSE))

model_rmses %>% knitr::kable()

# The RMSE was worsened compared to the simple linear regression approach.
# But can we do better? We try and tune our xgb model.


# As we will be using cross-validation, we can use the entire train dataset.
# We prepare it for xgboost usage with one-hot encoding.

variables <- names(subset(train, select = -c(Id, SalePrice)))

# The vtreat function designTreatmentsZ helps encode all variables numerically via one-hot-encoding
treatment_plan <- designTreatmentsZ(train, variables) # Devise a treatment plan for the variables
(newvars <- treatment_plan %>%
    use_series(scoreFrame) %>%  # use_series() works like a $, but within pipes, so we can access scoreFrame      
    filter(code %in% c("clean", "lev")) %>%  # We select only the rows we care about: catP is a "prevalence fact" and tells whether the original level was rare or common and not really useful in the model
    use_series(varName))           # We get the varName column

# The prepare() function prepares our data subsets according to the treatment plan
# we devised above and encodes all relevant variables "newvars" numerically
train_treated <- prepare(treatment_plan, train,  varRestriction = newvars)



# 1st set of tuning parameters for learning rate and maximum tree depth

grid_default <- expand.grid(
  nrounds = seq(from = 1000, to = 3000, 500),
  max_depth = 3,
  eta = 0.025,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Train control for caret train() function
train_control <- caret::trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = TRUE,
  allowParallel = TRUE
)

xgb_1st_tuning <- train(
  x = train_treated,
  y = train$SalePrice,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)


# By plotting the tune settings, we can observe that higher learning rates (0.6 and 1) actually don't improve RMSE
# at higher number of iterations (trees built).
plot_tunings(xgb_1st_tuning)

# We can select the best tuning values from the model like this
xgb_1st_tuning$bestTune

# We record the RMSE of the best tuned model

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_4_xgb_tuned", RMSE = min(xgb_1st_tuning$results$RMSE)))

model_rmses %>% knitr::kable()





########################
# Ranger random forest model
########################

tuneGrid <- data.frame(
  .mtry = 63, # Number of randomly chosen variables to possibly split at in each node
  .splitrule = "variance", # Default splitrule variance for regression
  .min.node.size = 5 # Default minimum node size for regression
)

ranger_model <- train(SalePrice ~ .,
                      method = "ranger",
                      data = subset(train_set, select = -Id),
                      trControl = trainControl(method = "repeatedcv",
                                               number = 5, repeats = 5, verboseIter = TRUE),
                      tuneGrid = tuneGrid,
                      num.trees = 2000)

print(ranger_model)

plot(ranger_model)


model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_5_ranger", RMSE = min(ranger_model$results$RMSE)))

model_rmses %>% knitr::kable()



###########################
# Glmnet - Lasso and/or Ride regression
###########################


# Train() fits only one model per Alpha value and passes all Lambda values at once for simultaneous fitting.
# We will explore pure Ridge, pure Lasso, and mixes between the two in 5% steps (e.g. 10% ridge and 90 % lasso regression etc... )
tuneGrid <- expand.grid(
  alpha = seq(0, 1, 0.05), # Mixing parameter between Lasso and Ridge regression; alpha = 0 equals pure Ridge regression, alpha = 1 equals pure Lasso regression
  lambda = seq(0.0001, 1, length = 100) # Strength of the penalty on the coefficients; A Lambda of 1 would shrink regression coefficients to 0, so that the model would only predict the intercept
)

myControl <- trainControl(
  method = "repeatedcv", number = 5, repeats = 5,
  verboseIter = TRUE
)

glm_model <- train(
  SalePrice ~ .,
  data = subset(train, select = -Id),
  method = "glmnet",
  trControl = myControl,
  tuneGrid = tuneGrid
)

print(glm_model)

print(glm_model$bestTune) # The optimal determined tuning parameters

plot(glm_model) # Plot of the tuning

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_6_glmnet", RMSE = min(glm_model$results$RMSE)))

model_rmses %>% knitr::kable()
