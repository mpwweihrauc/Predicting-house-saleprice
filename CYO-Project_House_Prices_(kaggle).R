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

# Is LotArea significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$LotArea) # We detect a very strong skew > 0.75; probably caused by a few houses with very large LotArea.
dataset %>%
  ggplot(aes(LotArea)) +
  geom_histogram(bins = 30, binwidth = 5000, fill = "orange", color = "black") +
  ggtitle("Histrogram of LotArea") +
  theme_bw()

# LotArea appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, LotArea now looks much more normal.
dataset %>% filter(LotArea != 0) %>%
  ggplot(aes(log1p(LotArea))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of LotArea") +
  theme_bw()

# We apply the log-transformation to LotArea.
dataset$LotArea <- log1p(dataset$LotArea)

#####
#####

# Alley is encoded as character, but should clearly be a facor variable. We change it accordingly.
dataset$Alley <- as.factor(dataset$Alley)

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

# OverallQual, OverallCond are ordinal, but encoded as integers. We change them into factor variables
dataset$OverallCond <- as.factor(dataset$OverallCond)
dataset$OverallQual <- as.factor(dataset$OverallQual)

#####
#####

# YearBuilt, YearRemodAdd, GarageYrBlt, MoSold, and YrSold would containt too large a number of levels as factor variables.
# Instead, we will normalize them to be between 0 and 1.
# GarageYrBlt contains missing values, most likely because some houses don't have a garage.
# We will inspect GarageYrBlt.

# GarageYrBuilt has some missing values, are these due to there being no Garage?
# Indeed, most houses with a missing GarageYrBlt simply have no garage, but zwo houses have a detached Garage.
dataset[which(is.na(dataset$GarageYrBlt)), ] %>% select(GarageYrBlt, GarageType, YearBuilt)

# We will impute the YearBuilt value for the detached garages with missing GarageYrBlt values.
# While it is quite possible that the detached garage was built later than the house itself, this is our best bet.
dataset[which(is.na(dataset$GarageYrBlt)), ] %>% select(Id, GarageYrBlt, GarageType, YearBuilt) %>% filter(GarageType == "Detchd")
dataset$GarageYrBlt[2127] <- dataset$YearBuilt[2127]
dataset$GarageYrBlt[2577] <- dataset$YearBuilt[2577]

# We then replace the remaining missing values for houses without any garage with 0 and afterwards scale the variable as described above.
dataset$GarageYrBlt[which(is.na(dataset$GarageYrBlt))] <- 0 # We impute 0 for the missing garage values
dataset$GarageYrBlt <- (dataset$GarageYrBlt - min(dataset$GarageYrBlt))/(max(dataset$GarageYrBlt) - min(dataset$GarageYrBlt)) # We normalize GarageYrBuilt to be between 0 and 1

# Furthermore, we scale the other columns mentioned above.
dataset$YearBuilt <- (dataset$YearBuilt - min(dataset$YearBuilt))/(max(dataset$YearBuilt) - min(dataset$YearBuilt))
dataset$YearRemodAdd <- (dataset$YearRemodAdd - min(dataset$YearRemodAdd))/(max(dataset$YearRemodAdd) - min(dataset$YearRemodAdd))
dataset$MoSold <- (dataset$MoSold - min(dataset$MoSold))/(max(dataset$MoSold) - min(dataset$MoSold))
dataset$YrSold <- (dataset$YrSold - min(dataset$YrSold))/(max(dataset$YrSold) - min(dataset$YrSold))


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

# Is BsmtFinSF1 significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$BsmtFinSF1) # We detect a strong skew > 0.75
dataset %>%
  ggplot(aes(BsmtFinSF1)) +
  geom_histogram(bins = 30, binwidth = 200, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtFinSF1") +
  theme_bw()

# BsmtFinSF1 appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, BsmtFinSF1 now looks much more normal.
dataset %>% filter(BsmtFinSF1 != 0) %>%
  ggplot(aes(log1p(BsmtFinSF1))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtFinSF1") +
  theme_bw()

# We apply the log-transformation to BsmtFinSF1.
dataset$BsmtFinSF1 <- log1p(dataset$BsmtFinSF1)

###

# Is BsmtFinSF2 significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$BsmtFinSF2) # We detect a strong skew > 0.75; most houses have very low or 0 values here
dataset %>% 
  ggplot(aes(BsmtFinSF2)) +
  geom_histogram(bins = 30, binwidth = 200, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtFinSF2") +
  theme_bw()

# BsmtFinSF2 appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, BsmtFinSF2 now looks much more normal.
dataset %>% filter(BsmtFinSF2 != 0) %>%
  ggplot(aes(log1p(BsmtFinSF2))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtFinSF2") +
  theme_bw()

# We apply the log-transformation to BsmtFinSF2.
dataset$BsmtFinSF2 <- log1p(dataset$BsmtFinSF2)

###

# Is TotalBsmtSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$TotalBsmtSF) # We detect a strong skew > 0.75
dataset %>% 
  ggplot(aes(TotalBsmtSF)) +
  geom_histogram(bins = 30, binwidth = 200, fill = "orange", color = "black") +
  ggtitle("Histrogram of TotalBsmtSF") +
  theme_bw()

# TotalBsmtSF appears to be skewed a little. We apply a log-transformation with log1p().
# Indeed, TotalBsmtSF now looks more normal.
dataset %>% filter(TotalBsmtSF != 0) %>%
  ggplot(aes(log1p(TotalBsmtSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of TotalBsmtSF") +
  theme_bw()

# We apply the log-transformation to TotalBsmtSF.
dataset$TotalBsmtSF <- log1p(dataset$TotalBsmtSF)

###

# Is BsmtUnfSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$BsmtUnfSF) # We detect a skew > 0.75
dataset %>% 
  ggplot(aes(BsmtUnfSF)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtUnfSF") +
  theme_bw()

# BsmtUnfSF appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, BsmtUnfSF now looks more normal.
dataset %>% filter(BsmtUnfSF != 0) %>%
  ggplot(aes(log1p(BsmtUnfSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of BsmtUnfSF") +
  theme_bw()

# We apply the log-transformation to BsmtUnfSF.
dataset$BsmtUnfSF <- log1p(dataset$BsmtUnfSF)


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

# Is MasVnrArea significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$MasVnrArea) # We detect a strong skew > 0.75
dataset %>%
  ggplot(aes(MasVnrArea)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of MasVnrArea") +
  theme_bw()

# MasVnrArea appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, MasVnrArea now looks much more normal.
dataset %>% filter(MasVnrArea != 0) %>%
  ggplot(aes(log1p(MasVnrArea))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of MasVnrArea") +
  theme_bw()

# We apply the log-transformation to MasVnrArea.
dataset$MasVnrArea <- log1p(dataset$MasVnrArea)


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

# Is GarageArea significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$GarageArea) # We detect a very weak skew < 0.25
dataset %>%
  ggplot(aes(GarageArea)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of GarageArea") +
  theme_bw()

# As GarageArea has little skew, we leave it untouched

#####
#####

# Is X1stFlrSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$X1stFlrSF) # We detect a strong skew > 0.75
dataset %>%
  ggplot(aes(X1stFlrSF)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of X1stFlrSF") +
  theme_bw()

# X1stFlrSF appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, X1stFlrSF now looks much more normal.
dataset %>% filter(X1stFlrSF != 0) %>%
  ggplot(aes(log1p(X1stFlrSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of X1stFlrSF") +
  theme_bw()

# We apply the log-transformation to X1stFlrSF.
dataset$X1stFlrSF <- log1p(dataset$X1stFlrSF)

###

# Is X2ndFlrSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$X2ndFlrSF) # We detect a strong skew > 0.75
dataset %>%
  ggplot(aes(X2ndFlrSF)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of X2ndFlrSF") +
  theme_bw()

# X2ndFlrSF appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, X2ndFlrSF now looks much more normal.
dataset %>% filter(X2ndFlrSF != 0) %>%
  ggplot(aes(log1p(X2ndFlrSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of X2ndFlrSF") +
  theme_bw()

# We apply the log-transformation to X2ndFlrSF.
dataset$X2ndFlrSF <- log1p(dataset$X2ndFlrSF)

###


#####
#####

# Is LowQualFinSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$LowQualFinSF) # We detect a strong skew > 0.75; but most houses don't have a value above 0 here
skewness(dataset$LowQualFinSF[dataset$LowQualFinSF > 0]) # If we remove all 0 values, the skew is 0.87
dataset %>% filter(LowQualFinSF != 0) %>%
  ggplot(aes(LowQualFinSF)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of LowQualFinSF") +
  theme_bw()

# LowQualFinSF appears to be skewed, although only a few houses actually have values above 0. We apply a log-transformation with log1p().
# Indeed, LowQualFinSF now looks much more normal.
dataset %>% filter(LowQualFinSF != 0) %>%
  ggplot(aes(log1p(LowQualFinSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of LowQualFinSF") +
  theme_bw()

# We apply the log-transformation to LowQualFinSF.
dataset$LowQualFinSF <- log1p(dataset$LowQualFinSF)

###


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


#####
#####


# Is GrLivArea significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$GrLivArea) # We detect a strong skew > 0.75
dataset %>%
  ggplot(aes(GrLivArea)) +
  geom_histogram(bins = 30, binwidth = 100, fill = "orange", color = "black") +
  ggtitle("Histrogram of GrLivArea") +
  theme_bw()

# GrLivArea appears to be skewed significantly. We apply a log-transformation with log1p().
# Indeed, GrLivArea now looks much more normal.
dataset %>% filter(GrLivArea != 0) %>%
  ggplot(aes(log1p(GrLivArea))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of GrLivArea") +
  theme_bw()

# We apply the log-transformation to GrLivArea.
dataset$GrLivArea <- log1p(dataset$GrLivArea)

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

dataset$GarageType <- as.factor(dataset$GarageType)
dataset$FireplaceQu <- as.factor(dataset$FireplaceQu)
dataset$GarageFinish <- as.factor(dataset$GarageFinish)
dataset$GarageQual <- as.factor(dataset$GarageQual)
dataset$GarageCond <- as.factor(dataset$GarageCond)
dataset$PoolQC <- as.factor(dataset$PoolQC)
dataset$Fence <- as.factor(dataset$Fence)
dataset$MiscFeature <- as.factor(dataset$MiscFeature)


#####
#####


# Is WoodDeckSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$WoodDeckSF) # We detect a strong skew > 0.75
dataset %>% filter(WoodDeckSF != 0) %>%
  ggplot(aes(WoodDeckSF)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of WoodDeckSF") +
  theme_bw()

# WoodDeckSF appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, WoodDeckSF now looks more normal.
dataset %>% filter(WoodDeckSF != 0) %>%
  ggplot(aes(log1p(WoodDeckSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of WoodDeckSF") +
  theme_bw()

# We apply the log-transformation to WoodDeckSF.
dataset$WoodDeckSF <- log1p(dataset$WoodDeckSF)

###


#####
#####


# Is OpenPorchSF significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$OpenPorchSF) # We detect a strong skew > 0.75
dataset %>% filter(OpenPorchSF != 0) %>%
  ggplot(aes(OpenPorchSF)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of OpenPorchSF") +
  theme_bw()

# OpenPorchSF appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, OpenPorchSF now looks more normal.
dataset %>% filter(OpenPorchSF != 0) %>%
  ggplot(aes(log1p(OpenPorchSF))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of OpenPorchSF") +
  theme_bw()

# We apply the log-transformation to OpenPorchSF.
dataset$OpenPorchSF <- log1p(dataset$OpenPorchSF)


###


#####
#####


# Is EnclosedPorch significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$EnclosedPorch) # We detect a strong skew > 0.75
dataset %>% filter(EnclosedPorch != 0) %>%
  ggplot(aes(EnclosedPorch)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of EnclosedPorch") +
  theme_bw()

# EnclosedPorch appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, EnclosedPorch now looks more normal.
dataset %>% filter(EnclosedPorch != 0) %>%
  ggplot(aes(log1p(EnclosedPorch))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of EnclosedPorch") +
  theme_bw()

# We apply the log-transformation to EnclosedPorch.
dataset$EnclosedPorch <- log1p(dataset$EnclosedPorch)


###


#####
#####


# Is X3SsnPorch significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$X3SsnPorch) # We detect a strong skew > 0.75
dataset %>% filter(X3SsnPorch != 0) %>%
  ggplot(aes(X3SsnPorch)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of X3SsnPorch") +
  theme_bw()

# X3SsnPorch appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, X3SsnPorch now looks more normal.
dataset %>% filter(X3SsnPorch != 0) %>%
  ggplot(aes(log1p(X3SsnPorch))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of X3SsnPorch") +
  theme_bw()

# We apply the log-transformation to X3SsnPorch.
dataset$X3SsnPorch <- log1p(dataset$X3SsnPorch)

###


#####
#####


# Is ScreenPorch significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$ScreenPorch) # We detect a strong skew > 0.75
dataset %>% filter(ScreenPorch != 0) %>%
  ggplot(aes(ScreenPorch)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of ScreenPorch") +
  theme_bw()

# ScreenPorch appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, ScreenPorch now looks more normal.
dataset %>% filter(ScreenPorch != 0) %>%
  ggplot(aes(log1p(ScreenPorch))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of ScreenPorch") +
  theme_bw()

# We apply the log-transformation to ScreenPorch.
dataset$ScreenPorch <- log1p(dataset$ScreenPorch)

###

#####
#####


# Is PoolArea significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$PoolArea) # We detect a strong skew > 0.75
dataset %>% filter(PoolArea != 0) %>%
  ggplot(aes(PoolArea)) +
  geom_histogram(bins = 30, binwidth = 50, fill = "orange", color = "black") +
  ggtitle("Histrogram of PoolArea") +
  theme_bw()

# PoolArea appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, PoolArea now looks more normal - the effect of this transformation is arguable.
dataset %>% filter(PoolArea != 0) %>%
  ggplot(aes(log1p(PoolArea))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of PoolArea") +
  theme_bw()

# We apply the log-transformation to PoolArea.
dataset$PoolArea <- log1p(dataset$PoolArea)


###


#####
#####


# Is MiscVal significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(dataset$MiscVal) # We detect a strong skew > 0.75
dataset %>% filter(MiscVal != 0) %>%
  ggplot(aes(MiscVal)) +
  geom_histogram(bins = 30, binwidth = 500, fill = "orange", color = "black") +
  ggtitle("Histrogram of MiscVal") +
  theme_bw()

# MiscVal appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, MiscVal now looks more normal.
dataset %>% filter(MiscVal != 0) %>%
  ggplot(aes(log1p(MiscVal))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of MiscVal") +
  theme_bw()

# We apply the log-transformation to MiscVal.
dataset$MiscVal <- log1p(dataset$MiscVal)

###


#####
#####

# Is LotFrontage significantly skewed?
# We can detect skewness with the e1071::skewness() function and plot a histrogram of the variable.
skewness(na.omit(dataset$LotFrontage)) # We detect a strong skew > 0.75
dataset %>% filter(LotFrontage != "NA") %>%
  ggplot(aes(LotFrontage)) +
  geom_histogram(bins = 30, binwidth = 10, fill = "orange", color = "black") +
  ggtitle("Histrogram of LotFrontage") +
  theme_bw()

# LotFrontage appears to be skewed quite a bit. We apply a log-transformation with log1p().
# Indeed, LotFrontage now looks more normal.
dataset %>% filter(LotFrontage != "NA") %>%
  ggplot(aes(log1p(LotFrontage))) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  ggtitle("Histrogram of LotFrontage") +
  theme_bw()

# We apply the log-transformation to LotFrontage.
dataset$LotFrontage <- log1p(dataset$LotFrontage)

###

#####
#####


#########################################################################################################
# Dealing with missing values in LotFrontage, which are the linear feet of street connected to property.
# LotFrontage might be closely correlated to the LotArea, the lot size in square feet.

# We plot LotArea against LotFrontage. Indeed, there seems to be a positive correlation
# between LotFrontage and LotArea as shown by the fitted general additive model explaining 
# LotFrontage as a smooth function of LotArea. NAs are automatically removed from the plot.
dataset %>%
  ggplot(aes(x = LotArea, y = LotFrontage)) +
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
  ggplot(aes(x = LotArea, y = LotFrontagePred)) +
  geom_point() +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  ggtitle("(Predicted) linear feet of street connected to property as a function of the lot size in square feet")



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



# 3. Model 3: Gradient Boosting Machine - XGBOOST


# We select all relevant predictors
variables <- names(train_set)[c(-1, -80)]

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

# The RMSE was worsened compared to  the simple linear regression approach.
# But can we do better? We try and tune our xgb model

# Default parameters
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Train control for caret train() function
train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE
)

xgb_untuned <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)

# We predict with the yet untuned xgb model and record the RMSE
xgb_untuned_pred <- predict(xgb_untuned, newdata = as.matrix(test_set_treated))
xgb_untuned_rmse <- RMSE(xgb_untuned_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_4_xgb_untuned", RMSE = xgb_untuned_rmse))

model_rmses %>% knitr::kable()


# We try a set of different tuning options to improve our xgb model


# 1st set pf tuning parameters for learning rate and maximum tree depth

grid_default <- expand.grid(
  nrounds = seq(from = 100, to = 1000, 100),
  max_depth = c(2,3,4,5,6),
  eta = c(0.05, 0.1, 0.3, 0.6, 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Train control for caret train() function
train_control <- caret::trainControl(
  method = "cv", # We use 3-fold cross-validation
  number = 3,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE
)

xgb_1st_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)


# By plotting the tune settings, we can observe that higher learning rates (0.6 and 1) actually don't improve RMSE
# at higher number of iterations (trees built).
# A learning rate of 0.1 appears to perform quite well. The higher learning rate values, while faster, are not a good option here.
plot_tunings(xgb_1st_tuning)

# We can select the best tuning values from the model like this
xgb_1st_tuning$bestTune

# We predict with the 1st tuning parameters and record the RMSE
xgb_1st_tuning_pred <- predict(xgb_1st_tuning, newdata = as.matrix(test_set_treated))
xgb_1st_tuning_rmse <- RMSE(xgb_1st_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_5_xgb_1st_tuning", RMSE = xgb_1st_tuning_rmse))

model_rmses %>% knitr::kable()



# 2nd set of tuning parameters

grid_default <- expand.grid(
  nrounds = seq(from = 200, to = 1000, 100),
  max_depth = 2, # We fix max tree depth to 2 from our 1st tuning round
  eta = 0.1, # We fix learning rate to 0.1 from our 1st tuning round
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0), # column sampling
  min_child_weight = c(1, 2, 3), # We look for optimal minimum child weight
  subsample = c(0.5, 0.75, 1.0) # subsampling
)

# Train control for caret train() function
train_control <- caret::trainControl(
  method = "cv", # We use 3-fold cross-validation
  number = 3,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE
)

xgb_2nd_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)

# We plot the 2nd tunings.
plot_tunings(xgb_2nd_tuning)

# We can select the best tuning values from the model like this
xgb_2nd_tuning$bestTune

# We predict with the 1st tuning parameters and record the RMSE
xgb_2nd_tuning_pred <- predict(xgb_2nd_tuning, newdata = as.matrix(test_set_treated))
xgb_2nd_tuning_rmse <- RMSE(xgb_2nd_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_6_xgb_2nd_tuning", RMSE = xgb_2nd_tuning_rmse))

model_rmses %>% knitr::kable()






# 3rd set of tuning parameters

grid_default <- expand.grid(
  nrounds = seq(from = 100, to = 2000, 100), # We search for the best number of rounds
  max_depth = 2, # We fix tree depth from 1st tune
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1), # We optimize learning rate
  gamma = 0,
  colsample_bytree = 0.4, # We set column sampling to 0.4 from our 2nd tuning
  min_child_weight = 1, # We set child weight to 1 from our 2nd tuning
  subsample = 1 # We set subsampling to 1 from our 2nd tuning
)

# Train control for caret train() function
train_control <- caret::trainControl(
  method = "cv", # We use 3-fold cross-validation
  number = 3,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE
)

xgb_3rd_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)

# We plot the 3rd tunings.
plot_tunings(xgb_3rd_tuning)

# We can select the best tuning values from the model like this
xgb_3rd_tuning$bestTune

# We predict with the 1st tuning parameters and record the RMSE
xgb_3rd_tuning_pred <- predict(xgb_3rd_tuning, newdata = as.matrix(test_set_treated))
xgb_3rd_tuning_rmse <- RMSE(xgb_3rd_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_7_xgb_3rd_tuning", RMSE = xgb_3rd_tuning_rmse))

model_rmses %>% knitr::kable()

# After the 3rd tuning, we arrive at the following optimal parameters
xgb_3rd_tuning$bestTune
