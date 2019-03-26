# Please download the datasets from the provided link or find them in my provided Github repository, links below, or in the report.
# Link to Kaggle page: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link to my Github page: https://github.com/mpwweihrauc/ML_House_Prices.git
# You will need the test.csv and the train.csv files. There is also a data_description.txt with descriptions for all the different parameters in the dataset.

# We begin by loading/installing all required libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org", dependencies = c("Depends", "Suggests"))
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(vtreat)) install.packages("vtreat", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(vip)) install.packages("vip", repos = "http://cran.us.r-project.org")
if(!require(VIM)) install.packages("VIM", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")


# Parallelization, which will be used to speed up hyperparameter tuning steps later.
cl <- makeCluster(detectCores(logical = FALSE))
registerDoParallel(cl = cl)

# We import the training and testing data subsets (files from Kaggle).
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


# To facilitate feature engineering and data cleaning we temporarily merge train and test into "dataset".
test$SalePrice <- 0 # Temporarily add SalePrice column with 0s to test
dataset <- rbind(train, test) # Merge train and test
summary(dataset) # There's a lot of NA values in many columns

# We will now systematically analyse each feature of the dataset.
# Whenever we work with the SalePrice variable, we will subset "dataset" with the train$Id, as "test" has
# no entries for it.



#####
# MSSubClass: Identifies the type of dwelling involved in the sale.
#####

# There are no missing values in MSSubClass.
summary(dataset$MSSubClass)

# MSSubClass should be a factor variable.
dataset$MSSubClass <- as.factor(dataset$MSSubClass)

# Boxplot of MSSubClass vs. SalePrice
MSSubClass_boxplot <- dataset[train$Id, ] %>%
  group_by(MSSubClass) %>%
  ggplot(aes(x = MSSubClass, y = SalePrice, color = MSSubClass)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_boxplot() +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of MSSubClass vs. SalePrice")

# Scatterplot of MSSubClass vs. SalePrice
MSSubClass_scatterplot <- dataset[train$Id, ] %>%
  group_by(MSSubClass) %>%
  ggplot(aes(x = MSSubClass, y = SalePrice, color = MSSubClass)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of MSSubClass vs. SalePrice")

grid.arrange(MSSubClass_boxplot, MSSubClass_scatterplot, nrow = 1)

# From the boxplot we can notice that some of the most expensive houses are from the "20", "50", and "60" category of MSSubClass.
# "20" are "1-STORY 1946 & NEWER ALL STYLES", "50" are "1-1/2 STORY FINISHED ALL AGES", and "60" are "2-STORY 1946 & NEWER".
# From the scatterplot we can see that only a few houses are in MSSubClass "40" and "180".



#####
# MSZoning: Identifies the general zoning classification of the sale.
#####

# There are some missing values in MSZoning.
summary(dataset$MSZoning)

# Boxplot of MSZoning vs. SalePrice.
MSZoning_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MSZoning, y = SalePrice, color = MSZoning)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_boxplot() +
  ggtitle("Boxplot of MSZoning vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

# Scatterplot of MSZoning vs. SalePrice.
MSZoning_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MSZoning, y = SalePrice, color = MSZoning)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Boxplot of MSZoning vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

grid.arrange(MSZoning_boxplot, MSZoning_scatterplot, nrow = 1)

# From the plots we can see that the most expensive houses belong to the "RL" category,
# while "C (all)" contains less valuable houses.
# Residential low ("RL") and medium ("RM") density can be predictive of a higher sale price,
# although floating village residential ("FL") has an even higher median sale price.

# Dealing with missing values in the MSZoning column.
# From the plot below we can see that "RL", or residential low-density is clearly the most common value.
dataset %>%
  ggplot(aes(x = MSZoning, fill = MSZoning)) +
  geom_histogram(stat = "count") +
  ggtitle("MSZoning classifications") +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("Number of houses")

# Does MSZoning depend on MSSubclass? Below we can see that for MSSubClass of 20,
# residential low-density is clearly the most common value, while it is less clear for MSSubClass of 30 or 70.
dataset %>% select(MSSubClass, MSZoning) %>%
  group_by(MSSubClass, MSZoning) %>%
  filter(MSSubClass %in% c(20, 30, 70)) %>%
  count()

# We use kNN-based missing value imputation
knn_model <- kNN(dataset, variable = "MSZoning", k = 9)

# Predicted MSZoning values
knn_model[knn_model$MSZoning_imp == TRUE, ]$MSZoning

# We impute the values
dataset$MSZoning[which(is.na(dataset$MSZoning))] <- knn_model[knn_model$MSZoning_imp == TRUE, ]$MSZoning




#####
# LotFrontage: Linear feet of street connected to property
#####

# There are a lot of missing values in LotFrontage
summary(dataset$LotFrontage)



# Scatterplot of LotFrontage vs. SalePrice (Regular and log-transformed).
# From the plots we can observe that LotFrontage doesn't seem to influence sale price a lot.
# Also, there are two houses with very large LotFrontage values, but comparatively low sale prices.
LotFrontage_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotFrontage, y = SalePrice)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Scatterplot of LotFrontage vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x))

LotFrontage_scatterplot_log <- dataset[train$Id, ] %>%
  ggplot(aes(x = log1p(LotFrontage), y = SalePrice)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Scatterplot of log-transformed LotFrontage vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  xlab("Log-transformed LotFrontage")

grid.arrange(LotFrontage_scatterplot, LotFrontage_scatterplot_log, nrow = 2)

# LotFrontage is quite predictive of sale price, as it is a measure of property size.
cor(na.omit(dataset$LotFrontage[train$Id]), dataset[train$Id, ]$SalePrice[-which(is.na(dataset$LotFrontage))])

# Does LotFrontage correlate well with LotArea? We plot the log-transformed LotFrontage vs. LotArea.
# Indeed, we find that LotFrontage correlates well with LotArea, although there are some houses with
# a larger deviation.
dataset %>%
  ggplot(aes(x = log1p(LotFrontage), y = log1p(LotArea))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  ggtitle("Scatterplot of log-transformed LotFrontage vs. LotArea") +
  xlab("Log-transformed LotFrontage") +
  ylab("Log-transformed LotArea") +
  theme_bw()

# We calculate the correlation between LotFrontage and LotArea. The correlation is quite strong, at almost 50%.
# This correlation is probably not stronger due to quite a few houses having noticably larger LotAreas
# while having lower LotFrontage values and vice-versa.
cor(na.omit(dataset$LotFrontage), dataset$LotArea[-which(is.na(dataset$LotFrontage))])

# It would be possible to impute the missing LotFrontage values by predicting them from LotArea
# and other potentially related features, such as LotShape, Neighborhood, LandShape etc...
# before we can d this, however, we need to wrangle the necessary features. LotFrontage value imputation will be conducted at a later point.

# We use kNN-based imputation for LotFrontage.
knn_model <- kNN(dataset, variable = "LotFrontage", k = 9)

# Predicted LotFrontage values
knn_model[knn_model$LotFrontage_imp == TRUE, ]$LotFrontage

# We impute the values
dataset$LotFrontage[which(is.na(dataset$LotFrontage))] <- knn_model[knn_model$LotFrontage_imp == TRUE, ]$LotFrontage

# Scatterplot after imputation

dataset %>%
  ggplot(aes(x = log1p(LotFrontage), y = log1p(LotArea))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  ggtitle("Scatterplot of log-transformed LotFrontage vs. LotArea after imputation") +
  xlab("Log-transformed LotFrontage") +
  ylab("Log-transformed LotArea") +
  theme_bw()

# We change variable encoding to numeric
dataset$LotFrontage <- as.numeric(dataset$LotFrontage)



#####
# LotArea: Lot size in square feet
#####


# There are no missing values in LotArea.
summary(dataset$LotArea)

# LotArea is encoded as an integer value, but it should be numeric.
dataset$LotArea <- as.numeric(dataset$LotArea)

# Scatterplot of LotArea vs. SalePrice (Regular and log-transformed).
# As there seem to be some outlying houses, the log-transformation of LotArea helps us visualize
# the relationship of LotArea with SalePrice better,
LotArea_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotArea, y = SalePrice)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("LotArea vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x))

LotArea_scatterplot_log <- dataset[train$Id, ] %>%
  ggplot(aes(x = log1p(LotArea), y = SalePrice)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Log-transformed LotArea vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  xlab("Log-transformed LotArea")

grid.arrange(LotArea_scatterplot, LotArea_scatterplot_log, nrow = 2)

# We calculate the correlation of LotArea with sale price.
# LotArea is quite predictive of sale price.
cor(dataset[train$Id, ]$LotArea, dataset[train$Id, ]$SalePrice)



#####
# Street: Type of road access to property
#####

# There are no missing values in Street.
summary(dataset$Street)

# Boxplot of Street vs. SalePrice.
Street_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Street, y = SalePrice, color = Street)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_boxplot() +
  ggtitle("Boxplot of Street vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

# Scatterplot of Street vs. SalePrice.
Street_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Street, y = SalePrice, color = Street)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Boxplot of Street vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

grid.arrange(Street_boxplot, Street_scatterplot, nrow = 1)

# From the plots we can see that the type of road access to the property matters in terms of sale price.
# However, only a few houses have gravel values in Street.



#####
# Alley: Type of alley access to property
#####

# There is a very large amount of missing values in Alley.
# From the data description, "No alley access" was encoded as NA.
# These NA entries are producing false missing value entries.
summary(dataset$Alley)

# We fix these wrong NA entries by replacing them with "None"
dataset$Alley <- str_replace_na(dataset$Alley, replacement = "None")
dataset$Alley <- factor(dataset$Alley, levels = c("None", "Grvl", "Pave"))

# Boxplot of Alley vs. SalePrice.
Alley_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Alley, y = SalePrice, color = Alley)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_boxplot() +
  ggtitle("Boxplot of Alley vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

# Scatterplot of Alley vs. SalePrice.
Alley_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Alley, y = SalePrice, color = Alley)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Boxplot of Alley vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none")

grid.arrange(Alley_boxplot, Alley_scatterplot, nrow = 1)



#####
# LotShape: General shape of property
#####

# There are no missing values in LotShape.
summary(dataset$LotShape)

# Boxplot of LotShape vs. SalePrice
LotShape_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotShape, y = SalePrice, color = LotShape)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of LotShape vs. SalePrice")

# Scatterplot of LotShape vs. SalePrice
LotShape_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotShape, y = SalePrice, color = LotShape)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of LotShape vs. SalePrice")

grid.arrange(LotShape_boxplot, LotShape_scatterplot, nrow = 1)

# From the plots we can see that some of the more expensive houses have a slightly irregular "IR1" LotShape.
# A regular "Reg" LotShape is indicative of a lower sale price, but the category still contains many houses with larger sale price as well.
# Only a small number of houses has a rally irregular "IR3" LotShape.
# LotShape doesn't seem to influence sale price too much.



#####
# LandContour: Flatness of the property
#####

# LandContour contains no missing values.
summary(dataset$LandContour)

# Boxplot of LandContour vs. SalePrice
LandContour_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LandContour, y = SalePrice, color = LandContour)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of LandContour vs. SalePrice")

# Scatterplot of LandContour vs. SalePrice
LandContour_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LandContour, y = SalePrice, color = LandContour)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of LandContour vs. SalePrice")

grid.arrange(LandContour_boxplot, LandContour_scatterplot, nrow = 1)

# From the plots we can see that a Banked - Quick and significant rise from street grade to building entry in LandContour
# can be indicative of a lower sale price of the house. The most expensive houses are on level, nearly flat terrain.
# This feature will help to distinguish some of the lower priced from the higher priced houses.



#####
# Utilities: Type of utilities available
#####

# There are a few missing values in Utilities.
summary(dataset$Utilities)

# Boxplot of Utilities vs. SalePrice
Utilities_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Utilities, y = SalePrice, color = Utilities)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Utilities vs. SalePrice")

# Scatterplot of Utilities vs. SalePrice
Utilities_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Utilities, y = SalePrice, color = Utilities)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of Utilities vs. SalePrice")

grid.arrange(Utilities_boxplot, Utilities_scatterplot, nrow = 1)

# From the plots we can see that there is only a single house with "NoSeWa" in Utilities, while all other houses have
# access to all public utilities.
# This feature is not helpful in predicting a houses sale price and can be removed from the dataset.
dataset <- subset(dataset, select = -Utilities)



#####
# LotConfig: Lot configuration
#####

# There are no missing values in LotConfig.
summary(dataset$LotConfig)


# Boxplot of LotConfig vs. SalePrice
LotConfig_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotConfig, y = SalePrice, color = LotConfig)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of LotConfig vs. SalePrice")

# Scatterplot of LotConfig vs. SalePrice
LotConfig_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LotConfig, y = SalePrice, color = LotConfig)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of LotConfig vs. SalePrice")

grid.arrange(LotConfig_boxplot, LotConfig_scatterplot, nrow = 1)

# From the plots we can see that LotConfig doesn't appear to influence sale price very much.



#####
# LandSlope: Slope of property
#####

# There are no missing values in LandSlope.
summary(dataset$LandSlope)

# Boxplot of LandSlope vs. SalePrice
LandSlope_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LandSlope, y = SalePrice, color = LandSlope)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("LandSlope vs. SalePrice")

# Scatterplot of LandSlope vs. SalePrice
LandSlope_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LandSlope, y = SalePrice, color = LandSlope)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("LandSlope vs. SalePrice")

grid.arrange(LandSlope_boxplot, LandSlope_scatterplot, nrow = 1)

# From the plots we can see that a gentle land slope can be predictive of a higher house sale price.



#####
# Neighborhood: Physical locations within Ames city limits
#####

# There are no missing values in Neighborhood.
summary(dataset$Neighborhood)

# Boxplot of Neighborhood vs. SalePrice
dataset[train$Id, ] %>%
  ggplot(aes(x = Neighborhood, y = SalePrice, color = Neighborhood)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Neighborhood vs. SalePrice")

# From the plot we can see that Northridge and Northridge High are where some of the most expensive houses are situated.
# The respective Neighborhood is a strong indicator for a houses sale price.



#####
# Condition1: Proximity to various conditions
#####

# There are no missing values in Condition1
summary(dataset$Condition1)

# Boxplot of Condition1 vs. SalePrice
Condition1_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition1, y = SalePrice, color = Condition1)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Condition1 vs. SalePrice")

# Scatterplot of Condition1 vs. SalePrice
Condition1_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition1, y = SalePrice, color = Condition1)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of Condition1 vs. SalePrice")

grid.arrange(Condition1_boxplot, Condition1_scatterplot, nrow = 1)

# From the plots we can see that adjacency to an artery or feeder street, as well as to a railroad may indicate a lower sale price.
# Adjacency to a positive off-site feature can be indicative of an increased value.
# Some categories are rather sparsely populated.

# We will combine the 2 different road-associated categories ("Artery", "Feedr") and the 4 different rail-associated categories ("RRAe, "RRAn", "RRNe", "RRNn") into
# 2 distinct categories. We will also combine the 2 psoitive off-site features.
# Before the conversion, we convert the factor into a character vector.
dataset$Condition1 <- as.character(dataset$Condition1)

dataset$Condition1[dataset$Condition1 %in% c("Artery", "Feedr")] <- "NearRoad"
dataset$Condition1[dataset$Condition1 %in% c("RRAe", "RRAn", "RRNe", "RRNn")] <- "NearRailroad"
dataset$Condition1[dataset$Condition1 %in% c("PosA", "PosN")] <- "NearPosOffSite"

# We convert back into a factor with appropriate levels.
dataset$Condition1 <- factor(dataset$Condition1, levels = c("Norm", "NearRoad", "NearRailroad", "NearPosOffSite"))

# Boxplot of changed Condition1 vs. SalePrice
Condition1_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition1, y = SalePrice, color = Condition1)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of changed Condition1 vs. SalePrice")

# Scatterplot of changed Condition1 vs. SalePrice
Condition1_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition1, y = SalePrice, color = Condition1)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of changed Condition1 vs. SalePrice")

grid.arrange(Condition1_boxplot, Condition1_scatterplot, nrow = 1)

# The plots of the changed Condition1 feature show that adjacency to a road or railroad
# tends to decrease sale price, while adjacency to a positive off-site leads leads to an increase.
# Almost all expensive houses are in the "normal" category.



#####
# Condition2: Proximity to various conditions (if more than one is present)
#####

# Some houses are adjacent to more than one condition, else they received a "normal" entry in Condition2.

# There are no missing values in Condition2. However, there is no "RRNe" entry.
summary(dataset$Condition2)

# We will apply the same changes to Condition2 as we did to Condition1.
dataset$Condition2 <- as.character(dataset$Condition2)

dataset$Condition2[dataset$Condition2 %in% c("Artery", "Feedr")] <- "NearRoad"
dataset$Condition2[dataset$Condition2 %in% c("RRAe", "RRAn", "RRNn")] <- "NearRailroad"
dataset$Condition2[dataset$Condition2 %in% c("PosA", "PosN")] <- "NearPosOffSite"

# We convert back into a factor with appropriate levels.
dataset$Condition2 <- factor(dataset$Condition2, levels = c("Norm", "NearRoad", "NearRailroad", "NearPosOffSite"))

# Boxplot of changed Condition2 vs. SalePrice.
# Clearly, being near a positive off-site raises the sale price, while adjacency to a busy road or railroad reduces it.
dataset[train$Id, ] %>%
  ggplot(aes(x = Condition2, y = SalePrice, color = Condition2)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of changed Condition2 vs. SalePrice")

# Are there houses which are adjacent to a road and a railroad simultaneously?
dataset[train$Id, ] %>%
  filter(Condition1 %in% c("NearRoad", "NearRailroad"), Condition2 %in% c("NearRoad", "NearRailroad")) %>%
  summarize(avg.value = mean(SalePrice))

# Indeed, the average SalePrice of the houses with a "NearRoad" and a "NearRailRoad" is rather low.


### Merging of Condition1 and Condition2 ###

# As Condition2 has a rather low amount of entries, we will merge Condition 1 and 2.
# Prior to this, we will merge "NearRoad" and "NearRailRoad" into a single category, "NearNegOffSite".
dataset$Condition1 <- as.character(dataset$Condition1)
dataset$Condition1[which(dataset$Condition1 %in% c("NearRoad", "NearRailroad"))] <- "NearNegOffSite"

# Next, we will determine which houses are nearby a second negative or positive off-site and create respective categories.
dataset$Condition2 <- as.character(dataset$Condition2)

# As there is only a single house which is near a positive as well as a negative off-site, we will assign it to the "normal" group, assuming that the effects cancel eachother out.
# As there are only two houses near two positive off-sites, we assign them to "NearPosOffSite".

dataset$Condition1[which(dataset$Condition2 %in% c("NearRoad", "NearRailRoad"))] <- "NearTwoNegOffSites"
index <- which(dataset$Condition2 %in% c("NearPosOffSite")) 
dataset$Condition1[index[which(dataset$Condition1[index] == "NearPosOffSite")]] <- "NearPosOffSite"
dataset$Condition1[index[which(dataset$Condition1[index] == "NearNegOffSite")]] <- "Normal"
dataset$Condition1[dataset$Condition1 == "Norm"] <- "Normal"

dataset$Condition1 <- factor(dataset$Condition1, levels = c("Normal", "NearNegOffSite", "NearTwoNegOffSites", "NearPosOffSite"))

# We rename the engineered feature and remove the previous ones.
dataset$Condition <- dataset$Condition1
dataset <- subset(dataset, select = -c(Condition1, Condition2))


# Boxplot of the merged Condition1 vs. SalePrice.
# Clearly, being near a positive off-site raises the sale price, while adjacency to a busy road or railroad reduces it.
Condition_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition, y = SalePrice, color = Condition)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered Condition vs. SalePrice")

Condition_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Condition, y = SalePrice, color = Condition)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered Condition vs. SalePrice")

grid.arrange(Condition_boxplot, Condition_scatterplot, nrow = 2)

# From the plots of the engineered Condition feature, we can see that being near one or even two negative off-sites (road, railroad) might indicate a lower sale price.
# Adjacency to a positive off-site is associated with a higher sale price, however all expensive houses are in the "Normal" category.



#####
# BldgType: Type of dwelling
#####

# There are no missing values in BldgType
summary(dataset$BldgType)

# Boxplot of BldgType vs. SalePrice.
BldgType_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BldgType, y = SalePrice, color = BldgType)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BldgType vs. SalePrice")

BldgType_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BldgType, y = SalePrice, color = BldgType)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BldgType vs. SalePrice")

grid.arrange(BldgType_boxplot, BldgType_scatterplot, nrow = 1)


# From the plots we can see that the most expensive houses are all detached single-family houses ("1Fam").
# Even then, there must be other distinguishing features behind those, as the median sale price of 1Fam houses isn't any higher compared to the other categories.



#####
# HouseStyle: Style of dwelling
#####

# There are no missing values in HouseStyle
summary(dataset$HouseStyle)

# Boxplot of HouseStyle vs. SalePrice.
HouseStyle_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = HouseStyle, y = SalePrice, color = HouseStyle)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of HouseStyle vs. SalePrice")

HouseStyle_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = HouseStyle, y = SalePrice, color = HouseStyle)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of HouseStyle vs. SalePrice")

grid.arrange(HouseStyle_boxplot, HouseStyle_scatterplot, nrow = 1)

# HouseStyle doesn't reveal very much about sale price, but unfinished 1.5 and 2.5 storey houses tend to have lower sale prices compared to completed ones.
# Unsurprisingly, the most expensive houses are in the 2Story category, while another large set of expensve houses are in the 1Story category.



#####
# OverallQual: Rates the overall material and finish of the house
#####

# There are no missing values in OverallQual
summary(dataset$OverallQual)

# OverallQual should be a numeric variable, not an integer.
dataset$OverallQual <- as.numeric(dataset$OverallQual)


# Boxplot of OverallQual vs. SalePrice.
p1 <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = factor(OverallQual))) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1, 10, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("OverallQual vs. SalePrice")

p2 <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = factor(OverallQual))) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1, 10, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("OverallQual vs. SalePrice")

grid.arrange(p1, p2, nrow = 1)

rm(p1, p2)

# OverallQual has tremendous influence on a houses sale price. In the highest category, 10, there are 2 houses with
# unexpectedly low, and extremely high sale prices. This increases the range a lot.
# OverallQual will be very important for predicting a houses value.

# There are only a few houses in the lowest OverallQual categories 1 and 2. Additionally, the higher quality levels 
# show quite a large range of values.

dataset$OverallQual <- as.numeric(dataset$OverallQual)



#####
# OverallCond: Rates the overall condition of the house
#####

# There are no missing values in OverallCond
summary(dataset$OverallCond)

# OverallCond should be numeric.
dataset$OverallCond <- as.numeric(dataset$OverallCond)

# Boxplot of OverallCond vs. SalePrice.
p1 <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = factor(OverallCond))) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1, 9, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallCond vs. SalePrice")

p2 <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = factor(OverallCond))) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1, 9, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallCond vs. SalePrice")

grid.arrange(p1, p2, nrow = 1)

rm(p1, p2)

# Contrary to intuition, OverallCond is much less predictive of a houses sale price and there is not a single house with a "very excellent" condition.
# We observe that the most expensive houses are between 4 - 9, with 5 having most of them. OverallCond below 5 holds houses with lower sale price.
# The lower two levels hold only very few houses.



#####
# YearBuilt: Original construction date
#####

# There are no missing values in YearBuilt
summary(dataset$YearBuilt)


# Scatterplot of YearBuilt. 
dataset[train$Id, ] %>%
  ggplot(aes(x = YearBuilt, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1870, 2010, 10)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of YearBuilt vs. SalePrice") +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  geom_smooth(method = "lm", color = "red") +
  annotate("text", x = 1900, y = 700000, label = "Correlation between YearBuilt and SalePrice: 0.52") +
  annotate("text", x = 1880, y = 225000, label = "Generalized additive model") +
  annotate("text", x = 1890, y = 40000, label = "Linear model")

# There is a large correlation between YearBuilt and SalePrice.
cor(dataset[train$Id, ]$YearBuilt, dataset[train$Id, ]$SalePrice)

# From the plot we can see that SalePrice is influenced by YearBuilt, especially houses built after the 1950's tend to go up in value.



#####
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#####

# There are no missing values in YearRemodAdd
summary(dataset$YearRemodAdd)

# Scatterplot of YearRemodAdd. 
dataset[train$Id, ] %>%
  ggplot(aes(x = YearRemodAdd, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(1950, 2010, 10)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of YearRemodAdd vs. SalePrice") +
  geom_smooth(method = "lm", color = "red") +
  annotate("text", x = 1970, y = 700000, label = "Correlation between YearRemodAdd and SalePrice: 0.50")

# There is a large correlation between YearRemodAdd and SalePrice.
cor(dataset[train$Id, ]$YearRemodAdd, dataset[train$Id, ]$SalePrice)

# There is also a large correlation between YearRemodAdd and YearBuilt.
cor(dataset$YearRemodAdd, dataset$YearBuilt)

# From the plot we can see that YearRemodAdd has a very similar correlation with SalePrice compared with YearBuilt.
# Remodellings were only recorded starting 1950.
# YearRemodAdd correlates > 61% with YearBuilt. We plot the two variables against eachother.

# Scatterplot of YearBuilt vs. YearRemodAdd. 
dataset[train$Id, ] %>%
  ggplot(aes(x = YearRemodAdd, y = YearBuilt)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(breaks = seq(1870, 2010, 10)) +
  scale_x_continuous(breaks = seq(1870, 2010, 10)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of YearRemodAdd vs. YearBuilt") +
  geom_vline(xintercept = 1950, color = "red", linetype = 2)


# From the plot it seems that for all houses built before 1950, even if there actually wasn't any remodelling done, they received an entry in YearRemodAdd at 1950.
# Most houses have identical YearBuilt and YearRemodAdd values, since there were no remodellings done.
# For predictive purposes, this variable isn't very helpful.
dataset <- subset(dataset, select = -YearRemodAdd)



#####
# RoofStyle: Type of roof
#####


# There are no missing values in RoofStyle
summary(dataset$RoofStyle)

# Boxplot of RoofStyle vs. SalePrice.
RoofStyle_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = RoofStyle, y = SalePrice, color = RoofStyle)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of RoofStyle vs. SalePrice")

RoofStyle_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = RoofStyle, y = SalePrice, color = RoofStyle)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of RoofStyle vs. SalePrice")

grid.arrange(RoofStyle_boxplot, RoofStyle_scatterplot, nrow = 1)

# The style of roof is not very predictive of sale price, but most expensive houses are either in the "Gable" or "Hip" category.


#####
# RoofMatl: Roof material
##### 

# There are no missing values in RoofMatl
summary(dataset$RoofMatl)

# Boxplot of RoofMatl vs. SalePrice.
RoofMatl_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = RoofMatl, y = SalePrice, color = RoofMatl)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of RoofMatl vs. SalePrice")

RoofMatl_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = RoofMatl, y = SalePrice, color = RoofMatl)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of RoofMatl vs. SalePrice")

grid.arrange(RoofMatl_boxplot, RoofMatl_scatterplot, nrow = 1)

# From the plots we can see that there are very few houses with roof materials differeing from the standard composite shingle.
# Most categories are populated by very few houses and are widely spread.



#####
# Exterior1st: Exterior covering on house
#####

# There is one missing value in Exterior1st
summary(dataset$Exterior1st)

# Boxplot of Exterior1st vs. SalePrice.
Exterior1st_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Exterior1st, y = SalePrice, color = Exterior1st)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Exterior1st vs. SalePrice")

Exterior1st_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Exterior1st, y = SalePrice, color = Exterior1st)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Exterior1st vs. SalePrice")

grid.arrange(Exterior1st_boxplot, Exterior1st_scatterplot, nrow = 1)

# From the plots it becomes apparent that asbestos shingles and asphalt shingles are mostly used with cheaper houses.

# Before we deal with the missing value, we take a look at Exterior2nd as well.



#####
# Exterior2nd: Exterior covering on house (if more than one material)
#####


# There is one missing value in Exterior2nd
summary(dataset$Exterior2nd)

# Boxplot of Exterior2nd vs. SalePrice.
Exterior2nd_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Exterior2nd, y = SalePrice, color = Exterior2nd)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Exterior2nd vs. SalePrice")

Exterior2nd_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Exterior2nd, y = SalePrice, color = Exterior2nd)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Exterior2nd vs. SalePrice")

grid.arrange(Exterior2nd_boxplot, Exterior2nd_scatterplot, nrow = 1)

# From the plots it becomes apparent that asbestos shingles and asphalt shingles are mostly used with cheaper houses.

# There is a missing value in Exterior1st and Exterior2nd, is it the same house?
dataset[which(is.na(dataset$Exterior1st)), ]
dataset[which(is.na(dataset$Exterior2nd)), ]

# Indeed, the info is missing for the same house. Which exterior material is most common?
dataset %>% select(Exterior1st) %>% group_by(Exterior1st) %>% tally() # VinylSd is the mode
dataset %>% select(Exterior2nd) %>% group_by(Exterior2nd) %>% tally() # VinylSd is the mode

# We use kNN-based imputation for Exterior1st and 2nd.
knn_model <- kNN(dataset, variable = c("Exterior1st", "Exterior2nd"), k = 5)
dataset$Exterior1st[2152] <- knn_model[knn_model$Exterior1st_imp == TRUE, ]$Exterior1st
dataset$Exterior2nd[2152] <- knn_model[knn_model$Exterior2nd_imp == TRUE, ]$Exterior2nd



#####
# MasVnrType: Masonry veneer type
#####

# There are some missing values in MasVnrType
summary(dataset$MasVnrType)


# Boxplot of MasVnrType vs. SalePrice.
MasVnrType_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrType, y = SalePrice, color = MasVnrType)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of MasVnrType vs. SalePrice")

MasVnrType_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrType, y = SalePrice, color = MasVnrType)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of MasVnrType vs. SalePrice")

grid.arrange(MasVnrType_boxplot, MasVnrType_scatterplot, nrow = 1)


# We will use kNN-based predictions to impute the missing values.
knn_model <- kNN(dataset, variable = "MasVnrType", k = 9)

# Missing value imputation of MasVnrType
dataset[which(is.na(dataset$MasVnrType)), ]$MasVnrType <- knn_model[knn_model$MasVnrType_imp == TRUE, ]$MasVnrType


# Boxplot of MasVnrType vs. SalePrice after missing value imputation.
MasVnrType_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrType, y = SalePrice, color = MasVnrType)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of MasVnrType vs. SalePrice after missing value imputation")

MasVnrType_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrType, y = SalePrice, color = MasVnrType)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of MasVnrType vs. SalePrice after missing value imputation")

grid.arrange(MasVnrType_boxplot, MasVnrType_scatterplot, nrow = 1)



#####
# MasVnrArea: Masonry veneer area in square feet
#####

# There are some missing values in MasVnrArea, presumably the same houses that had missing values in MasVnrType.
summary(dataset$MasVnrArea)

# MasVnrArea should be a numeric variable.
dataset$MasVnrArea <- as.numeric(dataset$MasVnrArea)

# Scatterplot of MasVnrArea vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrArea, y = SalePrice, color = MasVnrArea)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of MasVnrArea vs. SalePrice") +
  geom_smooth(method = "lm")

# From the plot we can see that many houses actually have 0 MasVnrArea.

# We will use kNN-based predictions to impute the missing values.
knn_model <- kNN(dataset, variable = "MasVnrArea", k = 9)

# Missing value imputation of MasVnrArea
dataset[which(is.na(dataset$MasVnrArea)), ]$MasVnrArea <- knn_model[knn_model$MasVnrArea_imp == TRUE, ]$MasVnrArea

# Scatterplot of MasVnrArea vs. SalePrice after missing value imputation.
dataset[train$Id, ] %>%
  ggplot(aes(x = MasVnrArea, y = SalePrice, color = MasVnrArea)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of MasVnrArea vs. SalePrice after missing value imputation") +
  geom_smooth(method = "lm")


# Are there houses with non-zero MasVnrArea but "None" as MasVnrType?
dataset[which(dataset$MasVnrArea != 0 & dataset$MasVnrType == "None"), ]

# We take the index.
ind1 <- dataset[which(dataset$MasVnrArea != 0 & dataset$MasVnrType == "None"), ]$Id


# Are there houses with zero MasVnrArea and anything but "None" as MasVnrType?
dataset[which(dataset$MasVnrArea == 0 & dataset$MasVnrType != "None"), ]

# We take the index.
ind2 <- dataset[which(dataset$MasVnrArea == 0 & dataset$MasVnrType != "None"), ]$Id


# We will set the area to zero and the type to "None" in these cases.
dataset$MasVnrArea[ind1] <- 0
dataset$MasVnrType[ind2] <- "None"



#####
# ExterQual: Evaluates the quality of the material on the exterior 
#####

# There are no missing values in ExterQual.
summary(dataset$ExterQual)

# Boxplot of ExterQual vs. SalePrice.
ExterQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = ExterQual, y = SalePrice, color = ExterQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of ExterQual vs. SalePrice")

ExterQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = ExterQual, y = SalePrice, color = ExterQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of ExterQual vs. SalePrice")

grid.arrange(ExterQual_boxplot, ExterQual_scatterplot, nrow = 1)

# From the plots, we can see that ExterQual is a weaker predictor of sale price.

# We transform the qualitative ordinal factor into a numeric containing the present levels.
dataset$ExterQual <- as.numeric(factor(dataset$ExterQual, levels=c("Fa", "TA", "Gd", "Ex")))

# Scatterplot of engineered ExterQual vs. SalePrice
dataset[train$Id, ] %>%
  ggplot(aes(x = ExterQual, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of engineered ExterQual vs. SalePrice") +
  geom_smooth(method = "lm")



#####
# ExterCond: Evaluates the quality of the material on the exterior 
#####

# There are no missing values in ExterCond.
summary(dataset$ExterCond)

# Boxplot of ExterCond vs. SalePrice.
ExterCond_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = ExterCond, y = SalePrice, color = ExterCond)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of ExterCond vs. SalePrice")

ExterCond_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = ExterCond, y = SalePrice, color = ExterCond)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of ExterCond vs. SalePrice")

grid.arrange(ExterCond_boxplot, ExterCond_scatterplot, nrow = 1)

# We plot "ExterCond" against "SalePrice" and observe that only a few houses have a "poor" or "excellent" 
# "ExterCond." "ExterCond" is a a weaker predictor of "SalePrice" compared to "ExterQual", similar
# to "OverallQual" vs. "OverallCond".



# We transform the qualitative ordinal factor into a numeric containing the present levels.
dataset$ExterCond <- as.numeric(factor(dataset$ExterCond, levels=c("Po", "Fa", "TA", "Gd", "Ex")))

# Scatterplot of engineered ExterCond vs. SalePrice
dataset[train$Id, ] %>%
  ggplot(aes(x = ExterCond, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of engineered ExterCond vs. SalePrice") +
  geom_smooth(method = "lm")



#####
# Foundation: Type of foundation
#####

# There are no missing values in Foundation.
summary(dataset$Foundation)

# Boxplot of Foundation vs. SalePrice.
Foundation_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Foundation, y = SalePrice, color = Foundation)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Foundation vs. SalePrice")

Foundation_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Foundation, y = SalePrice, color = Foundation)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Foundation vs. SalePrice")

grid.arrange(Foundation_boxplot, Foundation_scatterplot, nrow = 1)

# From the plots we can see that stone and wood Foundations are relatively rare. Poured concrete ("PConc") seems to
# be the material of choice for houses with higher sale prices, while slab seems to be associated with lower values.



#####
# BsmtQual: Evaluates the height of the basement
#####

# There are missing values in BsmtQual.
summary(dataset$BsmtQual)

# From the data description we know that "No Basement" was encoded as "NA" in this variable.
# Also, there is no house with a "poor" BsmtQual.
# We will therefore replace "NA" with "None" and fix the factor levels accordingly.

dataset$BsmtQual <- as.character(dataset$BsmtQual)
dataset$BsmtQual <- str_replace_na(dataset$BsmtQual, replacement = "None")
dataset$BsmtQual <- factor(dataset$BsmtQual, levels = c("None", "Fa", "TA", "Gd", "Ex"))


# Boxplot of BsmtQual vs. SalePrice.
BsmtQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtQual, y = SalePrice, color = BsmtQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtQual vs. SalePrice")

BsmtQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtQual, y = SalePrice, color = BsmtQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtQual vs. SalePrice")

grid.arrange(BsmtQual_boxplot, BsmtQual_scatterplot, nrow = 1)


# From the plots we can see that houses with no basement or a basement with only "fair" quality come with
# much lower sale prices. This might also be because a higher BsmtQual indicates a greater height of the basement, which 
# might be indicative of a larger house in general.

# We plot BsmtQual vs. GrLivArea, as an indication of house size. There seems to be a slight relationship.
dataset %>%
  ggplot(aes(x = BsmtQual, y = GrLivArea, color = BsmtQual)) +
  geom_boxplot() +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtQual vs. GrLivArea")



#####
# BsmtCond: Evaluates the general condition of the basement
#####

# There are missing values in BsmtCond.
# From the data description we know that "No Basement" was encoded as "NA" in this variable.
# Also, not a single house has an "excellent" basement condition.
summary(dataset$BsmtCond)


# We will therefore replace "NA" with "None" and fix the factor levels accordingly.

dataset$BsmtCond <- as.character(dataset$BsmtCond)
dataset$BsmtCond <- str_replace_na(dataset$BsmtCond, replacement = "None")
dataset$BsmtCond <- factor(dataset$BsmtCond, levels = c("None", "Po", "Fa", "TA", "Gd"))


# Boxplot of BsmtCond vs. SalePrice.
BsmtCond_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtCond, y = SalePrice, color = BsmtCond)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtCond vs. SalePrice")

BsmtCond_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtCond, y = SalePrice, color = BsmtCond)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtCond vs. SalePrice")

grid.arrange(BsmtCond_boxplot, BsmtCond_scatterplot, nrow = 1)

# Basement condition seems to be a good indicator of sale price. Although there are only a few houses
# with a really poor basement condition, all of these houses have exceptionally low sale prices.
# Overall, a typical or good basement condition is important for a houses sale price.



#####
# BsmtExposure: Refers to walkout or garden level walls
#####

# There are missing values in BsmtExposure.
# From the data description we know that "No Basement" was encoded as "NA" in this variable.
# "No exposure" is encoded as "No"
summary(dataset$BsmtExposure)

# We will therefore replace "NA" with "None" and fix the factor levels accordingly.

dataset$BsmtExposure <- as.character(dataset$BsmtExposure)
dataset$BsmtExposure <- str_replace_na(dataset$BsmtExposure, replacement = "None")
dataset$BsmtExposure <- factor(dataset$BsmtExposure, levels = c("None", "No", "Mn", "Av", "Gd"))


# Boxplot of BsmtExposure vs. SalePrice.
BsmtExposure_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtExposure, y = SalePrice, color = BsmtExposure)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtExposure vs. SalePrice")

BsmtExposure_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtExposure, y = SalePrice, color = BsmtExposure)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtExposure vs. SalePrice")

grid.arrange(BsmtExposure_boxplot, BsmtExposure_scatterplot, nrow = 1)

# From the plots we can see that greater basement exposure correlates slightly with larger sale prices.
# Houses with no basement ("None") have the lowest sale prices.



#####
# BsmtFinType1: Rating of basement finished area
#####

# There are missing values in BsmtExposure.
# From the data description we know that "No Basement" was encoded as "NA" in this variable.
summary(dataset$BsmtFinType1)

# We will therefore replace "NA" with "None" and fix the factor levels accordingly.
dataset$BsmtFinType1 <- as.character(dataset$BsmtFinType1)
dataset$BsmtFinType1 <- str_replace_na(dataset$BsmtFinType1, replacement = "None")
dataset$BsmtFinType1 <- factor(dataset$BsmtFinType1, levels = c("None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"))

# Boxplot of BsmtFinType1 vs. SalePrice.
BsmtFinType1_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtFinType1, y = SalePrice, color = BsmtFinType1)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtFinType1 vs. SalePrice")

BsmtFinType1_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtFinType1, y = SalePrice, color = BsmtFinType1)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtFinType1 vs. SalePrice")

grid.arrange(BsmtFinType1_boxplot, BsmtFinType1_scatterplot, nrow = 1)


# Houses with unfinished or good quality living quarter type of finish come with higher sale prices.



#####
# BsmtFinType2: Rating of basement finished area (if multiple types)
#####

# There are missing values in BsmtExposure.
# From the data description we know that "No Basement" was encoded as "NA" in this variable.
summary(dataset$BsmtFinType2)

# We will therefore replace "NA" with "None" and fix the factor levels accordingly.
dataset$BsmtFinType2 <- as.character(dataset$BsmtFinType2)
dataset$BsmtFinType2 <- str_replace_na(dataset$BsmtFinType2, replacement = "None")
dataset$BsmtFinType2 <- factor(dataset$BsmtFinType2, levels = c("None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"))

# Boxplot of BsmtFinType2 vs. SalePrice.
BsmtFinType2_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtFinType2, y = SalePrice, color = BsmtFinType2)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtFinType2 vs. SalePrice")

BsmtFinType2_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = BsmtFinType2, y = SalePrice, color = BsmtFinType2)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BsmtFinType2 vs. SalePrice")

grid.arrange(BsmtFinType2_boxplot, BsmtFinType2_scatterplot, nrow = 1)


# Houses with unfinished type of finish come with higher sale prices.



#####
# Basement square feet variables
#####

# We will deal with all basement-area related variables together.


# We replace the missing Bsmt-values of house 2121 with 0, as this house has no basement
dataset[2121, ] # This house has no basement
dataset$BsmtFinSF1[2121] <- 0
dataset$BsmtFinSF2[2121] <- 0
dataset$BsmtUnfSF[2121] <- 0
dataset$TotalBsmtSF[2121] <- 0

# We replace the remaining NAs in BsmtFullBath and BsmtHalfBath with 0 as the respective houses have no basement.
dataset[which(is.na(dataset$BsmtFullBath)), ]

dataset$BsmtFullBath[is.na(dataset$BsmtFullBath)] <- 0
dataset$BsmtHalfBath[is.na(dataset$BsmtHalfBath)] <- 0


# What is the correlation between TotalBsmtSF and the individual measurements of basement square feet?
cor(dataset$TotalBsmtSF, (dataset$BsmtFinSF1 + dataset$BsmtFinSF2 + dataset$BsmtUnfSF)) # It is exactly 1.

# Correlation between the variables and SalePrice: They are all mostly weak individually., while TotalBsmtSF is highly correlated.
cor(dataset[train$Id, ]$TotalBsmtSF, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtFinSF1, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtFinSF2, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$BsmtUnfSF, dataset[train$Id, ]$SalePrice) # Relatively strong correlation of unfinished basement with sale price

# The individual basement area variables seem to be redundant, but we observed that having an unfinished basement
# can be indiciative of a higher sale price earlier (e.g. relatively large correlation between BsmtUnfSF and SalePrice). We therefore leave these variables alone.



#####
# Bathroom variables
#####

# The number of total bathrooms can be indicative of a houses size.
# In the dataset there are 4 different variables describing bathrooms.
# It could be useful to combine these into a single feature.

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

# We will create a variable "TotalBaths", that sums up the various types of baths into one variable, taking into consideration the different correlations to SalePrice for full and half baths.
# We don't add BsmtHalfBaths, as their correlation with SalePrice is very low/negative and only very few houses have any.
dataset$TotalBaths <- dataset$FullBath + (dataset$BsmtFullBath * FullToBsmtFullBathRatio) + (dataset$HalfBath * FullToHalfBathRatio)
cor(dataset[train$Id, ]$TotalBaths, dataset[train$Id, ]$SalePrice) # Correlation of TotalBaths is higher than of just FullBaths

# The new TotalBaths variable has a correlation of almost 70% with SalePrice.

# We can remove the previous bath variables
dataset <- subset(dataset, select = -c(FullBath, HalfBath, BsmtFullBath, BsmtHalfBath))


# Scatterplot of engineered TotalBaths vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = TotalBaths, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(0, 5, 0.25)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of engineered TotalBaths vs. SalePrice") +
  geom_smooth(method = "lm")


# The combination of 4 different bath variables, "TotalBaths" has a strong relationship to sale price, as the total number of bathrooms is in itself indicative of a larger house.



#####
# Heating: Type of heating
#####

# There are no missing values in Heating, but there is only a single house with "Floor" and 2 houses with "OthW" heating.
summary(dataset$Heating)

# Boxplot of Heating vs. SalePrice.
Heating_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Heating, y = SalePrice, color = Heating)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Heating vs. SalePrice")

Heating_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Heating, y = SalePrice, color = Heating)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of Heating vs. SalePrice")

grid.arrange(Heating_boxplot, Heating_scatterplot, nrow = 1)


# From the plots we can see that the house with "floor furnace" heating has a low sale price and that basically any heating other than "GasA" and "GasW"
# appears to be associated with lower sale price. Most categories are woefully underrepresented, however.



#####
# HeatingQC: Heating quality and condition
#####

# There are no missing values in HeatingQC.
summary(dataset$HeatingQC)

# Boxplot of HeatingQC vs. SalePrice.
HeatingQC_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = HeatingQC, y = SalePrice, color = HeatingQC)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of HeatingQC vs. SalePrice")

HeatingQC_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = HeatingQC, y = SalePrice, color = HeatingQC)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of HeatingQC vs. SalePrice")

grid.arrange(HeatingQC_boxplot, HeatingQC_scatterplot, nrow = 1)

# From the plots we an see that very few houses have poor quality heating and that an excellent quality of heating 
# is associated with a higher sale price. The quality of the heating seems to be more important than its type.



#####
# CentralAir: Central air conditioning
#####

# There are no missing values in CentralAir.
summary(dataset$CentralAir)

# Boxplot of CentralAir vs. SalePrice.
CentralAir_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = CentralAir, y = SalePrice, color = CentralAir)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of CentralAir vs. SalePrice")

CentralAir_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = CentralAir, y = SalePrice, color = CentralAir)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of CentralAir vs. SalePrice")

grid.arrange(CentralAir_boxplot, CentralAir_scatterplot, nrow = 1)


# Clearly, having central air conditioning positively influences sale price.



#####
# Electrical: Electrical system
#####

# There is a missing value in Electrical.
summary(dataset$Electrical)

# There is one missing value in Electrical
which(is.na(dataset$Electrical))
dataset[1380, ] # The house with the missing Electrical value seems pretty normal

# From the plot we can see that SBrkr or standard circuit breakers are the most common value
plot(dataset[, "Electrical"],
     col = "orange",
     main = "Electrical",
     ylab = "Count"
)

# We will use kNN-based predictions to impute the missing values.
knn_model <- kNN(dataset, variable = "Electrical", k = 9)

# The kNN-model predicts SBrkr, the most common type of Electrical
knn_model[knn_model$Electrical_imp == TRUE, ]$Electrical

# Missing value imputation of Electrical
dataset[which(is.na(dataset$Electrical)), ]$Electrical <- knn_model[knn_model$Electrical_imp == TRUE, ]$Electrical

# Boxplot of Electrical vs. SalePrice.
Electrical_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Electrical, y = SalePrice, color = Electrical)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Electrical vs. SalePrice")

Electrical_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Electrical, y = SalePrice, color = Electrical)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Electrical vs. SalePrice")

grid.arrange(Electrical_boxplot, Electrical_scatterplot, nrow = 1)

# From the plots it becomes apparent that "SBrkr" is not only the most common,
# but also the most valuable kind of "Electrical".



#####
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
#####

# There are no missing values in the 3 variables.
summary(dataset$X1stFlrSF)
summary(dataset$X2ndFlrSF)
summary(dataset$LowQualFinSF)

# The above ground living area is composed of 1StFlrSF, 2ndFlrSF and LowQualFinSF.
# The correlation is 1.
cor(dataset$GrLivArea, (dataset$X1stFlrSF + dataset$X2ndFlrSF + dataset$LowQualFinSF))

# We will drop the redundant variables.
dataset <- subset(dataset, select = -c(X1stFlrSF, X2ndFlrSF, LowQualFinSF))



#####
# GrLivArea: Above grade (ground) living area square feet
#####

# There are no missing values in GrLivArea.
summary(dataset$GrLivArea)

# We change variable encoding to numeric
dataset$GrLivArea <- as.numeric(dataset$GrLivArea)


dataset[train$Id, ] %>%
  ggplot(aes(x = GrLivArea, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of GrLivArea vs. SalePrice") +
  geom_smooth(method = "lm")

# As to be expected, a larger above ground living area correlates strongly with sale price.



#####
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#####

# There are no missing values in BedroomAbvGr
summary(dataset$BedroomAbvGr)

# Scatterplot of BedroomAbvGr vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = BedroomAbvGr, y = SalePrice, color = BedroomAbvGr)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(0, 8, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of BedroomAbvGr vs. SalePrice") +
  geom_smooth(method = "lm")


# While a larger number of bedrooms probably denotes larger and therefore more expensive houses, the absolute number of bedrooms is not necessarily
# the strongest predictor of sale price.



#####
# Kitchen: Kitchens above grade
#####

# There are no missing values in KitchenAbvGr
summary(dataset$KitchenAbvGr)

# Scatterplot of BedroomAbvGr vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = KitchenAbvGr, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(0, 3, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of KitchenAbvGr vs. SalePrice") +
  geom_smooth(method = "lm")


# From the plot it becomes apparant that the number of kitchens above ground are not a good predictor of sale price.
# Most houses actually only have one kitchen above ground, while having two or more does not indicate increased value.

# A few houses actually have no kitchen above ground.
which(dataset$KitchenAbvGr == 0)



#####
# KitchenQual: Kitchen quality
#####

# There is one missing value in KitchenQual
summary(dataset$KitchenQual)

# From the above plot of KitchenAbvGr we know that a few houses actually don't have a kitchen above ground.
# But is it the same house? Actually, the house with missing KitchenQual has one kitchen above ground.
dataset[which(is.na(dataset$KitchenQual)), ]


# We will use kNN-based predictions to impute the missing values.
knn_model <- kNN(dataset, variable = "KitchenQual", k = 9)

# The kNN-model predicts TA, the most common type of KitchenQual
knn_model[knn_model$KitchenQual_imp == TRUE, ]$KitchenQual

# Missing value imputation of KitchenQual.
dataset[which(is.na(dataset$KitchenQual)), ]$KitchenQual <- knn_model[knn_model$KitchenQual_imp == TRUE, ]$KitchenQual

# Boxplot of KitchenQual vs. SalePrice.
KitchenQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = KitchenQual, y = SalePrice, color = KitchenQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("KitchenQual vs. SalePrice")

KitchenQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = KitchenQual, y = SalePrice, color = KitchenQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("KitchenQual vs. SalePrice")

grid.arrange(KitchenQual_boxplot, KitchenQual_scatterplot, nrow = 1)

# From the plots we can see that "KitchenQual" strongly correlates with "SalePrice".
# The quality of a kitchen is much more important than the total number.

# We transform this qualitative ordinal factor into a numeric containing the present levels and plot it again.

dataset$KitchenQual <- as.numeric(factor(dataset$KitchenQual, levels = c("Fa",
                                                                         "TA", "Gd", "Ex")))
# Plotting changed KitchenQual.
dataset[train$Id, ] %>%
  ggplot(aes(x = KitchenQual, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("KitchenQual vs. SalePrice") +
  geom_smooth(method = "lm")



#####
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#####

# There are no missing values in TotRmsAbvGrd
summary(dataset$TotRmsAbvGrd)

# Scatterplot of TotRmsAbvGrd vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = TotRmsAbvGrd, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(0, 14, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of TotRmsAbvGrd vs. SalePrice") +
  geom_smooth(method = "lm")

# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) is clearly an indicator of a houses size.
# The more rooms the higher the sale price in most cases.



#####
# Functional: Home functionality (Assume typical unless deductions are warranted)
#####

# There are a few missing values in Functional.
summary(dataset$Functional)

# We take a look at the houses with missing values in Functional.
dataset[which(is.na(dataset$Functional)), ]

# We will use kNN-based predictions to impute the missing values.
knn_model <- kNN(dataset, variable = "Functional", k = 9)

# The kNN-model predicts Typ, the most common type of Functional.
knn_model[knn_model$Functional_imp == TRUE, ]$Functional

# Missing value imputation of Functional.
dataset[which(is.na(dataset$Functional)), ]$Functional <- knn_model[knn_model$Functional_imp == TRUE, ]$Functional

# Boxplot of Functional vs. SalePrice.
Functional_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Functional, y = SalePrice, color = Functional)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Functional vs. SalePrice")

Functional_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Functional, y = SalePrice, color = Functional)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Functional vs. SalePrice")

grid.arrange(Functional_boxplot, Functional_scatterplot, nrow = 1)

# We plot "Functional" against "SalePrice" and observe that "Maj2" (major deductions 2) seems to influence "SalePrice" negatively. A few houses are "Sev" (severly damaged), but median "SalePrice" remaisn comparable in these cases. Most houses, also most expensive houses,have "Typ" (typical functionality) values.



#####
# Fireplaces: Number of fireplaces
#####

# There are no missing values in Fireplaces.
summary(dataset$Fireplaces)

# We change variable encoding to numeric
dataset$Fireplaces <- as.numeric(dataset$Fireplaces)

# Scatterplot of Fireplaces vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = Fireplaces, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  scale_x_continuous(breaks = seq(0, 14, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of Fireplaces vs. SalePrice") +
  geom_smooth(method = "lm")

# Fireplaces can be considered another measurement of house size. Having more fireplaces is possitively correalted with sale price.
# It seems likely that having 1 or more fireplaces of high quality would be strong predictors of sale price.
# To confirm this, we have to inspect FireplacesQu.



#####
# FireplaceQu: Fireplace quality
#####

# A large amount of values are missing in FireplaceQu
summary(dataset$FireplaceQu)

# From the data description we know that "no fireplace" was originally encoded as NA in the data.
# We can therefore replace all NAs with "None" and fix factor levels.

dataset$FireplaceQu <- as.character(dataset$FireplaceQu)
dataset$FireplaceQu[which(is.na(dataset$FireplaceQu))] <- "None"
dataset$FireplaceQu <- factor(dataset$FireplaceQu, levels = c("None", "Po", "Fa", "TA", "Gd", "Ex"))


# Boxplot of FireplaceQu vs. SalePrice.
FireplaceQu_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = FireplaceQu, y = SalePrice, color = FireplaceQu)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of FireplaceQu vs. SalePrice")

FireplaceQu_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = FireplaceQu, y = SalePrice, color = FireplaceQu)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of FireplaceQu vs. SalePrice")

grid.arrange(FireplaceQu_boxplot, FireplaceQu_scatterplot, nrow = 1)

# From the plots we can see that higher quality fireplaces are well-correlated with higher sale price.
# However, there are also many houses without any fireplaces that have larger sale prices than some of the houses with poor, fair, or typical quality fireplaces.



#####
# GarageType: Garage location
##### 

# There are a lot of missing values in GarageType.
summary(dataset$GarageType)

# From the data description we know that "No Garage" was originally encoded as NA.
# We will replace those with "None" and fix factor levels.

dataset$GarageType <- as.character(dataset$GarageType)
dataset$GarageType[which(is.na(dataset$GarageType))] <- "None"
dataset$GarageType <- factor(dataset$GarageType, levels = c("None", "Detchd", "CarPort", "BuiltIn", "Basment", "Attchd", "2Types"))

# Boxplot of GarageType vs. SalePrice.
GarageType_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageType, y = SalePrice, color = GarageType)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageType vs. SalePrice")

GarageType_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageType, y = SalePrice, color = GarageType)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageType vs. SalePrice")

grid.arrange(GarageType_boxplot, GarageType_scatterplot, nrow = 1)


# From the plots we can see that having no garage or just a carport can be predictive of a lower sale price.
# There is a certain difference between detached and attached garages, too.



#####
# GarageYrBlt: Year garage was built
#####

# There are a lot of missing values in GarageYrBlt, presumably from houses with no garages.
summary(dataset$GarageYrBlt)

# We plot GarageYrBlt vs. YearBuilt and color by GarageType.
dataset %>%
  ggplot(aes(x = GarageYrBlt, y = YearBuilt, color = GarageType)) +
  geom_point(alpha = 0.3) +
  theme_bw() +
  ggtitle("Boxplot of GarageType vs. SalePrice")

# From the plot we can see that it is mostly detached garages being built after the house itself, otherwise YearBuilt and GarageYrBlt are mostly identical.
# Also, there seems to be an outlier, with a garage built sometime in the far future.
dataset[which(dataset$GarageYrBlt > 2010), ]

# From this it seems most likely that the garage was built at the same time as the house and the outlier is based on a typo.
# We will fix this by changing the year 2207 to 2007. 
dataset$GarageYrBlt[which(dataset$GarageYrBlt > 2010)] <- 2007

# We plot GarageYrBlt vs. YearBuilt and color by GarageType without the outlier.
dataset %>%
  ggplot(aes(x = GarageYrBlt, y = YearBuilt, color = GarageType)) +
  geom_point(alpha = 0.6) +
  theme_bw() +
  ggtitle("Scatterplot of GarageType vs. SalePrice")

# Interestingly, there are some houses with garage building years prior to the house itself.
# Perhaps building the house took much longer than the garage.
# Overall, GarageYrBlt doesn't add much in terms of sale price predictive power.
# As we can't turn this variable into a factor with "None" for not having a garage, we have to either impute the house YearBuilt or remove
# GarageYrBlt entirely.
# We will do the latter, as there are more than enough variables about Garages in the dataset.
dataset <- subset(dataset, select = - GarageYrBlt)



#####
# GarageFinish: Interior finish of the garage
#####

# There are many missing values in GarageFinish, presumably from not having a garage in the first place, as seen before.
summary(dataset$GarageFinish)

# From the data description we know that "No Garage" was originally encoded as NA.
# We will replace those with "None" and fix factor levels.

dataset$GarageFinish <- as.character(dataset$GarageFinish)
dataset$GarageFinish[which(is.na(dataset$GarageFinish))] <- "None"
dataset$GarageFinish <- factor(dataset$GarageFinish, levels = c("None", "Unf", "RFn", "Fin"))

# Boxplot of GarageFinish vs. SalePrice.
GarageFinish_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageFinish, y = SalePrice, color = GarageFinish)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageFinish vs. SalePrice")

GarageFinish_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageFinish, y = SalePrice, color = GarageFinish)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageFinish vs. SalePrice")

grid.arrange(GarageFinish_boxplot, GarageFinish_scatterplot, nrow = 1)

# No or unfinished garages are predictive of lower sale price.



#####
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
#####

# GarageCars and GarageArea are two variables dealing with garage size.
# Both contain a missing value.
summary(dataset$GarageArea)
summary(dataset$GarageCars)

# One house is missing a value in both variables.
# While the house has a GarageType of "Detchd", the NAs in GarageQual and GarageCond hint at the house 
# actually not having any garage.
dataset[which(is.na(dataset$GarageArea)), ]
dataset[which(is.na(dataset$GarageCars)), ]

# It seems that this house wrongly has an entry in GarageType. We will change this to "None".
dataset$GarageType[which(is.na(dataset$GarageArea))] <- "None"

# This house has no garage and therefore no size value.
dataset$GarageArea[which(is.na(dataset$GarageArea))] <- 0
dataset$GarageCars[which(is.na(dataset$GarageCars))] <- 0

# We plot GarageArea vs. SalePrice and color by factorized GarageCars.
dataset[train$Id, ] %>%
  ggplot(aes(x = GarageArea, y = SalePrice, color = factor(GarageCars))) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  ggtitle("Boxplot of GarageArea vs. SalePrice")

# From the plot we can clearly see that garage size in terms of area strongly correlates with the number
# of cars that it is designed to fit.

# GarageArea and GarageCars are highly correlated.
cor(dataset$GarageArea, dataset$GarageCars)

# GarageCars is slightly better correlated with sale price.
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$GarageArea)
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$GarageCars)

# In order to avoid having two extremely colinear variables we will drop GarageArea, the one with slightly lower correlation to sale price.
dataset <- subset(dataset, select = -GarageArea)



#####
# GarageQual: Garage quality
#####

# There are many missing values, presumably from these houses having no garage.
summary(dataset$GarageQual)

# From the data description we know that "No Garage" was originally encoded as NA.
# We will replace those with "None" and fix factor levels.

dataset$GarageQual <- as.character(dataset$GarageQual)
dataset$GarageQual[which(is.na(dataset$GarageQual))] <- "None"
dataset$GarageQual <- factor(dataset$GarageQual, levels = c("None", "Po", "Fa", "TA", "Gd", "Ex"))

# Boxplot of GarageQual vs. SalePrice.
GarageQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageQual, y = SalePrice, color = GarageQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageQual vs. SalePrice")

GarageQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageQual, y = SalePrice, color = GarageQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageQual vs. SalePrice")

grid.arrange(GarageQual_boxplot, GarageQual_scatterplot, nrow = 1)

# Garage quality is associated with higher sale price.



#####
# GarageCond: Garage Condition
#####

# There are many missing values, presumably from these houses having no garage.
summary(dataset$GarageCond)

# From the data description we know that "No Garage" was originally encoded as NA.
# We will replace those with "None" and fix factor levels.

dataset$GarageCond <- as.character(dataset$GarageCond)
dataset$GarageCond[which(is.na(dataset$GarageCond))] <- "None"
dataset$GarageCond <- factor(dataset$GarageCond, levels = c("None", "Po", "Fa", "TA", "Gd", "Ex"))

# Boxplot of GarageCond vs. SalePrice.
GarageCond_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageCond, y = SalePrice, color = GarageCond)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageCond vs. SalePrice")

GarageCond_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = GarageCond, y = SalePrice, color = GarageCond)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of GarageCond vs. SalePrice")

grid.arrange(GarageCond_boxplot, GarageCond_scatterplot, nrow = 1)

# Garage condition is very similar to garage quality. In both cases, a typical value "TA" seems to be the mode and actually associated with highest sale price.



#####
# PavedDrive: Paved driveway
#####

# There are no missing values in PavedDrive.
summary(dataset$PavedDrive)

# Boxplot of PavedDrive vs. SalePrice.
PavedDrive_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = PavedDrive, y = SalePrice, color = PavedDrive)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of PavedDrive vs. SalePrice")

PavedDrive_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = PavedDrive, y = SalePrice, color = PavedDrive)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of PavedDrive vs. SalePrice")

grid.arrange(PavedDrive_boxplot, PavedDrive_scatterplot, nrow = 1)

# A paved driveway is predictive of a higher sale price.



# We will first look at all the porch-related variables in isolation.


#####
# WoodDeckSF: Wood deck area in square feet
#####

# There are no missing values in WoodDeckSF.
summary(dataset$WoodDeckSF)

# We change encoding to numerical.
dataset$WoodDeckSF <- as.numeric(dataset$WoodDeckSF)

# Scatterplot of WoodDeckSF vs. SalePrice
dataset[train$Id, ] %>%
  ggplot(aes(x = WoodDeckSF, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of WoodDeckSF vs. SalePrice")

# WoodDeckSF positively correelates with SalePrice.


#####
# OpenPorchSF: Open porch area in square feet
#####

# There are no missing values in OpenPorchSF.
summary(dataset$OpenPorchSF)

# We change encoding to numerical.
dataset$OpenPorchSF <- as.numeric(dataset$OpenPorchSF)

# Scatterplot of OpenPorchSF vs. SalePrice
dataset[train$Id, ] %>%
  filter(OpenPorchSF > 0) %>%
  ggplot(aes(x = OpenPorchSF, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OpenPorchSF vs. SalePrice")

# OpenPorchSF weakly positively correlates with SalePrice.


#####
# EnclosedPorch: Enclosed porch area in square feet
#####

# There are no missing values in EnclosedPorch.
summary(dataset$EnclosedPorch)

# We change encoding to numerical.
dataset$EnclosedPorch <- as.numeric(dataset$EnclosedPorch)

# Scatterplot of EnclosedPorch vs. SalePrice where EnclosedPorch > 0.
dataset[train$Id, ] %>%
  filter(EnclosedPorch > 0) %>%
  ggplot(aes(x = EnclosedPorch, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of EnclosedPorch vs. SalePrice")

# EnclosedPorch weakly positively correlates with SalePrice.


#####
# X3SsnPorch: Enclosed porch area in square feet
#####

# There are no missing values in X3SsnPorch.
summary(dataset$X3SsnPorch)

# We change encoding to numerical.
dataset$X3SsnPorch <- as.numeric(dataset$X3SsnPorch)

# Scatterplot of X3SsnPorch vs. SalePrice where X3SsnPorch > 0.
dataset[train$Id, ] %>%
  filter(X3SsnPorch > 0) %>%
  ggplot(aes(x = X3SsnPorch, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of X3SsnPorch vs. SalePrice")

# X3SsnPorch weakly positively correlates with SalePrice.


#####
# ScreenPorch: Enclosed porch area in square feet
#####

# There are no missing values in ScreenPorch.
summary(dataset$ScreenPorch)

# We change encoding to numerical.
dataset$ScreenPorch <- as.numeric(dataset$ScreenPorch)

# Scatterplot of ScreenPorch vs. SalePrice where ScreenPorch > 0.
dataset[train$Id, ] %>%
  filter(ScreenPorch > 0) %>%
  ggplot(aes(x = ScreenPorch, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of ScreenPorch vs. SalePrice")

# ScreenPorch weakly positively correlates with SalePrice.



# It seems that all the porch-related variables are individually not very predictive of sale price.
# We will try and combine them into one new variable, "TotalPorch".

dataset$TotalPorch <- dataset$WoodDeckSF + dataset$OpenPorchSF + dataset$EnclosedPorch + dataset$X3SsnPorch + dataset$ScreenPorch

# Scatterplot of TotalPorch vs. SalePrice where TotalPorch > 0.
dataset[train$Id, ] %>%
  filter(TotalPorch > 0) %>%
  ggplot(aes(x = TotalPorch, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of TotalPorch vs. SalePrice")

# TotalPorch as a decent correlation with sale price.
cor(dataset[train$Id, ]$SalePrice, dataset[train$Id, ]$TotalPorch)

# We will remove the individual porch-related variables from the dataset.
# This combined variable disregards any differenes in value between the individual variables.
dataset <- subset(dataset, select = -c(OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch, WoodDeckSF))



#####
# PoolArea: Pool area in square feet
#####

# There are no missing values in PoolArea
summary(dataset$PoolArea)

# We convert "PoolArea" to numerical.
dataset$PoolArea <- as.numeric(dataset$PoolArea)

# Scatterplot of PoolArea vs. SalePrice where PoolArea > 0.
dataset[train$Id, ] %>%
  filter(PoolArea > 0) %>%
  ggplot(aes(x = PoolArea, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of PoolArea vs. SalePrice")

# PoolArea seems to have little predictive power in terms of sale price. Also, only a few houses actually have a pool.
# Is having a pool of any size in general more important? We will explore PoolQC.



#####
# PoolQC: Pool quality
#####

# Almost all values are missing in PoolQC.
summary(dataset$PoolQC)

# We know from the data description that not having a pool is wrongly encoded as NA.
# We fill fix this problem and correct factor levels.
dataset$PoolQC <- as.character(dataset$PoolQC)
dataset$PoolQC[which(is.na(dataset$PoolQC))] <- "None"
dataset$PoolQC <- factor(dataset$PoolQC, levels = c("None", "Fa", "Gd", "Ex"))

# Scatterplot of PoolQC vs. SalePrice.
dataset[train$Id, ] %>%
  ggplot(aes(x = PoolQC, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of PoolQC vs. SalePrice")



#####
# Fence: Fence quality
#####

# Most values are missing from Fence.
summary(dataset$Fence)

# We know from the data description that not having a fence is wrongly encoded as NA.
# We fill fix this problem and correct factor levels.
dataset$Fence <- as.character(dataset$Fence)
dataset$Fence[which(is.na(dataset$Fence))] <- "None"
dataset$Fence <- factor(dataset$Fence, levels = c("None", "MnWw", "GdWo", "MnPrv", "GdPrv"))

# Boxplot of Fence vs. SalePrice.
Fence_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Fence, y = SalePrice, color = Fence)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplotplot of Fence vs. SalePrice")

# Scatterplot of Fence vs. SalePrice.
Fence_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = Fence, y = SalePrice, color = Fence)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of Fence vs. SalePrice")

grid.arrange(Fence_boxplot, Fence_scatterplot, nrow = 1)

# Fence is not giving much information at all. Fences seem to have little importance in terms of sale price.



#####
# MiscFeature: Miscellaneous feature not covered in other categories
#####

# Most values are missing from MiscFeature
summary(dataset$MiscFeature)

# We know from the data description that not having a MiscFeature is wrongly encoded as NA.
# We fill fix this problem and correct factor levels.
dataset$MiscFeature <- as.character(dataset$MiscFeature)
dataset$MiscFeature[which(is.na(dataset$MiscFeature))] <- "None"
dataset$MiscFeature <- factor(dataset$MiscFeature, levels = c("None", "TenC", "Shed", "Othr", "Gar2", "Elev"))

# Boxplot of MiscFeature vs. SalePrice.
MiscFeature_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MiscFeature, y = SalePrice, color = MiscFeature)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplotplot of MiscFeature vs. SalePrice")

# Scatterplot of MiscFeature vs. SalePrice.
MiscFeature_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = MiscFeature, y = SalePrice, color = MiscFeature)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of MiscFeature vs. SalePrice")

grid.arrange(MiscFeature_boxplot, MiscFeature_scatterplot, nrow = 1)

# While having a tennis court might increase sale price, there are simply not enough data points for the MiscFeature variable to be
# useful.



#####
# MiscVal: $Value of miscellaneous feature
#####

# There are no missing values in MiscVal
summary(dataset$MiscVal)

# We convert "MiscVal"" to numerical.
dataset$MiscVal <- as.numeric(dataset$MiscVal)

# Scatterplot of MiscVal vs. SalePrice
dataset[train$Id, ] %>%
  filter(MiscVal > 0) %>%
  ggplot(aes(x = MiscVal, y = SalePrice)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of MiscVal vs. SalePrice")



#####
# MoSold: Month Sold (MM)
# YrSold: Year Sold (YYYY)
#####

# We convert "YrSold" and "MoSold" to numerical.
dataset$YrSold <- as.numeric(dataset$YrSold)
dataset$MoSold <- as.numeric(dataset$MoSold)

# We plot MonthSold and YearSold - do these features influence SalePrice?


# We plot the amount of sold houses per month. Most houses are sold during the Spring/Summer months of the year.
dataset %>%
  group_by(MoSold) %>%
  count() %>%
  ggplot(aes(x = MoSold, y = n)) +
  geom_line(col = "blue") +
  ylab("Number of houses sold") +
  xlab("Month") +
  scale_x_continuous(breaks = seq(1,12,1)) +
  geom_vline(xintercept = 3, col = "red", linetype = "dashed") +
  geom_vline(xintercept = 9, col = "red", linetype = "dashed")

# Does this influence SalePrice? MoSold doesn't seem to influence SalePrice at all. There are simply more houses being sold in the summer months.
dataset[train$Id, ] %>%
  group_by(MoSold) %>%
  ggplot(aes(x = MoSold, y = SalePrice, group = MoSold)) +
  geom_boxplot() +
  ylab("Sale price") +
  xlab("Month") +
  scale_x_continuous(breaks = seq(1,12,1)) +
  ggtitle("Distribution of SalePrice over MoSold")
  
  
# We plot the amount of sold houses per year. The amount is relatively similar, but much less entries in 2010.
dataset %>%
  group_by(YrSold) %>%
  count() %>%
  ggplot(aes(x = YrSold, y = n)) +
  geom_line(col = "blue") +
  ylab("Number of houses sold") +
  xlab("Year")
  
  
# YrSold doesn't seem to influence SalePrice at all.
dataset[train$Id, ] %>%
  group_by(YrSold) %>%
  ggplot(aes(x = YrSold, y = SalePrice, group = YrSold)) +
  geom_boxplot() +
  ylab("Sale price") +
  xlab("Year") +
  ggtitle("Distribution of SalePrice over YrSold")


#####
# SaleType: Type of sale
#####

# There is a missing value in SaleType.
summary(dataset$SaleType)

# Boxplot of SaleType vs. SalePrice.
SaleType_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = SaleType, y = SalePrice, color = SaleType)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplotplot of SaleType vs. SalePrice")

# Scatterplot of SaleType vs. SalePrice.
SaleType_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = SaleType, y = SalePrice, color = SaleType)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of SaleType vs. SalePrice")

grid.arrange(SaleType_boxplot, SaleType_scatterplot, nrow = 1)

# From the plots we can see that some SaleTypes occur more seldom than others and that sales
# of new houses and warranty deed conventional ("WD") are indicative of higher sale prices.
# Some categories have simply too few entries.

# We use kNN-based missing value imputation
knn_model <- kNN(dataset, variable = "SaleType", k = 9)

# Predicted SaleType values
knn_model[knn_model$SaleType_imp == TRUE, ]$SaleType

# We impute the values
dataset$SaleType[which(is.na(dataset$SaleType))] <- knn_model[knn_model$SaleType_imp == TRUE, ]$SaleType



#####
# SaleCondition: Condition of sale
#####

# There are no missing values in SaleCondition.
summary(dataset$SaleCondition)

# Boxplot of SaleCondition vs. SalePrice.
SaleCondition_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = SaleCondition, y = SalePrice, color = SaleCondition)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplotplot of SaleCondition vs. SalePrice")

# Scatterplot of SaleCondition vs. SalePrice.
SaleCondition_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = SaleCondition, y = SalePrice, color = SaleCondition)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of SaleCondition vs. SalePrice")

grid.arrange(SaleCondition_boxplot, SaleCondition_scatterplot, nrow = 1)

# From the plots we can see that partial, as a measure of "newness" seems to indicate higher sale price.
# Most houses are sold under normal conditions.



###
# Skewedness of the outcome variable SalePrice
###

# Monetary values are often log-normally distributed. How does SalePrice look like?
# The histogram below indicates a certain skewedness.
dataset[train$Id, ] %>%
  ggplot(aes(SalePrice)) +
  geom_histogram(bins = 30, color = "black", fill = "orange") +
  ggtitle("Histogram of SalePrice distribution")

# We can check for skewedness with a function. The skew is considerably higher than 0.8.
# A log-transformation could potentially lead to SalePrice being more normal.
e1071::skewness(dataset[train$Id, ]$SalePrice) # Determine skew of SalePrice

dataset[train$Id, ] %>%
  ggplot(aes(log1p(SalePrice))) +
  geom_histogram(bins = 30, color = "black", fill = "orange") +
  xlab("Log-transformed SalePrice") +
  ggtitle("Histogram of log-transformed SalePrice distribution")

# We log-transform SalePrice and indeed, it looks much more normal now.
dataset$SalePrice <- log1p(dataset$SalePrice)



### All missing values have been dealt with and we can once again separate `dataset` into train and test ###
train <- dataset[train$Id, ]                                                                             ###
test <- subset(dataset[test$Id, ], select = -SalePrice)   # We remove the temporary SalePrice column again #                        ###                                                        ###
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
# test_set will receive 20% of the data, train_set will receive 80%.
# The test_set will serve as a "hold-out" set to check algorithm performance.
set.seed(1442)
test_index <- createDataPartition(train$SalePrice, p = 0.2, list = FALSE)
train_set <- train[-test_index, ]
temp <- train[test_index, ] # temporary test set

# We make sure there are no entries in test_set that aren't in train_set
test_set <- temp %>%
  semi_join(train_set, by = c("Electrical", "MiscFeature")) 

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

summary(train_set)
summary(test_set)


## Simple linear regression as a baseline model.

# As our first, very simple model we predict house "SalePrice" via simple linear regression with just "GrLivArea". We also generate a table to keep track of the RMSEs our various models generate.

# Linear regression with a single predictor, "GrLivArea".
model_1_lm <- lm(SalePrice ~ GrLivArea, data = train_set) 

# Predict on test_set.
model_1_pred <- predict(model_1_lm, newdata = test_set) 

# Calculate RMSE with our defined function.
model_1_lm_RMSE <- RMSE(model_1_pred, test_set$SalePrice) 

# Record RMSE of the model.
model_rmses <- data_frame(Model = "Simple_lm", RMSE = model_1_lm_RMSE)

# Print RMSEs.
model_rmses %>% knitr::kable()



#####
# Multivariate linear regression with several predictors.
model_2_multi_lm <- lm(SalePrice ~ GrLivArea + OverallQual + YearBuilt, data = train_set) 

# Predict on test_set.
model_2_multi_lm_pred <- predict(model_2_multi_lm, newdata = test_set) 

# Calculate RMSE with our defined function.
model_2_multi_lm_RMSE <- RMSE(model_2_multi_lm_pred, test_set$SalePrice) 

# Record RMSE of the model.
model_rmses <- bind_rows(model_rmses, data_frame(Model = "Multi_lm_several", RMSE = model_2_multi_lm_RMSE))

# Print RMSEs.
model_rmses %>% knitr::kable()


#####
# Multivariate linear regression utilizing all available predictors achieves an even lower RMSEm which is printed below.

# Multivariate linear regression with all predictors.
model_3_multi_lm <- lm(SalePrice ~ ., data = train_set) 

# Predict on test_set.
model_3_multi_lm_pred <- predict(model_3_multi_lm, newdata = test_set) 

# Calculate RMSE with our defined function.
model_3_multi_lm_RMSE <- RMSE(model_3_multi_lm_pred, test_set$SalePrice) 

# Record RMSE of the model.
model_rmses <- bind_rows(model_rmses, data_frame(Model = "Multi_lm_all", RMSE = model_3_multi_lm_RMSE))

# Print RMSEs.
model_rmses %>% knitr::kable()










# Gradient Boosting Machine - XGBOOST
# We will utilize the XGBoost algorithm to build models to predict "SalePrice". XGBoost requires the dataset to be entirely in numerical format, which can be achieved via so-called "one-hot-encoding" of the variables. One way to do this is to use the "vtreat" package and its function "designTreatmentsZ", which devises a "treatment plan" to "one-hot-encode" all relevant variables at once. This "treatment plan" is then used via the "prepare" function to do the "one-hot-encoding" on the `train_set` and on the `test_set`. We thus generate `train_set_treated` and `test_set_treated` for use with XGBoost.

# We select all relevant predictors.
variables <- names(subset(train_set, select = -c(Id, SalePrice)))

# The vtreat function "designTreatmentsZ" helps encode all variables numerically
# via one-hot-encoding.

# Devise a "treatment plan" for the variables selected above.
# use_series() works like a $, but within pipes, so we can access scoreFrame.
# We select only the rows we care about: catP is a "prevalence fact" and tells
# whether the original level was rare or common and not really useful in the model.
# We get the varName column.
treatment_plan <- designTreatmentsZ(train_set, variables, verbose = FALSE) 

newvars <- treatment_plan %>%
  use_series(scoreFrame) %>%        
  filter(code %in% c("clean", "lev")) %>%  
  use_series(varName)         

# The prepare() function prepares our data subsets according to the treatment plan
# we devised above and encodes all relevant variables "newvars" numerically.

# Treatment of train_set.
train_set_treated <- vtreat::prepare(treatment_plan, train_set,  varRestriction = newvars)

# Treatment of test_set.
test_set_treated <- vtreat::prepare(treatment_plan, test_set,  varRestriction = newvars)

# Next we will use the `xgb.cv()` function to determine the total number of rounds `nrounds` that improve RMSE until only the training RMSE reduces further, while the cross-validated RMSE already reached a minimum. While the training RMSE may continue to decrease on more and more rounds of boosting iterations ("overfitting"), the test RMSE usually does not after some point. After running `xgb.cv()` we can access its event log to find the optimal number of iterations. As the treated `train_set` no longer contains the outcome variable "SalePrice", we have to use the untreated `train_set` to provide it as a `label`. For our baseline XGBoost model, we will use mostly default parameters.

# xgb.cv only takes a matrix of the treated, all-numerical input data.
cv <- xgb.cv(data = as.matrix(train_set_treated),  
             label = train_set$SalePrice, # Outcome from untreated data
             nrounds = 500,
             nfold = 5, # We use 5 folds for cross-validation
             early_stopping_rounds = 10,
             verbose = 0)    # silent

# Get the evaluation log of the cross-validation and find the number
# of iterations that minimize RMSE without overfitting the training data
elog <- cv$evaluation_log 

# Finding the indexes.
elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean), 
            ntrees.test  = which.min(test_rmse_mean))

# Save the number of iterations that minimize test-RMSE in `niter`.
niter <- elog %>% 
  summarize(niter.train = which.min(train_rmse_mean),
            niter.test  = which.min(test_rmse_mean)) %>%
  use_series(niter.test)

# Next we run the actual modelling process with the information
# gained by running xgboost cross-validation above.
# The treated `train_set`has to be provided as a matrix.
XGBoost_baseline <- xgboost(data = as.matrix(train_set_treated),
                            label = train_set$SalePrice,
                            nrounds = niter,
                            objective = "reg:linear",
                            verbose = 0)  

# Now we can predict SalePrice in the test_set with the xgb-model
XGBoost_baseline_pred <- predict(XGBoost_baseline,
                                 newdata = as.matrix(test_set_treated))

# Calculate RMSE.
XGBoost_baseline_RMSE <- RMSE(XGBoost_baseline_pred, test_set$SalePrice)

# Record the RMSE.
model_rmses <- bind_rows(model_rmses,
                         data_frame(Model = "XGBoost_baseline", RMSE = XGBoost_baseline_RMSE))

# Print the RMSE table.
model_rmses %>% knitr::kable()


#####
# Hyperparameter tuning with Caret.
#####

# We will evaluate different values for the hyperparameters of the "xgbTree" algorithm.
# As expansive grid searches become computationally very expensive the more parameters are evaluated at the same time (easily into the thousands of models),
# we will instead evaluate no more than 2 parameters per tuning round.


#####
# xgbTree hyperparameter testing
#####

# We define a tune grid with selected ranges of hyperparameters to tune.
tuneGrid <- expand.grid(
  nrounds = seq(150, 1500, 50),
  max_depth = c(2, 3, 4, 5, 6, 7),
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3, 4, 5, 6),
  subsample = 1
)

# We define a custom train control for the caret train() function.
train_control <- trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = FALSE,
  allowParallel = TRUE
)

# We run the model with above parameters.
# Additionally, we add pre-processing which removes near-zero variance estimators,
# as well as centers and scales the data prior to training.
xgb_1st_tuning <- caret::train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = FALSE,
  preProcess = c("nzv", "center", "scale")
)

# Print the best tuning parameters.
xgb_1st_tuning$bestTune

# Visualization of the 1st tuning round.
ggplot(xgb_1st_tuning) + scale_y_continuous(limits = c(0.1225, 0.14))

# Visualization of the most important features.
vip(xgb_1st_tuning, num_features = 10) + ggtitle("Variable importance")


# We predict on the test_set and record the "out-of-bag" RMSE.
xgb_1st_tuning_pred <- predict(xgb_1st_tuning, test_set_treated)

xgb_1st_tuning_rmse <- RMSE(xgb_1st_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses,
                         data_frame(Model = "caret_xgbTree_1st_tune", RMSE = xgb_1st_tuning_rmse))

model_rmses %>% knitr::kable()




### 2nd tune ###

# We define a tune grid with selected ranges of hyperparameters to tune.
tuneGrid <- expand.grid(
  nrounds = seq(150, 1500, 50),
  max_depth = xgb_1st_tuning$bestTune$max_depth,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = seq(0.5, 1, 0.1),
  min_child_weight = xgb_1st_tuning$bestTune$min_child_weight,
  subsample = seq(0.5, 1, 0.1)
)

# We define a custom train control for the caret train() function.
train_control <- trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = FALSE,
  allowParallel = TRUE
)

# We run the model with above parameters.
# Additionally, we add pre-processing which removes near-zero variance estimators,
# as well as centers and scales the data prior to training.
xgb_2nd_tuning <- caret::train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = FALSE,
  preProcess = c("nzv", "center", "scale")
)

# Print the best tuning parameters.
xgb_2nd_tuning$bestTune

# Visualization of the 2nd tuning round.
ggplot(xgb_2nd_tuning) + scale_y_continuous(limits = c(0.125, 0.14))


### 3rd tune ###

# We define a tune grid with selected ranges of hyperparameters to tune.
tuneGrid <- expand.grid(
  nrounds = seq(150, 1500, 50),
  max_depth = xgb_1st_tuning$bestTune$max_depth,
  eta = c(0.01, 0.025, 0.05, 0.075, 0.1),
  gamma = 0,
  colsample_bytree = xgb_2nd_tuning$bestTune$colsample_bytree,
  min_child_weight = xgb_1st_tuning$bestTune$min_child_weight,
  subsample = xgb_2nd_tuning$bestTune$subsample
)

# We define a custom train control for the caret train() function.
train_control <- trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = FALSE,
  allowParallel = TRUE
)

# We run the model with above parameters.
# Additionally, we add pre-processing which removes near-zero variance estimators,
# as well as centers and scales the data prior to training.
xgb_3rd_tuning <- caret::train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = FALSE,
  preProcess = c("nzv", "center", "scale")
)

# Print the best tuning parameters.
xgb_3rd_tuning$bestTune

# Visualization of the 3rd tuning round.
ggplot(xgb_3rd_tuning) + scale_y_continuous(limits = c(0.1225, 0.14))


# We predict on the test_set and record the "out-of-bag" RMSE.
xgb_3rd_tuning_pred <- predict(xgb_3rd_tuning, test_set_treated)

xgb_3rd_tuning_rmse <- RMSE(xgb_3rd_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses,
                         data_frame(Model = "caret_xgbTree_3rd_tune", RMSE = xgb_3rd_tuning_rmse))

model_rmses %>% knitr::kable()




### Fitting the final XGBoost model on the entire train data ###

# We select all relevant predictors
variables <- names(subset(train, select = -c(Id, SalePrice)))

# Devise a "treatment plan"" for the variables.
treatment_plan <- designTreatmentsZ(train, variables, verbose = FALSE) 

newvars <- treatment_plan %>%
  use_series(scoreFrame) %>%      
  filter(code %in% c("clean", "lev")) %>%  
  use_series(varName)

# The prepare() function prepares our data subsets according to the treatment plan
# we devised above and encodes all relevant variables "newvars" numerically.
train_treated <- vtreat::prepare(treatment_plan, train,  varRestriction = newvars)
test_treated <- vtreat::prepare(treatment_plan, test,  varRestriction = newvars)

# We set the final tuning parameters.
tuneGrid <- expand.grid(
  nrounds = seq(150, 2500, 50),
  max_depth = xgb_1st_tuning$bestTune$max_depth,
  eta = xgb_3rd_tuning$bestTune$eta,
  gamma = 0,
  colsample_bytree = xgb_2nd_tuning$bestTune$colsample_bytree,
  min_child_weight = xgb_1st_tuning$bestTune$min_child_weight,
  subsample = xgb_2nd_tuning$bestTune$subsample
)

# Train control for caret train() function. We use k-fold cross-validation.
train_control <- caret::trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = FALSE,
  allowParallel = TRUE
)

# Run the model with above parameters.
xgb_final_model <- caret::train(
  x = train_treated,
  y = train$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = FALSE,
  preProcess = c("nzv", "center", "scale")
)

# Visualization of the final fitted model.
ggplot(xgb_final_model) + scale_y_continuous(limits = c(0.1225, 0.14))

# Lowest RMSE obtained in cross-validation.
min(xgb_final_model$results$RMSE)

# Predicting on test. We take the exp() as we log-transformed "SalePrice".
xgb_final_model_pred <- exp(predict(xgb_final_model, as.matrix(test_treated)))

# We create the submission file.
my_submission <- data.frame(Id = test$Id, SalePrice = xgb_final_model_pred)

write.table(my_submission, file = "submission.csv",
            col.names = TRUE,
            row.names = FALSE,
            sep = ",")