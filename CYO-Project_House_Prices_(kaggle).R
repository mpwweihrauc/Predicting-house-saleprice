# Please download the datasets from the provided link or find them in my provided Github repository, links below, or in the report.
# Link to Kaggle page: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link to my Github page: https://github.com/mpwweihrauc/ML_House_Prices.git
# You will need the test.csv and the train.csv files. There is also a data_description.txt with descriptions for all the different parameters in the dataset.

# We begin by loading/installing all required libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org", dependencies = c("Depends", "Suggests"))
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(vtreat)) install.packages("vtreat", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(vip)) install.packages("vip", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")


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


# To facilitate feature engineering and data cleaning we temporarily merge train and tet into "dataset".
test$SalePrice <- 0 # Temporarily add SalePrice column with 0s to test
dataset <- rbind(train, test) # Merge train and test
summary(dataset) # There's a lot of NA values in many columns

# We will now systematically analyse each feature of the dataset.
# Whenever we work with the SalePrice variable, we will subset "dataset" with the train$Id, as "test" has
# no entries for it.

#####
# Feature 1: MSSubClass
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
# Feature 2: MSZoning
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

# We will impute the most common value, "RL", for the houses with missing values in MSZoning.
dataset$MSZoning[is.na(dataset$MSZoning)] <- "RL"

#####
# Feature 3: LotFrontage
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

# We use kNN-based imputation for Exterior1st and 2nd.
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

#####
# Feature 4: LotArea
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
  ggtitle("Scatterplot of LotArea vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x))

LotArea_scatterplot_log <- dataset[train$Id, ] %>%
  ggplot(aes(x = log1p(LotArea), y = SalePrice)) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  geom_point(alpha = 0.3) +
  ggtitle("Scatterplot of LotArea vs. SalePrice") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  xlab("Log-transformed LotArea")

grid.arrange(LotArea_scatterplot, LotArea_scatterplot_log, nrow = 2)

# We calculate the correlation of LotArea with sale price.
# LotArea is quite predictive of sale price.
cor(dataset[train$Id, ]$LotArea, dataset[train$Id, ]$SalePrice)



#####
# Feature 5: Street
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
# Feature 6: Alley
# Alley: Type of alley access to property
#####

# There is a very large amount of missing values in Alley.
# From the data description, "No alley access" was encoded as NA.
# These NA entries are producing false missing value entries.
summary(dataset$Alley)

# We fix these wrong NA entries by replacing them with "None"
dataset$Alley <- str_replace_na(dataset$Alley, replacement = "None")
dataset$Alley <- factor(dataset$Alley, levels = c("None", "Grvl", "Pave"))

#####
# Feature 7: LotShape
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
# Feature 8: LandContour
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
# Feature 9: Utilities
# Utilities: Type of utilities available
#####

# There are a few missing values in Utilities.
summary(dataset$Utilities)

# LandContour contains no missing values.
summary(dataset$LandContour)

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
# Feature 10: LotConfig
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
# Feature 11: LandSlope
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
  ggtitle("Boxplot of LandSlope vs. SalePrice")

# Scatterplot of LandSlope vs. SalePrice
LandSlope_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = LandSlope, y = SalePrice, color = LandSlope)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of LandSlope vs. SalePrice")

grid.arrange(LandSlope_boxplot, LandSlope_scatterplot, nrow = 1)

# From the plots we can see that a gentle land slope can be predictive of a higher house sale price.

#####
# Feature 12: Neighbourhood
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
# Feature 13: Condition1
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
# Feature 14: Condition2
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
# Feature 15: BldgType
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
# Feature 16: HouseStyle
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
# Feature 17: OverallQual
# OverallQual: Rates the overall material and finish of the house
#####

# There are no missing values in OverallQual
summary(dataset$OverallQual)

# OverallQual should be an ordinal factor variable, not an integer.
dataset$OverallQual <- factor(dataset$OverallQual)


# Boxplot of OverallQual vs. SalePrice.
OverallQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = OverallQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallQual vs. SalePrice")

OverallQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = OverallQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallQual vs. SalePrice")

grid.arrange(OverallQual_boxplot, OverallQual_scatterplot, nrow = 1)

# OverallQual has tremendous influence on a houses sale price. In the highest category, 10, there are 2 houses with
# unexpectedly low, and extremely high sale prices. This increases the range a lot.
# OverallQual will be very important for predicting a houses value.

# There are only a few houses in the lowest OverallQual categories 1 and 2. Additionally, the higher quality levels 
# show quite a large range of values.
# It would be beneficial to bin OverallQual into a lower number of categories, by combining several levels into one.
dataset$OverallQual <- as.numeric(dataset$OverallQual)

dataset$OverallQual[dataset$OverallQual %in% c(1:2)] <- "Poor"
dataset$OverallQual[dataset$OverallQual %in% c(3:4)] <- "Fair"
dataset$OverallQual[dataset$OverallQual %in% c(5:6)] <- "Average"
dataset$OverallQual[dataset$OverallQual %in% c(7:8)] <- "Good"
dataset$OverallQual[dataset$OverallQual %in% c(9:10)] <- "Excellent"

dataset$OverallQual <- factor(dataset$OverallQual, levels = c("Poor", "Fair", "Average", "Good", "Excellent"))

# Boxplot of engineered OverallQual vs. SalePrice.
OverallQual_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = OverallQual)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered OverallQual vs. SalePrice")

OverallQual_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallQual, y = SalePrice, color = OverallQual)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered OverallQual vs. SalePrice")

grid.arrange(OverallQual_boxplot, OverallQual_scatterplot, nrow = 1)

# The engineered OverallQual levels are now very distinct in sale price.


#####
# Feature 18: OverallCond
# OverallCond: Rates the overall condition of the house
#####

# There are no missing values in OverallCond
summary(dataset$OverallCond)

# OverallCond should be an ordinal factor variable, not an integer.
dataset$OverallCond <- factor(dataset$OverallCond)


# Boxplot of OverallCond vs. SalePrice.
OverallCond_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = OverallCond)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallCond vs. SalePrice")

OverallCond_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = OverallCond)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of OverallCond vs. SalePrice")

grid.arrange(OverallCond_boxplot, OverallCond_scatterplot, nrow = 1)

# Contrary to intuition, OverallCond is much less predictive of a houses sale price and there is not a single house with a "very excellent" condition.
# We observe that the most expensive houses are between 4 - 9, with 5 having most of them. OverallCond below 5 holds houses with lower sale price.
# The lower two levels hold only very few houses.

# It makes sense to bin this feature into a lower number of levels.
# OverallCond 1-4 will become "Bad", OverallCond 5-7 will become "Good" and OverallCond >= 8 will become "Excellent".
dataset$OverallCond <- as.numeric(dataset$OverallCond)
dataset$OverallCond[dataset$OverallCond %in% c(1:4)] <- "Bad"
dataset$OverallCond[dataset$OverallCond %in% c(5:7)] <- "Good"
dataset$OverallCond[dataset$OverallCond %in% c(8:9)] <- "Excellent"

dataset$OverallCond <- factor(dataset$OverallCond, levels = c("Bad", "Good", "Excellent"))

# Boxplot of engineered OverallCond vs. SalePrice.
OverallCond_boxplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = OverallCond)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered OverallCond vs. SalePrice")

OverallCond_scatterplot <- dataset[train$Id, ] %>%
  ggplot(aes(x = OverallCond, y = SalePrice, color = OverallCond)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(labels = scales::comma, breaks = seq(0, 800000, 100000)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Boxplot of engineered OverallCond vs. SalePrice")

grid.arrange(OverallCond_boxplot, OverallCond_scatterplot, nrow = 1)

# We now have less categories for OverallCond, with more houses in each of them. A bad OverallCond can be idnicative of a lower sale price.







#####
# Feature 19: YearBuilt
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
# Feature 20: YearRemodAdd
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

# There is also a large correlation between YearRemodAdd and YearBuilt. In fact, it is one.
cor(dataset$YearRemodAdd, dataset$YearRemodAdd)

# From the plot we can see that YearRemodAdd has a very similar correlation with SalePrice compared with YearBuilt.
# Remodellings were only recorded starting 1950.
# YearRemodAdd correaltes 100% with YearBuilt. We plot the two variables against eachother.

# Scatterplot of YearBuilt vs. YearRemodAdd. 
dataset[train$Id, ] %>%
  ggplot(aes(x = YearRemodAdd, y = YearBuilt)) +
  geom_point(alpha = 0.3) +
  scale_y_continuous(breaks = seq(1870, 2010, 10)) +
  scale_x_continuous(breaks = seq(1870, 2010, 10)) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Scatterplot of YearRemodAdd vs. YearBuilt")


# From the plot it seems that for all houses built before 1950, even if there actually wasn't any remodelling done, they received an entry in YearRemodAdd at 1950.
# Most houses have identical YearBuilt and YearRemodAdd values, since there were no remodellings done.
# For predictive purposes, this variable isn't very helpful.
dataset <- subset(dataset, select = -YearRemodAdd)


#####
# Feature 21: RoofStyle
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
# Feature 22: RoofMatl
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
# Feature 23: Exterior1st
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
# Feature 24: Exterior2nd
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


# Feature 25: MasVnrType
# Feature 26: MasVnrArea

















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

# Define a grid of tuning parameters for the caret::train() function
# We select some empirically optimized parameters
grid_default <- expand.grid(
  nrounds = seq(from = 100, to = 800, 50), 
  max_depth = 4,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.75, 
  min_child_weight = 1, 
  subsample = 1 
)

# Train control for caret train() function; we use cross-validation to estimate out-of-sample error
train_control <- caret::trainControl(
  method = "repeatedcv", # We use 5-fold cross-validation
  number = 3,
  repeats = 3,
  verboseIter = TRUE,
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

ggplot(xgb_LF_tuned) # We use our defined function to visualize the tuning effects


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
dataset$BsmtExposure[is.na(dataset$BsmtExposure)] <- "No"
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
dataset$BsmtExposure <- as.numeric(factor(dataset$BsmtExposure, levels=c("No", "Mn", "Av", "Gd")))
dataset$HeatingQC <- as.numeric(factor(dataset$HeatingQC, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$KitchenQual <- as.numeric(factor(dataset$KitchenQual, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$FireplaceQu <- as.numeric(factor(dataset$FireplaceQu, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$GarageQual <- as.numeric(factor(dataset$GarageQual, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$GarageCond <- as.numeric(factor(dataset$GarageCond, levels=c("None","Po", "Fa", "TA", "Gd", "Ex")))
dataset$PoolQC <- as.numeric(factor(dataset$PoolQC, levels=c("None", "Fa", "TA", "Gd", "Ex")))

dataset$OverallCond <- as.numeric(dataset$OverallCond)
dataset$OverallQual <- as.numeric(dataset$OverallQual)

#####
#####





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


# There are 4 separate features describing porches: Open, enclosed, 3-season, and screened.
# We look at the individual correlation with SalePrice and build a combined feature "TotalPorch".
cor(dataset[train$Id, ]$OpenPorchSF, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$EnclosedPorch, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$X3SsnPorch, dataset[train$Id, ]$SalePrice)
cor(dataset[train$Id, ]$ScreenPorch, dataset[train$Id, ]$SalePrice)

dataset$TotalPorch <- as.numeric(dataset$OpenPorchSF + dataset$EnclosedPorch + dataset$X3SsnPorch + dataset$ScreenPorch)

cor(dataset[train$Id, ]$TotalPorch, dataset[train$Id, ]$SalePrice)

# We remove the individual features

dataset <- subset(dataset, select = -c(OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch))





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
  ggplot(aes(x = MoSold, y = exp(SalePrice), group = MoSold)) +
  geom_boxplot() +
  ylab("Sale price") +
  xlab("Month") +
  scale_x_continuous(breaks = seq(1,12,1)) +
  ggtitle("Distribution of SalePrice over MoSold") +


  # We plot the amount of sold houses per year. The amount is relatively similar, but much less entries in 2010.
  dataset %>%
  group_by(YrSold) %>%
  count() %>%
  ggplot(aes(x = YrSold, y = n)) +
  geom_line(col = "blue") +
  ylab("Number of houses sold") +
  xlab("Year") +


# YrSold doesn't seem to influence SalePrice at all.
dataset[train$Id, ] %>%
  group_by(YrSold) %>%
  ggplot(aes(x = YrSold, y = exp(SalePrice), group = YrSold)) +
  geom_boxplot() +
  ylab("Sale price") +
  xlab("Year") +
  ggtitle("Distribution of SalePrice over YrSold")
  
  
# As MoSold and YrSold have no predictive power over SalePrice, we drop them.
dataset <- subset(dataset, select = -c(MoSold, YrSold))
  
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
# test_set will receive 20% of the data, train_set will receive 80%.
# The test_set will serve as a "hold-out" set to check algorithm performance.
set.seed(1442)
test_index <- createDataPartition(train$SalePrice, p = 0.2, list = FALSE)
train_set <- train[-test_index, ]
temp <- train[test_index, ] # temporary test set

# We make sure there are no entries in test_set that aren't in train_set
test_set <- temp %>%
  semi_join(train_set, by = c("RoofMatl", "Exterior1st", "Exterior2nd", "Electrical", "Functional",
                              "MiscFeature", "Condition2")) # Variables were determined by trial and error with the lm() models below
# We return the removed entries from test to train
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

summary(train_set)
summary(test_set)


# 1. Model 1: Simple linear regression as a baseline.
# As our first, very simple model we predict house sale price via linear regression of the LotArea.
# We generate a table to keep track of the RMSEs our various models generate.

model_1_lm <- lm(SalePrice ~ OverallQual, data = train_set) # Linear regression with a single predictor
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
# Keep in mind that this model has not been regularized at all and most likely heavily overfits the training data.

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
             nrounds = 200, # We go up to 200 rounds of fitting models on the remaining residuals
             nfold = 5, # We use 5 folds for cross-validation
             objective = "reg:linear",
             eta = 0.3, # The learning rate; Closer to 0 is slower, but less prone to overfitting; Closer to 1 is faster, but more likely to overfit
             max_depth = 6,
             early_stopping_rounds = 20,
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


#####
#####



#####
# 1st set of hyperparameter tuning parameters for learning rate and maximum tree depth of XGBoost
#####


# We test different values for maximum tree depth and learning rate. 
# Learning rate can easily become too high, while too deep trees can make the model too complex and overfit.
# We keep the other parameters at their default for now.
tuneGrid <- expand.grid(
  nrounds = seq(200, 1000, 100),
  max_depth = c(2, 3, 4, 8),
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# Train control for caret train() function.
train_control <- trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE
)

# Run the model with above parameters.
xgb_1st_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = TRUE
)


# # Visualization of the 1st tuning round and the most important features
ggplot(xgb_1st_tuning)

vip(xgb_1st_tuning, num_features = 20) # We take a look at the most important features

# We can select the best tuning values from the model like this
xgb_1st_tuning$bestTune

# We predict on the test_set and record the "out-of-bag" RMSE
xgb_1st_tuning_pred <- predict(xgb_1st_tuning, test_set_treated)

xgb_1st_tuning_rmse <- RMSE(xgb_1st_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_4_xgb_1st_tune", RMSE = xgb_1st_tuning_rmse))

model_rmses %>% knitr::kable()


#####
#####


#####
# 2nd set of hyperparameter tuning parameters
#####

# We evaluate min_cild_weight values and check for a limited max tree depth with fixed learning rate
tuneGrid <- expand.grid(
  nrounds = seq(200, 1000, 100),
  max_depth = 2,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = seq(0.2, 0.7, 0.1),
  min_child_weight = 1,
  subsample = seq(0.4, 0.9, 0.1)
)

# Train control for caret train() function. We use 5-fold cross-validation.
train_control <- caret::trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE
)

# Run the model with above parameters.
xgb_2nd_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = TRUE
)


# # Visualization of the 2nd tuning round and the most important features
ggplot(xgb_2nd_tuning)

vip(xgb_2nd_tuning, num_features = 20) # We take a look at the most important features

# We can select the best tuning values from the model like this
xgb_2nd_tuning$bestTune

# We predict on the test_set and record the "out-of-bag" RMSE
xgb_2nd_tuning_pred <- predict(xgb_2nd_tuning, test_set_treated)

xgb_2nd_tuning_rmse <- RMSE(xgb_2nd_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_5_xgb_2nd_tune", RMSE = xgb_2nd_tuning_rmse))

model_rmses %>% knitr::kable()



#####
# 3rd set of hyperparameter tuning parameters
#####

# We evaluate column sampling with fixed depth, child weight and learning rate
tuneGrid <- expand.grid(
  nrounds = seq(200, 1000, 100),
  max_depth = 2,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.4,
  min_child_weight = seq(1, 8, length = 6),
  subsample = 0.9
)

# Train control for caret train() function. We use 5-fold cross-validation.
train_control <- caret::trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE
)

# Run the model with above parameters.
xgb_3rd_tuning <- train(
  x = train_set_treated,
  y = train_set$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = TRUE
)


# Visualization of the 3rd tuning round and the most important features
ggplot(xgb_3rd_tuning)

vip(xgb_3rd_tuning, num_features = 20) # We take a look at the most important features

# We can select the best tuning values from the model like this
xgb_3rd_tuning$bestTune

# We predict on the test_set and record the "out-of-bag" RMSE
xgb_3rd_tuning_pred <- predict(xgb_3rd_tuning, test_set_treated)

xgb_3rd_tuning_rmse <- RMSE(xgb_3rd_tuning_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_6_xgb_3rd_tune", RMSE = xgb_3rd_tuning_rmse))

model_rmses %>% knitr::kable()



#####
# 4th set of hyperparameter tuning parameters
#####

# We evaluate gamma and a few values of column sampling
tuneGrid <- expand.grid(
  nrounds = seq(100, 3000, 100),
  max_depth = 3,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = c(0.25, 0.5, 0.75, 1),
  min_child_weight = 1,
  subsample = 0.75
)

# Train control for caret train() function. We use 5-fold cross-validation.
train_control <- caret::trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 3,
  verboseIter = TRUE
)

# Run the model with above parameters.
xgb_final_tuning <- train(
  x = train_treated,
  y = train$SalePrice,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree",
  verbose = TRUE
)


# We plot the final thuning
ggplot(xgb_final_tuning)

# We can select the best tuning values from the model like this
xgb_final_tuning$bestTune

# We record the RMSE of the best tuned model
model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_7_xgb_final_tune", RMSE = min(xgb_final_tuning$results$RMSE)))

model_rmses %>% knitr::kable()

### Prediction on test with the final tune XGB model
xgb_final_tuning_pred <- exp(predict(xgb_final_tuning, as.matrix(test_treated)))

my_submission <- data.frame(Id = test$Id, SalePrice = xgb_final_tuning_pred)

write.table(my_submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")


### Fitting the final XGBoost model on the entire train data ###

# We select all relevant predictors
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
test_treated <- prepare(treatment_plan, test,  varRestriction = newvars)




########################
# Ranger random forest model
########################

# We conduct a first hyperparameter-search using Ranger to understand how the tuning parameters behave.
# For the ranger random forest algorithm, the only tunable hyperparameter is "mtry" (Number of randomly chosen variables to possibly split at in each node).

tuneGrid <- data.frame(
  .mtry = c(2,4,6,8,10,12,24,48,60,65), # Number of randomly chosen variables to possibly split at in each node; we screen values of 2 until 65 (the total number of predictors).
  .splitrule = "variance", # Default splitrule variance for regression
  .min.node.size = 5 # Default minimum node size for regression
)

ranger_model <- train(SalePrice ~ .,
                      method = "ranger",
                      data = subset(train_set, select = -Id),
                      trControl = trainControl(method = "cv",
                                               number = 5, verboseIter = TRUE),
                      tuneGrid = tuneGrid,
                      num.trees = 2000
                      )

print(ranger_model)

# The higher "mtry", the more complex the model gets. High values of "mtry" might lead to serious overfitting of the training data.
# We plot "mtry" against the cross-validated RMSE. We can see a clear "elbow" at around an mtry of 6 to 8.
plot(ranger_model)

ranger_model_pred <- predict(ranger_model, newdata = test_set) # We predict woth the ranger_model on test_set

ranger_model_rmse <- RMSE(ranger_model_pred, test_set$SalePrice)

model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_5_ranger_overtuned", RMSE = ranger_model_rmse))

model_rmses %>% knitr::kable()

###

# To prevent potential overfitting, we will choose the "mtry" at the "elbow" of the plot. After this value, RMSE decreases much slower with ever-increasing "mtry" values.
# Coincidentally, the default value for "mtry" in Ranger is the square root of the number of predictors (sqrt(65) = 8).

tuneGrid <- data.frame(
  .mtry = sqrt(ncol(train_set)), # Number of randomly chosen variables to possibly split at in each node; we use a value of 8 as determined above
  .splitrule = "variance", # Default splitrule variance for regression
  .min.node.size = 5 # Default minimum node size for regression
)

ranger_model2 <- train(SalePrice ~ .,
                      method = "ranger",
                      data = subset(train, select = -Id),
                      trControl = trainControl(method = "repeatedcv",
                                               number = 5, repeats = 3, verboseIter = TRUE),
                      tuneGrid = tuneGrid,
                      num.trees = 3000,
                      )

print(ranger_model2)

ranger_model2_pred <- predict(ranger_model2, newdata = test) # We predict woth the ranger_model on test_set


model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_6_ranger_properly_tuned", RMSE = ranger_model2_rmse))

model_rmses %>% knitr::kable()



###########################
# Glmnet - Lasso and/or Ride regression: Elastic Net Regression
###########################


# Train() fits only one model per Alpha value and passes all Lambda values at once for simultaneous fitting.
# We will explore pure Ridge, pure Lasso, and mixes between the two in 5% steps (e.g. 10% ridge and 90 % lasso regression etc... )

# tuneGrid for Elastic Net Regression
tuneGrid <- expand.grid(
  alpha = seq(0, 1, 0.05), # Mixing parameter between Lasso (L1) and Ridge (L2) regression; alpha = 0 equals pure Ridge regression, alpha = 1 equals pure Lasso regression
  lambda = seq(0.0001, 1, length = 100) # Strength of the penalty on the coefficients; A Lambda of 1 would shrink regression coefficients to 0, so that the model would only predict the intercept
)

myControl <- trainControl(
  method = "repeatedcv", number = 5, repeats = 3,
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

glm_model_pred <- predict(glm_model, newdata = test) 


model_rmses <- bind_rows(model_rmses, data_frame(Model = "Model_6_glmnet", RMSE = glm_model_rmse))

model_rmses %>% knitr::kable()

#####









# We fit the final Glmnet model #







my_ensemble <- data.frame(Id = test$Id, SalePrice = (xgb_final_tuning_pred + exp(ranger_model2_pred))/2)

write.table(my_ensemble, file = "submission_ensemble_2.csv", col.names = TRUE, row.names = FALSE, sep = ",")


