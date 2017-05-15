
# Start and connect to a H2O cluster (JVM)
suppressPackageStartupMessages(library(h2o))
h2o.init(nthreads = -1)

# Import pre-processed datasets

# locally (if you have the datasets in the 'data' sub-folder)
# hex_train <- h2o.importFile("./data/train.csv.gz")
# hex_test <- h2o.importFile("./data/test.csv.gz")

# or directly from the web (github)
hex_train <- h2o.importFile("https://github.com/woobe/h2o_demos/blob/master/human_activitiy_recognition_with_smartphones/data/train.csv.gz?raw=true")
hex_test <- h2o.importFile("https://github.com/woobe/h2o_demos/blob/master/human_activitiy_recognition_with_smartphones/data/test.csv.gz?raw=true")

# Quick summary of train dataset
dim(hex_train)
head(hex_train)
summary(hex_train$activity, exact_quantiles=TRUE)

# Quick summary of test dataset
dim(hex_test)
head(hex_test)
summary(hex_test$activity, exact_quantiles=TRUE)

# Define target and features for model training
target <- "activity"
features <- setdiff(colnames(hex_train), target) # i.e. using the records of all 561 sensors

# Build a GBM model
model <- h2o.gbm(x = features,
                 y = target,
                 training_frame = hex_train,                 
                 model_id = "h2o_gbm",
                 ntrees = 1000,
                 learn_rate = 0.05,
                 learn_rate_annealing = 0.999,
                 max_depth = 7,
                 sample_rate = 0.9,
                 col_sample_rate = 0.9,
                 nfolds = 3,
                 fold_assignment = "Stratified",
                 stopping_metric = "logloss",
                 stopping_rounds = 5,
                 score_tree_interval = 10,
                 #balance_classes = TRUE,
                 seed = 1234)

# Print out model summary
model

# Make predictions
yhat_test <- h2o.predict(model, hex_test)
head(yhat_test)

# Evaluate predictions
h2o.performance(model, newdata = hex_test)
