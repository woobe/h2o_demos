
# Pre-load all R packages
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(h2o))
suppressPackageStartupMessages(library(plotly))

# Start and connect to a H2O cluster (JVM)
h2o.init(nthreads = -1)

# Import pre-processed datasets

# locally (if you have the datasets in the 'data' sub-folder)
# hex_train <- h2o.importFile("./data/train.csv.gz")
# hex_test <- h2o.importFile("./data/test.csv.gz")

# or directly from the web (github)
hex_train <- h2o.importFile("https://github.com/woobe/h2o_demos/blob/master/human_activitiy_recognition_with_smartphones/data/train.csv.gz?raw=true")
hex_test <- h2o.importFile("https://github.com/woobe/h2o_demos/blob/master/human_activitiy_recognition_with_smartphones/data/test.csv.gz?raw=true")

# Dimensions
# 'Train' dataset has 7352 rows and 562 columns
# 'Test' dataset has 2947 rows and 562 columns
dim(hex_train)
dim(hex_test)

# First few records
# First column is the label 'activity'
# Rest of the columns (V1 to V561) are sensors data
head(hex_train)
head(hex_test)

# Look at 'activity' column
# Six classes (Carinality = 6)
# No missing value
h2o.describe(hex_train$activity)
h2o.describe(hex_test$activity)

# Extract 'activity' columns for other graphics packages in R
d_activity_train <- as.data.frame(hex_train$activity)
d_activity_test <- as.data.frame(hex_test$activity)

# Count acitivity 
d_freq_train <- as.data.frame(table(d_activity_train))
d_freq_test <- as.data.frame(table(d_activity_test))
d_freq <- merge(d_freq_train, d_freq_test, by.x = "d_activity_train", by.y = "d_activity_test", sort = FALSE)
colnames(d_freq) <- c("activity", "freq_train", "freq_test")
d_freq

# Visualize 'activity' in both 'train' and 'test'
p <- plot_ly(d_freq, x = ~activity, y = ~freq_train, type = 'bar', name = 'Frequency (Train)') %>%
  add_trace(y = ~freq_test, name = 'Frequency (Test)') %>%
  layout(title = "Activities in 'Train' and 'Test' Dataset") %>%
  layout(yaxis = list(title = 'Count'), xaxis = list(title = "")) %>%
  layout(margin = list(b = 90)) %>%
  layout(barmode = "group")
p

# Look at relationship between sensor data V1 and activity
d_v1 <- data.frame(V1_train = as.data.frame(hex_train$V1), activity = as.data.frame(hex_train$activity))
head(d_v1)

p <- plot_ly(d_v1, y = ~V1, color = ~activity, type = "box") %>%
     layout(title = "Relationship between Sensor Data V1 and Activities") %>%
     layout(yaxis = list(title = 'Sensor Data V1'), xaxis = list(title = "")) %>%
     layout(margin = list(b = 90))
p

# Principal Component Analysis
# 95% of variance in original data captured by first five principal components
suppressWarnings(
    model_pca <- h2o.prcomp(training_frame = hex_train, 
                        x = 2:562, 
                        model_id = "h2o_pca",
                        k = 5)    
)
model_pca                     

# Visualize principle components with activity labels
d_pca <- as.data.frame(h2o.predict(model_pca, hex_train))
d_pca <- data.frame(d_pca, as.data.frame(hex_train$activity))
head(d_pca)

p <- plot_ly(data = d_pca, x = ~PC2, y = ~PC3, color = ~activity, 
             type = "scatter", mode = "markers", marker = list(size = 3)) %>%
     layout(title = "Visualizing Principle Components")
p

# Define target and features for model training
target <- "activity"
features <- setdiff(colnames(hex_train), target) # i.e. using the records of all 561 sensors

# Build a GBM model
model <- h2o.gbm(x = features,
                 y = target,
                 training_frame = hex_train,                 
                 model_id = "h2o_gbm",
                 ntrees = 500,
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
                 seed = 1234)

# Print out model summary
model

# Make predictions
yhat_test <- h2o.predict(model, hex_test)
head(yhat_test)

# Evaluate predictions
h2o.performance(model, newdata = hex_test)

# Not Run
# Showing the syntax for now
# h2o.saveMojo(model, path = "")
