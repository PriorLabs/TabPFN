### --------- Installation and setup ---------

# Install CASdatasets - note this can no longer be done from CRAN
install.packages("CASdatasets", repos = "http://dutangc.perso.math.cnrs.fr/RRepository/pub/", type = "source")

#Install TabPFN
devtools::install_github("robintibor/R-tabpfn")
library(tabpfn)
install_tabpfn()

# Load required libraries
library(CASdatasets)
library(dplyr)
library(ggplot2)
library(lightgbm)
library(tabpfn)
library(tidyr)
library(gains)

#TabPFN setup - you will need an access token
#NB this access token is your own private token, do not share or commit to repo in error!!!
#sign up for TabPFN website, 
#go to Account Settings > Access Token >Your API access token. Keep this secure and do not share it with others.
#copy and paste this token below as a string between inverted commas
access_token = "Paste_Your_Token_Here"

# Load required datasets
data(freMTPL2freq); data(freMTPL2sev)

# Set RNG seed to ensure reproducibility
set.seed(100)

### --------- Dataset: freMTPL2freq + freMTPL2sev ---------

# Clean frequency dataset
# We want to ensure categorical vars are treated as factors
# Policies with exposures less than 1 month or greater than 1 year are removed
# Any cases where ClaimNb > 5 are considered unusual and removed

freMTPL2freq_clean <- freMTPL2freq %>%
  mutate(
    VehPower = as.factor(VehPower),
    VehBrand = as.factor(VehBrand),
    VehGas = as.factor(VehGas),
    Area = as.factor(Area),
    Region = as.factor(Region),
    IDpol = as.character(IDpol)  # Ensure IDpol is a string for consistent joining
  ) %>%
  filter(Exposure >= 1/12, Exposure <= 1.0, ClaimNb <= 5)

# Clean severity dataset
# Claim severity of less than 50 or greater than 10k is removed

freMTPL2sev_clean <- freMTPL2sev %>%
  filter(ClaimAmount >= 50, ClaimAmount <= 10000) %>%
  mutate(IDpol = as.character(IDpol))  # Ensure join works with freq_clean

# Join with frequency data
# Note: inner join to ensure no missing explanatory variables or claim severity
# Create a unique ClaimID, a random group for potential CV use, and a helper ClaimCount_row

sev_model_data2 <- freMTPL2sev_clean %>%
  inner_join(freMTPL2freq_clean, by = "IDpol") %>%
  mutate(
    ClaimCount_row = 1,
    ClaimID = paste0("C2_", row_number()),
    RandGroup = sample(1:5, n(), replace = TRUE)
  )

# Get unique IDpol values in sev_model_data2
# We split by IDpol, not ClaimID, to prevent data leakage
# If a policy had multiple claims, all its claims go into either train or test

unique_ids2 <- unique(sev_model_data2$IDpol)

# Random 60/40 split by IDpol
train_ids2 <- sample(unique_ids2, size = 0.6 * length(unique_ids2), replace = FALSE)
test_ids2 <- setdiff(unique_ids2, train_ids2)

# Create train/test sets directly from severity modeling data
train_data2 <- sev_model_data2[sev_model_data2$IDpol %in% train_ids2, ]
test_data2  <- sev_model_data2[sev_model_data2$IDpol %in% test_ids2, ]

# Confirm no IDpol values appearing across both train and test
# Could split on ClaimID instead, but better practice is to split on IDpol
length(intersect(train_data2$IDpol, test_data2$IDpol))  # Should be 0

### --------- EDA for Severity Data ---------

# 1. Distribution of Claim Amounts
ggplot(train_data2, aes(x = ClaimAmount)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Claim Amounts", x = "Claim Amount", y = "Count")

# 2. Claim Amount by Region
ggplot(train_data2, aes(x = Region, y = ClaimAmount)) +
  geom_boxplot(fill = "lightgreen") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Claim Amount by Region", x = "Region", y = "Claim Amount")

# 3. Claim Amount by VehPower
ggplot(train_data2, aes(x = VehPower, y = ClaimAmount)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Claim Amount by Vehicle Power", x = "VehPower", y = "Claim Amount")

# 4. Helper function for average claim with SE, volume, and low-n highlight
plot_mean_se <- function(data, age_var, title) {
  summary_data <- data %>%
    group_by({{age_var}}) %>%
    summarise(
      mean_claim = mean(ClaimAmount),
      sd_claim = sd(ClaimAmount),
      n = n(),
      se_claim = sd_claim / sqrt(n),
      low_volume = n < 5
    )
  
  ggplot(summary_data, aes(x = {{age_var}}, y = mean_claim)) +
    geom_col(aes(y = n * max(mean_claim) / max(n)), fill = "grey90", width = 1) +  # scaled volume
    geom_line(color = "steelblue", linewidth = 1) +
    geom_errorbar(aes(ymin = mean_claim - se_claim, ymax = mean_claim + se_claim),
                  width = 0.3, color = "darkgrey") +
    geom_point(aes(color = low_volume), size = 2) +
    scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"), labels = c("Sufficient n", "Low n")) +
    theme_minimal() +
    labs(
      title = title,
      x = as_label(enquo(age_var)),
      y = "Mean Claim Amount (± SE)",
      color = "Group Size"
    )
}

# 5. Average claim amount by DrivAge
plot_mean_se(train_data2, DrivAge, "Average Claim Amount by Driver Age (with Volume)")

# 6. Average claim amount by VehAge
plot_mean_se(train_data2, VehAge, "Average Claim Amount by Vehicle Age (with Volume)")

#Similar spike observed - allow this to remain for the moment

### --------- Gamma GLM for Severity Data (train_data2) ---------

# Fit Gamma GLM with log link on severity data (dataset 2)
glm_gamma_2 <- glm(
  ClaimAmount ~ VehPower + VehAge + DrivAge + VehBrand + VehGas + Region + Density,
  data = train_data2,
  family = Gamma(link = "log")
)

# Generate Gamma GLM predictions on test set
test_data2$glm_pred <- predict(glm_gamma_2, newdata = test_data2, type = "response")

### --------- LightGBM for Severity Data ---------

# Step 1: Prepare features and labels (convert factors to integers)
prepare_lgb_data2 <- function(data) {
  data <- data %>%
    mutate(across(c(VehPower, VehBrand, VehGas, Region), as.integer))  # Convert categorical to integers
  label <- data$ClaimAmount  # Target variable
  features <- data %>%
    select(VehPower, VehAge, DrivAge, VehBrand, VehGas, Region, Density)  # Features to include
  return(list(data = as.matrix(features), label = label))
}

train_lgb2 <- prepare_lgb_data2(train_data2)
test_lgb2 <- prepare_lgb_data2(test_data2)

# Step 2: Specify which variables are categorical
categorical_vars2 <- c("VehPower", "VehBrand", "VehGas", "Region")

# Step 3: Create LightGBM datasets
dtrain2 <- lgb.Dataset(
  data = train_lgb2$data,
  label = train_lgb2$label,
  categorical_feature = categorical_vars2  # Treat these as unordered factors
)

dtest2 <- lgb.Dataset(
  data = test_lgb2$data,
  label = test_lgb2$label,
  categorical_feature = categorical_vars2
)

# Step 4: Set model parameters
params2 <- list(
  objective = "regression",    # Predicting a continuous target
  metric = "l2",               # L2 loss = mean squared error
  learning_rate = 0.1,         # Step size during training
  num_leaves = 31,             # Tree complexity control
  verbosity = -1               # Suppress training logs
)

# Step 5: Train the model
lgb_model2 <- lgb.train(
  params = params2,            # Model settings
  data = dtrain2,              # Training data
  nrounds = 100,               # Max boosting iterations
  valids = list(test = dtest2),# Validation set for early stopping
  early_stopping_rounds = 10,  # Stop if no improvement for 10 rounds
  verbose = 0                  # Silence per-iteration output
)

# Step 6: Predict on the test set
test_data2$lgb_pred <- predict(lgb_model2, test_lgb2$data)

### --------- TabPFN for Severity Data ---------

# Ensure library loaded and access token is set if not already done
# set_tabpfn_access_token(access_token)

# Define predictor variables
tabpfn_features2 <- c("VehPower", "VehAge", "DrivAge", "VehBrand", "VehGas", "Region", "Density")

# Integer encoding function
encode_as_int <- function(x) as.integer(as.factor(x))

# Step 1: Sample 10,000 rows from training data
train_data2_sampled <- train_data2 %>% sample_n(size = 10000)

# Step 2: Prepare input features (encoded) and labels
X_train_tabpfn2 <- train_data2_sampled[, tabpfn_features2] %>%
  mutate(across(everything(), encode_as_int))

y_train_tabpfn2 <- train_data2_sampled$ClaimAmount

# Step 3: Prepare test set features (same columns)
X_test_tabpfn2 <- test_data2[, tabpfn_features2] %>%
  mutate(across(everything(), encode_as_int))

# Step 4: Fit TabPFN Regressor
regressor2 <- TabPFNRegressor$new()
regressor2$fit(X_train_tabpfn2, y_train_tabpfn2)

# Step 5: Predict on test set
test_data2$tabpfn_pred <- regressor2$predict(X_test_tabpfn2)


### --------- Model Evaluations for Severity (Gamma GLM, LightGBM, TabPFN) ---------

# Define metrics
mae <- function(actual, pred) mean(abs(actual - pred))
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mape <- function(actual, pred) mean(abs((actual - pred) / actual)) * 100
gamma_deviance <- function(y, mu) {
  mu <- pmax(mu, 1e-15)
  y <- pmax(y, 1e-15)
  2 * sum((y - mu) / mu - log(y / mu))
}

### --- Dataset ---

actual2 <- test_data2$ClaimAmount

mae_glm2   <- mae(actual2, test_data2$glm_pred)
rmse_glm2  <- rmse(actual2, test_data2$glm_pred)
mape_glm2  <- mape(actual2, test_data2$glm_pred)
gdev_glm2  <- gamma_deviance(actual2, test_data2$glm_pred)

mae_lgb2   <- mae(actual2, test_data2$lgb_pred)
rmse_lgb2  <- rmse(actual2, test_data2$lgb_pred)
mape_lgb2  <- mape(actual2, test_data2$lgb_pred)
gdev_lgb2  <- gamma_deviance(actual2, test_data2$lgb_pred)

mae_tab2   <- mae(actual2, test_data2$tabpfn_pred)
rmse_tab2  <- rmse(actual2, test_data2$tabpfn_pred)
mape_tab2  <- mape(actual2, test_data2$tabpfn_pred)
gdev_tab2  <- gamma_deviance(actual2, test_data2$tabpfn_pred)

results2 <- data.frame(
  Dataset = "Dataset 2",
  Model = c("Gamma GLM", "LightGBM", "TabPFN Regressor"),
  MAE = c(mae_glm2, mae_lgb2, mae_tab2),
  RMSE = c(rmse_glm2, rmse_lgb2, rmse_tab2),
  MAPE = c(mape_glm2, mape_lgb2, mape_tab2),
  Gamma_Deviance = c(gdev_glm2, gdev_lgb2, gdev_tab2),
  stringsAsFactors = FALSE
)

# Collate results
results_all <- results2

# Round for display
results_rounded <- results_all
results_rounded[, 3:6] <- round(results_rounded[, 3:6], 4)

# Print results
print(results_rounded)

### --------- Model Visualisations ---------

#Let's look at Claim Severity vs Driver Age
#Create Age Band variable (10-year bins)
test_data2 <- test_data2 %>%
  mutate(AgeBand = cut(DriverAge,
                       breaks = seq(15, 100, by = 10),
                       include.lowest = TRUE,
                       right = FALSE,
                       labels = c("15–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85–94")))

#Aggregate by Age Band
ageband_summary <- test_data2 %>%
  group_by(AgeBand) %>%
  summarise(
    Actual = mean(ClaimAmount, na.rm = TRUE),
    GLM = mean(glm_pred, na.rm = TRUE),
    LGB = mean(lgb_pred, na.rm = TRUE),
    TabPFN = mean(tabpfn_pred, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(AgeBand))

#Pivot longer for plotting
ageband_long <- pivot_longer(ageband_summary,
                             cols = c("Actual", "GLM", "LGB", "TabPFN"),
                             names_to = "Model", values_to = "MeanClaim")

#Plot
ggplot(ageband_long, aes(x = AgeBand, y = MeanClaim, color = Model, group = Model)) +
  geom_line(linewidth = 1.1) +
  geom_point(aes(size = Count), alpha = 0.6) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Average Claim Amount by Driver Age Band (Dataset 1)",
    x = "Driver Age Band", y = "Mean Claim Amount",
    size = "Observation Count"
  ) +
  scale_color_manual(values = c("Actual" = "black", "GLM" = "darkorange", "LGB" = "forestgreen", "TabPFN" = "steelblue")) +
  theme(legend.position = "bottom")

#looks like the TabPFN is underestimating the actuals on average!

hist(test_data1$tabpfn_pred)
#histogram of predictions is very concentrated around the data spike - consider removal

resids_1 <- test_data1$tabpfn_pred - test_data1$ClaimAmount
resids_2 <- test_data1$lgb_pred - test_data1$ClaimAmount
resids_3 <- test_data1$glm_pred - test_data1$ClaimAmount

hist(resids_1)
hist(resids_2)
hist(resids_3)