# cleanup

rm(list = ls())      # Removes all objects from the environment
gc()                 # Runs garbage collection to free up memory


### --- Load & Prepare Data ---

library(CASdatasets)
library(dplyr)
library(ggplot2)
library(lightgbm)
library(tabpfn)
library(tidyr)

# Load dataset
data(ausprivauto0405)

# Rename for ease
data_all <- ausprivauto0405

# Look at histogram of outcome variable 
hist(data_all$ClaimAmount)

hist(data_all$ClaimAmount,
     breaks = 100,
     xlim = c(0, 5000),
     col = "lightgray",
     main = "Histogram of Claim Amounts (Zoomed)",
     xlab = "Claim Amount")


# Ensure factors are properly set
data_all <- data_all %>%
  mutate(
    VehAge = as.factor(VehAge),
    VehBody = as.factor(VehBody),
    Gender = as.factor(Gender),
    DrivAge = as.numeric(DrivAge)
  )

set.seed(42)  # Reproducibility

### --- CLASSIFICATION TASK ---

# Classification: Predict ClaimOcc (binary)
class_data <- data_all %>% drop_na(ClaimOcc)

# Sample train/test splits with TabPFN limits
train_class <- class_data %>% sample_n(10000)
test_class <- class_data %>% setdiff(train_class) %>% sample_n(min(10000, nrow(class_data) - 10000))

# Logistic Regression
glm_cls <- glm(ClaimOcc ~ DrivAge + VehValue + VehAge + VehBody + Gender,
               data = train_class, family = binomial())

test_class$glm_pred <- predict(glm_cls, newdata = test_class, type = "response")

# LightGBM Classifier
prepare_lgb_data_cls <- function(data) {
  data <- data %>% mutate(across(c(VehAge, VehBody, Gender), as.integer))
  label <- data$ClaimOcc
  features <- data %>% select(DrivAge, VehValue, VehAge, VehBody, Gender)
  list(data = as.matrix(features), label = label)
}

train_lgb_cls <- prepare_lgb_data_cls(train_class)
test_lgb_cls <- prepare_lgb_data_cls(test_class)

dtrain_cls <- lgb.Dataset(data = train_lgb_cls$data, label = train_lgb_cls$label)

params_cls <- list(
  objective = "binary",
  metric = "binary_logloss",
  learning_rate = 0.1,
  num_leaves = 31,
  verbosity = -1
)

lgb_model_cls <- lgb.train(
  params = params_cls,
  data = dtrain_cls,
  nrounds = 100
)

test_class$lgb_pred <- predict(lgb_model_cls, test_lgb_cls$data)

# TabPFN Classifier
set_tabpfn_access_token("YOUR_ACCESS_TOKEN")
encode_as_int <- function(x) as.integer(as.factor(x))

X_train_tabpfn_cls <- train_class %>% select(DrivAge, VehValue, VehAge, VehBody, Gender) %>%
  mutate(across(everything(), encode_as_int))
X_test_tabpfn_cls <- test_class %>% select(DrivAge, VehValue, VehAge, VehBody, Gender) %>%
  mutate(across(everything(), encode_as_int))

tabpfn_cls <- TabPFNClassifier$new()
tabpfn_cls$fit(X_train_tabpfn_cls, train_class$ClaimOcc)
test_class$tabpfn_pred <- tabpfn_cls$predict(X_test_tabpfn_cls)

### --- REGRESSION TASK ---

sum(ausprivauto0405$ClaimOcc)

# Subset to rows where a claim occurred
reg_data <- data_all %>% filter(ClaimOcc == 1, !is.na(ClaimAmount), ClaimAmount > 0)

# Carry out train/test split
set.seed(42)
train_idx_reg <- sample(seq_len(nrow(reg_data)), size = 0.6 * nrow(reg_data))
train_reg <- reg_data[train_idx_reg, ]
test_reg <- reg_data[-train_idx_reg, ]

# Gamma GLM
glm_reg <- glm(ClaimAmount ~ DrivAge + VehValue + VehAge + VehBody + Gender,
               data = train_reg, family = Gamma(link = "log"))

test_reg$glm_pred <- predict(glm_reg, newdata = test_reg, type = "response")

# LightGBM Regressor
prepare_lgb_data_reg <- function(data) {
  data <- data %>% mutate(across(c(VehAge, VehBody, Gender), as.integer))
  label <- data$ClaimAmount
  features <- data %>% select(DrivAge, VehValue, VehAge, VehBody, Gender)
  list(data = as.matrix(features), label = label)
}

train_lgb_reg <- prepare_lgb_data_reg(train_reg)
test_lgb_reg <- prepare_lgb_data_reg(test_reg)

dtrain_reg <- lgb.Dataset(data = train_lgb_reg$data, label = train_lgb_reg$label)

params_reg <- list(
  objective = "regression",
  metric = "l2",
  learning_rate = 0.1,
  num_leaves = 31,
  verbosity = -1
)

lgb_model_reg <- lgb.train(
  params = params_reg,
  data = dtrain_reg,
  nrounds = 100
)

test_reg$lgb_pred <- predict(lgb_model_reg, test_lgb_reg$data)

# TabPFN Regressor
X_train_tabpfn_reg <- train_reg %>% select(DrivAge, VehValue, VehAge, VehBody, Gender) %>%
  mutate(across(everything(), encode_as_int))
X_test_tabpfn_reg <- test_reg %>% select(DrivAge, VehValue, VehAge, VehBody, Gender) %>%
  mutate(across(everything(), encode_as_int))

tabpfn_reg <- TabPFNRegressor$new()
tabpfn_reg$fit(X_train_tabpfn_reg, train_reg$ClaimAmount)
test_reg$tabpfn_pred <- tabpfn_reg$predict(X_test_tabpfn_reg)

### --- Evaluation Metrics ---
# Classification: Accuracy, AUC, etc.
# Regression: MAE, RMSE, MAPE (can be added next)

print("Pipeline complete. Ready for evaluation and visualization.")

### --- Evaluation & Visualisation ---

library(pROC)
library(caret)

### Classification Metrics ---

# Accuracy
acc_glm <- mean(round(test_class$glm_pred) == test_class$ClaimOcc)
acc_lgb <- mean(round(test_class$lgb_pred) == test_class$ClaimOcc)
acc_tab <- mean(test_class$tabpfn_pred == test_class$ClaimOcc)

# AUC (only for models with probabilities)
auc_glm <- roc(test_class$ClaimOcc, test_class$glm_pred)$auc
auc_lgb <- roc(test_class$ClaimOcc, test_class$lgb_pred)$auc

# Confusion Matrix
cm_tab <- table(Predicted = test_class$tabpfn_pred, Actual = test_class$ClaimOcc)

cat("\nClassification Accuracy:\n")
cat("GLM:", acc_glm, "\n")
cat("LightGBM:", acc_lgb, "\n")
cat("TabPFN:", acc_tab, "\n")

cat("\nClassification AUC:\n")
cat("GLM:", auc_glm, "\n")
cat("LightGBM:", auc_lgb, "\n")

cat("\nTabPFN Confusion Matrix:\n")
print(cm_tab)


### Regression Metrics ---

mae <- function(actual, pred) mean(abs(actual - pred))
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mape <- function(actual, pred) mean(abs((actual - pred) / actual)) * 100

mae_glm <- mae(test_reg$ClaimAmount, test_reg$glm_pred)
rmse_glm <- rmse(test_reg$ClaimAmount, test_reg$glm_pred)
mape_glm <- mape(test_reg$ClaimAmount, test_reg$glm_pred)

mae_lgb <- mae(test_reg$ClaimAmount, test_reg$lgb_pred)
rmse_lgb <- rmse(test_reg$ClaimAmount, test_reg$lgb_pred)
mape_lgb <- mape(test_reg$ClaimAmount, test_reg$lgb_pred)

mae_tab <- mae(test_reg$ClaimAmount, test_reg$tabpfn_pred)
rmse_tab <- rmse(test_reg$ClaimAmount, test_reg$tabpfn_pred)
mape_tab <- mape(test_reg$ClaimAmount, test_reg$tabpfn_pred)

cat("\nRegression Metrics (MAE / RMSE / MAPE):\n")
cat("GLM:", mae_glm, rmse_glm, mape_glm, "\n")
cat("LightGBM:", mae_lgb, rmse_lgb, mape_lgb, "\n")
cat("TabPFN:", mae_tab, rmse_tab, mape_tab, "\n")

#weighted mape

### Visualisation: Predicted vs Actual by DrivAge Band (Regression) ---

test_reg <- test_reg %>% mutate(AgeBand = as.factor(DrivAge))

summary_plot <- test_reg %>%
  group_by(AgeBand) %>%
  summarise(
    Actual = mean(ClaimAmount, na.rm = TRUE),
    GLM = mean(glm_pred, na.rm = TRUE),
    LGB = mean(lgb_pred, na.rm = TRUE),
    TabPFN = mean(tabpfn_pred, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c("Actual", "GLM", "LGB", "TabPFN"), names_to = "Model", values_to = "MeanClaim")

# Plot
library(ggplot2)

ggplot(summary_plot, aes(x = AgeBand, y = MeanClaim, color = Model, group = Model)) +
  geom_line(linewidth = 1.1) +
  geom_point(aes(size = Count), alpha = 0.6) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Average Claim Amount by Driver Age Band (Regression)",
    x = "Driver Age Band", y = "Mean Claim Amount",
    size = "Observation Count"
  ) +
  scale_color_manual(values = c("Actual" = "black", "GLM" = "darkorange", "LGB" = "forestgreen", "TabPFN" = "steelblue")) +
  theme(legend.position = "bottom")


#vs code 

#github copilot

#python

#tabpfn on kaggle datasets


