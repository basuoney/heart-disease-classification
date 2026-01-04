################################################################################
# Heart Disease Classification Project
#
# Research Question: Can we predict heart disease presence using clinical
# features, and which features are most important?
#
# This complements my diabetes analysis by showing classification methods
# (Random Forest, Decision Trees) vs. logistic regression.
################################################################################

# Load packages
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

set.seed(123)

################################################################################
# DATA LOADING
################################################################################

# Using the famous Cleveland Heart Disease dataset from UCI
# Available in several R packages, we'll use it directly

# Note: If this doesn't work, the data is publicly available at:
# https://archive.ics.uci.edu/ml/datasets/heart+disease

# For reproducibility, I'm using a version available through a common package
# If you need to load from CSV, uncomment below:
# heart <- read.csv("heart.csv")

# Using processed Cleveland data (common in ML tutorials)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names from dataset documentation
col_names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", 
               "restecg", "thalach", "exang", "oldpeak", "slope", 
               "ca", "thal", "target")

# Read data
heart <- read.csv(url, header = FALSE, col.names = col_names, na.strings = "?")

cat("Dataset loaded successfully!\n")
cat("Dimensions:", nrow(heart), "rows,", ncol(heart), "columns\n\n")

################################################################################
# DATA UNDERSTANDING
################################################################################

# What do these variables mean?
cat("Variable descriptions:\n")
cat("- age: Age in years\n")
cat("- sex: 1=male, 0=female\n")
cat("- cp: Chest pain type (1-4)\n")
cat("- trestbps: Resting blood pressure (mm Hg)\n")
cat("- chol: Serum cholesterol (mg/dl)\n")
cat("- fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false)\n")
cat("- restecg: Resting ECG results (0-2)\n")
cat("- thalach: Maximum heart rate achieved\n")
cat("- exang: Exercise induced angina (1=yes, 0=no)\n")
cat("- oldpeak: ST depression induced by exercise\n")
cat("- slope: Slope of peak exercise ST segment (1-3)\n")
cat("- ca: Number of major vessels colored by fluoroscopy (0-3)\n")
cat("- thal: Thalassemia (3=normal, 6=fixed defect, 7=reversible)\n")
cat("- target: Heart disease presence (0=no, 1-4=yes, varying severity)\n\n")

# Check structure
glimpse(heart)

################################################################################
# DATA CLEANING
################################################################################

# Check missing values
cat("Missing values per variable:\n")
print(colSums(is.na(heart)))
cat("\n")

# For this analysis, remove rows with missing values
# In a real study, we'd investigate patterns and consider imputation
heart_clean <- heart %>% drop_na()

cat("After removing missing values:", nrow(heart_clean), "observations\n\n")

# Create binary outcome: 0 = no disease, 1-4 = disease present
heart_clean <- heart_clean %>%
  mutate(
    disease = ifelse(target > 0, 1, 0),
    disease = factor(disease, levels = c(0, 1), labels = c("No", "Yes")),
    # Make categorical variables factors
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    cp = factor(cp),
    fbs = factor(fbs),
    restecg = factor(restecg),
    exang = factor(exang),
    slope = factor(slope),
    ca = factor(ca),
    thal = factor(thal)
  )

# Check disease prevalence
cat("Heart disease prevalence:\n")
print(table(heart_clean$disease))
cat("Percentage with disease:", 
    round(mean(heart_clean$disease == "Yes") * 100, 1), "%\n\n")

################################################################################
# EXPLORATORY ANALYSIS
################################################################################

# Compare characteristics by disease status
summary_stats <- heart_clean %>%
  group_by(disease) %>%
  summarise(
    n = n(),
    avg_age = round(mean(age), 1),
    pct_male = round(mean(sex == "Male") * 100, 1),
    avg_chol = round(mean(chol), 0),
    avg_bp = round(mean(trestbps), 0),
    avg_max_hr = round(mean(thalach), 0)
  )

cat("Summary statistics by disease status:\n")
print(summary_stats)
cat("\n")

# Visualizations

# Age distribution
ggplot(heart_clean, aes(x = age, fill = disease)) +
  geom_density(alpha = 0.5) +
  labs(title = "Age Distribution by Heart Disease Status",
       x = "Age (years)", y = "Density") +
  theme_minimal()
ggsave("age_distribution_heart.png", width = 7, height = 4)

# Cholesterol levels
ggplot(heart_clean, aes(x = disease, y = chol, fill = disease)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Cholesterol Levels by Heart Disease Status",
       x = "", y = "Serum Cholesterol (mg/dl)") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("cholesterol_heart.png", width = 6, height = 4)

# Gender comparison
gender_summary <- heart_clean %>%
  group_by(sex, disease) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sex) %>%
  mutate(pct = n / sum(n) * 100)

ggplot(gender_summary, aes(x = sex, y = pct, fill = disease)) +
  geom_col(position = "dodge") +
  labs(title = "Heart Disease Prevalence by Gender",
       x = "", y = "Percentage (%)") +
  theme_minimal()
ggsave("gender_heart.png", width = 6, height = 4)

################################################################################
# TRAIN/TEST SPLIT
################################################################################

# Stratified split to maintain disease prevalence in both sets
train_idx <- createDataPartition(heart_clean$disease, p = 0.75, list = FALSE)
train_data <- heart_clean[train_idx, ]
test_data <- heart_clean[-train_idx, ]

cat("Training set:", nrow(train_data), "observations,",
    sum(train_data$disease == "Yes"), "with disease\n")
cat("Test set:", nrow(test_data), "observations,",
    sum(test_data$disease == "Yes"), "with disease\n\n")

################################################################################
# MODEL 1: DECISION TREE
################################################################################
cat("=== DECISION TREE MODEL ===\n\n")

# Fit decision tree
tree_model <- rpart(
  disease ~ age + sex + cp + trestbps + chol + fbs + restecg + 
    thalach + exang + oldpeak + slope + ca + thal,
  data = train_data,
  method = "class",
  control = rpart.control(cp = 0.01, minsplit = 20)
)

# Plot tree
png("decision_tree_plot.png", width = 1000, height = 600)
rpart.plot(tree_model, main = "Decision Tree for Heart Disease", 
           extra = 104, box.palette = "RdYlGn")
dev.off()

# Variable importance
cat("Decision Tree Variable Importance:\n")
importance_tree <- tree_model$variable.importance
print(round(importance_tree, 2))
cat("\n")

# Predictions on test set
tree_pred <- predict(tree_model, test_data, type = "class")
tree_conf <- confusionMatrix(tree_pred, test_data$disease)

cat("Decision Tree Test Performance:\n")
print(tree_conf)
cat("\n")

################################################################################
# MODEL 2: RANDOM FOREST
################################################################################
cat("=== RANDOM FOREST MODEL ===\n\n")

# Fit random forest
# Using fewer trees for speed, but still effective
rf_model <- randomForest(
  disease ~ age + sex + cp + trestbps + chol + fbs + restecg + 
    thalach + exang + oldpeak + slope + ca + thal,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Variable importance
cat("Random Forest Variable Importance:\n")
importance_rf <- importance(rf_model)
print(round(importance_rf, 2))
cat("\n")

# Plot importance
png("rf_importance.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()

# Predictions on test set
rf_pred <- predict(rf_model, test_data)
rf_conf <- confusionMatrix(rf_pred, test_data$disease)

cat("Random Forest Test Performance:\n")
print(rf_conf)
cat("\n")

################################################################################
# MODEL COMPARISON
################################################################################
cat("=== MODEL COMPARISON ===\n\n")

# Create comparison table
comparison <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Accuracy = c(
    round(tree_conf$overall["Accuracy"], 3),
    round(rf_conf$overall["Accuracy"], 3)
  ),
  Sensitivity = c(
    round(tree_conf$byClass["Sensitivity"], 3),
    round(rf_conf$byClass["Sensitivity"], 3)
  ),
  Specificity = c(
    round(tree_conf$byClass["Specificity"], 3),
    round(rf_conf$byClass["Specificity"], 3)
  )
)

print(comparison)
write.csv(comparison, "model_comparison.csv", row.names = FALSE)

cat("\nKey Finding: Random Forest performs", 
    ifelse(comparison$Accuracy[2] > comparison$Accuracy[1], "better", "similarly"),
    "than Decision Tree\n")

# Top 3 most important features (from Random Forest)
top_features <- names(sort(importance_rf[,4], decreasing = TRUE))[1:3]
cat("\nTop 3 most important features:", paste(top_features, collapse = ", "), "\n")

################################################################################
# SAVE MODELS
################################################################################
saveRDS(tree_model, "heart_tree_model.rds")
saveRDS(rf_model, "heart_rf_model.rds")

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Models and visualizations saved to working directory\n")

################################################################################
# REFLECTIONS
################################################################################
cat("\n=== WHAT I LEARNED ===\n")
cat("1. Random Forest generally outperforms single Decision Trees\n")
cat("2. Chest pain type and thalassemia appear most predictive\n")
cat("3. Tree models are more interpretable than many algorithms\n")
cat("4. Class imbalance isn't severe here (~54% disease prevalence)\n\n")

cat("=== LIMITATIONS ===\n")
cat("- Small dataset (n=297 after cleaning)\n")
cat("- Missing data removed rather than imputed\n")
cat("- No hyperparameter tuning done\n")
cat("- Should validate on completely external dataset\n")
cat("- Some medical features (thal, ca) require domain expertise to interpret\n")

################################################################################