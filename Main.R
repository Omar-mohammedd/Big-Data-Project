# ------------------------------------------------------------------------------
# STEP 1: INSTALL AND LOAD REQUIRED LIBRARIES
# ------------------------------------------------------------------------------

library(tidyverse)   # Data manipulation (dplyr) and visualization (ggplot2)
library(caret)       # Machine learning utilities (data splitting, evaluation)
library(rpart)       # Decision Tree building algorithm
library(rpart.plot)  # Advanced plotting utilities for decision trees
library(scales)      # For clean number formats (e.g., adding commas to prices)

# ------------------------------------------------------------------------------
# STEP 2: DATA LOADING & EXAMINING STRUCTURE
# ------------------------------------------------------------------------------
raw_car_data <- read.csv("car details v4.csv", stringsAsFactors = FALSE)

cat("--- Raw Data Structure Preview ---\n")
str(raw_car_data)

# ------------------------------------------------------------------------------
# STEP 3: DATA PREPROCESSING, CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
cat("\n--- Running Preprocessing and Cleaning Steps ---\n")

# Check for missing values across all columns
missing_summary <- colSums(is.na(raw_car_data))
print("Missing values per attribute:")
print(missing_summary)

# Apply Transformations
cleaned_car_data <- raw_car_data %>%
  # 1. Deduplication: Remove any redundant or exactly identical rows
  distinct() %>%
  
  # 2. Outlier/Sanity Filter: Filter out any unrealistic km readings
  filter(Kilometer < 500000) %>%
  
  # 3. Feature Engineering - Vehicle Age: Convert 'year' into an actionable 'car_age' column
  mutate(car_age = 2026 - Year) %>%
  
  # 4. Feature Engineering - Brand Extraction: Take the first word from the 'name' column
  mutate(brand = Make) %>%
  
  # 5. Data Type Formatting: Convert textual descriptions into categorical Factors
  mutate(
    fuel = as.factor(Fuel.Type),
    seller_type = as.factor(Seller.Type),
    transmission = as.factor(Transmission),
    owner = as.factor(Owner),
    brand = as.factor(brand),
    selling_price = Price,
    km_driven = Kilometer
  )

cat("\n--- Cleaned Data Summary ---\n")
summary(cleaned_car_data)

# ------------------------------------------------------------------------------
# STEP 4: HYPOTHESIS TESTING (Statistical Validation)
# ------------------------------------------------------------------------------
cat("\n--- Executing Hypothesis Test (Welch's Two-Sample t-Test) ---\n")
# Testing whether Transmission Type significantly influences the vehicle's market value
# Null Hypothesis (H0): Mean Price (Automatic) = Mean Price (Manual)
# Alternative Hypothesis (H1): Mean Price (Automatic) != Mean Price (Manual)

t_test_results <- t.test(selling_price ~ transmission, data = cleaned_car_data)
print(t_test_results)

# ------------------------------------------------------------------------------
# STEP 5: EXPLORATORY DATA ANALYSIS & VISUALIZATION (EDA)
# ------------------------------------------------------------------------------
cat("\n--- Generating Visualizations (Check the Plots pane) ---\n")

# Plot 1: Boxplot of Price distribution across Fuel configurations
plot1 <- ggplot(cleaned_car_data, aes(x = fuel, y = selling_price, fill = fuel)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.2) +
  scale_y_continuous(labels = comma, limits = c(0, 2000000)) + 
  labs(
    title = "Analysis of Car Selling Prices across Fuel Categories",
    subtitle = "Visualizing distribution profiles and median variance",
    x = "Fuel Class",
    y = "Selling Price (INR)",
    fill = "Fuel Type"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(plot1)

# Plot 2: Scatter plot mapping Age against Depreciation values
plot2 <- ggplot(cleaned_car_data, aes(x = car_age, y = selling_price)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "darkred", linetype = "solid", size = 1.2, se = TRUE) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Vehicle Depreciation Curve Analysis",
    subtitle = "Relationship between Car Age and expected Selling Price market values",
    x = "Vehicle Age (Years)",
    y = "Selling Price (INR)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(plot2)

# ------------------------------------------------------------------------------
# STEP 6: DATASET PREPARATION FOR MACHINE LEARNING
# ------------------------------------------------------------------------------
cat("\n--- Splitting Dataset into Training and Testing Sets ---\n")

# Set seed for pseudo-random reproducibility
set.seed(786)

# Create a stratified partition index based on the outcome variable (80% Train, 20% Test)
train_indices <- createDataPartition(cleaned_car_data$selling_price, p = 0.8, list = FALSE)

# Generate subsets
training_set <- cleaned_car_data[train_indices, ]
testing_set  <- cleaned_car_data[-train_indices, ]

cat("Observations inside Training Matrix: ", nrow(training_set), "\n")
cat("Observations inside Testing Matrix:  ", nrow(testing_set), "\n")

# ------------------------------------------------------------------------------
# STEP 7: MODEL TRAINING (APPLYING MACHINE LEARNING ALGORITHMS)
# ------------------------------------------------------------------------------
cat("\n--- Fitting Modeling Frameworks ---\n")

# Technique A: Parametric Modeling - Multiple Linear Regression
regression_model <- lm(
  selling_price ~ car_age + km_driven + transmission + fuel + seller_type, 
  data = training_set
)
cat("\n--- Linear Regression Coefficients summary ---\n")
summary(regression_model)

# Technique B: Non-Parametric Modeling - Decision Tree Regressor (ANOVA Method)
decision_tree_model <- rpart(
  selling_price ~ car_age + km_driven + transmission + fuel + seller_type + owner, 
  data = training_set, 
  method = "anova",
  control = rpart.control(cp = 0.005) # Complexity Parameter boundary to check overfitting
)

# Render the graphical Decision Tree flow layout
cat("\n--- Plotting Tree Diagram structures ---\n")
rpart.plot(
  decision_tree_model, 
  type = 2, 
  extra = 101, 
  fallen.leaves = TRUE, 
  box.palette = "BuGn",
  main = "Used Car Pricing Strategy - Regression Decision Rules"
)

# ------------------------------------------------------------------------------
# STEP 8: EVALUATION & COMPILING METRIC ARRAYS
# ------------------------------------------------------------------------------
cat("\n--- Evaluation Engine Processing ---\n")

# Predict prices using unseen Test data
reg_predictions <- predict(regression_model, newdata = testing_set)
tree_predictions <- predict(decision_tree_model, newdata = testing_set)

# Define calculation function for standard validation metrics
calculate_performance <- function(actual, predicted) {
  mae  <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  
  # R-squared computation
  total_ss     <- sum((actual - mean(actual))^2)
  residual_ss  <- sum((actual - predicted)^2)
  r_squared    <- 1 - (residual_ss / total_ss)
  
  return(c(MAE = mae, RMSE = rmse, R_Squared = r_squared))
}

# Run evaluation on test outcomes
reg_metrics  <- calculate_performance(testing_set$selling_price, reg_predictions)
tree_metrics <- calculate_performance(testing_set$selling_price, tree_predictions)

# Bind metrics into a reporting performance frame
performance_comparison <- data.frame(
  Evaluation_Metric = c("Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R-Squared Score (R2)"),
  Linear_Regression   = c(reg_metrics["MAE"], reg_metrics["RMSE"], reg_metrics["R_Squared"]),
  Decision_Tree       = c(tree_metrics["MAE"], tree_metrics["RMSE"], tree_metrics["R_Squared"])
)

print("\n=== FINAL EVALUATION METRIC REPORT ===")
print(performance_comparison)