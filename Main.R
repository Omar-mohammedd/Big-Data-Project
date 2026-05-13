# ==========================================
# Cleaning
# ==========================================
raw_data <- read.csv("car details v4.csv", stringsAsFactors = TRUE)

columns_to_keep <- c("Make", "Price", "Year", "Kilometer", "Engine", 
                     "Max.Power", "Max.Torque", "Transmission", "Fuel.Type",
                     "Owner", "Length", "Width", "Seating.Capacity",
                     "Fuel.Tank.Capacity", "Drivetrain", "Seller.Type", "Height" , "Location")

clean_data <- na.omit(raw_data[, columns_to_keep])

# Clean Max Power & Max Torque
clean_data$Max.Power.bhp <- as.numeric(sub("^([0-9.]+).*", "\\1", clean_data$Max.Power))
clean_data$Max.Power.rpm <- as.numeric(sub(".*@\\s*([0-9]+).*", "\\1", clean_data$Max.Power))

clean_data$Max.Torque.Nm  <- as.numeric(sub("^([0-9.]+).*", "\\1", clean_data$Max.Torque))
clean_data$Max.Torque.rpm <- as.numeric(sub(".*@\\s*([0-9]+).*", "\\1", clean_data$Max.Torque))

# Clean Engine
clean_data$Engine <- as.numeric(gsub(" cc", "", clean_data$Engine))
clean_data <- na.omit(clean_data)

# Remove Outliers : Price
Q1 <- quantile(clean_data$Price, 0.25)
Q3 <- quantile(clean_data$Price, 0.75)
IQR_val <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR_val
upper_bound <- Q3 + 1.5 * IQR_val
clean_data <- subset(clean_data, Price >= lower_bound & Price <= upper_bound)

# Remove Outliers : Kilometer
Q1_k <- quantile(clean_data$Kilometer, 0.25)
Q3_k <- quantile(clean_data$Kilometer, 0.75)
IQR_val_k <- Q3_k - Q1_k
lower_bound_k <- Q1_k - 1.5 * IQR_val_k
upper_bound_k <- Q3_k + 1.5 * IQR_val_k
clean_data <- subset(clean_data, Kilometer >= lower_bound_k & Kilometer <= upper_bound_k)

# To fix Prices Format
options(scipen = 999)

# ==========================================
# Hypothesis Test
# ==========================================
# Ensure using 'Manual' and 'Automatic' only
filtered_cars <- subset(clean_data, Transmission %in% c("Manual", "Automatic"))
test_result <- wilcox.test(Price ~ Transmission, data = filtered_cars)
print(test_result)

# Store the p-value
p_value <- test_result$p.value
cat("\n--- Conclusion ---\n")
cat("P-value is:", p_value, "\n")
if (p_value < 0.05) {
  cat("Conclusion: There is a statistically significant difference in the median prices of Manual and Automatic cars.\n")
} else {
  cat("Because the p-value is greater than 0.05, we FAIL TO REJECT the Null Hypothesis.\n")
  cat("Conclusion: There is no significant difference in the average prices of Manual and Automatic cars.\n")
}


# ==========================================
# EDA
# ==========================================
# Histogram : Distribution of Price
hist(clean_data$Price , col = "red" , main = "Distribution of Price" , xlab = "Car prices" , breaks = 50)

# Pie Chart : Top 10 Makers + Others
make_pie <- sort(table(clean_data$Make), decreasing = TRUE)
top_10_pie <- head(make_pie, 10)
others_count <- sum(make_pie) - sum(top_10_pie)
final_pie <- c(top_10_pie, Others = others_count)
pct <- round(final_pie / sum(final_pie) * 100)
lbls <- paste(names(final_pie), " ", pct, "%", sep="")

pie(final_pie, 
    labels = lbls, 
    col = rainbow(length(lbls)), 
    main = "Pie Chart of Top 10 Car Makes + Others")

# Scatter plot : Year vs Price
plot(clean_data$Year, clean_data$Price, 
     main = "Year vs Price", 
     xlab = "Year", 
     ylab = "Price", 
     col = "darkgreen")

# Bar Plot: Count of Fuel Types
fuel_counts <- table(clean_data$Fuel.Type)
barplot(fuel_counts, main = "Count of Cars by Fuel Type", col = "orange", ylab = "Count")


# ==========================================
# Prediction
# ==========================================
if(!require(party)) install.packages("party")
library(party)

# Split Data
set.seed(42) 
ind <- sample(2, nrow(clean_data), prob = c(0.7, 0.3), replace = TRUE)
train.data <- clean_data[ind == 1, ]
test.data  <- clean_data[ind == 2, ]

# Predict
price_tree <- ctree(Price ~ Make + Year + Kilometer + Fuel.Type + Transmission + Engine +
                      Seating.Capacity  + Max.Power.bhp + Max.Torque.Nm + Max.Torque.rpm +
                      Fuel.Tank.Capacity + Drivetrain + Seller.Type  + Max.Power.rpm ,data = train.data)

plot(price_tree, type="simple")

# Metrics
price_preds <- predict(price_tree, newdata = test.data)
rmse <- sqrt(mean((test.data$Price - price_preds)^2))
mae <- mean(abs(test.data$Price - price_preds))
r_sq <- cor(test.data$Price, price_preds)^2

print(paste("Final RMSE:", rmse))
print(paste("Final MAE:", mae))
print(paste("Final R2:", r_sq))
print(summary(clean_data))