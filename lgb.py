import lightgbm as lgb #light gradient boosting
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
symbol = "ΔG" #ΔG
# Load your data from a CSV file or any other source
data = pd.read_csv('data.csv')

data = data.iloc[:, 1:]

# Separate the features (X) and the target (y)
X = data.drop(symbol, axis=1)  # Replace 'target_column' with your target variable name
y = data[symbol]  # Replace 'target_column' with your target variable name

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset for training
train_data = lgb.Dataset(X_train, label=y_train)

# Define hyperparameters for the LightGBM model
params = {
    "objective": "regression",  # For regression tasks
    "metric": "mse",  # Mean Squared Error
    "boosting_type": "dart",
    "num_leaves": 25,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "max_depth": 10,
    "subsample": 0.2,  # Adjust to a smaller value
    "min_data_in_leaf": 20,  # Adjust to a larger value
}

# Train the LightGBM model with early stopping
num_round = 300  # Number of boosting rounds (adjust as needed)

bst = lgb.train(params, train_data, num_round, valid_sets=[train_data])

# Make predictions on the test set
y_pred_test = bst.predict(X_test)
y_pred_train = bst.predict(X_train)

# Calculate MSE for test and train
mse_test = mean_squared_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (MSE) - Test: {mse_test}")
print(f"Mean Squared Error (MSE) - Train: {mse_train}")

# Calculate R-squared (R^2) for test and train
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
print(f"R-squared (R^2) - Test: {r2_test}")
print(f"R-squared (R^2) - Train: {r2_train}")

# Calculate the standard error for R-squared and MSE for both training and test data
n_train = len(y_train)
n_test = len(y_test)

se_r2_train = np.sqrt((1 - r2_train) / (n_train - 2))
se_r2 = np.sqrt((1 - r2) / (n_test - 2))
se_mse_train = np.sqrt(mse_train / n_train)
se_mse = np.sqrt(mse / n_test)

print("Training R-squared (R^2) Standard Error:", se_r2_train)
print("Test R-squared (R^2) Standard Error:", se_r2)
print("Training Mean Squared Error (MSE) Standard Error:", se_mse_train)
print("Test Mean Squared Error (MSE) Standard Error:", se_mse)

# Scatter plot for test data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Test Data",s=100)  # Scatter plot test data
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# # Best-fit line for test data
# fit = np.polyfit(y_test, y_pred, 1)
# line = fit[0] * y_test + fit[1]
# plt.plot(y_test, line, color='red', lw=2, label="Best Fit Line (Test Data)")

# Scatter plot for training data

plt.scatter(y_train, y_train_pred, alpha=0.5, label="Training Data", color='green',s=100)  # Scatter plot training data

#Best-fit line
# combined_y = np.concatenate((y_test, y_train))
# combined_y_pred = np.concatenate((y_pred, regressor.predict(X_train)))
# fit = np.polyfit(combined_y, combined_y_pred, 1)
# line = fit[0] * combined_y + fit[1]
# plt.plot(combined_y,line,color='red', lw=2,  label="Best Fit Line")

plt.axline((0, 0), slope=1)

plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
