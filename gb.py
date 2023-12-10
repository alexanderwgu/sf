from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data from the CSV file
symbol = "TRUE" #"ΔG" #ΔG

df = pd.read_csv('data.csv')

df = df.iloc[:, 1:]
# Assuming the target variable is symbol
X = df.drop(symbol, axis=1)
y = df[symbol]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the GradientBoostingRegressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on both the training and test datasets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate and print R-squared and Mean Squared Error for the test dataset
r_squared = r2_score(y_test, y_test_pred)
print(f'R-squared (Test): {r_squared}')

mse = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error (Test): {mse}')

# Calculate R-squared and Mean Squared Error for the training dataset
r_squared_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f'R-squared (Train): {r_squared_train}')
print(f'Mean Squared Error (Train): {mse_train}')

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


# Visualize feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

# Create a bar chart to visualize feature importance
plt.figure(figsize=(14, 6))
plt.bar(range(len(sorted_feature_importance)), sorted_feature_importance)
plt.xticks(range(len(sorted_feature_importance)), sorted_feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.show()

# Scatter plot for test data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Test Data", s=100, color = 'blue', marker='^', edgecolors='none')  # Scatter plot test data
plt.xlabel('Actual ΔG', fontsize=18)
plt.ylabel('Predicted ΔG', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Scatter plot for training data
plt.scatter(y_train, y_train_pred, alpha=0.5, label="Training Data", color = 'green', s=100, marker='o')  # Scatter plot training data

# Best-fit line for test data
# fit = np.polyfit(y_test, y_pred, 1)
# line = fit[0] * y_test + fit[1]
# plt.plot(y_test, line, color='red', lw=2, label="Best Fit Line (Test Data)")

# Best-fit line for combined data
# combined_y = np.concatenate((y_test, y_train))
# combined_y_pred = np.concatenate((y_pred, regressor.predict(X_train)))
# fit = np.polyfit(combined_y, combined_y_pred, 1)
# line = fit[0] * combined_y + fit[1]
# plt.plot(combined_y, line, color='red', lw=2, label="Best Fit Line (Combined Data)")

plt.axline((0, 0), slope=1)

plt.title('Gradient Boosting',fontsize=24)
plt.legend()
plt.grid(True)
plt.show()

# Calculate the Pearson correlation matrix
correlation_matrix = data.corr()
