import numpy as np          #KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
symbol = "ΔG" #ΔG
data = pd.read_csv('data.csv')

data = data.iloc[:, 1:]
# Separate the features (X) and the target (y)
X = data.drop(symbol, axis=1)  # Replace 'target_column' with your target variable name
y = data[symbol]  # Replace 'target_column' with your target variable name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Fit the regressor with your training data
regressor.fit(X_train, y_train)

# Make predictions on the training and test data
y_train_pred = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

# Calculate the R-squared score for training and test data
r2_train = r2_score(y_train, y_train_pred)
r2 = r2_score(y_test, y_pred)
print("Training R-squared (R^2) Score:", r2_train)
print("Test R-squared (R^2) Score:", r2)

# Calculate the Mean Squared Error for training and test data
mse_train = mean_squared_error(y_train, y_train_pred)
mse = mean_squared_error(y_test, y_pred)
print("Training Mean Squared Error:", mse_train)
print("Test Mean Squared Error:", mse)

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
