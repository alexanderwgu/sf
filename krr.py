import pandas as pd      #KRR
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
symbol = "TRUE" #ΔG
# Load data from "data.csv"
data = pd.read_csv("data.csv")

data = data.iloc[:, 1:]
# Assuming that your CSV file contains features in columns "X1", "X2", ..., and the target in "y"
X = data.drop(symbol, axis=1).values
y = data[symbol].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Kernel Ridge Regression model
krr_model = KernelRidge(kernel='rbf', alpha=1.0)
krr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = krr_model.predict(X_test)


# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
