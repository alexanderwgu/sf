import numpy as np        #Ridge
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
symbol = "TRUE" #ΔG
data = pd.read_csv('data.csv')

data = data.iloc[:, 1:]
# Separate the features (X) and the target (y)
X = data.drop(symbol, axis=1)  # Replace 'target_column' with your target variable name
y = data[symbol]  # Replace 'target_column' with your target variable name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
regressor = Ridge(alpha=1.0)  # You can adjust the alpha (regularization strength) as needed

# Fit the Ridge model with your training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared (R^2) Score:", r2)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
