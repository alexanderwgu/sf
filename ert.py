import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor  #Extremely Randomized Trees
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import seaborn as sns

symbol = "TRUE" #"ΔG"
# Load your data from a CSV file or any other source
data = pd.read_csv('data.csv')

data = data.iloc[:, 1:]
# Separate the features (X) and the target (y)
X = data.drop(symbol, axis=1)
y = data[symbol]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regressor = ExtraTreesRegressor(n_estimators=25, random_state=1, max_depth=7)

# Fit the regressor with your training data
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)



# externalSet = pd.read_csv("test.csv")
# externalSet = externalSet.iloc[:, 1:]
# eX = externalSet.drop(symbol, axis=1)
# ey = externalSet[symbol]

# ey_pred = regressor.predict(eX)
# print(eX)
# print(ey_pred)


# Calculate the R-squared score
r2_train = r2_score(y_train, y_train_pred)
r2 = r2_score(y_test, y_pred)
print("Test R-squared (R^2) Score:", r2)
print("Train R-squared (R^2) score:", r2_train)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error:", mse)
print("Train Mean-Squared Error:", mse_train)

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

# Get feature importances
feature_importance = regressor.feature_importances_

# Get the feature names (column names)
feature_names = X_train.columns

# Sort feature importances and feature names in descending order
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]


# Limit the display to the top n features
top_n = 4
top_feature_importance = sorted_feature_importance[:top_n]
top_feature_names = sorted_feature_names[:top_n]


light_blue = '#6A9AFA'  # A lighter shade of blue
plt.figure(figsize=(6, 5))  # Adjust the figure size as needed
plt.barh(range(top_n, 0, -1), top_feature_importance, color=light_blue)
plt.yticks(range(top_n, 0, -1), top_feature_names)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.show()
print(sorted_feature_importance)
print(sorted_feature_names)

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

plt.title('Extremely Randomized Trees',fontsize=24)
plt.legend()
plt.grid(True)
plt.show()

# Calculate the Pearson correlation matrix
correlation_matrix = data.corr()

# Visualize the Pearson correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, square=True)
plt.title('Pearson Correlation Matrix')
plt.show()




# Create a new DataFrame with the top 3 attributes and symbol values for both the test and training data
top_3_attributes_test = X_test[sorted_feature_names[:3]]
top_3_attributes_test[symbol] = y_test

top_3_attributes_train = X_train[sorted_feature_names[:3]]
top_3_attributes_train[symbol] = y_train

# Combine test and training data
top_3_attributes_combined = pd.concat([top_3_attributes_test, top_3_attributes_train])

# Create a figure with custom subplot proportions
fig = plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
ax = fig.add_subplot(121, projection='3d', proj_type='ortho', adjustable='box', aspect='auto')  # Create a 3D subplot with custom proportions

x_test = top_3_attributes_test[sorted_feature_names[0]]
y_test = top_3_attributes_test[sorted_feature_names[1]]
z_test = top_3_attributes_test[sorted_feature_names[2]]
c_test = top_3_attributes_test[symbol]  # Color encodes symbol values for test data

x_train = top_3_attributes_train[sorted_feature_names[0]]
y_train = top_3_attributes_train[sorted_feature_names[1]]
z_train = top_3_attributes_train[sorted_feature_names[2]]
c_train = top_3_attributes_train[symbol]  # Color encodes symbol values for training data


# Scatter plot for combined data with the same marker shape
scatter_combined = ax.scatter(np.concatenate([x_test, x_train]),
                              np.concatenate([y_test, y_train]),
                              np.concatenate([z_test, z_train]),
                              c=np.concatenate([c_test, c_train]),
                              cmap='viridis',
                              marker='o',  # Use circles for all points
                              label="Combined Data",
                              s=20)

ax.set_xlabel(sorted_feature_names[0])
ax.set_ylabel(sorted_feature_names[1])
ax.set_zlabel(sorted_feature_names[2])

# Create a 2D scatter plot as a color bar beside the 3D plot with a thinner width
colorbar_ax = fig.add_subplot(122)
cbar = plt.colorbar(scatter_combined, cax=colorbar_ax, label='symbol values (Combined Data)')
colorbar_ax.set_position([0.55, 0.2, 0.03, 0.6])  # Adjust the position and size for the color bar

fig.suptitle('4D Scatter Plot with Top 3 Features and Color Encoding for symbol (Combined Data)', fontsize=16, x=0.32)
plt.show()




# Get the top 2 feature names
top_2_feature_names = sorted_feature_names[:2]

# Create a 3D scatter plot with the top 2 features swapped
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

x_test = top_3_attributes_test[top_2_feature_names[1]]  # Swap the attributes
y_test = top_3_attributes_test[top_2_feature_names[0]]  # Swap the attributes
z_test = top_3_attributes_test[symbol]

x_train = top_3_attributes_train[top_2_feature_names[1]]  # Swap the attributes
y_train = top_3_attributes_train[top_2_feature_names[0]]  # Swap the attributes
z_train = top_3_attributes_train[symbol]

scatter_test = ax.scatter(x_test, y_test, z_test, c=c_test, cmap='viridis', marker='o', label="Test Data", s=20)
scatter_train = ax.scatter(x_train, y_train, z_train, c=c_train, cmap='viridis', marker='o', label="Training Data", s=20)

ax.set_xlabel(top_2_feature_names[1])  # Swap the attribute names
ax.set_ylabel(top_2_feature_names[0])   # Swap the attribute names
ax.set_zlabel(symbol)


plt.legend()
plt.title('3D Scatter Plot with Top 2 Features and symbol')
plt.show()
