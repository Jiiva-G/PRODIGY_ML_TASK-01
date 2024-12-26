import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("D:/ML_Intern_Task/Task-1/Housing.csv")

# Display dataset structure
print("First 5 rows of the dataset:")
print(data.head())
print("\nColumn names in the dataset:")
print(data.columns)

# Check data types and missing values
print("\nDataset Info:")
print(data.info())
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Handle missing values (drop rows with missing values or impute them)
data = data.dropna()

# Convert categorical columns to numeric
categorical_columns = data.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print(f"\nCategorical Columns to Encode: {categorical_columns}")
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    print("\nData after encoding:")
    print(data.head())

# Add more features to improve the model
data['price_per_area'] = data['price'] / data['area']
data['rooms_total'] = data['bedrooms'] + data['bathrooms']

# Separate features and target
X = data[['area', 'bedrooms', 'bathrooms', 'price_per_area', 'rooms_total']]
y = data['price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Feature Importance (Coefficients)
print("\nFeature Importance (Coefficients):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Visualize Actual vs Predicted Prices
y_pred = model.predict(X_test_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Visualizing Coefficients
plt.figure(figsize=(8, 6))
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='viridis')
plt.title("Feature Importance via Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.grid(True)
plt.show()

# Feature Analysis - Correlation Matrix
plt.figure(figsize=(10, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Pairplot to visualize relationships
sns.pairplot(data[['area', 'bedrooms', 'bathrooms', 'price_per_area', 'rooms_total', 'price']])
plt.show()
