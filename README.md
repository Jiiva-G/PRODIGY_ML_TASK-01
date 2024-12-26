# PRODIGY_ML_TASK-01

Concept of the Linear Regression Code
Purpose:
The code implements a linear regression model to predict house prices based on various features such as area, number of bedrooms, bathrooms, and additional engineered features. It also evaluates the model's performance and visualizes the relationships between features and predictions.

Steps in the Code:
1. Data Loading and Inspection
The dataset is loaded using pandas.read_csv().
First 5 rows, column names, and dataset structure are displayed.
Missing values are identified and handled by dropping rows with null values.
2. Handling Categorical Data
Non-numeric (categorical) columns are automatically detected.
These columns are converted into numeric format using LabelEncoder, which assigns unique integers to each category.
3. Feature Engineering
New features are created to improve the model's predictive power:
Price per Area: Price of the house divided by its area (price_per_area).
Total Rooms: Sum of the number of bedrooms and bathrooms (rooms_total).
4. Exploratory Data Analysis (EDA)
Correlation Matrix:
A heatmap is created to show relationships between features and the target variable (price).
Pairplot:
Visualizes pairwise relationships between important features and the target variable for better insight.
5. Feature Selection and Target Variable
Selects the independent variables (X) and the target variable (y):
Features: area, bedrooms, bathrooms, price_per_area, rooms_total.
Target: price.
6. Data Splitting and Scaling
Splits the dataset into training and testing sets using train_test_split().
Standardizes the feature values using StandardScaler to ensure that all features have equal importance in the model.
7. Model Training
Initializes and trains a linear regression model using LinearRegression() on the scaled training data.
8. Feature Importance
Displays the coefficients of the features to understand their impact on the target variable.
Visualizes the coefficients as a bar chart for better interpretation.
9. Prediction and Visualization
Predicts house prices on the test set using the trained model.
Compares actual vs. predicted prices using a scatter plot:
The diagonal line indicates perfect predictions.
Deviations from the line highlight the model's errors.
10. Model Evaluation
Calculates performance metrics:
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted prices.
R-Squared (R²): Indicates how well the features explain the variability in the target variable
