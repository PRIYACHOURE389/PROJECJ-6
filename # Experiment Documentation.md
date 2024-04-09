# Experiment Documentation
"""
Experiment Name: Sales Prediction Model
Author: [Your Name]
Date: [Date]

Description:
This experiment aims to build a machine learning model to predict sales based on various features such as store information, day of the week, promotions, and holidays. The dataset used for training and evaluation contains historical sales data from multiple stores.

Dataset:
- The dataset contains the following columns: ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday_encoded', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsMonthEnd']
- Features: 'Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday_encoded', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsMonthEnd'
- Target: 'Sales'

Preprocessing:
- No preprocessing steps were applied to the dataset.

Model:
- Model Type: Linear Regression
- Evaluation Metric: Mean Squared Error (MSE)

Experiment Steps:
1. Load the dataset and split it into features (X) and target variable (y).
2. Split the data into training and testing sets.
3. Initialize MLflow experiment and start a new run.
4. Train the Linear Regression model on the training data.
5. Make predictions on the test set and calculate MSE.
6. Log parameters, metrics, and the trained model to MLflow.
7. Save the trained model using pickle.
8. Visualize the model's residuals and log the plot as an artifact.
9. Log additional evaluation metrics such as mean error, max error, and min error.
10. End the MLflow run.

Additional Notes:
- This experiment serves as a baseline model for sales prediction. Future iterations may explore more advanced models and feature engineering techniques to improve performance.
"""
