# Pharma Project-6

## Business Need

You work at Nexthikes as a Machine Learning Engineer, and Rossmann Pharmaceuticals has given you a project on sales forecasting. The finance team wants to forecast sales in all their stores across several cities six weeks ahead of time.

## Objectives

- Build and serve an end-to-end product for sales prediction to analysts in the finance team.
- Explore customer purchasing behavior and identify factors affecting sales.
- Predict store sales using machine learning and deep learning approaches.
- Serve predictions on a web interface for easy access by managers of the stores.

## Data and Features

The data for this project includes various features such as Store, Sales, Customers, Open, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo, Promo2, and others.

## Data Fields

- **Id**: An Id that represents a (Store, Date) tuple within the test set.
- **Store**: A unique Id for each store.
- **Sales**: The turnover for any given day (this is what you are predicting).
- **Customers**: The number of customers on a given day.
- **Open**: An indicator for whether the store was open: 0 = closed, 1 = open.
- **StateHoliday**: Indicates a state holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None).
- **SchoolHoliday**: Indicates if the (Store, Date) was affected by the closure of public schools.
- **StoreType**: Differentiates between 4 different store models: a, b, c, d.
- **Assortment**: Describes an assortment level: a = basic, b = extra, c = extended.
- **CompetitionDistance**: Distance in meters to the nearest competitor store.
- **CompetitionOpenSince[Month/Year]**: Gives the approximate year and month of the time the nearest competitor was opened.
- **Promo**: Indicates whether a store is running a promo on that day.
- **Promo2**: Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating.
- **Promo2Since[Year/Week]**: Describes the year and calendar week when the store started participating in Promo2.
- **PromoInterval**: Describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.

## Learning Outcomes

- Technical Skills: Pandas, Matplotlib, Numpy, HTML and CSS, Flask. Interns will also improve their code modularization skills.
- Creation of new features
- Predictive pipeline: Exploratory data analysis, data wrangling, building and fine-tuning models
- Building model using MLOps Techniques
- Deployment: Interns will know how to serve predictions in a web app.

## Setup
To set up the project, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. [Add any additional setup steps here].

## Directory Structure
- `.venv`: Virtual environment for the project.
- `mlruns`: MLflow tracking directory.
- `static`: Static files for web application (if applicable).
- `templates`: HTML templates for web application (if applicable).
- [Add descriptions for other directories and files here].

## Files
- `app.py`: Main Python script for running the application.
- `README.md`: This file.
- `static/style.css`:
- `templates/index.html`:

## Data
- `store.csv`: [Description of the dataset].
- `train.csv`: [Description of the dataset].
- `test.csv`: [Description of the dataset].
- `sample_submission.csv`: [Description of the dataset].

## Models
- `linear_model_02-04-2024-22-33-46-00.pkl`: Trained linear regression model.
- `lstm_regression_model.pkl`: Trained LSTM regression model.
-  


## Analysis
- `Comparing LSTM and Linear Regression Model.md`: Analysis comparing LSTM and linear regression models.
- `decision_tree_model_02-04-2024-22-33-46-00.pkl`: Trained decision tree model.
- 


## Overview
`requirements.txt`

## Instructions

### Task 1 - Exploration of Customer Purchasing Behavior
...
### Task 1.2 - Logging
...
### Task 2 - Prediction of Store Sales
...
### Task 3 - Serving Predictions on a Web Interface
...

## Links 

Running on INFO:waitress:Serving on http://127.0.0.1:5000

MLFLOW : http://127.0.0.1:5000/#/experiments/253508274149322182?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

SALES PREDICTION DASHBOARD : (file:///C:/Users/admin/OneDrive/Desktop/pharma%20project-6/templates/index.html)

## Conclusion
This project focuses on pharmaceutical sales prediction across multiple stores. It aims to forecast sales in all Rossmann Pharmaceuticals stores across several cities six weeks ahead of time.