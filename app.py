import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'

from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Sample data (replace with your actual data)
eda_stats = {
    'Mean Sales': 12000,
    'Median Sales': 10000,
    'Max Sales': 15000,
}

feature_coefficients = {
    'Store ID': 0.5,
    'Is Holiday': -0.2,
    'Is Weekend': 0.3,
    'Is Promo': 0.4,
}

comparisons = {
    'Comparison 1': 'Value 1',
    'Comparison 2': 'Value 2',
}

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input parameters from the form (not implemented yet)
    store_id = request.form.get('store_id', '')
    is_holiday = request.form.get('is_holiday', '')
    is_weekend = request.form.get('is_weekend', '')
    is_promo = request.form.get('is_promo', '')

    # Perform prediction (not implemented yet)
    sales_amount = 10000  # Sample prediction
    num_customers = 500   # Sample prediction

    # Generate plot for prediction
    x = ['Sales Amount', 'Number of Customers']
    y = [sales_amount, num_customers]
    prediction_plot = generate_bar_chart(x, y, 'Predicted Sales Metrics')

    # Generate additional plots (EDA, feature coefficients, comparisons)
    eda_plot = generate_eda_plot(eda_stats)
    coefficients_plot = generate_coefficients_plot(feature_coefficients)
    comparisons_plot = generate_comparisons_plot(comparisons)

    return render_template('index.html', prediction_text='Prediction:', prediction_plot=prediction_plot,
                           eda_plot=eda_plot, feature_coefficients_plot=coefficients_plot,
                           comparisons_plot=comparisons_plot)

def generate_bar_chart(x, y, title):
    plt.bar(x, y)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(title)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def generate_eda_plot(eda_stats):
    labels = list(eda_stats.keys())
    values = list(eda_stats.values())
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Exploratory Data Analysis (EDA)')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def generate_coefficients_plot(feature_coefficients):
    labels = list(feature_coefficients.keys())
    values = list(feature_coefficients.values())
    plt.figure(figsize=(8, 6))
    plt.barh(labels, values, color='lightgreen')
    plt.xlabel('Coefficients')
    plt.ylabel('Features')
    plt.title('Feature Coefficients')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def generate_comparisons_plot(comparisons):
    labels = list(comparisons.keys())
    values = list(comparisons.values())
    plt.figure(figsize=(8, 6))
    plt.plot(labels, values, marker='o', color='orange')
    plt.xlabel('Comparisons')
    plt.ylabel('Values')
    plt.title('Comparisons')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

if __name__ == '__main__':
    app.run(debug=True)
