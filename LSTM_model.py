import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load the dataset
file_path = r'C:\Users\admin\OneDrive\Desktop\pharma project-6\Pharma-project-6\sales_data.csv'
sales_data = pd.read_csv(file_path, low_memory=False)

# Extract features and target
dates = sales_data['Date']
sales = sales_data['Sales']

# Split data into train and test sets
train_size = int(len(sales) * 0.8)
test_size = len(sales) - train_size
train, test = sales[0:train_size], sales[train_size:len(sales)]

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(np.array(train).reshape(-1, 1))
test_scaled = scaler.transform(np.array(test).reshape(-1, 1))

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 1
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate MSE
train_mse = np.mean(np.square(train - train_predict))
test_mse = np.mean(np.square(test - test_predict))
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

# Save the model
with open('lstm_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save predictions to CSV files
train_predictions_df = pd.DataFrame({'Date': dates[:len(train)], 'Predicted_Sales': train_predict.flatten()})
test_predictions_df = pd.DataFrame({'Date': dates[len(train):], 'Predicted_Sales': test_predict.flatten()})
train_predictions_df.to_csv('train_predictions.csv', index=False)
test_predictions_df.to_csv('test_predictions.csv', index=False)

# Visualize the data and predictions
plt.figure(figsize=(14, 7))
plt.plot(dates[:len(train)], train, label='Actual Train Sales', color='blue')
plt.plot(dates[:len(train)], train_predict, label='Predicted Train Sales', color='orange')
plt.title('Training Data and Predictions')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(dates[len(train):], test, label='Actual Test Sales', color='blue')
plt.plot(dates[len(train):], test_predict, label='Predicted Test Sales', color='orange')
plt.title('Test Data and Predictions')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
