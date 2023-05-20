# This is a basic workflow to help you get started with Actions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv('Apple Stocks.csv')

# Reverse the DataFrame to read the data in reverse order
df_reverse = df.iloc[::-1].reset_index(drop=True)

# Extract the 'Date' and 'Close' columns from the reversed DataFrame
dates = pd.to_datetime(df_reverse['Date'])
prices = df_reverse['Close']

# Convert the prices to a numpy array
prices = np.array(prices).reshape(-1, 1)

# Scale the prices using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Split the data into training and testing sets
train_size = int(len(scaled_prices) * 0.8)
test_size = len(scaled_prices) - train_size
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

def prepare_data(data, num_steps):
    X, y = [], []
    for i in range(len(data) - num_steps):
        X.append(data[i : (i + num_steps)])
        y.append(data[i + num_steps])
    return np.array(X), np.array(y)

# Define the number of time steps for the LSTM model
num_steps = 30

# Prepare the training and testing data
X_train, y_train = prepare_data(train_data, num_steps)
X_test, y_test = prepare_data(test_data, num_steps)

model = Sequential()
model.add(LSTM(64, input_shape=(num_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Combine the training and testing data for making predictions
combined_data = np.concatenate((train_data, test_data), axis=0)
inputs = combined_data[-num_steps:]
predicted_prices = []

# Generate predictions for the next thirty days
for _ in range(30):
    inputs_reshaped = np.reshape(inputs, (1, num_steps, 1))
    prediction = model.predict(inputs_reshaped)
    predicted_prices.append(prediction)
    inputs = np.append(inputs[1:], prediction)

# Inverse transform the predicted prices to their original scale
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Generate predictions for the next thirty days
future_dates = pd.date_range(start=dates.iloc[-1], periods=30, freq='D')
future_inputs = test_data[-num_steps:]

predicted_prices = []

for _ in range(30):
    future_inputs_reshaped = np.reshape(future_inputs, (1, num_steps, 1))
    future_prediction = model.predict(future_inputs_reshaped)
    predicted_prices.append(future_prediction)
    future_inputs = np.append(future_inputs[1:], future_prediction)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Create a DataFrame with future dates and predicted prices
predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})

# Display the predicted prices for the next thirty days
print(predictions_df)
