import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv('train.csv')
column_list = df['close'].tolist()

    
# Assuming you have your data in a list or numpy array called 'prices'
# where prices[i] represents the price at time i.

# Example data generation (replace with your actual data)
np.random.seed(0)
prices = np.array(column_list)#np.random.randint(100, 1000, 1000)  # Random prices between 100 and 200 for 1000 time points
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Function to create dataset for LSTM
def create_dataset(prices, look_back=200):
    X, y = [], []
    for i in range(len(prices) - look_back):
        X.append(prices[i:i+look_back])
        y.append(prices[i+look_back])
    return np.array(X), np.array(y)

# Create dataset
X, y = create_dataset(prices_scaled)

# Reshape input data to be 3D for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting the dataset into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict next 5 prices
last_200_prices = prices_scaled[-200:].reshape(1, -1, 1)
next_5_prices_scaled = model.predict(last_200_prices)
next_5_prices = scaler.inverse_transform(next_5_prices_scaled)

print("Next Price is = ", next_5_prices.flatten())

# print("#"*77)

# next_5_prices_scaled = []
# current_prices = prices_scaled[-200:].reshape(1, -1, 1)

# for _ in range(5):
#     next_price_scaled = model.predict(current_prices)
#     next_5_prices_scaled.append(next_price_scaled[0, 0])  # Append the predicted price
#     current_prices = np.roll(current_prices, -1, axis=1)  # Roll the array to remove the oldest price
#     current_prices[0, -1, 0] = next_price_scaled  # Add the predicted price at the end

# # Inverse transform the scaled prices to get the actual prices
# next_5_prices = scaler.inverse_transform(np.array(next_5_prices_scaled).reshape(-1, 1))

# print("Predicted next 5 prices:", next_5_prices.flatten())