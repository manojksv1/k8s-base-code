import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  output import a

# Assuming you have your data in a list or numpy array called 'prices'
# where prices[i] represents the price at time i.

# Example data generation (replace with your actual data)
np.random.seed(0)
#prices = np.random.randint(100, 1000, 1000)  # Random prices between 100 and 200 for 1000 time points
prices = np.array(a)
# Function to create dataset for regression
def create_dataset(prices, window_size=200):
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    return np.array(X), np.array(y)

# Create dataset
X, y = create_dataset(prices)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict next 5 prices
last_200_prices = prices[-200:].reshape(1, -1)  # Reshape to match the input format for prediction
next_5_prices = model.predict(last_200_prices)

print("Predicted next 5 prices:", next_5_prices)

print("*"*77)

next_5_prices = []
current_prices = prices[-200:]  # Initial 200 prices

for _ in range(5):
    # Predict the next price based on the current 200 prices
    next_price = model.predict(current_prices.reshape(1, -1))
    print("Predicted price:", next_price)
    next_5_prices.append(next_price[0])
    
    # Update the current prices by removing the oldest price and adding the predicted price
    current_prices = np.roll(current_prices, -1)
    current_prices[-1] = next_price[0]
