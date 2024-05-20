import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('train.csv')
column_list = df['close'].tolist()

# Assuming you have your data in a list or numpy array called 'prices'
# where prices[i] represents the price at time i.

# Example data generation (replace with your actual data)
np.random.seed(0)
#prices = np.random.randint(100, 1000, 1000)  # Random prices between 100 and 200 for 1000 time points
prices = np.array(column_list)
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

print("Predicted price:", next_5_prices)
