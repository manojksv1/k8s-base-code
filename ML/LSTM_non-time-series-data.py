import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample line break chart data (replace this with your actual data)
# For demonstration purposes, I'm generating random data
np.random.seed(42)
data_length = 100
line_break_chart_data = np.random.rand(data_length)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(line_break_chart_data.reshape(-1, 1))

# Function to create dataset for LSTM
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Hyperparameters
time_steps = 10
epochs = 100
batch_size = 16

# Create dataset
X, y = create_dataset(normalized_data, time_steps)

# Reshape input data for LSTM (samples, time_steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Predict future values
future_time_steps = 10
future_predictions = []
current_batch = X[-1].reshape((1, time_steps, 1))

for i in range(future_time_steps):
    current_prediction = model.predict(current_batch)[0]
    future_predictions.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis=1)

# Inverse transform predictions to original scale
future_predictions = scaler.inverse_transform(future_predictions)

# Plot original data and future predictions
plt.plot(line_break_chart_data, label='Original Data')
plt.plot(np.arange(data_length, data_length + future_time_steps), future_predictions, label='Future Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Line Break Chart Future Prediction using LSTM')
plt.legend()
plt.show()

#***********************************************************************************************************************************************************************

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # Generate sample line break chart data (replace this with your actual data)
# # For demonstration purposes, I'm generating random data
# np.random.seed(42)
# data_length = 100
# line_break_chart_data = np.random.rand(data_length)

# # Normalize the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# normalized_data = scaler.fit_transform(line_break_chart_data.reshape(-1, 1))

# # Function to create dataset for LSTM
# def create_dataset(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:(i + time_steps)])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# # Hyperparameters
# time_steps = 10
# epochs = 100
# batch_size = 16

# # Create dataset
# X, y = create_dataset(normalized_data, time_steps)

# # Reshape input data for LSTM (samples, time_steps, features)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build LSTM model
# model = Sequential([
#     LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
#     LSTM(units=50),
#     Dense(units=1)
# ])

# # Compile model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train model
# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# # Predictions on test data
# predictions = model.predict(X_test)

# # Inverse transform predictions and actual values to original scale
# predictions = scaler.inverse_transform(predictions)
# y_test_original = scaler.inverse_transform(y_test)

# # Calculate accuracy (you may use other metrics depending on your task)
# mse = np.mean((predictions - y_test_original) ** 2)
# accuracy = 100 - mse  # Example metric, higher values indicate better accuracy

# print("Mean Squared Error (MSE):", mse)
# print("Accuracy:", accuracy)

# # Plot actual vs predicted values
# plt.plot(y_test_original, label='Actual Data')
# plt.plot(predictions, label='Predicted Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Line Break Chart Future Prediction using LSTM (Test Data)')
# plt.legend()
# plt.show()


#***********************************************************************************************************************************************************************

#                                                                ***
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # Generate sample line break chart data (replace this with your actual data)
# # For demonstration purposes, I'm generating random data
# np.random.seed(42)
# data_length = 1000
# line_break_chart_data = np.random.rand(data_length)

# # Normalize the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# normalized_data = scaler.fit_transform(line_break_chart_data.reshape(-1, 1))

# # Function to create dataset for LSTM
# def create_dataset(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:(i + time_steps)])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# # Hyperparameters
# time_steps = 10
# epochs = 200
# batch_size = 16

# # Create dataset
# X, y = create_dataset(normalized_data, time_steps)

# # Reshape input data for LSTM (samples, time_steps, features)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build LSTM model
# model = Sequential([
#     LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)),
#     Dropout(0.2),
#     LSTM(units=100),
#     Dropout(0.2),
#     Dense(units=1)
# ])

# # Compile model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train model
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# # Predictions on test data
# predictions = model.predict(X_test)

# # Inverse transform predictions and actual values to original scale
# predictions = scaler.inverse_transform(predictions)
# y_test_original = scaler.inverse_transform(y_test)

# # Calculate accuracy (you may use other metrics depending on your task)
# mse = np.mean((predictions - y_test_original) ** 2)
# accuracy = 100 - mse  # Example metric, higher values indicate better accuracy

# print("Mean Squared Error (MSE):", mse)
# print("Accuracy:", accuracy)

# # Plot actual vs predicted values
# plt.plot(y_test_original, label='Actual Data')
# plt.plot(predictions, label='Predicted Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Line Break Chart Future Prediction using LSTM (Test Data)')
# plt.legend()
# plt.show()

#***********************************************************************************************************************************************************************
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # Generate sample time series data (replace this with your actual data)
# # For demonstration purposes, I'm generating random data
# np.random.seed(42)
# data_length = 100
# time_series_data = np.random.randn(data_length)

# # Function to create dataset for LSTM
# def create_dataset(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:(i + time_steps)])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# # Hyperparameters
# time_steps = 10
# epochs = 200
# batch_size = 16

# # Create dataset
# X, y = create_dataset(time_series_data, time_steps)

# # Reshape input data for LSTM (samples, time_steps, features)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build LSTM model
# model = Sequential([
#     LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)),
#     Dropout(0.2),
#     LSTM(units=100),
#     Dropout(0.2),
#     Dense(units=1)
# ])

# # Compile model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train model
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# # Predictions on test data
# predictions = model.predict(X_test)

# # Plot actual vs predicted values
# plt.plot(y_test, label='Actual Data')
# plt.plot(predictions, label='Predicted Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Time Series Future Prediction using LSTM (Test Data)')
# plt.legend()
# plt.show()

