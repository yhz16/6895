import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Load the dataset
data_path = "/home/yhz/Downloads/02-14-2018.csv"
data = pd.read_csv(data_path)

# Convert the timestamp to hour of the day
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M:%S')
data['Time'] = data['Timestamp'].dt.hour
data = data.drop(['Timestamp'], axis=1)

# Step 3: Preprocess the data
# Convert categorical features to numeric
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Replace infinity values with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Fill NaN values with the mean or median of the corresponding column
data = data.fillna(data.mean())

# Normalize the dataset
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Step 4: Split the dataset into training and testing sets
X = data.drop(['Label'], axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input to fit LSTM
X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

# Step 5: Create the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Step 7: Evaluate the model
y_pred = model.predict_classes(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
