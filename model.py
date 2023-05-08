import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
data_path = "/home/yhz/Downloads/02-14-2018.csv"
data = pd.read_csv(data_path)

# Convert the timestamp to hour of the day
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M:%S')
data['Time'] = data['Timestamp'].dt.hour
data = data.drop(['Timestamp'], axis=1)

# Preprocess the data
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
num_classes = len(le.classes_)

# Replace infinity values with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Fill NaN values with the mean or median of the corresponding column
data = data.fillna(data.mean())

# Normalize the dataset
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Split the dataset into training and testing sets
X = data.drop(['Label'], axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Reshape the input to fit LSTM
X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

# Create the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=-1)
y_test = np.argmax(y_test, axis=-1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
