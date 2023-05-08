import pandas as pd
import numpy as np
import json
from kafka import KafkaConsumer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the saved model
model = load_model('model.h5')

# Define the LabelEncoder and MinMaxScaler from the previous steps
le = LabelEncoder()
le.classes_ = np.load('labelencoder_classes.npy', allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('minmax_scaler_params.npy', allow_pickle=True)

# Set up the Kafka consumer
consumer = KafkaConsumer('ids', bootstrap_servers='localhost:9092', value_deserializer=lambda m: json.loads(m.decode('utf-8')))

i = 0
# Process the events and print the received messages
for msg in consumer:
    event = msg.value
    # print(f"Received event {event}")

    event = pd.DataFrame([msg.value])
    
    # Preprocess the event
    event['Timestamp'] = pd.to_datetime(event['Timestamp'], format='%d/%m/%Y %H:%M:%S')
    event['Time'] = event['Timestamp'].dt.hour
    event = event.drop(['Timestamp'], axis=1)
    event['Label'] = le.transform(event['Label'])
    event = event.replace([np.inf, -np.inf], np.nan)
    event = event.fillna(event.mean())
    event[event.columns] = scaler.transform(event[event.columns])
    
    # Predict the label
    X = event.drop(['Label'], axis=1)
    X = np.array(X).reshape(X.shape[0], 1, X.shape[1])
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    # If the predicted label is not 'BENIGN', print the anomaly
    if y_pred[0] != le.transform(['Benign'])[0]:
        print(f"Anomaly detected: {le.inverse_transform(y_pred)[0]}")
