import pandas as pd
import numpy as np
import json
from kafka import KafkaConsumer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

# Define the LabelEncoder and MinMaxScaler from the previous steps
le = LabelEncoder()
le.classes_ = np.load('labelencoder_classes.npy', allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('minmax_scaler_params.npy', allow_pickle=True)

# Set up the Kafka consumer
consumer = KafkaConsumer('ids', bootstrap_servers='localhost:9092', value_deserializer=lambda m: json.loads(m.decode('utf-8')))

@app.route('/events', methods=['GET'])
def get_events():
    event_data = []
    try:
        i = 0
        counter = 10  # Change this value to control the number of events you want to process

        while counter > 0:
            msg_pack = consumer.poll(timeout_ms=1000)

            for tp, messages in msg_pack.items():
                for message in messages:
                    print(f"Received event", i)
                    i += 1

                    event = message.value

                    event = pd.DataFrame([event])

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

                    result = {
                        'event': message.value,
                        'Prediction': le.inverse_transform(y_pred)[0]
                    }

                    event_data.append(result)
                    # print("JSON data:", json.dumps({"events": event_data}))

                    counter -= 1
                    if counter <= 0:
                        break

        return jsonify(event_data)
    except Exception as e:
        print(f"Error: {e}")
        error_message = {'error': str(e)}
        return jsonify(error_message), 500

# @app.route('/events', methods=['GET'])
# def get_events():
#     event_data = []
#     try:
#         i = 0

#         for msg in consumer:
#             print(f"Received event", i)
#             i += 1
            
#             event = msg.value

#             event = pd.DataFrame([msg.value])

#             # Preprocess the event
#             event['Timestamp'] = pd.to_datetime(event['Timestamp'], format='%d/%m/%Y %H:%M:%S')
#             event['Time'] = event['Timestamp'].dt.hour
#             event = event.drop(['Timestamp'], axis=1)
#             event['Label'] = le.transform(event['Label'])
#             event = event.replace([np.inf, -np.inf], np.nan)
#             event = event.fillna(event.mean())
#             event[event.columns] = scaler.transform(event[event.columns])

#             # Predict the label
#             X = event.drop(['Label'], axis=1)
#             X = np.array(X).reshape(X.shape[0], 1, X.shape[1])
#             y_pred_probs = model.predict(X)
#             y_pred = np.argmax(y_pred_probs, axis=-1)

#             # If the predicted label is not 'BENIGN', print the anomaly
#             if y_pred[0] != le.transform(['Benign'])[0]:
#                 print(f"Anomaly detected: {le.inverse_transform(y_pred)[0]}")

#             result = {
#                 'event': msg.value,
#                 'prediction': le.inverse_transform(y_pred)[0]
#             }

#             event_data.append(result)
#             # print("JSON data:", json.dumps({"events": event_data}))
            
#             if len(event_data) >= 100:
#                 break

#         return jsonify(event_data)
#     except Exception as e:
#         print(f"Error: {e}")
#         return str(e), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
