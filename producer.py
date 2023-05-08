import pandas as pd
import time
from kafka import KafkaProducer
import json

data_path = "/home/yhz/6895/test.csv"
data = pd.read_csv(data_path)

# print("finish reading.")
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda v: json.dumps(v).encode('utf-8'))

i = 0
# print("start sending...")
for index, row in data.iterrows():
    event = row.to_dict()
    producer.send('ids', event)
    print(f"sent", i)
    i += 1
    time.sleep(0.5)

producer.close()
