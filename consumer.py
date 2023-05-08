import json
from kafka import KafkaConsumer

# Set up the Kafka consumer
consumer = KafkaConsumer('ids', bootstrap_servers='localhost:9092', value_deserializer=lambda m: json.loads(m.decode('utf-8')))

i = 0
# Process the events and print the received messages
for msg in consumer:
    event = msg.value
    print(f"Received event {event}")
